from functools import partial
from jax import jit, lax, random, vmap
import jax
import jax.numpy as jnp
from numpy import ndarray
import chompax as cx
from .structures import RRTParams, Tree
from dataclasses import dataclass
from .traj import resample_along_traj
from jax.typing import ArrayLike

class JitableRRT:

    @staticmethod
    @jit
    def get_init_tree(q_init: ArrayLike, params: RRTParams):
        tree = (
            jnp.ones((params.max_iter * params.leaf_batch_size, params.ri.dim))
            * jnp.inf
        )
        tree = tree.at[0].set(q_init)
        edges = (
            jnp.ones((params.max_iter * params.leaf_batch_size, 2), dtype=jnp.int32)
            * -1
        )
        num_nodes = 1
        num_edges = 0
        return Tree(tree, edges, num_nodes, num_edges)

    @staticmethod
    @jit
    def add_node_to_tree(node: ArrayLike, tree: Tree, connection_node_idx: int):
        tree.tree = tree.tree.at[tree.num_nodes].set(node)
        tree.edges = tree.edges.at[tree.num_edges].set(
            jnp.array([connection_node_idx, tree.num_nodes], dtype=jnp.int32)
        )
        tree.num_nodes += 1
        tree.num_edges += 1
        return tree

    @staticmethod
    @jit
    def extend(sample: ArrayLike, tree: Tree, sdf: ArrayLike, params: RRTParams):
        # will produce inf for uninitialized nodes
        half_db_norm_sq = jnp.linalg.norm(tree.tree, axis=1) ** 2 / 2
        # will produce inf-dists for uninitialized nodes
        nearest_idx = JitableRRT.ann_l2(sample, tree.tree, half_db_norm_sq)
        nearest_node = tree.tree[nearest_idx]
        new_node, trapped = JitableRRT.steer(nearest_node, sample, params)
        tree = lax.cond(
            JitableRRT.is_collision_free(nearest_node, new_node, sdf, params),
            lambda _: JitableRRT.add_node_to_tree(new_node, tree, nearest_idx),
            lambda _: tree,
            None,
        )
        return tree, trapped

    @staticmethod
    @jit
    def ann_l2(query: ArrayLike, db: ArrayLike, half_db_norms: ArrayLike):
        dists = half_db_norms - lax.dot(query, db.transpose())
        return jnp.nanargmin(dists)

    @staticmethod
    @jit
    def steer(q_from, q_to, params):
        step_size = params.step_size
        direction = q_to - q_from
        distance = jnp.linalg.norm(direction, ord=jnp.inf)
        return (
            lax.cond(
                distance < step_size,
                lambda dir: q_to,
                lambda dir: q_from + step_size * (dir / distance),
                direction,
            ),
            distance < step_size,
        )

    @staticmethod
    @jit
    def is_collision_free(q1, q2, sdf, params):
        pseudo_traj = jnp.linspace(q1, q2, params.coll_check_interval)
        return JitableRRT.is_trajectory_feasible(pseudo_traj, sdf, params)

    @staticmethod
    @jit
    def grow_tree(
        q_init: ArrayLike,
        q_goal: ArrayLike,
        sdf: ArrayLike,
        key: ArrayLike,
        params: RRTParams,
    ):
        _, key = random.split(key)
        tree = JitableRRT.get_init_tree(q_init, params)

        def cond_fun(carry):
            _, goal_reached, iters, _ = carry
            return jnp.logical_not(goal_reached) & (iters < params.max_iter)

        def body_fun(carry):
            tree, goal_reached, iters, key = carry
            tree = JitableRRT.sample_and_extend_batch(key, tree, sdf, params)
            sample = lax.cond(
                iters % params.goal_sampling_interval == 0,
                lambda _: q_goal,
                lambda _: JitableRRT.get_valid_sample(sdf, key, params),
                operand=None,
            )  # random_samples[iters]

            tree, _ = JitableRRT.extend(sample, tree, sdf, params)

            goal_reached = jnp.allclose(tree.tree[tree.num_nodes - 1], q_goal)
            _, key = random.split(key)
            return tree, goal_reached, iters + 1, key

        carry = (tree, False, 0, key)
        tree, goal_reached, _, _ = lax.while_loop(cond_fun, body_fun, carry)
        return tree, goal_reached

    @staticmethod
    def run(
        q_init: ArrayLike,
        q_goal: ArrayLike,
        sdf: ArrayLike,
        key: ArrayLike,
        params: RRTParams,
    ):
        tree, goal_reached = JitableRRT.grow_tree(q_init, q_goal, sdf, key, params)
        if goal_reached:
            path, path_slice = JitableRRT.get_path(tree)
            return path[slice(*path_slice)]
        return None

    @staticmethod
    @jit
    def sample_and_extend_batch(
        key: ArrayLike, tree: Tree, sdf: ArrayLike, params: RRTParams
    ):
        keys = random.split(key, params.leaf_batch_size)
        samples = JitableRRT.get_valid_samples(sdf, keys, params)
        half_db_norm_sq = jnp.linalg.norm(tree.tree, axis=1) ** 2 / 2
        # will produce inf-dists for uninitialized nodes

        nearest_ids = vmap(JitableRRT.ann_l2, in_axes=(0, None, None))(
            samples, tree.tree, half_db_norm_sq
        )
        nearest_nodes = tree.tree[nearest_ids]
        new_nodes, _ = vmap(JitableRRT.steer, in_axes=(0, 0, None))(
            nearest_nodes, samples, params
        )
        collision_free = vmap(JitableRRT.is_collision_free, in_axes=(0, 0, None, None))(
            nearest_nodes, new_nodes, sdf, params
        )

        # sizes must be static, hence we introduces infs and resort them later so that it all aligns nicely.
        new_nodes = jnp.where(collision_free[:, None], new_nodes, jnp.inf)
        nearest_nodes = jnp.where(collision_free[:, None], nearest_nodes, jnp.inf)
        nearest_ids = jnp.where(collision_free, nearest_ids, -1)
        # we resort new_nodes so that all infs are at the end. Then we apply the same sorting to nearest_ids and nearest_nodes
        sort_idx = jnp.argsort(jnp.any(jnp.isinf(new_nodes), axis=1))
        new_nodes = new_nodes[sort_idx]
        nearest_nodes = nearest_nodes[sort_idx]
        nearest_ids = nearest_ids[sort_idx]

        root_node_id = tree.num_nodes
        root_edge_id = tree.num_edges
        # tree.tree = tree.tree.at[root_id:root_id+params.leaf_batch_size].set(new_nodes)
        tree.tree = lax.dynamic_update_slice(
            tree.tree, new_nodes, (root_node_id, root_node_id + params.leaf_batch_size)
        )
        edges = jnp.ones((params.leaf_batch_size, 2), dtype=jnp.int32) * -1
        edges = edges.at[:, 0].set(nearest_ids)
        new_ids = lax.iota(jnp.int32, params.leaf_batch_size) + root_node_id
        edges = edges.at[:, 1].set(
            lax.dynamic_update_slice_in_dim(edges[:, 1], new_ids, 0, 0)
        )
        tree.edges = lax.dynamic_update_slice(
            tree.edges, edges, (root_edge_id, root_edge_id + params.leaf_batch_size)
        )
        num_new_nodes = jnp.sum(jnp.any(~jnp.isinf(new_nodes), axis=1))
        tree.num_nodes += num_new_nodes
        tree.num_edges += num_new_nodes
        return tree

    @staticmethod
    @jit
    def get_path(tree):
        """
        Reconstruct the path from the tree and edges.

        Args:
            tree: The tree containing the nodes.
            edges: The edges connecting the nodes.

        Returns:
            path: The reconstructed path.
            path_idx: The index of the path.
        """
        # reconstruct path in a jit compatible way
        edges = tree.edges
        tree = tree.tree
        path = jnp.ones_like(tree) * jnp.inf
        path = path.at[-1].set(tree[-1])

        def cond_fun(carry):
            _, node_idx, _, _ = carry
            return node_idx > 0

        def body_fun(carry):
            path, _, edge_idx, path_idx = carry
            parent_node_idx, child_node_idx = edges[edge_idx]
            path = path.at[path_idx].set(tree[parent_node_idx])
            # find first edge that ends in the parent node
            # edge_idx = jnp.argmin(jnp.where(edges[:, 1] == parent_node_idx, 0, jnp.inf))
            edge_idx = jnp.flatnonzero(edges[:, 1] == parent_node_idx, size=1).squeeze()

            return path, parent_node_idx, edge_idx, path_idx - 1

        # find last node and last edge in the tree without using num_nodes and num_edges
        path_idx = path.shape[0] - 1
        node_idx = jnp.searchsorted(~jnp.all(jnp.isfinite(tree), axis=1), True) - 1
        edge_idx = jnp.searchsorted(~jnp.all(edges >= 0, axis=1), True) - 1

        path = path.at[path_idx].set(tree[node_idx])
        carry = (path, node_idx, edge_idx, path_idx - 1)
        path, _, _, _ = lax.while_loop(cond_fun, body_fun, carry)
        path_idx = jnp.searchsorted(jnp.all(jnp.isfinite(path), axis=1), True)
        start_idx = path_idx
        stop_idx = len(path)
        # path = path[path_idx:]
        return path, (start_idx, stop_idx)

    @staticmethod
    @jit
    def is_trajectory_feasible(traj, sdf, params):
        feas = vmap(cx.is_configuration_feasible, in_axes=(0, None, None, None))(
            traj, sdf, params.ri, params.wi
        )
        return jnp.all(feas)

    @staticmethod
    @jit
    def get_valid_sample(sdf, key, params):
        ri = params.ri
        wi = params.wi

        def is_feasible(q):
            return cx.is_configuration_feasible(q, sdf, ri, wi)

        def body_fun(carry):
            key, q = carry
            key, subkey = random.split(key)
            q = random.uniform(
                subkey, (ri.dim,), minval=ri.limits[:, 0], maxval=ri.limits[:, 1]
            )
            return key, q

        def cond_fun(carry):
            key, q = carry
            return jnp.logical_not(is_feasible(q))

        q = random.uniform(
            key, (ri.dim,), minval=ri.limits[:, 0], maxval=ri.limits[:, 1]
        )
        carry = (key, q)
        _, q = lax.while_loop(cond_fun, body_fun, carry)
        return q

    @staticmethod
    @jit
    def get_valid_samples(sdf, keys, params):
        return vmap(JitableRRT.get_valid_sample, in_axes=(None, 0, None))(
            sdf, keys, params
        )


class JitableRRTConnect(JitableRRT):

    @staticmethod
    @jit
    def grow_tree(
        q_init: ArrayLike,
        q_goal: ArrayLike,
        sdf: ArrayLike,
        key: ArrayLike,
        params: RRTParams,
    ):
        key, subkey = random.split(key, 2)
        tree_start = JitableRRT.get_init_tree(q_init, params)
        tree_goal = JitableRRT.get_init_tree(q_goal, params)
        # combine the trees so that they are a struct of arrays
        tree = JitableRRTConnect.repack_trees(tree_start, tree_goal)

        def sample_treewise(key, tree):  # -> tuple[Any, Any, Tree, Tree] | None:
            node_ids = random.randint(
                key, (2,), minval=jnp.array([0, 0]), maxval=jnp.array(tree.num_nodes)
            )
            sample_start = tree.tree[0][node_ids[0]]
            sample_goal = tree.tree[1][node_ids[1]]
            samples = jnp.stack(
                (sample_goal, sample_start)
            )  # Note that the order is reversed here!
            tree, connection = vmap(
                JitableRRTConnect.connect, in_axes=(0, 0, None, None)
            )(samples, tree, sdf, params)
            tree_start, tree_goal = JitableRRTConnect.unpack_trees(tree)
            start_connected_to_goal, goal_connected_to_start = connection
            tree_start, tree_goal, reverse, done = lax.cond(
                start_connected_to_goal,
                lambda _: (
                    JitableRRTConnect.unite_trees(tree_start, tree_goal, node_ids[1])[
                        0
                    ],
                    tree_goal,
                    False,
                    True,
                ),
                lambda _: lax.cond(
                    goal_connected_to_start,
                    lambda _: (
                        tree_start,
                        JitableRRTConnect.unite_trees(
                            tree_goal, tree_start, node_ids[0]
                        )[0],
                        True,
                        True,
                    ),
                    lambda _: (tree_start, tree_goal, False, False),
                    operand=None,
                ),
                operand=None,
            )
            return JitableRRTConnect.repack_trees(tree_start, tree_goal), reverse, done

        def sample_random(key, tree):
            keys = random.split(key, 2)
            tree = vmap(JitableRRT.sample_and_extend_batch, in_axes=(0, 0, None, None))(
                keys, tree, sdf, params
            )
            # samples = vmap(JitableRRTConnect.get_valid_sample, in_axes=(None, 0, None))(sdf, keys, params)
            # tree, _ = vmap(JitableRRTConnect.extend, in_axes=(0, 0, None, None))(samples, tree, sdf, params)
            reverse = False
            return tree, reverse, False

        def cond_fun(carry):
            # break when max iters is reached or trees are connected (done)
            done, iters, *_ = carry
            return jnp.logical_not(done) & (iters < params.max_iter)

        def body_fun(carry):
            done, iters, key, tree, reverse = carry
            tree, reverse, done = lax.cond(
                (iters + 1) % params.goal_sampling_interval == 0,
                lambda _: sample_treewise(key, tree),
                lambda _: sample_random(key, tree),
                operand=None,
            )
            key, subkey = random.split(key, 2)
            return done, iters + 1, subkey, tree, reverse

        carry = (False, 0, subkey, tree, False)
        done, iters, _, tree, reverse = lax.while_loop(cond_fun, body_fun, carry)
        return *JitableRRTConnect.unpack_trees(tree), done, reverse

    @staticmethod
    def run(
        q_init: ArrayLike,
        q_goal: ArrayLike,
        sdf: ArrayLike,
        key: ArrayLike,
        params: RRTParams,
    ):
        tree_start, tree_goal, goal_reached, reverse = JitableRRTConnect.grow_tree(
            q_init, q_goal, sdf, key, params
        )
        if goal_reached:
            if reverse:
                path, (path_idx, _) = JitableRRT.get_path(tree_goal)
                path = jnp.flip(path, axis=0)
                # path = path.at[path_idx:].set(rev_path[:len(path)-path_idx])
                start_idx = 0
                stop_idx = len(path) - path_idx
            else:
                path, (start_idx, stop_idx) = JitableRRT.get_path(tree_start)
            return path[slice(start_idx, stop_idx)]
        return None

    @staticmethod
    @jit
    def unpack_trees(tree_pair):
        tree_start = Tree(
            tree_pair.tree[0],
            tree_pair.edges[0],
            tree_pair.num_nodes[0],
            tree_pair.num_edges[0],
        )
        tree_goal = Tree(
            tree_pair.tree[1],
            tree_pair.edges[1],
            tree_pair.num_nodes[1],
            tree_pair.num_edges[1],
        )
        return tree_start, tree_goal

    @staticmethod
    @jit
    def repack_trees(tree_start, tree_goal):
        tree = Tree(
            jnp.stack((tree_start.tree, tree_goal.tree)),
            jnp.stack((tree_start.edges, tree_goal.edges)),
            jnp.array([tree_start.num_nodes, tree_goal.num_nodes]),
            jnp.array([tree_start.num_edges, tree_goal.num_edges]),
        )
        return tree

    @staticmethod
    @jit
    def connect(sample: ArrayLike, tree: Tree, sdf: ArrayLike, params: RRTParams):
        """
        Attempts to connect the given sample to the tree until the tree is trapped or the goal is reached.
        Args:
            sample (jnp.ndarray): The sample point to connect to the tree.
            tree (Tree): The current tree structure.
            sdf (jnp.ndarray): The signed distance field representing the environment.
            params (RRTParams): Parameters for the RRT algorithm.
        Returns:
            tuple: A tuple containing the updated tree, a boolean indicating if the tree is trapped,
                   and a boolean indicating if the goal has been reached.
        """

        def cond_fun(carry):
            _, trapped, goal_reached, i = carry
            return jnp.logical_not(trapped) & jnp.logical_not(goal_reached) & (i < 100)

        def body_fun(carry):
            tree, trapped, goal_reached, i = carry
            tree, trapped = JitableRRT.extend(sample, tree, sdf, params)
            goal_reached = jnp.allclose(tree.tree[tree.num_nodes - 1], sample)
            return tree, trapped, goal_reached, i + 1

        carry = (tree, False, False, 0)
        tree, _, goal_reached, _ = lax.while_loop(cond_fun, body_fun, carry)
        return tree, goal_reached

    @staticmethod
    def unite_trees(tree_from: Tree, tree_to: Tree, connect_node_id_to: int):
        """
        Unite two trees by connecting a node from one tree to another.

        This function takes two trees and connects a node from `tree_to` to `tree_from`
        by following the parent nodes in `tree_to` until the root is reached.
        The nodes from `tree_to` are added to `tree_from` during this process.

        Args:
            tree_from (Tree): The tree to which nodes will be added.
            tree_to (Tree): The tree from which nodes will be taken.
            connect_node_id_to (int): The ID of the node in `tree_to` that will be connected to `tree_from`.

        Returns:
            Tuple[Tree, Tree]: A tuple containing the updated `tree_from` and `tree_to`.
        """

        def cond_fun(carry):
            parent_idx_to, *_ = carry
            return parent_idx_to > 0  # stop if parent is root

        def body_fun(carry):
            parent_idx_to, tree_from, tree_to = carry
            # edge_idx_to = jnp.argmin(
            #     jnp.where(tree_to.edges[:, 1] == parent_idx_to, 0, jnp.inf)
            # )
            edge_idx_to = jnp.flatnonzero(
                tree_to.edges[:, 1] == parent_idx_to, size=1
            ).squeeze()
            parent_idx_to, _ = tree_to.edges[edge_idx_to]
            target_node = tree_to.tree[parent_idx_to]
            tree_from = JitableRRT.add_node_to_tree(
                target_node, tree_from, tree_from.num_nodes - 1
            )
            return parent_idx_to, tree_from, tree_to

        carry = (connect_node_id_to, tree_from, tree_to)
        carry = lax.while_loop(cond_fun, body_fun, carry)
        _, tree_from, tree_to = carry
        return tree_from, tree_to

    @staticmethod
    @jit
    def find_leafs(tree: Tree):
        """
        Find the leaf nodes in the tree.

        Args:
            tree (Tree): The tree to search for leaf nodes.

        Returns:
            jnp.ndarray: The indices of the leaf nodes.
        """
        # leaf nodes are nodes with no children -> no outgoing edges.
        # edges are stores as (from_id,  to_id). Hence, we need to find the nodes that are not in the from_id column, but are less than tree.num_nodes
        node_ids = jnp.arange(len(tree.tree))  # needs to be statically sized...
        node_ids = jnp.where(node_ids < tree.num_nodes, node_ids, -1)
        return jnp.setdiff1d(
            node_ids, tree.edges[:, 0], size=len(tree.tree), fill_value=-1
        )

    @staticmethod
    @partial(jit, static_argnames=("num_starts",))
    def __generate_multistarts(
        start: ArrayLike,
        goal: ArrayLike,
        num_starts: int,
        sdf: ArrayLike,
        key: ArrayLike,
        params: RRTParams,
    ):
        # we try to find a path using RRT-Connect. We grow the trees for set number of iterations and then randomly sample connection pairs from the trees.
        # these then form the multi-starts for the subsequent chomp run.
        key, subkey = random.split(key)
        tree_start, tree_goal, goal_reached, reverse = JitableRRTConnect.grow_tree(
            start, goal, sdf, key, params
        )
        # to get suitable leaf nodes, we sample from the goal tree with probability proportional to the inverse of the distance to the start
        # this is to ensure that we get nodes that are close to the other tree
        num_starts_1 = num_starts // 2
        num_starts_2 = num_starts - num_starts_1
        dists = jnp.linalg.norm(
            tree_goal.tree - start, axis=1
        )  # infs will be assigned zero probability
        probs = 1 / (dists + 1e-6)
        probs /= jnp.sum(probs)
        leaf_ids_goal = random.choice(
            subkey, tree_goal.tree.shape[0], (num_starts_1,), replace=False, p=probs
        )

        # infs will be assigned zero probability
        dists = jnp.linalg.norm(tree_start.tree - goal, axis=1)
        probs = 1 / (dists + 1e-6)
        probs /= jnp.sum(probs)
        leaf_ids_start = random.choice(
            subkey, tree_start.tree.shape[0], (num_starts_2,), replace=False, p=probs
        )

        # for now, we connect different nodes from goal tree to one node in start tree. TODO: Connect leaves from both trees
        trees_start, trees_goal = vmap(
            JitableRRTConnect.unite_trees, in_axes=(None, None, 0)
        )(tree_start, tree_goal, leaf_ids_goal)
        paths, (start_ids, stop_ids) = vmap(JitableRRTConnect.get_path, in_axes=(0))(
            trees_start
        )

        # caution reversed order of trees
        trees_goal, trees_start = vmap(
            JitableRRTConnect.unite_trees, in_axes=(None, None, 0)
        )(tree_goal, tree_start, leaf_ids_start)
        paths_goal, (start_ids_goal, stop_ids_goal) = vmap(
            JitableRRTConnect.get_path, in_axes=(0, None)
        )(trees_goal, True)
        paths = jnp.concatenate((paths, paths_goal), axis=0)
        start_ids = jnp.concatenate((start_ids, start_ids_goal), axis=0)
        stop_ids = jnp.concatenate((stop_ids, stop_ids_goal), axis=0)

        return paths, (start_ids, stop_ids)

    @staticmethod
    def multi_starts(
        start: ArrayLike,
        goal: ArrayLike,
        num_starts: int,
        num_points: int,
        sdf: ArrayLike,
        key: ArrayLike,
        params: RRTParams,
    ):
        paths, (start_ids, stop_ids) = JitableRRTConnect.__generate_multistarts(
            start, goal, num_starts, sdf, key, params
        )
        paths_out = jnp.zeros((num_starts, num_points, params.ri.dim))

        for i in range(
            num_starts
        ):  # becasue of dynamic shapes, we need to loop over the paths
            paths_out = paths_out.at[i].set(
                resample_along_traj(
                    paths[i][slice(start_ids[i], stop_ids[i]), :], num_points
                )
            )

        return paths_out

    @staticmethod
    @partial(jit, static_argnames=("reverse"))
    def get_path(tree: Tree, reverse=False):
        """
        Reconstruct the path from the tree and edges.

        Args:
            tree: The tree containing the nodes.
            edges: The edges connecting the nodes.

        Returns:
            path: The reconstructed path.
            path_idx: The index of the path.
        """
        # reconstruct path in a jit compatible way
        edges = tree.edges
        tree = tree.tree
        path = jnp.ones_like(tree) * jnp.inf
        path = path.at[-1].set(tree[-1])

        def cond_fun(carry):
            _, node_idx, _, _ = carry
            return node_idx > 0

        def body_fun(carry):
            path, _, edge_idx, path_idx = carry
            parent_node_idx, child_node_idx = edges[edge_idx]
            path = path.at[path_idx].set(tree[parent_node_idx])
            # find first edge that ends in the parent node
            # edge_idx = jnp.argmin(jnp.where(edges[:, 1] == parent_node_idx, 0, jnp.inf))
            edge_idx = jnp.flatnonzero(edges[:, 1] == parent_node_idx, size=1).squeeze()
            return path, parent_node_idx, edge_idx, path_idx - 1

        # find last node and last edge in the tree without using num_nodes and num_edges
        path_idx = path.shape[0] - 1
        node_idx = jnp.searchsorted(~jnp.all(jnp.isfinite(tree), axis=1), True) - 1
        edge_idx = jnp.searchsorted(~jnp.all(edges >= 0, axis=1), True) - 1

        path = path.at[path_idx].set(tree[node_idx])
        path, _, _, _ = lax.while_loop(
            cond_fun, body_fun, (path, node_idx, edge_idx, path_idx - 1)
        )
        path_idx = jnp.searchsorted(jnp.all(jnp.isfinite(path), axis=1), True)
        if reverse:
            path = jnp.flip(path, axis=0)
            # path = path.at[path_idx:].set(rev_path[:len(path)-path_idx])
            start_idx = 0
            stop_idx = len(path) - path_idx
        else:
            start_idx = path_idx
            stop_idx = len(path)
        # path = path[path_idx:]
        return path, (start_idx, stop_idx)
