from .obs import *
from .traj import *
from .utils import *
from .optim import *
from .structures import *
from .twod_plotly import *
from .benchmark import *
from .limits import handle_joint_limits
from .obs import get_dists_nondiff, subtract_radius
from .traj import get_K_1, get_Abc, traj_cost
from .utils import construct_kinematic_chains
from .optim import get_A_inv, covariant_update
from .structures import WorldInfo, OptimInfo, RobotInfo, Spheres_Jax
from .twod_plotly import RobotVis2DPlotly
from .benchmark import sample_pose, compute_hardness_score
from jax import random, vmap, lax, jit
import jax.numpy as jnp
from functools import partial
from jax.typing import ArrayLike
from typing import Callable
from yaspin import yaspin


class Chompax:
    loop = None
    opt_state = None

    @staticmethod
    def setup(
        optim_info: OptimInfo,
        robot_info: RobotInfo,
        world_info: WorldInfo,
    ):
        """
        Combine the bits and pieces to set up the optimization problem
        Args:
            optim_info: Optimization information.
            robot_info: Robot information.
            world_info: World information.
        Returns:
            function: The optimization loop.
            opt_state: The optimizer state.
        """
        cost_fn = Chompax.get_cost_fun(robot_info, world_info, optim_info)
        # build example traj
        example_traj = jnp.zeros((optim_info.num_points, robot_info.dim))
        K_1 = get_K_1(robot_info, optim_info)
        A = lax.dot(K_1.T, K_1)
        A_inv = get_A_inv(A)
        opt, opt_states = Chompax.get_optimizer(
            example_traj, A_inv, robot_info, optim_info
        )
        optim_loop = Chompax.__get_optim_loop(
            cost_fn, opt, A_inv, optim_info, robot_info
        )
        return jit(vmap(optim_loop, in_axes=(0, 0, 0, None, None, None))), opt_states
    
    @staticmethod
    def run(start : ArrayLike, end : ArrayLike, world : ArrayLike, key : ArrayLike, robot_info : RobotInfo, optim_info : OptimInfo, world_info : WorldInfo, init_traj : ArrayLike = None, sdf : ArrayLike = None, grad_sdf : ArrayLike = None):
        """
        Run the optimization loop.
        Args:
            start : The start configuration.
            end : The end configuration.
            world : The world image.
            key : The random key for initialization.
            robot_info : Robot information.
            optim_info : Optimization information.
            world_info : World information.
            init_traj : Optional initial trajectory to use instead of random starts.
            sdf : Optional pre-computed signed distance field.
            grad_sdf : Optional pre-computed gradient of the signed distance field.
        Returns:
            Feasible trajectories, cost history, and cost.
        """
        sp = yaspin(text="Assembling...", timer=True)
        sp.start()
        lambdas = jnp.ones((optim_info.num_starts, optim_info.num_points)) * 200.0
        normalizers = Normalizers(traj_cost=1.0, col_cost=1.0) #no normalization
        
        # Use provided initial trajectory or generate random starts
        if init_traj is not None:
            # Repeat the initial trajectory for each start
            init_trajs = jnp.repeat(init_traj[None, ...], optim_info.num_starts, axis=0)
            #jax.debug.print("init_trajs: {}", init_trajs)
        else:
            init_trajs = get_random_starts(key, start, end, optim_info, robot_info)
        
        # Use provided SDF or compute it
        if sdf is None:
            sdf = Chompax.get_sdf(world, world_info)
        
        # Use provided gradient or compute it
        if grad_sdf is None:
            grad_sdf = Chompax.get_sdf_grad(sdf, world_info)

        if Chompax.loop is not None:
            sp.write("Reusing previous optimization loop.")
            optim_loop = Chompax.loop
            opt_states = Chompax.opt_state
        else:
            optim_loop, opt_states = Chompax.setup(
                optim_info, robot_info, world_info)
            Chompax.loop = optim_loop
            Chompax.opt_state = opt_states
        sp.stop()
        sp.ok("✨")

        sp.text = "Optimizing..."
        sp.start()
        trajs_optim, cost_hist, cost = optim_loop(
            init_trajs, lambdas, opt_states, sdf, grad_sdf, normalizers)
        feas = is_batch_feasible(trajs_optim, sdf,
                                    robot_info, world_info, optim_info)
        sp.stop()
        sp.ok("✨")

        sp.write(f"Optimization finished. Found {sum(feas)} feasible trajectories.")
        return trajs_optim, cost_hist, cost, feas
    
    @staticmethod
    def reset():
        Chompax.loop = None
        Chompax.opt_state = None

    @staticmethod
    def get_sdf(world: ArrayLike, world_info: WorldInfo):
        """
        Computes the signed distance field (SDF) for the current world.
        Calls into normal SciPy functions to compute the SDF (not differentiable, not jittable, not GPU compatible).

        The SDF is calculated using the `obstacle_img2dist_img` function, which
        takes the world and world information from the class as inputs and returns the SDF.
        Args:
            world : The world image.
            world_info : The world information.
        Returns:
            jnp.ndarray : The signed distance field of the world.
        """
        return obstacle_img2dist_img(world, world_info)

    @staticmethod
    def get_sdf_grad(sdf: ArrayLike, world_info: WorldInfo):
        """
        Computes the gradient of the signed distance function (SDF).

        Args:
            sdf : The signed distance field.
            world_info : The world information.

        Returns:
            grad_sdf : The gradient of the signed distance field.
        """
        grad_sdf = img_grad(sdf, world_info)
        return grad_sdf

    @staticmethod
    def get_cost_fun(
        robot_info: RobotInfo,
        world_info: WorldInfo,
        optim_info: OptimInfo,
    ):
        """
        Generates a cost function for a given trajectory based on signed distance fields (SDF) and their gradients.

        Returns:
            function: A cost function that takes a trajectory and a lambda parameter, and returns the computed cost.

            The cost function computes the following:
            - Distance of the trajectory points from obstacles using the signed distance field and the robot's spheres.
            - Collision cost based on the distance from obstacles per timestep and the moved distance between timesteps.
            - Computes trajectory cost using L2 norm of the velocity.
            - Returns the weighted sum of collision cost and trajectory cost.
        """
        radius = robot_info.spheres.r[-1] + robot_info.safety_margin
        K_1 = get_K_1(robot_info, optim_info)

        def costs(traj, sdf, grad_sdf):
            dists_traj = vmap(get_dists, in_axes=(0, None, None, None, None))(
                traj, sdf, grad_sdf, robot_info, world_info
            )
            dists_traj = subtract_radius_traj(dists_traj, radius)
            obstacle_cost_per_step = obstacle_cost_traj(dists_traj, optim_info)
            obstacle_cost_per_step = apply_collision_heuristic(obstacle_cost_per_step)
            collision_cost = velocity_weighting(
                obstacle_cost_per_step, traj, robot_info
            )
            A, b, c = get_Abc(traj, K_1)
            trajectory_cost = traj_cost(traj, A, b, c)
            from sqp.problem_definition import dynamic_constraints
            dynamic_cost = jnp.sum(jnp.maximum(0.0, dynamic_constraints(traj, robot_info, optim_info)))


            return trajectory_cost, collision_cost + dynamic_cost

        return costs

    @staticmethod
    def get_optimizer(
        example_traj: ArrayLike,
        A_inv: ArrayLike,
        robot_info: RobotInfo,
        optim_info: OptimInfo,
    ):
        """
        Generates and inits an optimizer for a given batch of trajectories (used for initialization).

        Args:
            example_traj : Example traj for optimizer initialization.
            A_inv : Inverse of the matrix A. Used as covariant metric.
            robot_info : Robot information.
            optim_info : Optimization information.
        """
        # this is the metric to use
        schedule = optax.constant_schedule(optim_info.step_size)
        opt = optax.chain(
            optax.flatten(scale_by_metric(A_inv)),
            optax.scale_by_learning_rate(schedule),
            zero_grads_at(0, -1),
            optax.clip(optim_info.clip_size),
        )
        return opt, vmap(opt.init)(
            jnp.repeat(example_traj[None, ...], optim_info.num_starts, axis=0)
        )

    @staticmethod
    def __get_optim_loop(
        cost_fn: Callable,
        optimizer : optax.GradientTransformationExtraArgs,
        A_inv: ArrayLike,
        optim_info: OptimInfo,
        robot_info: RobotInfo,
    ):
        """
        Generates the optimization loop.
        Args:
            cost_fn : The cost function.
            optimizer : The optimizer.
            A_inv : The inverse of the matrix A.
            optim_info : The optimization information.
            robot_info : The robot information.
        Returns:
            function: The optimization (while) loop.
        """

        cost_hist = jnp.ones((optim_info.num_iters, 2)) * 0.  # dummy value
        def normalized_cost_fun(cost_fun, traj, lambda_, sdf, grad_sdf, normalizers):
            traj_cost, col_cost = cost_fun(traj, sdf, grad_sdf)
            traj_cost_norm = traj_cost / normalizers.traj_cost
            col_cost_norm = col_cost / normalizers.col_cost
            return traj_cost_norm + lambda_ * col_cost_norm, (traj_cost_norm, col_cost_norm)

        def cond_fun(c):
            return jnp.logical_and(
                c.i < optim_info.num_iters, c.patience < optim_info.patience_limit
            )

        def body_fun(c):
            c.patience = lax.cond(
                jnp.any(jnp.abs(c.traj - c.traj_old) > optim_info.convergence_tol),
                lambda p: 0,
                lambda p: p + 1,
                c.patience,
            )
            c.traj_old = c.traj
            (cost, (traj_cost, col_cost)), DcostDq = value_and_grad(normalized_cost_fun, has_aux=True, argnums=1)(
                cost_fn, c.traj, c.lambdas[c.i], c.sdf, c.grad_sdf, c.normalizers
            )
            updates, c.optimizer_state = optimizer.update(
                DcostDq,
                c.optimizer_state,
                c.traj,
                value=cost,
                grad=DcostDq,
                value_fn=normalized_cost_fun,
                cost_fun=cost_fn,
                lambda_=c.lambdas[c.i],
                sdf=c.sdf,
                grad_sdf=c.grad_sdf,
                normalizers=c.normalizers
            )
            traj = optax.apply_updates(c.traj, updates)
            traj = handle_joint_limits(traj, A_inv, robot_info, optim_info)
            c.traj = jnp.where(
                (c.i + 1) % optim_info.resample_interval == 0,
                resample_along_traj(traj, optim_info.num_points),
                traj,
            )
            c.cost_hist = c.cost_hist.at[c.i].set(jnp.array([traj_cost, col_cost]))
            c.cost = cost
            c.i += 1
            return c

        # anything that is passed into the loop can be changed between runs without rejitting
        def init(traj, lambdas, opt_state, sdf, grad_sdf, normalizers):
            carry = DynamicParams(
                i=0,
                patience=0,
                traj=traj,
                traj_old=traj * jnp.inf,
                cost=0.0,
                cost_hist=cost_hist,
                lambdas=lambdas,
                sdf=sdf,
                grad_sdf=grad_sdf,
                optimizer_state=opt_state,
                normalizers=normalizers
            )
            return carry

        def finalize_fun(c: DynamicParams):
            return c.traj, c.cost_hist, c.cost

        return lambda traj, lambdas, opt_state, sdf, grad_sdf, normalizers: finalize_fun(
            lax.while_loop(
                cond_fun,
                body_fun,
                init(traj, lambdas, opt_state, sdf, grad_sdf, normalizers),
            )
        )

        

