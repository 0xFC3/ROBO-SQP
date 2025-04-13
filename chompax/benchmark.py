import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
import numpy as np
import sqlite3 as sql
from pathlib import Path
from zlib import decompress
from .structures import RobotInfo, WorldInfo
from .obs import is_configuration_feasible
from .rob import get_tcp_pos
from .obs import get_dists_nondiff, obstacle_img2dist_img
from .twod_plotly import RobotVis2DPlotly
from jax.typing import ArrayLike
from collections import namedtuple
from yaspin import yaspin
SHAPE = (64, 64)

class BenchmarkData:
    """
    Class that contains the worlds and the corresponding start and goal configurations and provides a generator for sampling.
    """
    def __init__(self, db_path : str, n_worlds : int, n_configs_per_world : int, key : ArrayLike, robot_info : RobotInfo, world_info : WorldInfo):
        db_path = db_path
        assert Path(db_path).exists(), f"Database at {db_path} does not exist."
        key_worlds, key_configs = jax.random.split(key)
        self.worlds = get_n_unique_worlds(db_path, n_worlds, key_worlds)
        self.sdfs = jnp.array([obstacle_img2dist_img(world, world_info) for world in self.worlds])

        Params = namedtuple("Params", ["sdfs", "n_worlds", "n_configs_per_world", "robot_info", "world_info"])
        self.params = Params(self.sdfs, n_worlds, n_configs_per_world, robot_info, world_info)
        with yaspin(text=f"Sampling {n_worlds * n_configs_per_world} start and goal configurations...") as sp:
            self.configs = self.__get_configs(key_configs, self.params)
            self.configs.block_until_ready()
            sp.ok("✓")
        
    @staticmethod
    def __get_configs(key, Params) -> ArrayLike:
        """
        Samples start and goal configurations for the robot in the worlds.
        """
        keys = jax.random.split(key, Params.n_worlds * Params.n_configs_per_world)
        keys = jnp.reshape(keys, (Params.n_worlds, Params.n_configs_per_world, 2))
        #for every world, there is n_configs_per_world start and goal configurations.
        get_configs_for_world = vmap(sample_start_goal, in_axes=(0, None, None, None))
        starts, goals = vmap(get_configs_for_world, in_axes=(0, 0, None, None))(keys, Params.sdfs, Params.robot_info, Params.world_info)
        return jnp.stack([starts, goals], axis=-1)
    
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, idx):
        return self.worlds[idx], self.sdfs[idx], self.configs[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"Benchmark with {self.params.n_worlds} worlds and {self.params.n_configs_per_world} corresponding start and goal configurations."

        
        
def get_n_unique_worlds(db_path : str, n : int, key : ArrayLike) -> np.ndarray:
    """
    Returns n unique worlds from the database at db_path.
    """
    #from table "worlds" read column "img_cmp"
    conn = sql.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT img_cmp FROM worlds")
    worlds = cur.fetchall()
    conn.close()
    
    sample_idx = jax.random.choice(key, len(worlds), (n,), replace=False)
    worlds = [jnp.frombuffer(decompress(worlds[i][0]), dtype=bool).reshape(SHAPE) for i in sample_idx]
    return jnp.array(worlds)

def sample_pose(key : ArrayLike, sdf : ArrayLike, robot_info : RobotInfo, world_info : WorldInfo) -> ArrayLike:
    """
    Samples a configuration for the robot in the world.
    """
    def sample_fun(key):
        q = jax.random.uniform(key, (robot_info.dim,), minval=robot_info.limits[:, 0], maxval=robot_info.limits[:, 1])
        return q
    
    def cond_fun(carry):
        q,_, i = carry
        return jnp.logical_not(is_configuration_feasible(q, sdf, robot_info, world_info) & (i < 10000))
    def body_fun(carry):
        _, key, i = carry
        key, _ = jax.random.split(key)
        q = sample_fun(key)
        return q, key, i+1
    
    carry = (sample_fun(key), key, 0)
    q, key, i = lax.while_loop(cond_fun, body_fun, carry)
    q = lax.select(i < 10000, q, jnp.nan * jnp.ones_like(q))
    return q

@jit    
def sample_start_goal(key : ArrayLike, sdf : ArrayLike, robot_info : RobotInfo, world_info : WorldInfo) -> ArrayLike:
    """
    Samples a start and goal configuration for the robot in the world.
    """ 
    key_start, key_goal = jax.random.split(key)
    start = sample_pose(key_start, sdf, robot_info, world_info)
    #sample goals until they are sufficiently far away from start. Use TCP distance as metric. Make couple of GD steps for refinement
    def tcp_dist(q1, q2):
        tcp_1, tcp_2 = get_tcp_pos(q1, robot_info), get_tcp_pos(q2, robot_info)
        return jnp.linalg.norm(tcp_1 - tcp_2)

    def sum_of_dists(q_goal):
        obs_dist = get_dists_nondiff(q_goal, sdf, robot_info, world_info)
        return jnp.sum(obs_dist)
    
    def refine_goal_step(q_goal):
        grad = jax.grad(sum_of_dists)(q_goal)
        #max clip grad to 0.2 rad
        grad = jnp.clip(0.01 * grad, -0.2, 0.2)
        q_goal = q_goal + grad
        return q_goal
    
    refine_goal = lambda q_goal: lax.fori_loop(0, 3, lambda i, q_goal: refine_goal_step(q_goal), q_goal)

    #strategy 1: sample heaps of goals and take the one with the highest distance
    #strategy 2: sample goals and refine them until they are far enough and valid

    #strategy 1
    n_goals_tries = 1000
    keys_goal = jax.random.split(key_goal, n_goals_tries)
    goals = vmap(sample_pose, in_axes=(0, None, None, None))(keys_goal, sdf, robot_info, world_info)
    dists = vmap(tcp_dist, in_axes=(0, None))(goals, start)
    goal = goals[jnp.nanargmax(dists)] #unsuccesful samples are nan
    #start = refine_goal(start)
    #goal = refine_goal(goal)
    return start, goal

def plot_start_goal(start, goal, world, robot_info, world_info):
    traj = jnp.linspace(start, goal, 20)
    viz = RobotVis2DPlotly(traj, world, world_info, robot_info)
    viz.plot()

@jit
def compute_hardness_score(sdf, start, goal, robot_info, world_info):
    """
    Computes the hardness score of a given start and goal configuration in a given world by evaluating
    collisions for a linearly interpolated start.
    """
    traj = jnp.linspace(start, goal, 100)
    feas = vmap(is_configuration_feasible, in_axes=(0, None, None, None))(traj, sdf, robot_info, world_info)
    return 1.0 - jnp.mean(feas)
    

