import jax
import numpy as np
from .benchmark import *
import jax.numpy as jnp
from .configsTesting import *
import os
#test if db_path is environment variable
def test_db_path():
    assert os.getenv('ROBDATA_PATH') is not None, "Environment variable ROBDATA_PATH is not set."
db_path = os.getenv('ROBDATA_PATH') + '/StaticArm04_02.db'


def test_sample_pose():
    key = jax.random.PRNGKey(0)
    sdf = jnp.ones((64, 64)) * 10 #free space

    start_config = sample_pose(key, sdf, robot_info, world_info)
    
    assert start_config.shape == (robot_info.dim,), f"Expected shape (robot_info.dim,), got {start_config.shape}."
    assert jnp.all(start_config >= robot_info.limits[:, 0]), f"Expected all elements >= {robot_info.limits[:, 0]}, got {start_config}."
    assert jnp.all(start_config <= robot_info.limits[:, 1]), f"Expected all elements <= {robot_info.limits[:, 1]}, got {start_config}."

def test_worlds_database():
    n = 100
    import os
    assert os.path.exists(db_path), f"Database file {db_path} does not exist."
    worlds = get_n_unique_worlds(db_path, n)

    assert len(worlds) == n, f"Expected {n} worlds, got {len(worlds)}."
    assert worlds[0].shape == (64, 64), f"Expected shape (64, 64), got {worlds[0].shape}."
    assert worlds[0].dtype == bool, f"Expected dtype bool, got {worlds[0].dtype}."

def test_start_goal():
    key = jax.random.PRNGKey(0)
    sdf = jnp.ones((64, 64)) * 10 #free space
    start, goal = sample_start_goal(key, sdf, robot_info, world_info)

    assert start.shape == (robot_info.dim,), f"Expected shape (robot_info.dim,), got {start.shape}."
    assert jnp.all(start >= robot_info.limits[:, 0]), f"Expected all elements >= {robot_info.limits[:, 0]}, got {start}."
    assert jnp.all(start <= robot_info.limits[:, 1]), f"Expected all elements <= {robot_info.limits[:, 1]}, got {start}."
    assert goal.shape == (robot_info.dim,), f"Expected shape (robot_info.dim,), got {goal.shape}."
    assert jnp.all(goal >= robot_info.limits[:, 0]), f"Expected all elements >= {robot_info.limits[:, 0]}, got {goal}."
    assert jnp.all(goal <= robot_info.limits[:, 1]), f"Expected all elements <= {robot_info.limits[:, 1]}, got {goal}."
