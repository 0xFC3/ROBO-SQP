from time import perf_counter as time
import numpy as np
from dataclasses import dataclass
from functools import partial
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike
from typing import NamedTuple

@partial(
    register_dataclass,
    data_fields=["traj_cost", "col_cost"],
    meta_fields=[],
)
@dataclass
class Normalizers:
    traj_cost: float = 1.0
    col_cost: float = 1.0

@partial(
    register_dataclass,
    data_fields=[
        "i",
        "patience",
        "traj",
        "traj_old",
        "cost",
        "cost_hist",
        "lambdas",
        "optimizer_state",
        "sdf",
        "grad_sdf",
        "normalizers",
    ],
    meta_fields=[],
)
@dataclass
class DynamicParams:
    i: int
    patience: int
    traj: ArrayLike
    traj_old: ArrayLike
    cost: float
    cost_hist: ArrayLike
    lambdas: ArrayLike
    sdf: ArrayLike
    grad_sdf: ArrayLike
    optimizer_state: ArrayLike
    normalizers : Normalizers


@partial(register_dataclass, data_fields=["x", "r", "f_idx"], meta_fields=[])
@dataclass
class Spheres_Jax:
    x: ArrayLike
    r: ArrayLike
    f_idx: ArrayLike


@partial(
    register_dataclass,
    data_fields=["limits", "voxel_size"],
    meta_fields=["num_vox", "add_boundary"],
)
@dataclass
class WorldInfo:
    limits: ArrayLike
    voxel_size: float
    num_vox: int
    add_boundary: bool = True


@partial(
    register_dataclass,
    data_fields=["mask", "step_size"],
    meta_fields=[
        "num_points",
        "num_iters",
        "num_starts",
        "resample_interval",
        "initialguess_points",
        "collcheck_upsampling_factor",
        "obstacle_cost_eps",
        "max_iter_limits",
        "patience_limit",
        "convergence_tol",
        "clip_size",
        "qp_tol",
        "eq_tol",
        "ineq_tol",
        "time_scaling",
    ],
)
@dataclass
class OptimInfo:
    mask: ArrayLike
    initialguess_points: int = 3
    num_points: int = 20
    num_iters: int = 100
    patience_limit: int = 5
    convergence_tol: float = 1e-3
    qp_tol: float = 1e-3
    eq_tol: float = 1e-3
    ineq_tol: float = 1e-3
    time_scaling: float = 5.0
    num_starts: int = 10
    step_size: float = 0.001
    resample_interval: int = 20
    collcheck_upsampling_factor: int = 10
    obstacle_cost_eps: float = 0.1
    max_iter_limits: int = 10
    clip_size: float = 0.3  # clips the gradient updates elementwise


@partial(
    register_dataclass,
    data_fields=["spheres", "limits", "safety_margin", "next_frame_idx"],
    meta_fields=["dim", "type", "link_length", "link_masses", "max_joint_forces"],
)
@dataclass
class RobotInfo:
    dim: int
    limits: ArrayLike
    spheres: Spheres_Jax
    type: str
    link_length: float
    safety_margin: float = 0.01
    next_frame_idx: ArrayLike = None
    link_masses: ArrayLike = None
    max_joint_forces: ArrayLike = None


@partial(
    register_dataclass,
    data_fields=[
        "ri",
        "wi",
        "step_size",
        "goal_sampling_interval",
    ],  # "tree", "edges", "sdf", "num_nodes", "num_edges"],
    meta_fields=["max_iter", "coll_check_interval", "leaf_batch_size"],
)
@dataclass
class RRTParams:
    ri: RobotInfo
    wi: WorldInfo
    step_size: float = 0.5  # we use inf-norm for the distance.
    max_iter: int = 10_000
    goal_sampling_interval: int = 10
    coll_check_interval: int = 100
    leaf_batch_size: int = 10


@partial(
    register_dataclass,
    data_fields=["tree", "edges", "num_nodes", "num_edges"],
    meta_fields=[],
)
@dataclass
class Tree:
    tree: ArrayLike
    edges: ArrayLike
    num_nodes: int
    num_edges: int
