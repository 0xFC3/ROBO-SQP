# Description: Obstacle related functions for CHOMPAX
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, custom_jvp, lax, jacobian
from functools import partial
from scipy.ndimage import distance_transform_edt
from jax.scipy.ndimage import map_coordinates

from .rob import get_sphere_pos
from .traj import resample_along_traj
from .structures import *
from jax.typing import ArrayLike
from optax import safe_norm


def obstacle_img2dist_img(img, world_info: WorldInfo, add_boundary=True):
    """
    Calculate the signed distance field from an 2D/3D image of the world.
    Obstacles are 1/True, free space is 0/False.
    The distance image is of the same shape as the input image and has positive values outside objects and negative
    values inside objects see 'CHOMP - signed distance field' (10.1177/0278364913488805)
    The voxel_size is used to scale the distance field correctly (the shape of a single pixel / voxel)
    Args:
        img: binary image of the world
        world_info: WorldInfo object
        add_boundary: if True, the boundary is filled with obstacles
    """
    n_voxels = np.array(img.shape)
    voxel_size = world_info.voxel_size

    if not add_boundary:
        # Main function
        #                                         # EDT wants objects as 0, rest as 1
        dist_img = distance_transform_edt(-img.astype(int) + 1, sampling=voxel_size)
        dist_img_complement = distance_transform_edt(
            img.astype(int), sampling=voxel_size
        )
        dist_img[img] = -dist_img_complement[img]  # Add interior information

    else:
        # Additional branch, to include boundary filled with obstacles
        obstacle_img_wb = np.ones(n_voxels + 2, dtype=bool)
        inner_image_idx = tuple(
            map(slice, np.ones(img.ndim, dtype=int), (n_voxels + 1))
        )
        obstacle_img_wb[inner_image_idx] = img

        dist_img = obstacle_img2dist_img(
            img=obstacle_img_wb, world_info=world_info, add_boundary=False
        )
        dist_img = dist_img[inner_image_idx]

    return jnp.asarray(dist_img)


def obstacle_cost(dist: ArrayLike, optim_info: OptimInfo) -> jnp.ndarray:
    """
    Calculate the cost of single distance to the obstacle.
    Args:
        dist: distance to the obstacle
        optim_info: OptimInfo object
    Returns:
        cost of the distance to the obstacle
    """
    eps = optim_info.obstacle_cost_eps
    return jnp.where(
        dist < 0,
        -dist + eps / 2,
        jnp.where(dist < eps, 0.5 / eps * jnp.power(dist - eps, 2), 0),
    )


def take_dist_sdf_grad(pos: ArrayLike, sdf_grad: ArrayLike, world_info: WorldInfo):
    """
    Calculate the gradient of the distance to the obstacles at a given position. Indexes the precomputed sdf gradient with linear interpolation.
    Args:
        pos: position
        sdf_grad: gradient of the signed distance field
        world_info: WorldInfo object
    Returns:
        gradient of the distance field (grad_x, grad_y)
    """
    limits = world_info.limits
    voxel_size = world_info.voxel_size

    p_indices = (pos - limits[:, 0]) / voxel_size
    func = lambda g, p: map_coordinates(g, p, order=1, mode="nearest")
    grads = vmap(func, in_axes=(0, None))(sdf_grad, p_indices)
    return grads[::-1]


def get_DdistsDq(
    q: ArrayLike, grad_sdf: ArrayLike, rob_info: RobotInfo, world_info: WorldInfo
):
    """Get the gradient of sphere distances with respect to the joint angles
    Args:
        q: joint angles
        grad_sdf : Gradient of the signed distance field obtained by finite diff.
        rob_info : Robot information
    Returns:
        DdistDq: gradient of the distance field with respect to the joint angles
    """
    sphere_positions = get_sphere_pos(q, rob_info)
    DdistsDx = vmap(take_dist_sdf_grad, in_axes=(0, None, None))(
        sphere_positions, grad_sdf, world_info
    )  # (N_spheres x 2)
    DxDq = jacobian(get_sphere_pos, argnums=0)(q, rob_info)
    # contract over sphere dims
    return jnp.einsum("ij,ijk->ik", DdistsDx, DxDq)


def take_dist_sdf(pos: ArrayLike, sdf: ArrayLike, world_info: WorldInfo):
    """
    Calculate the distance to the obstacles at a given position. Indexes the precomputed sdf (no gradient!) with linear interpolation.
    Args:
        pos: position(s)
        sdf: signed distance field
        world_info: WorldInfo object
    """
    limits = world_info.limits
    pos_idx =  (pos - limits[:, 0]) * (world_info.num_vox-0.5 - -0.5) / (limits[:, 1] - limits[:, 0]) + -0.5 # normalize to [-0.5, num_vox-0.5]
    dists = map_coordinates(sdf, pos_idx.T, order=1, mode="nearest")
    return dists


@partial(custom_jvp, nondiff_argnums=(1, 2, 3, 4))
@jit  # we have to jit here because otherwise we get weird errors
def get_dists(
    q: ArrayLike,
    sdf: ArrayLike,
    grad_sdf: ArrayLike,
    rob_info: RobotInfo,
    world_info: WorldInfo,
):
    """
    Get the differentiable distances of the spheres to the obstacles
    Args:
        q: joint angles
        sdf: signed distance field
        grad_sdf: gradient of the signed distance field (for autodiff)
        rob_info: RobotInfo object
        world_info: WorldInfo object
    Returns:
        Distances of the spheres to the obstacles
    """

    pos = get_sphere_pos(q, rob_info)
    return vmap(take_dist_sdf, in_axes=(0, None, None))(pos, sdf, world_info)


@get_dists.defjvp
def get_dists_jvp(sdf, grad_sdf, robot_info, world_info, primals, tangents):
    (q,) = primals
    (q_dot,) = tangents
    DdistsDq = get_DdistsDq(q, grad_sdf, robot_info, world_info)
    primals_out = get_dists_nondiff(q, sdf, robot_info, world_info)
    tangents_out = lax.dot(DdistsDq, q_dot)
    return primals_out, tangents_out


def get_dists_nondiff(
    q: ArrayLike, sdf: ArrayLike, rob_info: RobotInfo, world_info: WorldInfo
):
    """
    Get the distances of the spheres to the obstacles
    Note that technically this is differentiable, because we use map_coordinates, but we want to rely on numeric gradient with finite differences for this.
    Args:
        q: joint angles
        sdf: signed distance field
        rob_info: RobotInfo object
        world_info: WorldInfo object
    Returns:
        Distances of the spheres to the obstacles
    """

    pos = get_sphere_pos(q, rob_info)
    return vmap(take_dist_sdf, in_axes=(0, None, None))(pos, sdf, world_info)


def velocity_weighting(
    cost_per_step: ArrayLike, traj: ArrayLike, robot_info: RobotInfo
):
    """
    Calculate the weighted sum of pairwise costs and distances for a given trajectory.
    Args:
        cost_per_step: cost per step
        traj: trajectory
        robot_info: RobotInfo object
    Returns:
        jnp.ndarray: The weighted sum of pairwise costs and distances.
    """
    cost_pairs = cost_per_step[:-1] + cost_per_step[1:]
    # for the dist pairs, we have to get the spheres and then calculate the distance between them over the time
    pos = vmap(get_sphere_pos, in_axes=(0, None))(traj, robot_info)
    velocities = pos[1:] - pos[:-1]
    # dist_pairs = jnp.linalg.norm(dist_pairs, axis=-1, ord=1)
    velocities = safe_norm(
        velocities, axis=-1, min_norm=1e-4, ord=2
    )  # will differentiate at 0

    return jnp.sum(cost_pairs * velocities)


def subtract_radius(dist, radius):
    """
    Subtract the radius from the distance to get the distance to the surface of the obstacle
    Args:
        dist: distance to the obstacle
        radius: radius of the sphere
    Returns:
        distance to the surface of the obstacle
    """
    return dist - radius


def apply_collision_heuristic(obstacle_cost_per_step):
    first_nonzero_idx = jnp.apply_along_axis(
        jnp.flatnonzero, axis=1, arr=obstacle_cost_per_step, size=1
    ).squeeze()
    mask = jnp.arange(obstacle_cost_per_step.shape[1]) <= first_nonzero_idx[:, None]
    return obstacle_cost_per_step * mask


def is_configuration_feasible(
    q: ArrayLike, sdf: ArrayLike, rob_info: RobotInfo, world_info: WorldInfo
):
    """
    Check if the configuration is feasible
    Args:
        q: joint angles
        sdf: signed distance field
        rob_info: RobotInfo object
        world_info: WorldInfo object
    Returns:
        feasibility of the configuration
    """
    radius = rob_info.spheres.r[-1]
    dists = get_dists_nondiff(q, sdf, rob_info, world_info)
    dists = subtract_radius(dists, radius + rob_info.safety_margin)
    return jnp.all(dists > 0)


def is_trajectory_feasible(
    traj: ArrayLike,
    sdf: ArrayLike,
    rob_info: RobotInfo,
    world_info: WorldInfo,
    optim_info: OptimInfo,
):
    """
    Check if the upsampled trajectory is feasible
    Args:
        traj: trajectory to check
        sdf: signed distance field
        rob_info: RobotInfo object
        world_info: WorldInfo object
        optim_info: OptimInfo object
    Returns:
        feasibility of the trajectory
    """
    upsamplingfactor = optim_info.collcheck_upsampling_factor
    upsampled_traj = resample_along_traj(traj, traj.shape[0] * upsamplingfactor)
    feas = vmap(is_configuration_feasible, in_axes=(0, None, None, None))(
        upsampled_traj, sdf, rob_info, world_info
    )
    return jnp.all(feas)

@jit
def is_batch_feasible(
    trajs: ArrayLike,
    sdf: ArrayLike,
    robot_info: RobotInfo,
    world_info: WorldInfo,
    optim_info: OptimInfo,
):
    return vmap(is_trajectory_feasible, in_axes=(0, None, None, None, None))(
        trajs, sdf, robot_info, world_info, optim_info
    )

def worst_obstacle_cost_per_step(sdf: ArrayLike, robot_info: RobotInfo, world_info: WorldInfo, optim_info: OptimInfo):
    # worst case is when the sphere is at the lowest point in the sdf
    sdf_min = jnp.min(sdf)
    sdf_min = lax.select(sdf_min < 0, -sdf_min + robot_info.spheres.r[-1] + robot_info.safety_margin, 1)
    return sdf_min
