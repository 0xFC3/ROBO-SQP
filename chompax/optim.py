# Description: Optimization related functions for CHOMPAX
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad, lax
from jax.scipy.ndimage import map_coordinates
from functools import partial
from jax.typing import ArrayLike
import optax
from .traj import traj_cost
from .obs import (
    get_dists_nondiff,
    subtract_radius,
    obstacle_cost,
    velocity_weighting,
)
from .structures import *


def symetric_numeric_jacobian(f, x, eps=1e-8):
    """
    Calculate the jacobian of a function f at x using finite differences.
    """

    J = jnp.array(
        [(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))]
    ).T
    return J


def get_img_grad_interp(grad_sdf: ArrayLike, world_info: WorldInfo):
    """
    Get the interpolated gradient function of the signed distance field.
    Args:
        grad_sdf : Gradient of the signed distance field obtained by finite diff.
        world_info : World information.

    Returns:
        Interpolated gradient function.
    """
    limits = world_info.limits
    voxel_size = world_info.voxel_size

    def img_grad_interp_x(p):
        p_indices = (p - limits[:, 0]) / voxel_size
        return map_coordinates(grad_sdf[1], p_indices, order=1, mode="nearest")

    def img_grad_interp_y(p):
        p_indices = (p - limits[:, 0]) / voxel_size
        return map_coordinates(grad_sdf[0], p_indices, order=1, mode="nearest")

    return jit(lambda p: jnp.stack([img_grad_interp_x(p), img_grad_interp_y(p)]))


def img_grad(img: ArrayLike, world_info: WorldInfo) -> jnp.ndarray:
    """
    Calculate the gradient of an image in each direction of the image, using the sobel filter.
    Args:
        img : Image.
        world_info : World information.

    Returns:
        Gradient of the image.
    """
    voxel_size = world_info.voxel_size
    kernel = jnp.zeros((2, 1, 3, 3))  # I, O, H, W
    kernel = kernel.at[0, 0, :, :].set(  # Gx
        jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    )
    kernel = kernel.at[1, 0, :, :].set(  # Gy
        jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    )
    # apply sensible padding to sdf. -> Outside the world we are inside an obstacle.
    img_padded = jnp.pad(
        img, ((1, 1), (1, 1)), mode="constant", constant_values=-world_info.voxel_size
    )
    out = lax.conv(img_padded[None, None, ...], kernel, (1, 1), "VALID") / (
        8 * voxel_size
    )
    return out.squeeze()


def get_A_inv(A):
    """
    Calculate the inverse of a matrix A
    """
    return jnp.linalg.inv(A)


def covariant_update(
    traj: ArrayLike,
    A_inv: ArrayLike,
    grad: ArrayLike,
    step_size: float,
    rob_info: RobotInfo,
    optim_info: OptimInfo,
):
    """
    Update the joint angles using the covariant update rule traj = traj - mask * A_inv @ grad * step_size
    Args:
        traj: trajectory
        A_inv: Inverse of the matrix A
        grad: gradient of the cost function
        rob_info : Robot information
        optim_info : Optimization information
    """
    dim = rob_info.dim
    mask = optim_info.mask
    return traj - mask * (lax.dot(A_inv, grad.ravel())).reshape(-1, dim) * step_size


def covariant_update_lambda(
    traj: ArrayLike,
    A_inv: ArrayLike,
    grad: ArrayLike,
    step_size: float,
    lambda_: float,
    rob_info: RobotInfo,
    optim_info: OptimInfo,
):
    dim = rob_info.dim
    mask = optim_info.mask
    return (
        traj
        - mask * (lax.dot(A_inv, grad.ravel())).reshape(-1, dim) * step_size / lambda_
    )


def scale_by_metric(metric) -> optax.GradientTransformation:
    """Returns a covariant gradient transformation."""

    def update_fn(updates, state, params=None):
        del params
        return jnp.dot(metric, updates), state

    def init_fn(_):
        return optax.EmptyState()

    return optax.GradientTransformation(init_fn, update_fn)


def zero_grads_at(*indices):
    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        return updates.at[jnp.array(indices)].set(0), state

    return optax.GradientTransformation(init_fn, update_fn)


# Vectorized functions
get_dists_traj = vmap(get_dists_nondiff, in_axes=(0, None, None, None))
subtract_radius_traj = vmap(subtract_radius, in_axes=(0, None))
grad_obstacle_cost_traj = vmap(grad(obstacle_cost))
obstacle_cost_traj = vmap(obstacle_cost, in_axes=(0, None))
valgrad_pairwise_cost_traj = value_and_grad(velocity_weighting, argnums=(0, 1))
valgrad_traj_cost_traj = value_and_grad(traj_cost, argnums=0)
