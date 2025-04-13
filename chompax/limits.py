# Description: Functions to handle joint limits in CHOMPAX
import jax.numpy as jnp
from jax import jit, lax
from .optim import covariant_update
from .structures import RobotInfo, OptimInfo
from jax.typing import ArrayLike


def get_limit_correction(traj: ArrayLike, rob_info: RobotInfo):
    """
    Handle joint limits by L1-clipping the trajectory q back into the limits.
    Args:
        q: trajectory
        rob_info: robot information
    Returns:
        corrections: corrections to the trajectory by L1-clipping
    """
    # compute the l1 projection for all limit violations
    joint_limits = rob_info.limits
    traj_clipped = jnp.clip(traj, joint_limits[:, 0], joint_limits[:, 1])
    corrections = traj_clipped - traj
    return corrections


def handle_joint_limits(traj: ArrayLike, A_inv: ArrayLike, rob_info: RobotInfo, optim_info: OptimInfo):
    """
    Handle joint limits by projecting the trajectory q back into the limits
    Args:
        traj: trajectory
        A_inv: inverse of the metric
        rob_info: robot information
        optim_info: optimization information
    Returns:
        traj_corr: corrected trajectory
    """
    def cond_fun(carry):
        i, largest_correction, _ = carry
        return jnp.logical_and(i < optim_info.max_iter_limits, jnp.max(jnp.abs(largest_correction)) > 0.)
    
    def body_fun(carry):
        i, _, traj = carry
        corrections = get_limit_correction(traj, rob_info)
        largest_violation_idx = jnp.argmax(jnp.abs(corrections))
        transformed_correction = A_inv @ corrections.ravel()
        numerator = corrections.ravel()[largest_violation_idx]
        denom = transformed_correction[largest_violation_idx]
        alpha = lax.cond(jnp.isclose(denom, 0), lambda _: 0., lambda _: -numerator/denom, None)
        traj = covariant_update(traj, A_inv, corrections, alpha, rob_info, optim_info)
        return i+1, numerator, traj
    
    _, _, traj_corr = lax.while_loop(cond_fun, body_fun, (0., jnp.inf, traj))
    final_corrections = get_limit_correction(traj_corr, rob_info)

    return traj_corr + final_corrections
