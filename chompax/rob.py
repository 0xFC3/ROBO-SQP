# Description: Robotics related functions for CHOMPAX
import jax.numpy as jnp
from jax import jit
from functools import partial
from .structures import *
from jax.typing import ArrayLike


@partial(jit, static_argnames=["return_all"])
def forward_kinematic_StaticArm(q: ArrayLike, return_all: bool, rob_info: RobotInfo):
    """
    Compute the forward kinematics of a 2D arm with n joints

    Args:
        q: joint angles of shape (..., n)
        return_all: if true return all joint positions, else only the end effector position
        rob_info : RobotInfo object
    Returns:
        joint positions of shape (..., n+1, 2) if return_all else (..., 2)
    """
    link_length = rob_info.link_length
    q_cum = jnp.cumsum(q, axis=-1)  # cumulative sum of joint angles
    x = jnp.cos(q_cum) * link_length
    y = jnp.sin(q_cum) * link_length
    x = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(x, axis=-1)), axis=-1)
    y = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(y, axis=-1)), axis=-1)
    if return_all:
        return jnp.stack((x, y), axis=-1)
    else:
        return jnp.stack((x[..., -1], y[..., -1]), axis=-1)


@partial(jit, static_argnames=["return_all"])
def forward_kinematic_SingleSphere(
    q_in: ArrayLike, return_all: bool, rob_info: RobotInfo
):
    """
    Compute the forward kinematics of a 2D sphere where q is the position of the sphere

    Args:
        q_in: position of the sphere
        return_all: if true return all joint positions, else only the end effector position (not used)
        rob_info : RobotInfo object (not used)

    """
    return q_in


@jit
def get_frames_StaticArm(q: ArrayLike, rob_info: RobotInfo):
    """Get the frames of the robot, from first link to end effector
    Args:
        q: joint angles
        rob_info : RobotInfo object
    Returns:
        frames: frames of the robot
    """
    t = forward_kinematic_StaticArm(q, True, rob_info)
    frames = jnp.zeros((q.shape[0] + 1, 3, 3))
    q_cum = jnp.cumsum(q, axis=-1)
    frames = frames.at[:-1, 0, :2].set(jnp.array([jnp.cos(q_cum), -jnp.sin(q_cum)]).T)
    frames = frames.at[:-1, 1, :2].set(jnp.array([jnp.sin(q_cum), jnp.cos(q_cum)]).T)

    frames = frames.at[:, :, 2].set(
        jnp.concat([t, jnp.ones((q.shape[0] + 1, 1))], axis=-1)
    )

    # last frame is the end effector
    frames = frames.at[-1, 0, :2].set(
        jnp.array([jnp.cos(q_cum[-1]), -jnp.sin(q_cum[-1])]).T
    )
    frames = frames.at[-1, 1, :2].set(
        jnp.array([jnp.sin(q_cum[-1]), jnp.cos(q_cum[-1])]).T
    )

    return frames


@jit
def get_frames_SingleSphere(q: ArrayLike, rob_info: RobotInfo):
    """Get the frame of the sphere
    Args:
        q: sphere position
        rob_info : RobotInfo object
    Returns:
        frames: frame of the sphere
    """
    t = forward_kinematic_SingleSphere(q, True, rob_info)
    frame = jnp.zeros((1, 3, 3))
    frame = frame.at[0, 0, 0].set(1)
    frame = frame.at[0, 1, 1].set(1)
    frame = frame.at[0, 2, 2].set(1)
    frame = frame.at[0, 0, 2].set(t[0])
    frame = frame.at[0, 1, 2].set(t[1])
    return frame


@jit
def get_sphere_pos(q: ArrayLike, rob_info: RobotInfo):
    """Get the sphere positions for each joint angle
    Args:
        q: joint angles
        rob_info : RobotInfo object
    Returns:
        sphere positions
    """
    type = rob_info.type
    spheres = rob_info.spheres
    if type.lower() == "staticarm":
        frames = get_frames_StaticArm
    else:
        frames = get_frames_SingleSphere
    frames = frames(q, rob_info)[spheres.f_idx]
    return jnp.einsum("bij,bj->bi", frames, spheres.x)[..., :-1]

@jit
def get_tcp_pos(q: ArrayLike, rob_info: RobotInfo):
    """Get the tcp position for the joint angles
    Args:
        q: joint angles
        rob_info : RobotInfo object
    Returns:
        tcp position
    """
    type = rob_info.type
    if type.lower() == "staticarm":
        frames = get_frames_StaticArm
    else:
        frames = get_frames_SingleSphere
    frames = frames(q, rob_info)
    return frames[-1, :2, 2]
    

