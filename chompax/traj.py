#Description: Trajectory related functions for CHOMPAX
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad, random, lax, jacobian
from functools import partial
from .structures import *
from jax.typing import ArrayLike

def get_K_1(rob_info : RobotInfo, optim_info : OptimInfo):
    """
    Get the finite difference matrix for a trajectory of length n
    Args:
        rob_info: robot information
        optim_info: optimization information
    Returns:
        K_1: finite difference matrix
    """
    n = optim_info.num_points
    dim = rob_info.dim
    D = np.zeros((n+1, n))
    D[range(n), range(n)] = 1
    D[range(1, n+1), range(n)] = -1
    I_m = np.eye(dim)
    K_1 = np.kron(D, I_m)
    return jnp.array(K_1)

def get_bc(traj: ArrayLike, K_1: ArrayLike):
    """
    Get the b and c terms of the quadratic cost function for a trajectory q
    Args:
        traj: trajectory
        K_1: finite difference matrix
    Returns:
        b: b term
        c: c term
    """
    n = traj.shape[0]
    m = traj.shape[1]
    e_1 = jnp.zeros(((n+1)*m, 1))
    start = traj[0]
    end = traj[-1]
    e_1 = e_1.at[:m].set(-1 * start[..., None])
    e_1 = e_1.at[-m:].set(end[..., None])
    b = K_1.T @ e_1
    c = e_1.T @ e_1
    return b, c


def get_Abc(traj: ArrayLike, K_1: ArrayLike):
    """
    Get the quadratic cost function for a trajectory q
    Args:
        traj: trajectory
        K_1: finite difference matrix
    Returns:
        A: A term
        b: b term
        c: c term
    """
    n = traj.shape[0]
    m = traj.shape[1]
    e_1 = jnp.zeros(((n+1)*m, 1))
    # batch_size = q.batch
    start = traj[0]
    end = traj[-1]
    e_1 = e_1.at[:m].set(-1 * start[..., None])
    e_1 = e_1.at[-m:].set(end[..., None])
    A = K_1.T @ K_1
    b = K_1.T @ e_1
    c = e_1.T @ e_1
    return A, b, c

def traj_cost(traj : ArrayLike, A : ArrayLike, b : ArrayLike, c : ArrayLike):
    """
    Calculate the cost of a trajectory q given the quadratic cost function parameters A, b, c.
    Args:
        traj: trajectory
        A: A term
        b: b term
        c: c term
    Returns:
        cost: cost of the trajectory (1/2 * q^T A q + b^T q + 1/2 * c)
    """
    xi = traj.ravel()[:, None]
    return (0.5 * lax.dot(xi.T, lax.dot(A, xi)) + lax.dot(xi.T, b) + 0.5 * c).squeeze()


@partial(jit, static_argnames=["n_points"])
def resample_along_traj(q, n_points=20):
    """
    Resample a trajectory q to have equidistant points along the original linear path.
    
    Args:
        q (jnp.ndarray): The input trajectory of shape (N, D).
        n_points (int): The number of points in the resampled trajectory.
        
    Returns:
        jnp.ndarray: The resampled trajectory of shape (n_point, D).
    """
    # Calculate cumulative arc length
    distances = jnp.linalg.norm(q[1:] - q[:-1], axis=-1)
    cumulative_distances = jnp.cumsum(distances)
    cumulative_distances = jnp.insert(cumulative_distances, 0, 0)

    # Normalize to [0, 1]
    normalized_distances = cumulative_distances / cumulative_distances[-1]

    # Equally spaced points in the normalized distance
    alpha = jnp.linspace(0, 1, n_points)

    # Linear interpolation
    def linear_interpolate(x, xp, fp):
        indices = jnp.searchsorted(xp, x) - 1
        indices = jnp.clip(indices, 0, len(xp) - 2)
        x0 = xp[indices]
        x1 = xp[indices + 1]
        y0 = fp[indices]
        y1 = fp[indices + 1]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

    q_resampled = jnp.vstack([linear_interpolate(
        alpha, normalized_distances, q[:, dim]) for dim in range(q.shape[-1])]).T
    return q_resampled


def get_linear_traj(start, end):
    traj = jnp.linspace(start, end, 2)
    return traj

def get_rand_traj(key, start, end, robot_info : RobotInfo, optim_info : OptimInfo):
    """
    Generate a random trajectory using (subsampled) randomly sampled points.
    Args:
        key (jax.random.PRNGKey): The random key.
        start (jnp.ndarray): The start point.
        end (jnp.ndarray): The end point.
        robot_info (RobotInfo): The robot information.
        optim_info (OptimInfo): The optimization information.
    Returns:
        jnp.ndarray: The random trajectory of shape (num_points, dim).
    """
    num_points = optim_info.initialguess_points
    joint_limits = robot_info.limits
    dim = robot_info.dim
    assert dim == joint_limits.shape[0], "Dimension mismatch"
    def get_rand_points(key, lower, upper, len): return random.uniform(
        key, (len, dim), minval=lower, maxval=upper)
    traj = jnp.zeros((num_points+2, dim))
    traj = traj.at[0].set(start)
    traj = traj.at[-1].set(end)
    traj = traj.at[1:-1].set(get_rand_points(key, joint_limits[:,0], joint_limits[:,1], num_points))
    return traj

def get_initial_guess(key : ArrayLike, start : ArrayLike, end : ArrayLike, rob_info : RobotInfo, optim_info : OptimInfo):
    """
    Generate an initial guess for the trajectory optimization.
    Args:
        key (jax.random.PRNGKey): The random key.
        num_points (int): The number of points in the trajectory.
        start (jnp.ndarray): The start point.
        end (jnp.ndarray): The end point.
        rob_info (RobotInfo): The robot information.
        optim_info (OptimInfo): The optimization information.
        random (bool): Whether to generate a random trajectory.
    Returns:
        jnp.ndarray: The initial guess trajectory of shape (num_points, dim).
    """
    # we reduce the numbe of point in the inner part so we get piecewise linear trajectories
    resample = partial(resample_along_traj, n_points=optim_info.num_points)
    return resample(get_rand_traj(key, start, end, rob_info, optim_info=optim_info))

def best_traj_cost(start, end, robot_info: RobotInfo, optim_info: OptimInfo):
    # best trajectory is a linear interpolation between start and goal
    best_traj = jnp.linspace(start, end, num=optim_info.num_points)
    A, b, c = get_Abc(best_traj, get_K_1(robot_info, optim_info))
    return traj_cost(best_traj, A, b, c)

def worst_traj_cost(start, end, robot_info: RobotInfo, optim_info: OptimInfo):
    # worst traj oscilates between joint limits. We ignore the start and end here, as it shouldnt be significant.
    lower_lim = robot_info.limits[:, 0]
    upper_lim = robot_info.limits[:, -1]
    lowers = jnp.repeat(lower_lim[None], optim_info.num_points, axis=0)
    uppers = jnp.repeat(upper_lim[None], optim_info.num_points, axis=0)
    worst_traj = jnp.empty((optim_info.num_points * 2, robot_info.dim))
    worst_traj = worst_traj.at[::2].set(lowers)
    worst_traj = worst_traj.at[1::2].set(uppers)
    worst_traj = worst_traj.at[:optim_info.num_points].get()
    A, b, c = get_Abc(worst_traj, get_K_1(robot_info, optim_info))
    return traj_cost(worst_traj, A, b, c)

def get_best_traj(traj: ArrayLike, num: int, cost: ArrayLike):
    """
    Selects the best trajectories based on cost.

    Args:
        q (array-like): Array of trajectories.
        num (int): Number of best trajectories to return.
        cost (array-like): Array of costs associated with each trajectory.

    Returns:
        array-like: Array of the best trajectories sorted by cost.
    """
    feasible_q = traj
    feasible_cost = cost
    indices = jnp.argsort(feasible_cost)
    return feasible_q[indices][:num]

def get_best_feas_traj(
    traj: ArrayLike, num: int, feasibility: ArrayLike, cost: ArrayLike
):
    """
    Selects the best trajectories based on feasibility and cost.

    Args:
        q (array-like): Array of trajectories.
        num (int): Number of best trajectories to return.
        feasibility (array-like): Boolean array indicating the feasibility of each trajectory.
        cost (array-like): Array of costs associated with each trajectory.

    Returns:
        array-like: Array of the best feasible trajectories sorted by cost.
    """
    feasible_q = traj[feasibility]
    feasible_cost = cost[feasibility]
    indices = jnp.argsort(feasible_cost)
    return feasible_q[indices][:num]

def get_random_starts(
    key: ArrayLike,
    start: ArrayLike,
    end: ArrayLike,
    optim_info: OptimInfo,
    robot_info: RobotInfo,
):
    keys = random.split(key, optim_info.num_starts)
    starts = jnp.repeat(start[None, :], optim_info.num_starts, axis=0)
    ends = jnp.repeat(end[None, :], optim_info.num_starts, axis=0)
    traj = vmap(get_initial_guess, in_axes=(0, 0, 0, None, None))(
        keys, starts, ends, robot_info, optim_info
    )
    return traj
