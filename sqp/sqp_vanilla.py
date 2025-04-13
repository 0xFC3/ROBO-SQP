import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Callable
import qpax
from functools import partial
from jax.lib import xla_bridge
from contextlib import contextmanager
from chompax.structures import WorldInfo, OptimInfo, RobotInfo
from sqp.problem_definition import objective_function, inequality_constraints, equality_constraints
from jax.nn import softplus

# Print which backend we're using
print(f"JAX is using {xla_bridge.get_backend().platform} backend")

class OptState(NamedTuple):
    x: jnp.ndarray
    lambda_: jnp.ndarray  
    mu_: jnp.ndarray     
    iter_num: jnp.int32
    converged: jnp.bool_
    progress: jnp.ndarray  

class OptResult(NamedTuple):
    x: jnp.ndarray
    lambda_: jnp.ndarray  
    mu_: jnp.ndarray     
    success: jnp.bool_
    n_iter: jnp.int32
    f_val: jnp.float32
    eq_violation: jnp.float32
    ineq_violation: jnp.float32
    progress_history: jnp.ndarray  

class QPSolution(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    error: jnp.float32

@contextmanager
def profile_context(enable_profiling=False):
    if enable_profiling:
        try:
            print("Starting profiling...")
            jax.profiler.start_server(9999)  
            jax.profiler.start_trace("./tensorboard_logs")
            yield
        finally:
            print("Stopping profiling...")
            jax.profiler.stop_trace()
    else:
        yield

@partial(jax.jit, static_argnums=(6))
def solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq, qp_tol):
    x, s, z, y, converged, iters = qpax.solve_qp(
        H, g, A_eq, b_eq, A_ineq, b_ineq,
        solver_tol=qp_tol
    )
    error = jnp.where(converged, jnp.float32(1e-4), jnp.float32(1.0))
    return QPSolution(x=x, y=y, z=z, error=error)


@partial(jax.jit, static_argnums=(1,2,3,4,5,6,7,8,9,10,11,12))  
def step(state: OptState, f, grad_f, c_eq, c_ineq, jac_c_eq, jac_c_ineq, dim: int, num_points: int, qp_tol:float, eq_tol:float, ineq_tol:float, convergence_tol:float) -> OptState:
    x, lambda_, mu_, iter_num, converged, progress = state

    
    g = jnp.squeeze(grad_f(x))      

    def lagrangian(x_):
        return f(x_) + jnp.sum(lambda_ * c_eq(x_)) + jnp.sum(mu_ * c_ineq(x_))

    hessian_fn = jax.jit(jax.hessian(lagrangian))
    H = jnp.squeeze(hessian_fn(x)) 

    A_eq = jnp.squeeze(jac_c_eq(x))   
    b_eq = c_eq(x)
    A_ineq = jnp.squeeze(jac_c_ineq(x)) 
    b_ineq = c_ineq(x)
    
    # solve QP subproblem
    qp_sol = solve_qp(H, g, A_eq, -b_eq, A_ineq, -b_ineq, qp_tol)
    
    dx = qp_sol.x
    lambda_next = qp_sol.y
    mu_next = qp_sol.z
    
    # Apply final step size
    x_next = x + dx
    
    # Check convergence
    
    b_eq_next = c_eq(x_next)
    b_ineq_next = c_ineq(x_next)


    # Calculate feasibility for convergence check regardless of track_progress
    eq_violations = jnp.abs(b_eq_next)
    ineq_violations = jnp.maximum(0.0, b_ineq_next)
    feasibilty = jnp.logical_and(jnp.max(eq_violations) < eq_tol, jnp.max(ineq_violations) < ineq_tol)


    objective_improvement = jnp.abs(f(x_next) - f(x))
    
    converged = jnp.logical_or(jnp.linalg.norm(dx) < convergence_tol,
                               jnp.logical_and(objective_improvement < 1e-3, feasibilty))
    
    


    # Split inequality constraints into dynamic and obstacle violations
    n_dyn = dim * num_points  # Dynamic constraints
    obs_violations = ineq_violations[:-n_dyn]
    dyn_violations = ineq_violations[-n_dyn:]
    
    progress_next = jnp.array([
        f(x_next),  # objective value
        jnp.max(eq_violations),  # max equality violation
        jnp.max(ineq_violations),  # max inequality violation
        jnp.mean(eq_violations),  # mean equality violation
        jnp.mean(ineq_violations),  # mean inequality violation
        jnp.max(dyn_violations),  # max dynamic violation
        jnp.mean(dyn_violations),  # mean dynamic violation
        jnp.max(obs_violations),  # max obstacle violation
        jnp.mean(obs_violations),  # mean obstacle violation
        jnp.nan,
        jnp.nan,
        jnp.nan,
        jnp.nan,
        dx@H@dx
    ])


    progress = progress.at[iter_num + 1].set(progress_next)

    
    return OptState(
        x=x_next,
        lambda_=lambda_next,
        mu_=mu_next,
        iter_num=iter_num + jnp.array(1, dtype=jnp.int32),
        converged=jnp.array(converged, dtype=jnp.bool_),
        progress=progress
    )

@partial(jax.jit, static_argnums=(5)) 
def sqp_solve(
    world_info: WorldInfo,
    optim_info: OptimInfo,
    robot_info: RobotInfo,
    sdf: jnp.ndarray,   
    x0: jnp.ndarray,
    max_iter: int = 3
) -> OptResult:
    # Extract start and end from flattened trajectory
    traj_shape = (optim_info.num_points, robot_info.dim)
    x0_reshaped = x0.reshape(traj_shape)
    start = x0_reshaped[0]  # First configuration
    end = x0_reshaped[-1]   # Last configuration

    def f(traj):
        return objective_function(traj, robot_info, optim_info)
    
    def grad_f(traj):
        return jax.grad(f)(traj)
    
    def c_eq(traj):
        return equality_constraints(traj, start, end, optim_info, robot_info)
    
    def c_ineq(traj):
        return inequality_constraints(traj, sdf, robot_info, world_info, optim_info)
    
    def jac_c_eq(traj):
        jac = jax.jacfwd(lambda x: equality_constraints(x, start, end, optim_info, robot_info))(traj)
        return jnp.atleast_2d(jac)
    
    def jac_c_ineq(traj):
        jac = jax.jacfwd(lambda x: inequality_constraints(x, sdf, robot_info, world_info, optim_info))(traj)
        return jnp.atleast_2d(jac)
    

    m_eq = c_eq(x0).shape[0] if c_eq(x0).size > 0 else 0
    m_ineq = c_ineq(x0).shape[0]
    
    progress = jnp.zeros((max_iter + 1, 14))  
    
    eq_violations_init = jnp.abs(c_eq(x0))
    ineq_violations_init = jnp.maximum(0.0, c_ineq(x0))

    n_dyn = robot_info.dim * optim_info.num_points  # Dynamic constraints
    dyn_violations_init = ineq_violations_init[-n_dyn:]
    obs_violations_init = ineq_violations_init[:-n_dyn]
    
    initial_progress = jnp.array([
        f(x0),  # objective value
        jnp.max(eq_violations_init),
        jnp.max(ineq_violations_init),
        jnp.mean(eq_violations_init),
        jnp.mean(ineq_violations_init),
        jnp.max(dyn_violations_init),
        jnp.mean(dyn_violations_init),
        jnp.max(obs_violations_init),
        jnp.mean(obs_violations_init),
        jnp.nan,
        jnp.nan,
        jnp.nan,
        jnp.nan,
        0.0
    ])
    
    progress = progress.at[0].set(initial_progress)


    init_state = OptState(
        x=x0,
        lambda_= jnp.zeros(m_eq),
        mu_= jnp.zeros(m_ineq),
        iter_num=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(False, dtype=jnp.bool_),
        progress=progress
    )
    
    # Create step function with fixed parameters
    step_fn = lambda state: step(state, f, grad_f, c_eq, c_ineq, jac_c_eq, jac_c_ineq, robot_info.dim, optim_info.num_points, optim_info.qp_tol, optim_info.eq_tol, optim_info.ineq_tol, optim_info.convergence_tol)
    
    def cond_fn(state: OptState) -> bool:
        return jnp.logical_and(
            state.iter_num < max_iter,
            jnp.logical_not(state.converged)
        )
    

    final_state = jax.lax.while_loop(cond_fn, step_fn, init_state)
    
    final_eq_violations = jnp.abs(c_eq(final_state.x))
    final_ineq_violations = jnp.maximum(0.0, c_ineq(final_state.x))
    
    return OptResult(
        x=final_state.x,
        lambda_=final_state.lambda_,
        mu_=final_state.mu_,
        success=jnp.array(final_state.converged, dtype=jnp.bool_),
        n_iter=jnp.array(final_state.iter_num, dtype=jnp.int32),
        f_val=jnp.array(f(final_state.x), dtype=jnp.float32),
        eq_violation=jnp.array(jnp.max(final_eq_violations) if final_eq_violations.size > 0 else 0.0, dtype=jnp.float32),
        ineq_violation=jnp.array(jnp.max(final_ineq_violations) if final_ineq_violations.size > 0 else 0.0, dtype=jnp.float32),
        progress_history=final_state.progress
    )

@partial(jax.jit, static_argnums=(5)) 
def sqp_solve_batch(world_info: WorldInfo,
    optim_info: OptimInfo,
    robot_info: RobotInfo,
    sdf: jnp.ndarray,
    x0: jnp.ndarray,
    max_iter: int = 3):

    batched_solve = jax.vmap(sqp_solve, in_axes=(None, None, None, None, 0, None, None))
    return batched_solve(world_info, optim_info, robot_info, sdf, x0, max_iter)
