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


class OptState(NamedTuple):
    x: jnp.ndarray
    lambda_: jnp.ndarray  
    mu_: jnp.ndarray     
    sigma: jnp.float32
    trust_radius: jnp.float32
    B: jnp.ndarray
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
    slacks_eq_pos: jnp.ndarray  
    slacks_eq_neg: jnp.ndarray  
    slacks_ineq: jnp.ndarray    

@contextmanager
def profile_context(enable_profiling=False):
    if enable_profiling:
        try:
            print("Starting profiling...")
            jax.profiler.start_server(9999)  # Start the profiling server
            jax.profiler.start_trace("./tensorboard_logs")
            yield
        finally:
            print("Stopping profiling...")
            jax.profiler.stop_trace()
    else:
        yield

@partial(jax.jit, static_argnums=(8))
def solve_elastic_qp(H, g, A_eq, b_eq, A_ineq, b_ineq, sigma, trust_radius, qp_tol):

    n = g.shape[0]
    m_eq = A_eq.shape[0] if A_eq.size > 0 else 0
    m_ineq = A_ineq.shape[0] if A_ineq.size > 0 else 0
    
    
    H_aug = jnp.zeros((n + 2*m_eq + m_ineq, n + 2*m_eq + m_ineq))
    H_aug = H_aug.at[:n, :n].set(H)
    
    g_aug = jnp.zeros(n + 2*m_eq + m_ineq)
    g_aug = g_aug.at[:n].set(g)
    g_aug = g_aug.at[n:n+2*m_eq].set(sigma * jnp.ones(2*m_eq))  
    g_aug = g_aug.at[n+2*m_eq:].set(sigma * jnp.ones(m_ineq))   # Cost for t
    
    A_eq_aug = jnp.zeros((m_eq, n + 2*m_eq + m_ineq))
    A_eq_aug = A_eq_aug.at[:, :n].set(A_eq)
    
    row_indices = jnp.arange(m_eq)
    v_col_indices = n + row_indices
    w_col_indices = n + m_eq + row_indices
    
    A_eq_aug = A_eq_aug.at[row_indices, v_col_indices].set(-1.0)  # -v_i
    A_eq_aug = A_eq_aug.at[row_indices, w_col_indices].set(1.0)   # +w_i
    b_eq_aug = -b_eq
    
    
    total_ineq = m_ineq + 2*n + 2*m_eq + m_ineq
    
    A_ineq_aug = jnp.zeros((total_ineq, n + 2*m_eq + m_ineq))
    b_ineq_aug = jnp.zeros(total_ineq)
    
    A_ineq_aug = A_ineq_aug.at[:m_ineq, :n].set(A_ineq)
    
    row_indices = jnp.arange(m_ineq)
    t_col_indices = n + 2*m_eq + row_indices
    
    A_ineq_aug = A_ineq_aug.at[row_indices, t_col_indices].set(-1.0)  # -t_i
    b_ineq_aug = b_ineq_aug.at[:m_ineq].set(-b_ineq)
    
    upper_bound_rows = jnp.arange(m_ineq, m_ineq + n)
    upper_bound_cols = jnp.arange(n)
    A_ineq_aug = A_ineq_aug.at[upper_bound_rows, upper_bound_cols].set(1.0)
    b_ineq_aug = b_ineq_aug.at[upper_bound_rows].set(trust_radius)
    
    lower_bound_rows = jnp.arange(m_ineq + n, m_ineq + 2*n)
    lower_bound_cols = jnp.arange(n)
    A_ineq_aug = A_ineq_aug.at[lower_bound_rows, lower_bound_cols].set(-1.0)
    b_ineq_aug = b_ineq_aug.at[lower_bound_rows].set(trust_radius)
    
    start_idx = m_ineq + 2*n
    
    v_rows = jnp.arange(start_idx, start_idx + m_eq)
    v_cols = jnp.arange(n, n + m_eq)
    A_ineq_aug = A_ineq_aug.at[v_rows, v_cols].set(-1.0)
    
    w_rows = jnp.arange(start_idx + m_eq, start_idx + 2*m_eq)
    w_cols = jnp.arange(n + m_eq, n + 2*m_eq)
    A_ineq_aug = A_ineq_aug.at[w_rows, w_cols].set(-1.0)
    
    t_rows = jnp.arange(start_idx + 2*m_eq, total_ineq)
    t_cols = jnp.arange(n + 2*m_eq, n + 2*m_eq + m_ineq)
    A_ineq_aug = A_ineq_aug.at[t_rows, t_cols].set(-1.0)
    
    b_ineq_aug = b_ineq_aug.at[start_idx:total_ineq].set(0.0)
    
    sol, s, z, y, converged, iters = qpax.solve_qp(
        H_aug, g_aug, A_eq_aug, b_eq_aug, A_ineq_aug, b_ineq_aug,
        solver_tol=qp_tol
    )
    
    d = sol[:n]
    v = sol[n:n+m_eq]
    w = sol[n+m_eq:n+2*m_eq]
    t = sol[n+2*m_eq:]
    
    error = jnp.where(converged, jnp.float32(1e-4), jnp.float32(1.0))
    
    return QPSolution(
        x=d, 
        y=y[:m_eq] if m_eq > 0 else jnp.array([]), 
        z=z[:m_ineq] if m_ineq > 0 else jnp.array([]),  
        error=error,
        slacks_eq_pos=v,
        slacks_eq_neg=w,
        slacks_ineq=t
    )

@partial(jax.jit, static_argnums=(1,2,3,4,5,6,7,8,9,10,11,12))
def step(state: OptState, f, grad_f, c_eq, c_ineq, jac_c_eq, jac_c_ineq, dim: int, num_points: int, qp_tol:float, eq_tol:float, ineq_tol:float, convergence_tol:float) -> OptState:
    x, lambda_, mu_, sigma_prev, trust_radius, B, iter_num, _, progress = state
    
    g = jnp.squeeze(grad_f(x))      # Squeeze out extra dimensions

    A_eq = jnp.squeeze(jac_c_eq(x))    
    b_eq = c_eq(x)
    A_ineq = jnp.squeeze(jac_c_ineq(x)) 
    b_ineq = c_ineq(x)
    
    qp_sol = solve_elastic_qp(B, g, A_eq, b_eq, A_ineq, b_ineq, sigma_prev, trust_radius, qp_tol)
    
    dx = qp_sol.x
    lambda_next = qp_sol.y
    mu_next = qp_sol.z
    v = qp_sol.slacks_eq_pos
    w = qp_sol.slacks_eq_neg
    t = qp_sol.slacks_ineq
    
    eq_viol = jnp.sum(v + w) if v.size > 0 else 0.0
    ineq_viol = jnp.sum(t) if t.size > 0 else 0.0
    total_viol = eq_viol + ineq_viol
    
    nu = jnp.float32(0.01)  
    psi = jnp.float32(0.5)  

    def h_function(x):
        return jnp.sum(jnp.abs(c_eq(x))) + jnp.sum(jnp.maximum(0.0, c_ineq(x)))

    def choose_sigma(x, dx, g, H, p):
        hes = dx.T @ H @ dx
        m = 1.0
        h_val = h_function(x)
        denominator = (1-p) * jnp.maximum(h_val, 1e-10)
        sigma = (jnp.dot(g, dx) + (m/2) * hes) / denominator
        return jnp.where(sigma_prev > sigma, sigma_prev, sigma + 1)  # some slack
    
    sigma = choose_sigma(x, dx, g, B, 0.5)

    def merit(x, sigma):
        return f(x) + sigma * h_function(x)
    
    def model_function(d):
        obj_model = f(x) + jnp.dot(g, d) + 0.5 * d.T @ B @ d
        
        eq_const = c_eq(x)
        eq_linearized = eq_const + A_eq @ d
        eq_model = sigma_prev * jnp.sum(jnp.abs(eq_linearized))
        
        ineq_const = c_ineq(x)
        ineq_linearized = ineq_const + A_ineq @ d
        ineq_model = sigma_prev * jnp.sum(jnp.maximum(0.0, ineq_linearized))
        
        return obj_model + eq_model + ineq_model
    
    model_at_zero = model_function(jnp.zeros_like(dx))
    model_at_dx = model_function(dx)
    pred_reduction = model_at_zero - model_at_dx

    merit_current = merit(x, sigma_prev)
    merit_next = merit(x + dx, sigma_prev)
    
    actual_reduction = merit_current - merit_next
    
    rho = jnp.where(pred_reduction > 0, 
                   actual_reduction / pred_reduction, 
                   jnp.float32(-1.0))
    
    trust_radius_next = jnp.where(rho > nu, 
                                 2.0 * trust_radius,  
                                 psi * trust_radius)                     
    
    x_next = jnp.where(rho > nu, x + dx, x)
    lambda_next = jnp.where(rho > nu, lambda_next, lambda_)
    mu_next = jnp.where(rho > nu, mu_next, mu_)
    
    def lagrangian(x_, lambda_, mu_):
        return f(x_) + jnp.sum(lambda_ * c_eq(x_)) + jnp.sum(mu_ * c_ineq(x_))

    def update_B():
        s = dx
        
        grad_L_cur = jax.grad(lambda x_: lagrangian(x_, lambda_next, mu_next))(x)
        grad_L_next = jax.grad(lambda x_: lagrangian(x_, lambda_next, mu_next))(x_next)
        y = grad_L_next - grad_L_cur
        
        theta = jnp.float32(1.0)
        sy = jnp.dot(s, y)
        sBs = s.T @ B @ s
        
        theta = jnp.where(sy >= 0.2 * sBs, 
                         jnp.float32(1.0),
                         (0.8 * sBs) / (sBs - sy))
        
        r = theta * y + (1.0 - theta) * (B @ s)
        
        Bs = B @ s
        return B + jnp.outer(r, r) / jnp.dot(s, r) - jnp.outer(Bs, Bs) / sBs
    
    B_next = jnp.where(rho > nu, update_B(), B)
    
    eq_violations = jnp.abs(c_eq(x_next))
    ineq_violations = jnp.maximum(0.0, c_ineq(x_next))


    feasibilty = jnp.logical_and(jnp.max(eq_violations) < eq_tol, jnp.max(ineq_violations) < ineq_tol)

    #sigma = jnp.where(feasibilty, jnp.maximum(sigma / 2.0, 1.0), sigma)

    
    # Check for convergence
    converged = jnp.logical_or(
        jnp.linalg.norm(dx) < convergence_tol, 
        trust_radius_next < convergence_tol
    )
    
    n_dyn = dim * num_points  
    obs_violations = ineq_violations[:-n_dyn]
    dyn_violations = ineq_violations[-n_dyn:]
    
    # Update progress metrics
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
        merit_current,  # current merit value
        trust_radius,  # trust radius
        jnp.where(rho > nu, 1.0, 0.0),  # step accepted
        sigma,
        dx@B@dx
    ])
    
    # Update progress history
    progress = progress.at[iter_num + 1].set(progress_next)
    
    return OptState(
        x=x_next,
        lambda_=lambda_next,
        mu_=mu_next,
        sigma=sigma,
        trust_radius=trust_radius_next,
        B=B_next,
        iter_num=iter_num + jnp.array(1, dtype=jnp.int32),
        converged=jnp.array(converged, dtype=jnp.bool_),
        progress=progress
    )

@partial(jax.jit, static_argnums=(5))  # Make max_iter static
def sqp_solve(
    world_info: WorldInfo,
    optim_info: OptimInfo,
    robot_info: RobotInfo,
    sdf: jnp.ndarray,
    x0: jnp.ndarray,
    max_iter: int = 3,
    initial_trust_radius: float = 0.01,
    initial_hessian_scale: float = 5.0,
    initial_sigma: float = 100.0
) -> OptResult:
    traj_shape = (optim_info.num_points, robot_info.dim)
    x0_reshaped = x0.reshape(traj_shape)
    start = x0_reshaped[0]  
    end = x0_reshaped[-1]  

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
    
    n_dyn = robot_info.dim * optim_info.num_points  
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
        0.0,  # initial merit value 
        initial_trust_radius,  # initial trust radius
        0.0,   # step accepted yes\no
        initial_sigma,
        0.0
    ])
    
    progress = progress.at[0].set(initial_progress)


    def calculate_initial_hessian(x):
        hessian = jax.hessian(f)(x)
        
        eigenvalues = jnp.linalg.eigvalsh(hessian)
        min_eig = jnp.min(eigenvalues)
        
        min_eig_threshold = 1e-4
        shift = jnp.maximum(0.0, min_eig_threshold - min_eig)
        
        n = hessian.shape[0]
        regularized_hessian = hessian + shift * jnp.eye(n)
        
        return regularized_hessian
    


    B0 = jnp.where(initial_hessian_scale == -1.0, calculate_initial_hessian(x0), initial_hessian_scale * jnp.eye(len(x0)))

    init_state = OptState(
        x=x0,
        lambda_=jnp.zeros(m_eq),
        mu_=jnp.zeros(m_ineq),
        sigma=initial_sigma,
        trust_radius=jnp.float32(initial_trust_radius),
        B=B0,
        iter_num=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(False, dtype=jnp.bool_),
        progress=progress
    )
    
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
    max_iter: int = 3,
    initial_trust_radius: float = 0.01,
    initial_hessian_scale: float = 5.0,
    initial_sigma: float = 100.0):

    batched_solve = jax.vmap(sqp_solve, in_axes=(None, None, None, None, 0, None))
    return batched_solve(world_info, optim_info, robot_info, sdf, x0, max_iter, initial_trust_radius, initial_hessian_scale, initial_sigma)