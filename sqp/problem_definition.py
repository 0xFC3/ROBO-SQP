import jax.numpy as jnp
from jax import jit, vmap, lax
from chompax.rob import get_sphere_pos, get_tcp_pos, forward_kinematic_StaticArm
from chompax.obs import get_dists_nondiff, subtract_radius
from chompax.traj import get_K_1, get_Abc, traj_cost
from chompax.structures import RobotInfo, WorldInfo, OptimInfo
from jax.typing import ArrayLike
from functools import partial
import jax

def _reshape_traj(flat_traj: ArrayLike, num_points: int, dim: int) -> jnp.ndarray:
    return flat_traj.reshape(num_points, dim)

@jit
def objective_function(flat_traj: ArrayLike, robot_info: RobotInfo, optim_info: OptimInfo) -> jnp.ndarray:

    traj = _reshape_traj(flat_traj, optim_info.num_points, robot_info.dim)
    
    # Calculate smoothness cost
    K_1 = get_K_1(robot_info, optim_info)
    A, b, c = get_Abc(traj, K_1)
    smoothness_cost = traj_cost(traj, A, b, c)
    

    # Combine costs
    total_cost = smoothness_cost 
    return total_cost

@jit
def inequality_constraints(flat_traj: ArrayLike, sdf: ArrayLike, robot_info: RobotInfo, world_info: WorldInfo, optim_info: OptimInfo) -> jnp.ndarray:

    traj = _reshape_traj(flat_traj, optim_info.num_points, robot_info.dim)
    
    dists = vmap(get_dists_nondiff, in_axes=(0, None, None, None))(
        traj, sdf, robot_info, world_info
    )
    radius = robot_info.spheres.r[-1] + robot_info.safety_margin
    dists = subtract_radius(dists, radius)
    
    obstacle_scale = 10.0   # Scale factor to make obstacle constraints more significant
    obstacle_constraints = -dists.ravel()
    obstacle_constraints = obstacle_constraints * obstacle_scale
    
    joint_limits_lower = robot_info.limits[:, 0]
    joint_limits_upper = robot_info.limits[:, 1]
    
    lower_bound_violations = (joint_limits_lower) - traj
    upper_bound_violations = traj - (joint_limits_upper)
    lower_bound_violations = lower_bound_violations.reshape(-1) 
    upper_bound_violations = upper_bound_violations.reshape(-1) 
    
    force_constraints = dynamic_constraints(flat_traj, robot_info, optim_info)
    
    return jnp.concatenate([
        obstacle_constraints, 
        lower_bound_violations, 
        upper_bound_violations,
        force_constraints
    ])

@jit
def equality_constraints(flat_traj: ArrayLike, start: ArrayLike, end: ArrayLike, optim_info: OptimInfo, robot_info: RobotInfo) -> jnp.ndarray:

    traj = _reshape_traj(flat_traj, optim_info.num_points, robot_info.dim)
    start_error = traj[0] - start
    end_error = traj[-1] - end
    return jnp.concatenate([start_error, end_error])

@jit
def forward_kinematics_custom(q: ArrayLike, link_lengths: ArrayLike, anchor: ArrayLike) -> jnp.ndarray:

    q_cum = jnp.cumsum(q)  # cumulative sum of joint angles
    x = jnp.cos(q_cum) * link_lengths
    y = jnp.sin(q_cum) * link_lengths
    x = anchor[0] + jnp.sum(x)
    y = anchor[1] + jnp.sum(y)
    return jnp.array([x, y])

@jit
def compute_joint_velocities_accelerations(traj: ArrayLike, time_scaling: float) -> tuple:
    dt = time_scaling
    
    velocities = jnp.zeros_like(traj)
    
    central_velocities = (traj[2:] - traj[:-2]) / (2 * dt)
    velocities = velocities.at[1:-1].set(central_velocities)
    
    accelerations = jnp.zeros_like(traj)
    
    second_derivatives = (traj[2:] - 2 * traj[1:-1] + traj[:-2]) / (dt * dt)
    accelerations = accelerations.at[1:-1].set(second_derivatives)
        
    first_accel = (2 * traj[0] - 5 * traj[1] + 4 * traj[2] - traj[3]) / (dt * dt)
    accelerations = accelerations.at[0].set(first_accel)

    last_accel = (2 * traj[-1] - 5 * traj[-2] + 4 * traj[-3] - traj[-4]) / (dt * dt)
    accelerations = accelerations.at[-1].set(last_accel)
    
    return velocities, accelerations

@jit
def compute_link_transforms(q: ArrayLike, robot_info: RobotInfo) -> tuple:

    positions = forward_kinematic_StaticArm(q, True, robot_info)
    
    q_cum = jnp.cumsum(q)
    cos_q = jnp.cos(q_cum)
    sin_q = jnp.sin(q_cum)
    
    rotations = jnp.stack([
        jnp.stack([cos_q, -sin_q], axis=-1),
        jnp.stack([sin_q, cos_q], axis=-1)
    ], axis=-2)
    
    return positions, rotations

@jit
def rnea_forward_pass(q: ArrayLike, qd: ArrayLike, qdd: ArrayLike, robot_info: RobotInfo) -> tuple:

    num_joints = q.shape[0]
    link_length = robot_info.link_length
    
    positions, rotations = compute_link_transforms(q, robot_info)
    

    w = jnp.zeros(num_joints)
    alpha = jnp.zeros(num_joints)
    v = jnp.zeros((num_joints, 2))
    a = jnp.zeros((num_joints, 2))
    
    def body_fun(i, carry):
        w_prev, alpha_prev, v_prev, a_prev, w, alpha, v, a = carry
        
        qi = q[i]
        qdi = qd[i]
        qddi = qdd[i]

        R_i = rotations[i]
        r_cm = R_i @ jnp.array([link_length / 2.0, 0.0])

        w_i = w_prev + qdi
        
        alpha_i = alpha_prev + qddi
        
        v_joint = v_prev

        r_cm = jnp.array([link_length / 2.0, 0.0])
        
        v_i = v_joint + jnp.array([-w_i * r_cm[1], w_i * r_cm[0]])

       
        a_i = a_prev + jnp.array([-alpha_i * r_cm[1], alpha_i * r_cm[0]]) + \
              jnp.array([-w_i**2 * r_cm[0], -w_i**2 * r_cm[1]])
        
        w = w.at[i].set(w_i)
        alpha = alpha.at[i].set(alpha_i)
        v = v.at[i].set(v_i)
        a = a.at[i].set(a_i)
        
        return w_i, alpha_i, v_i, a_i, w, alpha, v, a
    
    w_0 = 0.0  
    alpha_0 = 0.0  
    v_0 = jnp.zeros(2)  
    a_0 = jnp.zeros(2)  
    
    _, _, _, _, w, alpha, v, a = lax.fori_loop(
        0, num_joints, 
        lambda i, carry: body_fun(i, carry),
        (w_0, alpha_0, v_0, a_0, w, alpha, v, a)
    )
    
    return v, w, alpha, a, positions, rotations

@jit
def rnea_backward_pass(v: ArrayLike, w: ArrayLike, alpha: ArrayLike, a: ArrayLike, 
                       positions: ArrayLike, rotations: ArrayLike, 
                       robot_info: RobotInfo) -> jnp.ndarray:

    num_joints = w.shape[0]
    link_length = robot_info.link_length
    link_masses = robot_info.link_masses
    
    f = jnp.zeros((num_joints, 2))
    tau = jnp.zeros(num_joints)
    
    def body_fun(i, carry):
        f_next, tau_next, f, tau = carry
        
        j = num_joints - i - 1
        
        m_j = link_masses[j]
        
        I_j = (1.0/12.0) * m_j * link_length**2
        
        f_j = m_j * a[j]
        
        tau_j = I_j * alpha[j]

        r_next = positions[j+1] - positions[j]
        
        f_j = jnp.where(j < num_joints - 1, f_j + f_next, f_j)
        tau_j = jnp.where(j < num_joints - 1, tau_j + tau_next, tau_j)
        
        r_next = jnp.array([link_length, 0.0])
        
        cross_product = r_next[0] * f_next[1] - r_next[1] * f_next[0]
        tau_j = tau_j + jnp.where(j < num_joints - 1, cross_product, 0.0)
        
        f = f.at[j].set(f_j)
        tau = tau.at[j].set(tau_j)
        
        return f_j, tau_j, f, tau
    
    f_n = jnp.zeros(2)
    tau_n = 0.0
    
    _, _, f, tau = lax.fori_loop(
        0, num_joints,
        lambda i, carry: body_fun(i, carry),
        (f_n, tau_n, f, tau)
    )
    
    return tau

@jit
def compute_joint_forces(traj: ArrayLike, time_scaling: float, robot_info: RobotInfo) -> jnp.ndarray:
    velocities, accelerations = compute_joint_velocities_accelerations(traj, time_scaling)
    
    def compute_forces_at_point(point_idx):
        q = traj[point_idx]
        qd = velocities[point_idx]
        qdd = accelerations[point_idx]
        
        v, w, alpha, a, positions, rotations = rnea_forward_pass(q, qd, qdd, robot_info)
        tau = rnea_backward_pass(v, w, alpha, a, positions, rotations, robot_info)
        
        return jnp.abs(tau)
    
    forces = vmap(compute_forces_at_point)(jnp.arange(traj.shape[0]))
    
    return forces

@jit
def dynamic_constraints(flat_traj: ArrayLike, robot_info: RobotInfo, optim_info: OptimInfo) -> jnp.ndarray:
    traj = _reshape_traj(flat_traj, optim_info.num_points, robot_info.dim)
    
    forces = compute_joint_forces(traj, optim_info.time_scaling, robot_info)
    
    def create_joint_constraints(joint_idx):
        max_force = robot_info.max_joint_forces[joint_idx]
        joint_forces = forces[:, joint_idx]
        return jnp.concatenate([
            -max_force + joint_forces,  # force <= max_force
        ])
    
    constraints = vmap(create_joint_constraints)(jnp.arange(robot_info.dim))
    return jnp.ravel(constraints)
