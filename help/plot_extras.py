import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jax.numpy as jnp
from chompax.obs import get_dists_nondiff
from sqp.problem_definition import compute_joint_forces

def plot_joint_forces(traj, robot_info, optim_info, save_path):
    """
    Plot joint forces over time with max force limits.
    
    Args:
        traj: Joint trajectory of shape (num_points, num_joints)
        robot_info: Robot information containing max joint forces
        optim_info: Optimization information
        save_path: Path where to save the plot
    """
    # Ensure trajectory has correct shape
    if len(traj.shape) != 2:
        raise ValueError(f"Expected trajectory shape (num_points, num_joints), got {traj.shape}")
    
    # Compute forces for the trajectory
    time_scaling = optim_info.time_scaling
    forces = compute_joint_forces(traj, time_scaling, robot_info)
    
    # Create figure
    fig = go.Figure()
    
    # Define a color sequence for the joints
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot force for each joint
    time_points = jnp.arange(forces.shape[0])
    for joint_idx in range(robot_info.dim):
        color = colors[joint_idx % len(colors)]  # Cycle through colors if more joints than colors
        
        # Plot joint forces
        fig.add_trace(go.Scatter(
            x=time_points,
            y=jnp.abs(forces[:, joint_idx]),
            name=f'Joint {joint_idx+1}',
            mode='lines',
            line=dict(width=2, color=color)
        ))
        
        # Plot max force limit as a horizontal line with same color
        max_force = robot_info.max_joint_forces[joint_idx]
        fig.add_trace(go.Scatter(
            x=[time_points[0], time_points[-1]],
            y=[max_force, max_force],
            name=f'Max Force Joint {joint_idx+1}',
            line=dict(dash='dash', width=1, color=color),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title='Joint Forces Over Time',
        xaxis_title='Time Step',
        yaxis_title='Force Magnitude',
        hovermode='x unified',
        showlegend=True
    )
    
    # Save the plot
    fig.write_html(save_path)

def plot_min_obstacle_distance(traj, sdf, robot_info, world_info, save_path, init_traj=None):
    """
    Plot minimum distance to obstacles over time.
    
    Args:
        traj: Joint trajectory of shape (num_points, num_joints)
        sdf: Signed distance field
        robot_info: Robot information
        world_info: World information
        save_path: Path where to save the plot
        init_traj: Initial trajectory for comparison (optional), shape (num_points, num_joints)
    """
    # Ensure trajectory has correct shape
    if len(traj.shape) != 2:
        raise ValueError(f"Expected trajectory shape (num_points, num_joints), got {traj.shape}")
    
    if init_traj is not None and len(init_traj.shape) != 2:
        raise ValueError(f"Expected initial trajectory shape (num_points, num_joints), got {init_traj.shape}")
    
    # Compute distances for final trajectory
    distances = jnp.array([get_dists_nondiff(q, sdf, robot_info, world_info) for q in traj])
    
    # Subtract robot radius and safety margin
    radius = robot_info.spheres.r[-1]
    distances = distances - radius
    
    # Get minimum distance at each timestep
    min_distances = jnp.min(distances, axis=1)
    
    # Create figure
    fig = go.Figure()
    
    # Plot minimum distance for optimized trajectory
    time_points = jnp.arange(len(min_distances))
    fig.add_trace(go.Scatter(
        x=time_points,
        y=min_distances,
        name='Optimized Path',
        mode='lines',
        line=dict(width=2)
    ))
    
    # Plot minimum distance for initial trajectory if provided
    if init_traj is not None:
        init_distances = jnp.array([get_dists_nondiff(q, sdf, robot_info, world_info) for q in init_traj])
        init_distances = init_distances - radius
        init_min_distances = jnp.min(init_distances, axis=1)
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=init_min_distances,
            name='Initial Path',
            mode='lines',
            line=dict(width=2, dash='dot')
        ))
    
    # Add safety margin line to show minimum safe distance
    fig.add_trace(go.Scatter(
        x=[time_points[0], time_points[-1]],
        y=[robot_info.safety_margin, robot_info.safety_margin],
        name='Safety Margin',
        line=dict(dash='dash', color='red', width=1),
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title='Minimum Distance to Obstacles Over Time',
        xaxis_title='Time Step',
        yaxis_title='Distance',
        hovermode='x unified',
        showlegend=True
    )
    
    # Save the plot
    fig.write_html(save_path)

def plot_optimization_progress(optim_info, progress_history, n_iter, save_path):
    """
    Plot the optimization progress metrics over iterations.
    
    Args:
        progress_history: Array of shape (max_iter, 14) containing:
            [obj, max_eq, max_ineq, mean_eq, mean_ineq, max_dyn, mean_dyn, max_obs, mean_obs,
            merit_value, merit_derivative, line_search_loops, sigma, hessian]
        n_iter: Number of actual iterations performed
        save_path: Path where to save the plot
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=10, cols=1,
        subplot_titles=(
            'Objective Value',
            'Equality Constraint Violations',
            'Total Inequality Constraint Violations',
            'Dynamic Constraint Violations',
            'Obstacle Constraint Violations',
            'Merit Function Value',
            'Merit Function Derivative',
            'Line Search Iterations',
            'Sigma',
            'Hessian'
        ),
        vertical_spacing=0.05
    )
    
    # Get iteration range
    iterations = jnp.arange(n_iter)
    
    # Plot objective value
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 0],
                  name='Objective', line=dict(width=2)),
        row=1, col=1
    )
    
    # Plot equality violations
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 1],
                  name='Max Equality', line=dict(width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 3],
                  name='Mean Equality', line=dict(width=2, dash='dot')),
        row=2, col=1
    )
    # Add equality tolerance line
    fig.add_trace(
        go.Scatter(x=[iterations[0], iterations[-1]], 
                  y=[optim_info.eq_tol, optim_info.eq_tol],
                  name='Equality Tolerance',
                  line=dict(dash='dash', color='red', width=1)),
        row=2, col=1
    )
    
    # Plot total inequality violations
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 2],
                  name='Max Total Inequality', line=dict(width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 4],
                  name='Mean Total Inequality', line=dict(width=2, dash='dot')),
        row=3, col=1
    )
    # Add inequality tolerance line
    fig.add_trace(
        go.Scatter(x=[iterations[0], iterations[-1]], 
                  y=[optim_info.ineq_tol, optim_info.ineq_tol],
                  name='Inequality Tolerance',
                  line=dict(dash='dash', color='red', width=1)),
        row=3, col=1
    )
    
    # Plot dynamic violations
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 5],
                  name='Max Dynamic', line=dict(width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 6],
                  name='Mean Dynamic', line=dict(width=2, dash='dot')),
        row=4, col=1
    )
    
    # Plot obstacle violations
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 7],
                  name='Max Obstacle', line=dict(width=2)),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 8],
                  name='Mean Obstacle', line=dict(width=2, dash='dot')),
        row=5, col=1
    )
    
    # Plot merit function value
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 9],
                  name='Merit Value', line=dict(width=2)),
        row=6, col=1
    )
    
    # Plot merit function derivative
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 10],
                  name='Merit Derivative', line=dict(width=2)),
        row=7, col=1
    )
    
    # Plot line search iterations
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 11],
                  name='Line Search Iterations', line=dict(width=2)),
        row=8, col=1
    )
    
    # Plot sigma
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 12],
                  name='Sigma', line=dict(width=2)),
        row=9, col=1
    )
    
    # Plot Hessian
    fig.add_trace(
        go.Scatter(x=iterations, y=progress_history[:n_iter, 13],
                  name='Hessian', line=dict(width=2)),
        row=10, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Optimization Progress',
        height=2200,  # Make plot taller to accommodate 10 subplots
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Violation", row=2, col=1)
    fig.update_yaxes(title_text="Violation", row=3, col=1)
    fig.update_yaxes(title_text="Violation", row=4, col=1)
    fig.update_yaxes(title_text="Violation", row=5, col=1)
    fig.update_yaxes(title_text="Value", row=6, col=1)
    fig.update_yaxes(title_text="Value", row=7, col=1)
    fig.update_yaxes(title_text="Count", row=8, col=1)
    fig.update_yaxes(title_text="Value", row=9, col=1)
    fig.update_yaxes(title_text="Value", row=10, col=1)
    
    # Update x-axes labels (only show label on bottom plot)
    for i in range(1, 10):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    fig.update_xaxes(title_text="Iteration", row=10, col=1)
    
    # Save the plot
    fig.write_html(save_path)

def plot_average_joint_forces(trajectories, robot_info, optim_info, save_path):
    """
    Plot average joint forces over time with max force limits, averaged across all runs.
    
    Args:
        trajectories: List of joint trajectories, each of shape (num_points, num_joints)
        robot_info: Robot information containing max joint forces
        optim_info: Optimization information
        save_path: Path where to save the plot
    """
    # Compute forces for each trajectory
    all_forces = []
    for traj in trajectories:
        if traj is not None and not jnp.any(jnp.isnan(traj)):  # Skip failed runs and NaN trajectories
            forces = compute_joint_forces(traj, optim_info.time_scaling, robot_info)
            if not jnp.any(jnp.isnan(forces)):  # Only include non-NaN forces
                all_forces.append(forces)
    
    if not all_forces:  # If no valid forces, return
        print("No valid forces available for plotting")
        return
    
    # Convert to array and compute mean, ignoring NaNs
    all_forces = jnp.stack(all_forces)
    mean_forces = jnp.nanmean(all_forces, axis=0)
    
    # Skip plotting if all means are NaN
    if jnp.all(jnp.isnan(mean_forces)):
        print("All force means are NaN, skipping plot")
        return
    
    # Create figure
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    time_points = jnp.arange(mean_forces.shape[0])
    
    # Plot force for each joint
    for joint_idx in range(robot_info.dim):
        color = colors[joint_idx % len(colors)]
        
        # Plot mean force if not all NaN
        joint_forces = mean_forces[:, joint_idx]
        if not jnp.all(jnp.isnan(joint_forces)):
            fig.add_trace(go.Scatter(
                x=time_points,
                y=joint_forces,
                name=f'Joint {joint_idx+1} (Mean)',
                mode='lines',
                line=dict(width=2, color=color)
            ))
        
        # Plot max force limit
        max_force = robot_info.max_joint_forces[joint_idx]
        fig.add_trace(go.Scatter(
            x=[time_points[0], time_points[-1]],
            y=[max_force, max_force],
            name=f'Max Force Joint {joint_idx+1}',
            line=dict(dash='dash', width=1, color=color)
        ))
    
    fig.update_layout(
        title='Average Joint Forces Over Time (Across All Runs)',
        xaxis_title='Time Step',
        yaxis_title='Force Magnitude',
        hovermode='x unified',
        showlegend=True
    )
    
    fig.write_html(save_path)

def plot_average_min_obstacle_distance(trajectories, sdf, robot_info, world_info, save_path, init_trajectories=None):
    """
    Plot average minimum distance to obstacles over time, averaged across all runs.
    
    Args:
        trajectories: List of joint trajectories, each of shape (num_points, num_joints)
        sdf: Signed distance field
        robot_info: Robot information
        world_info: World information
        save_path: Path where to save the plot
        init_trajectories: List of initial trajectories for comparison (optional)
    """
    # Compute distances for all final trajectories
    all_min_distances = []
    for traj in trajectories:
        if traj is not None and not jnp.any(jnp.isnan(traj)):  # Skip failed runs and NaN trajectories
            distances = jnp.array([get_dists_nondiff(q, sdf, robot_info, world_info) for q in traj])
            if not jnp.any(jnp.isnan(distances)):  # Only include non-NaN distances
                radius = robot_info.spheres.r[-1]
                distances = distances - radius
                min_distances = jnp.min(distances, axis=1)
                if not jnp.any(jnp.isnan(min_distances)):  # Only include non-NaN min distances
                    all_min_distances.append(min_distances)
    
    if not all_min_distances:  # If no valid distances, return
        print("No valid distances available for plotting")
        return
    
    # Convert to array and compute mean, ignoring NaNs
    all_min_distances = jnp.stack(all_min_distances)
    mean_distances = jnp.nanmean(all_min_distances, axis=0)
    
    # Skip plotting if all means are NaN
    if jnp.all(jnp.isnan(mean_distances)):
        print("All distance means are NaN, skipping plot")
        return
    
    # Create figure
    fig = go.Figure()
    time_points = jnp.arange(len(mean_distances))
    
    # Plot mean distance for optimized trajectories
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_distances,
        name='Optimized Path (Mean)',
        mode='lines',
        line=dict(width=2)
    ))
    
    # Do the same for initial trajectories if provided
    if init_trajectories is not None:
        all_init_min_distances = []
        for init_traj in init_trajectories:
            if init_traj is not None and not jnp.any(jnp.isnan(init_traj)):  # Skip NaN trajectories
                init_distances = jnp.array([get_dists_nondiff(q, sdf, robot_info, world_info) for q in init_traj])
                if not jnp.any(jnp.isnan(init_distances)):  # Only include non-NaN distances
                    init_distances = init_distances - radius
                    init_min_distances = jnp.min(init_distances, axis=1)
                    if not jnp.any(jnp.isnan(init_min_distances)):  # Only include non-NaN min distances
                        all_init_min_distances.append(init_min_distances)
        
        if all_init_min_distances:  # Only plot if we have valid initial distances
            all_init_min_distances = jnp.stack(all_init_min_distances)
            mean_init_distances = jnp.nanmean(all_init_min_distances, axis=0)
            
            if not jnp.all(jnp.isnan(mean_init_distances)):  # Only plot if not all NaN
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=mean_init_distances,
                    name='Initial Path (Mean)',
                    mode='lines',
                    line=dict(width=2, dash='dot')
                ))
    
    # Add safety margin line
    fig.add_trace(go.Scatter(
        x=[time_points[0], time_points[-1]],
        y=[robot_info.safety_margin, robot_info.safety_margin],
        name='Safety Margin',
        line=dict(dash='dash', color='red', width=1)
    ))
    
    fig.update_layout(
        title='Average Minimum Distance to Obstacles Over Time (Across All Runs)',
        xaxis_title='Time Step',
        yaxis_title='Distance',
        hovermode='x unified',
        showlegend=True
    )
    
    fig.write_html(save_path)

def add_metric_plot(fig, data, name, row):
    """Helper function to add a metric plot to the figure."""
    fig.add_trace(
        go.Scatter(
            x=jnp.arange(len(data)),
            y=data,
            name=name,
            line=dict(width=2)
        ),
        row=row, col=1
    )

def plot_average_optimization_progress(optim_info, all_progress_histories, all_n_iters, save_path):
    """Plot average optimization progress across multiple runs."""
    # Filter out None or empty histories
    valid_histories = []
    valid_n_iters = []
    for history, n_iter in zip(all_progress_histories, all_n_iters):
        if history is not None and len(history) > 0:
            valid_histories.append(history)
            valid_n_iters.append(n_iter)
    
    if not valid_histories:
        print("No valid optimization histories to plot")
        return
        
    max_n_iter = max(valid_n_iters)
    
    # Pad shorter histories by repeating last state
    padded_histories = []
    for history, n_iter in zip(valid_histories, valid_n_iters):
        # Get the last state
        last_state = history[n_iter - 1]
        # Create padding by repeating last state
        padding = jnp.tile(last_state, (max_n_iter - n_iter, 1))
        # Concatenate original history with padding
        padded = jnp.concatenate([history[:n_iter], padding], axis=0)
        padded_histories.append(padded)
    
    # Stack and compute mean, ignoring NaNs
    all_histories = jnp.stack(padded_histories)
    mean_history = jnp.nanmean(all_histories, axis=0)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=10, cols=1,
        subplot_titles=(
            "Cost",
            "Obstacle Cost",
            "Joint Limit Cost",
            "Boundary Cost",
            "Force Cost",
            "Smoothness Cost",
            "Total Cost",
            "Step Size",
            "Lambda",
            "Convergence",
        ),
        vertical_spacing=0.05,
    )
    
    # Add each metric plot
    add_metric_plot(fig, mean_history[:, 0], "Cost", 1)
    add_metric_plot(fig, mean_history[:, 1], "Obstacle Cost", 2)
    add_metric_plot(fig, mean_history[:, 2], "Joint Limit Cost", 3)
    add_metric_plot(fig, mean_history[:, 3], "Boundary Cost", 4)
    add_metric_plot(fig, mean_history[:, 4], "Force Cost", 5)
    add_metric_plot(fig, mean_history[:, 5], "Smoothness Cost", 6)
    add_metric_plot(fig, mean_history[:, 6], "Total Cost", 7)
    add_metric_plot(fig, mean_history[:, 7], "Step Size", 8)
    add_metric_plot(fig, mean_history[:, 8], "Lambda", 9)
    add_metric_plot(fig, mean_history[:, 9], "Convergence", 10)
    
    # Update layout
    fig.update_layout(
        height=2000,
        showlegend=True,
        title_text="Average Optimization Progress",
    )
    
    # Save plot
    fig.write_html(save_path) 