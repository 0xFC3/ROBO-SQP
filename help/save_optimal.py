from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", False)
config.update('jax_platform_name', 'gpu')

import os
import argparse
import numpy as np
from rokin import robots
from wzk import sql2
import jax
import jax.numpy as jnp
from jax import random
import sys
# Go up one level from the current file (help/) to the root, where chompax/ is located
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chompax.structures import WorldInfo, OptimInfo, RobotInfo, Spheres_Jax
from chompax.obs import obstacle_img2dist_img
from chompax.twod_plotly import RobotVis2DPlotly
from sqp.problem_definition import objective_function, inequality_constraints, equality_constraints
from sqp_solver import SQP
import time
import plotly.graph_objects as go
import sqlite3

def save_trajectory_plot(traj, world, world_info, robot_info, optim_info, path_id, save_dir='plots'):
    """Generate and save a plot of the trajectory."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visualization
    viz = RobotVis2DPlotly(traj, world, world_info, robot_info, optim_info)
    try:
        viz.plot(100, show_plot=False)
        viz.save_fig(os.path.join(save_dir, f'optimal_path_{path_id}.html'))
    except Exception as e:
        print(f"Error saving plot: {e}")

def optimize_and_save_paths(num_paths=10, random_selection=False, plot_paths=False, 
                           max_iter=100, batch_name='optimal_paths', calculate_both=True,
                           plot_interval=1):
    """
    Optimize paths from the database and save the optimal solutions back to the database.
    
    Args:
        num_paths (int): Number of paths to optimize
        random_selection (bool): If True, select random paths from the database
        plot_paths (bool): If True, generate and save plots of the optimal paths
        max_iter (int): Maximum number of iterations for optimization
        batch_name (str): Name for the output directory if plotting
        calculate_both (bool): If True, calculate both normal and high time scaling optimizations
        plot_interval (int): Plot every Nth path (e.g., 10 means plot every 10th path)
    """
    # Load database
    assert os.getenv('ROBDATA_PATH') is not None, "Set the environment variable ROBDATA_PATH"
    db_path = os.getenv('ROBDATA_PATH') + '/StaticArm04_2.db'
    
    # Constants
    n_waypoints = 20
    n_dof = 4

    # Create mask for optimization (only intermediate points are optimized)
    mask = jnp.vstack(
        [jnp.zeros((1, 1)),
         jnp.ones((n_waypoints-2, 1)),
         jnp.zeros((1, 1))]
    )
    
    # Load worlds
    worlds = sql2.get_values_sql(file=db_path, table='worlds', columns="img_cmp")
    worlds = sql2.compressed2img(img_cmp=worlds, shape=(64, 64), dtype='bool')
    
    # Get total number of paths in the database
    total_paths = sql2.get_values_sql(file=db_path, table='paths', columns="COUNT(*)")
    total_paths = int(total_paths)
    print(f"Total paths in database: {total_paths}")
    
    # Select paths to optimize
    if random_selection:
        # Select random path indices
        np.random.seed(42)  # For reproducibility
        path_indices = np.random.choice(total_paths, size=min(num_paths, total_paths), replace=False)
        path_indices = sorted(path_indices)  # Sort for cleaner output
    else:
        # Select first num_paths
        path_indices = range(min(num_paths, total_paths))
    
    # Load selected paths
    rows = list(path_indices)
    i_world, q, path_ids = sql2.get_values_sql(file=db_path, table='paths', 
                                  columns=["world_i32", "q_f32", "rowid"], rows=rows)
    q = q.reshape(-1, n_waypoints, n_dof)
    
    # Setup robot and world parameters
    robot = robots.StaticArm(n_dof, 0.15, widths=0.05)
    limits = np.array([[-1, 1], [-1, 1]])  # Cube world
    voxel_size = (limits[:, 1] - limits[:, 0]) / np.array(worlds[0].shape)
    voxel_size = np.min(voxel_size)
    
    # Create info structures
    world_info = WorldInfo(
        limits=limits,
        voxel_size=voxel_size,
        num_vox=worlds[0].shape[0]
    )
    
    # Create output directory for plots if needed
    if plot_paths:
        output_dir = f"optimal_results/{batch_name}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Define time scaling parameters
    time_scaling_normal = 0.15
    time_scaling_high = 100.0
    
    # Create optimization info structures for both time scaling parameters
    optim_info_normal = OptimInfo(
        num_points=n_waypoints,
        num_iters=max_iter,
        mask=mask,
        convergence_tol=1e-4,
        qp_tol=1e-4,
        eq_tol=1e-3,
        ineq_tol=1e-4,
        time_scaling=time_scaling_normal
    )
    
    optim_info_high = OptimInfo(
        num_points=n_waypoints,
        num_iters=max_iter,
        mask=mask,
        convergence_tol=1e-4,
        qp_tol=1e-4,
        eq_tol=1e-3,
        ineq_tol=1e-4,
        time_scaling=time_scaling_high
    )
    
    sph = Spheres_Jax(robot.spheres.x, robot.spheres.r, robot.spheres.f_idx)
    robot_info = RobotInfo(
        dim=robot.n_dof,
        limits=robot.limits,
        link_length=robot.lengths,
        spheres=sph,
        type="staticarm",
        safety_margin=0.04,
        next_frame_idx=robot.chain.next_f_idx,
        link_masses=jnp.array([5.0, 5.0, 5.0, 5.0]),  
        max_joint_forces=jnp.array([1.0, 0.5, 0.2, 0.1])
    )

    sqp_solver_normal = SQP(method="trust_region", max_iter=max_iter, optim_info=optim_info_normal)
    sqp_solver_high = SQP(method="trust_region", max_iter=max_iter, optim_info=optim_info_high)
    
    # Warm-up runs
    print(f"Performing warm-up runs...")
    world = worlds[i_world[0]]
    sdf = obstacle_img2dist_img(world, world_info)
    init_traj = q[0].reshape(-1)
    
    print(f"  Warm-up with normal time scaling (time_scaling={time_scaling_normal})...")
    _ = sqp_solver_normal.solve(
        world_info=world_info,
        robot_info=robot_info,
        sdf=sdf,
        x0=init_traj

    )
    
    print(f"  Warm-up with high time scaling (time_scaling={time_scaling_high})...")
    _ = sqp_solver_high.solve(
        world_info=world_info,
        robot_info=robot_info,
        sdf=sdf,
        x0=init_traj
    )
    
    print("Warm-up completed")
    
    # Optimize paths and save results
    normal_optimal_paths = []
    high_optimal_paths = []
    normal_success_count = 0
    high_success_count = 0
    
    print(f"\nOptimizing {len(path_indices)} paths with both normal and high time scaling...")
    for i, path_idx in enumerate(path_indices):
        path_id = int(path_ids[i])
        world_id = int(i_world[i])
        world = worlds[world_id]
        initial_path = q[i]
        
        print(f"\nPath {i+1}/{len(path_indices)} (Database ID: {path_id}, World ID: {world_id})")
        
        # Compute signed distance field
        sdf = obstacle_img2dist_img(world, world_info)
        
        # Check if start and end positions are legal (not in collision)
        start_pos = initial_path[0]
        end_pos = initial_path[-1]
        
        # Create dummy trajectories for start and end
        start_traj = jnp.tile(start_pos, (optim_info_normal.num_points, 1))
        end_traj = jnp.tile(end_pos, (optim_info_normal.num_points, 1))
        
        # Get obstacle constraints for start and end positions
        def ineq_fn(traj, optim_info):
            return inequality_constraints(traj, sdf, robot_info, world_info, optim_info)
            
        start_ineq = ineq_fn(start_traj.reshape(-1), optim_info_normal)
        end_ineq = ineq_fn(end_traj.reshape(-1), optim_info_normal)
        
        # Split constraints - only consider obstacle constraints (not dynamic constraints)
        n_dyn = robot_info.dim * optim_info_normal.num_points  # Number of dynamic constraints
        start_obs_ineq = start_ineq[:-n_dyn]  # Exclude dynamic constraints
        end_obs_ineq = end_ineq[:-n_dyn]      # Exclude dynamic constraints
        
        # Since we repeated the configuration, we only need to check the first set of obstacle constraints
        n_obs_per_point = len(start_obs_ineq) // optim_info_normal.num_points
        start_illegal = bool(jnp.any(start_obs_ineq[:n_obs_per_point] > 0))
        end_illegal = bool(jnp.any(end_obs_ineq[:n_obs_per_point] > 0))
        
        if start_illegal or end_illegal:
            print(f"  Skipping path - {'Start' if start_illegal else 'End'} position in collision with obstacle")
            continue
        
        # Prepare initial trajectory
        init_traj = initial_path.reshape(-1)
        
        # Run optimization with normal time scaling
        print("  Running optimization with normal time scaling...", end='', flush=True)
        start_time = time.time()
        try:
            # Run optimization
            opt_results_normal = sqp_solver_normal.solve(
                world_info=world_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=init_traj
            )
            
            jax.block_until_ready(opt_results_normal.x)
            duration_normal = time.time() - start_time
            
            # Extract optimal trajectory
            optimal_traj_normal = opt_results_normal.x.reshape(n_waypoints, n_dof)

            # Check for NaN values
            if jnp.any(jnp.isnan(optimal_traj_normal)):
                print(f" failed - NaN values in result (took {duration_normal:.2f}s)")
            else:
                # Compute violations to check feasibility
                def eq_fn(traj, optim_info):
                    return equality_constraints(traj, initial_path[0], initial_path[-1], optim_info, robot_info)
                    
                eq_violation_max = float(jnp.max(jnp.abs(eq_fn(opt_results_normal.x.reshape(-1), optim_info_normal))))
                ineq_violation_max = float(jnp.max(jnp.maximum(0.0, ineq_fn(opt_results_normal.x.reshape(-1), optim_info_normal))))
                
                # Check feasibility
                feasibility = jnp.logical_and(
                    eq_violation_max < optim_info_normal.eq_tol,
                    ineq_violation_max < optim_info_normal.ineq_tol
                )
                
                success_normal = bool(feasibility)
                
                if success_normal:
                    print(f" success in {duration_normal:.2f}s")
                    normal_success_count += 1
                    
                    # Save optimal trajectory to database
                    optimal_traj_flat_normal = np.array(optimal_traj_normal.reshape(-1), dtype=np.float32)
                    
                    # Store trajectory as binary blob using direct SQL
                    try:
                        # Connect to the database
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        # First check if the column exists
                        cursor.execute("PRAGMA table_info(paths)")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        if 'optimal_f32' not in columns:
                            print("    Adding optimal_f32 column to paths table")
                            cursor.execute("ALTER TABLE paths ADD COLUMN optimal_f32 BLOB")
                        
                        # Convert array to binary blob
                        blob_data = optimal_traj_flat_normal.tobytes()
                        
                        # Update the database with the binary data
                        cursor.execute("UPDATE paths SET optimal_f32 = ? WHERE rowid = ?", 
                                      (sqlite3.Binary(blob_data), path_id))
                        
                        conn.commit()
                        conn.close()
                        print("    Normal trajectory stored successfully")
                    except Exception as storage_error:
                        print(f"    Error storing normal trajectory: {type(storage_error).__name__}: {str(storage_error)}")
                    
                    # Store for potential plotting
                    normal_optimal_paths.append({
                        'path_id': path_id,
                        'world_id': world_id,
                        'world': world,
                        'initial_path': initial_path,
                        'optimal_path': optimal_traj_normal
                    })
                    
                    # Generate plot if requested - only plot every plot_interval paths
                    if plot_paths and (i % plot_interval == 0):
                        save_trajectory_plot(
                            optimal_traj_normal, world, world_info, robot_info, optim_info_normal,
                            path_id=f"{path_id}_normal", save_dir=f"{output_dir}/plots"
                        )
                else:
                    print(f" failed - constraints not satisfied (took {duration_normal:.2f}s)")
                    print(f"    Max equality violation: {eq_violation_max:.2e}")
                    print(f"    Max inequality violation: {ineq_violation_max:.2e}")
        except Exception as e:
            print(f" error: {str(e)}")
            
        # Run optimization with high time scaling
        print("  Running optimization with high time scaling...", end='', flush=True)
        start_time = time.time()
        try:
            # Run optimization
            opt_results_high = sqp_solver_high.solve(
                world_info=world_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=init_traj
            )
            
            jax.block_until_ready(opt_results_high.x)
            duration_high = time.time() - start_time
            
            # Extract optimal trajectory
            optimal_traj_high = opt_results_high.x.reshape(n_waypoints, n_dof)

            # Check for NaN values
            if jnp.any(jnp.isnan(optimal_traj_high)):
                print(f" failed - NaN values in result (took {duration_high:.2f}s)")
            else:
                # Compute violations to check feasibility
                eq_violation_max = float(jnp.max(jnp.abs(eq_fn(opt_results_high.x.reshape(-1), optim_info_high))))
                ineq_violation_max = float(jnp.max(jnp.maximum(0.0, ineq_fn(opt_results_high.x.reshape(-1), optim_info_high))))
                
                # Check feasibility
                feasibility = jnp.logical_and(
                    eq_violation_max < optim_info_high.eq_tol,
                    ineq_violation_max < optim_info_high.ineq_tol
                )
                
                success_high = bool(feasibility)
                
                if success_high:
                    print(f" success in {duration_high:.2f}s")
                    high_success_count += 1
                    
                    # Save optimal trajectory to database
                    optimal_traj_flat_high = np.array(optimal_traj_high.reshape(-1), dtype=np.float32)
                    
                    # Store trajectory as binary blob using direct SQL
                    try:
                        # Connect to the database
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        # First check if the column exists
                        cursor.execute("PRAGMA table_info(paths)")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        if 'optimal_only_obstacle' not in columns:
                            print("    Adding optimal_only_obstacle column to paths table")
                            cursor.execute("ALTER TABLE paths ADD COLUMN optimal_only_obstacle BLOB")
                        
                        # Convert array to binary blob
                        blob_data = optimal_traj_flat_high.tobytes()
                        
                        # Update the database with the binary data
                        cursor.execute("UPDATE paths SET optimal_only_obstacle = ? WHERE rowid = ?", 
                                      (sqlite3.Binary(blob_data), path_id))
                        
                        conn.commit()
                        conn.close()
                        print("    High time scaling trajectory stored successfully")
                    except Exception as storage_error:
                        print(f"    Error storing high time scaling trajectory: {type(storage_error).__name__}: {str(storage_error)}")
                    
                    # Store for potential plotting
                    high_optimal_paths.append({
                        'path_id': path_id,
                        'world_id': world_id,
                        'world': world,
                        'initial_path': initial_path,
                        'optimal_path': optimal_traj_high
                    })
                    
                    # Generate plot if requested - only plot every plot_interval paths
                    if plot_paths and (i % plot_interval == 0):
                        save_trajectory_plot(
                            optimal_traj_high, world, world_info, robot_info, optim_info_high,
                            path_id=f"{path_id}_high_ts", save_dir=f"{output_dir}/plots"
                        )
                else:
                    print(f" failed - constraints not satisfied (took {duration_high:.2f}s)")
                    print(f"    Max equality violation: {eq_violation_max:.2e}")
                    print(f"    Max inequality violation: {ineq_violation_max:.2e}")
        except Exception as e:
            print(f" error: {str(e)}")
    
    # Print summary
    print("\nOptimization completed!")
    print(f"Normal time scaling: Successfully optimized {normal_success_count} out of {len(path_indices)} paths")
    print(f"Normal time scaling: Success rate: {normal_success_count/len(path_indices)*100:.2f}%")
    print(f"High time scaling: Successfully optimized {high_success_count} out of {len(path_indices)} paths")
    print(f"High time scaling: Success rate: {high_success_count/len(path_indices)*100:.2f}%")
    
    if plot_paths and (normal_optimal_paths or high_optimal_paths):
        print(f"Plots saved to {output_dir}/plots")
        if plot_interval > 1:
            print(f"Note: Only every {plot_interval}th path was plotted")
    
    return normal_optimal_paths, high_optimal_paths

def main():
    
    optimize_and_save_paths(
        num_paths=2000,
        random_selection=True,
        plot_paths=True,
        max_iter=200,
        batch_name='optimal_big_run_trust_region',
        plot_interval=50  # Plot every 10th path
    )

if __name__ == "__main__": 
    main() 