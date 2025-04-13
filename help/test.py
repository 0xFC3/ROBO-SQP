from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", False)
config.update('jax_platform_name', 'gpu')



import os
from rokin import robots
import numpy as np
from wzk import sql2
import jax
import jax.numpy as jnp
from jax import random, grad, jacfwd
import sys
# Go up one level from the current file (help/) to the root, where chompax/ is located
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chompax.structures import WorldInfo, OptimInfo, RobotInfo, Spheres_Jax
from chompax.obs import obstacle_img2dist_img
from chompax.twod_plotly import RobotVis2DPlotly
from sqp.problem_definition import objective_function, inequality_constraints, equality_constraints
import time
from pandas import DataFrame
import plotly.graph_objects as go
from help.plot_extras import plot_joint_forces, plot_min_obstacle_distance, plot_optimization_progress, plot_average_joint_forces, plot_average_min_obstacle_distance, plot_average_optimization_progress
from help.noise_generator import generate_noise_trajectory
from sqp_solver import SQP
import argparse


class MotionPlanningTest:
    def __init__(self, sqp_variant='line_search', base_set='dynamic',
                 convergence_tol=1e-4, qp_tol=1.0, 
                 eq_tol=1e-3, ineq_tol=1e-4, time_scaling=0.15,
                 max_iter=100, initial_sigma=100.0, initial_trust_radius=0.01, initial_hessian_scale=5.0):
        # Check environment variable
        assert os.getenv('ROBDATA_PATH') is not None, "Set the environment variable ROBDATA_PATH"
        self.db_path = os.getenv('ROBDATA_PATH') + '/StaticArm04_2.db'
        
        # Constants
        self.n_waypoints = 20
        self.n_dof = 4

        # Solver specifics
        self.sqp_variant = sqp_variant
        self.max_iter = max_iter
        self.initial_sigma = initial_sigma
        self.initial_trust_radius = initial_trust_radius
        self.initial_hessian_scale = initial_hessian_scale
        self.base_set = base_set
         # Create mask
        self.mask = jnp.vstack(
            [jnp.zeros((1, 1)),
             jnp.ones((self.n_waypoints-2, 1)),
             jnp.zeros((1, 1))]
        )

        self.optim_info = OptimInfo(
            num_points=self.n_waypoints,
            mask=self.mask,
            convergence_tol=convergence_tol,
            qp_tol=qp_tol,
            eq_tol=eq_tol,
            ineq_tol=ineq_tol,
            time_scaling=time_scaling
            )

        # Initialize robot
        self.robot = robots.StaticArm(self.n_dof, 0.15, widths=0.05)
              
        # Initialize random key
        self.key = random.PRNGKey(0)

        self.solver = SQP(method=self.sqp_variant, optim_info=self.optim_info, max_iter=max_iter, 
                          initial_sigma=initial_sigma, initial_trust_radius=initial_trust_radius, initial_hessian_scale=initial_hessian_scale)




    def save_trajectory_plot(self, traj, world, world_info, robot_info, optim_info, test_id, save_dir='plots'):
        """Generate and save a plot of the trajectory."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create visualization
        viz = RobotVis2DPlotly(traj, world, world_info, robot_info, optim_info)
        try:
            viz.plot(100, show_plot=False)
            viz.save_fig(os.path.join(save_dir, f'trajectory_test_{test_id}.html'))
        except Exception as e:
            print(e)
            raise e

    def add_noise_to_trajectory(self, traj, noise_level, control_points=3, smoothing_factor=0.9):
        """Add natural-looking noise to a trajectory, preserving start and end points."""
        # Convert JAX array to numpy for compatibility with noise_generator

        if noise_level == 0.0:
            return traj
        else:
            traj_np = np.array(traj)
            
            # Apply noise with our generator function
            noisy_traj_np = generate_noise_trajectory(
                traj_np,
                noise_level=noise_level,
                control_points=control_points,
                smoothing_factor=smoothing_factor
            )
            # Convert back to JAX array
            return jnp.array(noisy_traj_np)

    def run_test(self, batch_name='test_batch', num_tests=10, num_noise_starts=5, noise_level=0.1, control_points=3, smoothing_factor=0.9,
                  plot_indices=None, plot_initial=True):
        """
        Run a batch of motion planning tests.
        
        Parameters:
            batch_name: Name of the test batch (used for file naming)
            num_tests: Number of test cases to run
            num_noise_starts: Number of noisy initializations per test case
            noise_level: Amount of noise to add to initial trajectories
            plot_indices: List of test indices to generate plots for
            plot_initial: Whether to plot initial trajectories
        """

       # Load worlds and paths
        worlds = sql2.get_values_sql(file=self.db_path, table='worlds', columns="img_cmp")
        worlds = sql2.compressed2img(img_cmp=worlds, shape=(64, 64), dtype='bool')
        
        # Fetch more rows than needed to handle potential NULL values
        fetch_rows = range(num_tests * 10000)  # Fetch twice as many rows as needed

        if self.base_set == 'dynamic':
            column = "optimal_f32"
        elif self.base_set == 'obstacle':
            column = "optimal_only_obstacle"
        else:
            column = "q_f32"

        i_world, q = sql2.get_values_sql(file=self.db_path, table='paths', 
                                        columns=["world_i32", column], rows=fetch_rows)
        
        # Filter out rows where either column is NULL
        valid_indices = [i for i in range(len(i_world)) if i_world[i] is not None and q[i] is not None]

        
        # Take only the required number of rows
        valid_indices = valid_indices[:num_tests]
        if len(valid_indices) < num_tests:
            raise ValueError(f"Could not find {num_tests} valid rows. Only found {len(valid_indices)}.")
        
        i_world = i_world[valid_indices]
        q = q[valid_indices]
        
        # Convert binary data to numpy arrays
        q_arrays = []
        for binary_data in q:
            # Convert binary data to numpy array of float32 values
            q_array = np.frombuffer(binary_data, dtype=np.float32)
            # Check if the array has the expected size
            if len(q_array) == self.n_waypoints * self.n_dof:
                q_arrays.append(q_array)
            else:
                print(f"Warning: Skipping trajectory with unexpected size: {len(q_array)} (expected {self.n_waypoints * self.n_dof})")
        
        # Make sure we have enough valid trajectories
        if len(q_arrays) < num_tests:
            raise ValueError(f"Could not find {num_tests} valid trajectories. Only found {len(q_arrays)}.")
        
        # Take only the required number of trajectories and reshape them
        q_arrays = q_arrays[:num_tests]
        q = np.array(q_arrays).reshape(-1, self.n_waypoints, self.n_dof)
        
        # Setup world parameters
        limits = np.array([[-1, 1], [-1, 1]])  # Cube world
        voxel_size = (limits[:, 1] - limits[:, 0]) / np.array(worlds[0].shape)
        voxel_size = np.min(voxel_size)
        
        # Create info structures
        world_info = WorldInfo(
            limits=limits,
            voxel_size=voxel_size,
            num_vox=worlds[0].shape[0]
        )
        

        sph = Spheres_Jax(self.robot.spheres.x, self.robot.spheres.r, self.robot.spheres.f_idx)
        robot_info = RobotInfo(
            dim=self.robot.n_dof,
            limits=self.robot.limits,
            link_length=self.robot.lengths,
            spheres=sph,
            type="staticarm",
            safety_margin=0.04,
            next_frame_idx=self.robot.chain.next_f_idx,
            link_masses=jnp.array([5.0, 5.0, 5.0, 5.0]),  
            max_joint_forces=jnp.array([1.0, 0.5, 0.2, 0.1])
        )
        
        # Results storage
        results = []
        
        # Create output directory
        # Extract the base name without path for file naming
        if '/' in batch_name:
            base_name = batch_name.split('/')[-1]
            output_dir = batch_name
        else:
            base_name = batch_name
            output_dir = f"test_results/{batch_name}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Run tests
        print(f"\nStarting test batch '{batch_name}'")
        print(f"Running {num_tests} tests with {num_noise_starts} noise starts each")
        print(f"Noise level: {noise_level}, Max iterations: {self.max_iter}\n")
        
        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests} (World ID: {int(i_world[i])})")
            world = worlds[i_world[i]]
            start = q[i, 0]
            end = q[i, -1]

            # Compute signed distance field
            sdf = obstacle_img2dist_img(world, world_info)

            def obj_fn(traj):
                return objective_function(traj, robot_info, self.optim_info)
            def eq_fn(traj):
                return equality_constraints(traj, start, end, self.optim_info, robot_info)
            def ineq_fn(traj):
                return inequality_constraints(traj, sdf, robot_info, world_info, self.optim_info)
            
            # Run optimization for each noisy start
            for j in range(num_noise_starts):
                
                print(f"  Noise start {j+1}/{num_noise_starts}:")
                noisy_traj = self.add_noise_to_trajectory(q[i], noise_level, control_points, smoothing_factor)
                init_traj = noisy_traj.reshape(-1)
                
                # Warm up run for the first test case only
                if i == 0 and j == 0:
                    print("    Performing warm-up run...")
                    _ = self.solver.solve(
                        world_info=world_info,
                        robot_info=robot_info,
                        sdf=sdf,
                        x0=init_traj,
                    )
                    print("    Warm-up completed")
                
                # Check if start and end positions are legal (not in collision)
                start_pos = noisy_traj[0]
                end_pos = noisy_traj[-1]
                
                # Create dummy trajectories for start and end (repeating the same configuration)
                start_traj = jnp.tile(start_pos, (self.optim_info.num_points, 1))
                end_traj = jnp.tile(end_pos, (self.optim_info.num_points, 1))
                
                # Get obstacle constraints for start and end positions
                start_ineq = inequality_constraints(start_traj.reshape(-1), sdf, robot_info, world_info, self.optim_info)
                end_ineq = inequality_constraints(end_traj.reshape(-1), sdf, robot_info, world_info, self.optim_info)
                
                # Split constraints - only consider obstacle constraints (not dynamic constraints)
                n_dyn = robot_info.dim * self.optim_info.num_points  # Number of dynamic constraints
                start_obs_ineq = start_ineq[:-n_dyn]  # Exclude dynamic constraints
                end_obs_ineq = end_ineq[:-n_dyn]      # Exclude dynamic constraints
                
                # Since we repeated the configuration, we only need to check the first set of obstacle constraints
                # as they will all be the same
                n_obs_per_point = len(start_obs_ineq) // self.optim_info.num_points
                start_illegal = bool(jnp.any(start_obs_ineq[:n_obs_per_point] > 0))
                end_illegal = bool(jnp.any(end_obs_ineq[:n_obs_per_point] > 0))
                
                if start_illegal or end_illegal:
                    print(f"    Skipping iteration - {'Start' if start_illegal else 'End'} position in collision with obstacle")
                    continue
                
                # Evaluate initial trajectory
                init_obj = float(obj_fn(init_traj))
                init_eq_viol_sum = float(jnp.sum(jnp.abs(eq_fn(init_traj))))
                init_ineq_viol_sum = float(jnp.sum(jnp.maximum(0.0, ineq_fn(init_traj))))
                init_eq_viol_max = float(jnp.max(jnp.abs(eq_fn(init_traj))))
                init_ineq_viol_max = float(jnp.max(jnp.maximum(0.0, ineq_fn(init_traj))))
                
                # Plot initial trajectory if requested
                if plot_indices is not None and i in plot_indices and plot_initial:
                    print(f"    Generating plot for initial trajectory...")
                    self.save_trajectory_plot(
                        noisy_traj, world, world_info, robot_info, self.optim_info,  
                        test_id=f"{i}_{j}_initial", save_dir=f"{output_dir}/plots"
                    )
                    print(f"    Initial trajectory:")
                    print(f"      Cost: {init_obj:.4e}")
                    print(f"      Eq violation (sum): {init_eq_viol_sum:.2e}")
                    print(f"      Ineq violation (sum): {init_ineq_viol_sum:.2e}")
                    print(f"      Max Eq violation: {init_eq_viol_max:.2e}")
                    print(f"      Max Ineq violation: {init_ineq_viol_max:.2e}")
                
                
                start_time = time.time()
                try:
                    print("    Running optimization...", end='', flush=True)
                    start_time = time.time()
                    # Run optimization
                    opt_results = self.solver.solve(
                        world_info=world_info,
                        robot_info=robot_info,
                        sdf=sdf,
                        x0=init_traj,
                    )
                    
                    jax.block_until_ready(opt_results.x)
                    duration = time.time() - start_time

                    # Compute total violations for reporting and saving
                    eq_violation_sum = float(jnp.sum(jnp.abs(eq_fn(opt_results.x))))
                    ineq_violation_sum = float(jnp.sum(jnp.maximum(0.0, ineq_fn(opt_results.x))))
                    
                    # Check feasibility using maximum violations
                    eq_violation_max = float(jnp.max(jnp.abs(eq_fn(opt_results.x))))
                    ineq_violation_max = float(jnp.max(jnp.maximum(0.0, ineq_fn(opt_results.x))))
                    
                    # Check feasibility
                    feasibility = jnp.logical_and(
                        eq_violation_max < self.optim_info.eq_tol,
                        ineq_violation_max < self.optim_info.ineq_tol
                    )
                    
                    success = bool(feasibility)
                    best_cost = float(opt_results.f_val)
                    best_traj = opt_results.x.reshape(self.n_waypoints, self.n_dof)
                    
                    print(f" done in {duration:.2f}s")
                    print(f"    Status: {'Success' if success else 'Failed'}")
                    print(f"    Final trajectory:")
                    print(f"      Cost: {best_cost:.4e}")
                    print(f"      Iterations: {int(opt_results.n_iter)}")
                    print(f"      Eq violation (sum): {eq_violation_sum:.2e}")
                    print(f"      Ineq violation (sum): {ineq_violation_sum:.2e}")
                    print(f"      Max Eq violation: {eq_violation_max:.2e}")
                    print(f"      Max Ineq violation: {ineq_violation_max:.2e}")
                    
                    # Generate plots if this test is in plot_indices
                    if plot_indices is not None and i in plot_indices and best_traj is not None:
                        print(f"    Generating plots...")
                        plots_dir = f"{output_dir}/plots"
                        os.makedirs(plots_dir, exist_ok=True)
                        
                        # Check if trajectory contains NaN values
                        if not jnp.any(jnp.isnan(best_traj)):
                            try:
                                # Save trajectory plot
                                self.save_trajectory_plot(
                                    best_traj, world, world_info, robot_info, self.optim_info, 
                                    test_id=f"{i}_{j}_final", save_dir=plots_dir
                                )
                                
                                # Generate additional plots
                                plot_joint_forces(
                                    best_traj, robot_info, self.optim_info,
                                    save_path=os.path.join(plots_dir, f'joint_forces_{i}_{j}.html')
                                )
                                plot_min_obstacle_distance(
                                    best_traj, sdf, robot_info, world_info,
                                    save_path=os.path.join(plots_dir, f'min_obstacle_distance_{i}_{j}.html'),
                                    init_traj=noisy_traj
                                )
                                plot_optimization_progress(self.optim_info,
                                    opt_results.progress_history,
                                    int(opt_results.n_iter),
                                    save_path=os.path.join(plots_dir, f'optimization_progress_{i}_{j}.html')
                                )
                            except Exception as plot_error:
                                print(f"    Warning: Failed to generate some plots due to error: {str(plot_error)}")
                        else:
                            print("    Skipping plot generation - trajectory contains NaN values")
                    
                except Exception as e:
                    success = False
                    best_cost = float('inf')
                    best_traj = None
                    eq_violation_sum = float('inf')
                    ineq_violation_sum = float('inf')
                    eq_violation_max = float('inf')
                    ineq_violation_max = float('inf')
                    print(f"\n    Error in test {i}, noise start {j}:")
                    print(f"    Error type: {type(e).__name__}")
                    print(f"    Error message: {str(e)}")
                    import traceback
                    print(f"    Traceback:")
                    print(''.join(['    ' + line for line in traceback.format_tb(e.__traceback__)]))
                    duration = time.time() - start_time
                    
                results.append({
                    'test_id': i,
                    'noise_start_id': j,
                    'world_id': int(i_world[i]),
                    'success': success,
                    'best_cost': best_cost,
                    'duration': duration,
                    'eq_violation_sum': eq_violation_sum,
                    'ineq_violation_sum': ineq_violation_sum,
                    'eq_violation_max': eq_violation_max,
                    'ineq_violation_max': ineq_violation_max,
                    'n_iter': int(opt_results.n_iter),
                    'noise_level': noise_level,
                    'max_iter': self.max_iter,
                    'init_cost': init_obj,
                    'init_eq_violation_sum': init_eq_viol_sum,
                    'init_ineq_violation_sum': init_ineq_viol_sum,
                    'init_eq_violation_max': init_eq_viol_max,
                    'init_ineq_violation_max': init_ineq_viol_max,
                    'initial_trajectory': noisy_traj.tolist(),  # Save initial trajectory
                    'final_trajectory': best_traj.tolist() if best_traj is not None else None,  # Save final trajectory
                    'progress_history': opt_results.progress_history.tolist() if hasattr(opt_results, 'progress_history') else None,  # Save progress history
                    'control_points': control_points,
                    'smoothing_factor': smoothing_factor,
                    'plot_indices': plot_indices,
                    'plot_initial': plot_initial,
                    'num_tests': num_tests,
                    'num_noise_starts': num_noise_starts
                })
                
        print("\nAll tests completed!")
        
        # Create results DataFrame and calculate statistics
        results_df = DataFrame(results)
        
        # Save results to CSV with a consistent naming pattern
        csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
        # Generate summary and plots
        self.generate_summary_and_plots(results_df, batch_name, output_dir, world_info, robot_info, self.optim_info, sdf)
        
        return results_df

    def generate_summary_and_plots(self, results_df, batch_name, output_dir, world_info, robot_info, optim_info, sdf):
        """Generate summary statistics and plots from test results."""
        # Split results into successful and failed runs
        success_df = results_df[results_df['success']]
        failure_df = results_df[~results_df['success']]
        
        n_success = len(success_df)
        n_failure = len(failure_df)
        total_runs = len(results_df)
        success_rate = (n_success / total_runs) * 100 if total_runs > 0 else 0
        
        # Extract the last part of the batch_name for file naming (e.g., "noise_0.0" from "batch_test_vanilla/noise_0.0")
        file_prefix = os.path.basename(batch_name)
        
        # Create summary string with all parameters
        summary = f"""Test Results:
        Test Configuration:
            # SQP Solver Parameters
            SQP Variant: {self.sqp_variant}
            Base Set: {self.base_set}
            Max Iterations: {self.max_iter}
            Initial Sigma: {self.initial_sigma}
            Initial Trust Radius: {self.initial_trust_radius}
            Initial Hessian Scale: {self.initial_hessian_scale}
            
            # Optimization Parameters
            Convergence Tolerance: {self.optim_info.convergence_tol}
            QP Tolerance: {self.optim_info.qp_tol}
            Equality Tolerance: {self.optim_info.eq_tol}
            Inequality Tolerance: {self.optim_info.ineq_tol}
            Time Scaling: {self.optim_info.time_scaling}
            
            # Test Parameters
            Batch Name: {batch_name}
            Number of Tests: {len(results_df['test_id'].unique()) if not results_df.empty else 0}
            Noise Starts per Test: {len(results_df['noise_start_id'].unique()) if not results_df.empty else 0}
            Noise Level: {results_df['noise_level'].iloc[0] if not results_df.empty else 'N/A'}
            Control Points: {results_df.iloc[0].get('control_points', 'N/A') if not results_df.empty else 'N/A'}
            Smoothing Factor: {results_df.iloc[0].get('smoothing_factor', 'N/A') if not results_df.empty else 'N/A'}
            Plot Initial Trajectories: {results_df.iloc[0].get('plot_initial', 'N/A') if not results_df.empty else 'N/A'}
            Plot Indices: {results_df.iloc[0].get('plot_indices', 'N/A') if not results_df.empty else 'N/A'}
            Total Optimization Runs: {total_runs}

        Overall Results:
            Success Rate: {success_rate:.2f}% ({n_success}/{total_runs} runs)
            Failed Runs: {n_failure}/{total_runs} runs ({100-success_rate:.2f}%)
        """

        if n_success > 0:
            success_summary = f"""
            Successful Runs Statistics ({n_success} runs):
                Duration:
                    Mean: {success_df['duration'].mean():.2f} seconds
                    Std: {success_df['duration'].std():.2f} seconds
                    Min: {success_df['duration'].min():.2f} seconds
                    Max: {success_df['duration'].max():.2f} seconds
                
                Iterations:
                    Mean: {success_df['n_iter'].mean():.2f}
                    Std: {success_df['n_iter'].std():.2f}
                    Min: {success_df['n_iter'].min():.0f}
                    Max: {success_df['n_iter'].max():.0f}
                
                Final Violations:
                    Equality (Sum):
                        Mean: {success_df['eq_violation_sum'].mean():.2e}
                        Max: {success_df['eq_violation_sum'].max():.2e}
                    Inequality (Sum):
                        Mean: {success_df['ineq_violation_sum'].mean():.2e}
                        Max: {success_df['ineq_violation_sum'].max():.2e}
                
                Improvements:
                    Cost:
                        Initial Mean: {success_df['init_cost'].mean():.2e}
                        Final Mean: {success_df['best_cost'].mean():.2e}
                        Average Reduction: {(success_df['init_cost'] - success_df['best_cost']).mean():.2e} ({((success_df['init_cost'] - success_df['best_cost'])/success_df['init_cost']).mean()*100:.1f}%)
                    
                    Equality Violation:
                        Initial Mean: {success_df['init_eq_violation_sum'].mean():.2e}
                        Final Mean: {success_df['eq_violation_sum'].mean():.2e}
                        Average Reduction: {(success_df['init_eq_violation_sum'] - success_df['eq_violation_sum']).mean():.2e} ({((success_df['init_eq_violation_sum'] - success_df['eq_violation_sum'])/success_df['init_eq_violation_sum']).mean()*100:.1f}%)
                    
                    Inequality Violation:
                        Initial Mean: {success_df['init_ineq_violation_sum'].mean():.2e}
                        Final Mean: {success_df['ineq_violation_sum'].mean():.2e}
                        Average Reduction: {(success_df['init_ineq_violation_sum'] - success_df['ineq_violation_sum']).mean():.2e} ({((success_df['init_ineq_violation_sum'] - success_df['ineq_violation_sum'])/success_df['init_ineq_violation_sum']).mean()*100:.1f}%)
            """
            summary += success_summary

        if n_failure > 0:
            failure_summary = f"""
            Failed Runs Statistics ({n_failure} runs):
                Duration:
                    Mean: {failure_df['duration'].mean():.2f} seconds
                    Std: {failure_df['duration'].std():.2f} seconds
                    Min: {failure_df['duration'].min():.2f} seconds
                    Max: {failure_df['duration'].max():.2f} seconds
                
                Final Violations:
                    Equality (Sum):
                        Mean: {failure_df['eq_violation_sum'].mean():.2e}
                        Max: {failure_df['eq_violation_sum'].max():.2e}
                    Inequality (Sum):
                        Mean: {failure_df['ineq_violation_sum'].mean():.2e}
                        Max: {failure_df['ineq_violation_sum'].max():.2e}
                
                Initial Conditions:
                    Cost:
                        Mean: {failure_df['init_cost'].mean():.2e}
                        Std: {failure_df['init_cost'].std():.2e}
                    Equality Violation:
                        Mean: {failure_df['init_eq_violation_sum'].mean():.2e}
                        Max: {failure_df['init_eq_violation_sum'].max():.2e}
                    Inequality Violation:
                        Mean: {failure_df['init_ineq_violation_sum'].mean():.2e}
                        Max: {failure_df['init_ineq_violation_sum'].max():.2e}
            """
            summary += failure_summary
        
        # Save results DataFrame to CSV
        csv_path = os.path.join(output_dir, f"{file_prefix}_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
        # Create averaged plots across all runs
        print("\nGenerating averaged plots across all runs...")
        
        # Extract trajectories and other data needed for plotting
        final_trajectories = [jnp.array(result['final_trajectory']) for result in results_df.to_dict('records')]
        initial_trajectories = [jnp.array(result['initial_trajectory']) for result in results_df.to_dict('records')]
        progress_histories = []
        n_iters = []
        for result in results_df.to_dict('records'):
            # Convert progress history list to array and ensure it has correct shape
            # Squeeze out the batch dimension to get shape (n_iter, 9)
            history = jnp.array(result['progress_history']).squeeze()
            progress_histories.append(history)
            n_iters.append(result['n_iter'] + 1)
        
        # Create plots for all runs
        plot_average_joint_forces(
            final_trajectories, robot_info, optim_info,
            os.path.join(output_dir, f'{file_prefix}_average_joint_forces.html')
        )
        
        plot_average_min_obstacle_distance(
            final_trajectories, sdf, robot_info, world_info,
            os.path.join(output_dir, f'{file_prefix}_average_min_obstacle_distance.html'),
            init_trajectories=initial_trajectories
        )
        
        plot_average_optimization_progress(
            optim_info, progress_histories, n_iters,
            os.path.join(output_dir, f'{file_prefix}_average_optimization_progress.html')
        )
        
        # Generate separate plots for successful and failed runs
        if n_success > 0:
            print("\nGenerating averaged plots for successful runs...")
            
            # Extract data for successful runs
            success_indices = success_df.index.tolist()
            success_final_trajectories = [final_trajectories[i] for i in success_indices]
            success_initial_trajectories = [initial_trajectories[i] for i in success_indices]
            success_progress_histories = [progress_histories[i] for i in success_indices]
            success_n_iters = [n_iters[i] for i in success_indices]
            
            # Create plots for successful runs
            plot_average_joint_forces(
                success_final_trajectories, robot_info, optim_info,
                os.path.join(output_dir, f'{file_prefix}_success_average_joint_forces.html')
            )
            
            plot_average_min_obstacle_distance(
                success_final_trajectories, sdf, robot_info, world_info,
                os.path.join(output_dir, f'{file_prefix}_success_average_min_obstacle_distance.html'),
                init_trajectories=success_initial_trajectories
            )
            
            plot_average_optimization_progress(
                optim_info, success_progress_histories, success_n_iters,
                os.path.join(output_dir, f'{file_prefix}_success_average_optimization_progress.html')
            )
        
        if n_failure > 0:
            print("\nGenerating averaged plots for failed runs...")
            
            # Extract data for failed runs
            failure_indices = failure_df.index.tolist()
            failure_final_trajectories = [final_trajectories[i] for i in failure_indices]
            failure_initial_trajectories = [initial_trajectories[i] for i in failure_indices]
            failure_progress_histories = [progress_histories[i] for i in failure_indices]
            failure_n_iters = [n_iters[i] for i in failure_indices]
            
            # Create plots for failed runs
            plot_average_joint_forces(
                failure_final_trajectories, robot_info, optim_info,
                os.path.join(output_dir, f'{file_prefix}_failure_average_joint_forces.html')
            )
            
            plot_average_min_obstacle_distance(
                failure_final_trajectories, sdf, robot_info, world_info,
                os.path.join(output_dir, f'{file_prefix}_failure_average_min_obstacle_distance.html'),
                init_trajectories=failure_initial_trajectories
            )
            
            plot_average_optimization_progress(
                optim_info, failure_progress_histories, failure_n_iters,
                os.path.join(output_dir, f'{file_prefix}_failure_average_optimization_progress.html')
            )
        
        print(f"Saved averaged plots to {output_dir}")
        
        # Save summary to text file
        summary_path = os.path.join(output_dir, f"{file_prefix}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"Saved summary to {summary_path}")
        
        # Print results to console
        print(f"\n{summary}")





def run_singular_test(tester, batch_name='single_test', num_tests=10, num_noise_starts=1, 
                     noise_level=0.0, control_points=3, smoothing_factor=0.9,
                     plot_indices=None, plot_initial=False):
    """
    Run a single test with the specified parameters.
    """
    results = tester.run_test(
        batch_name=batch_name,
        num_tests=num_tests,
        num_noise_starts=num_noise_starts,
        noise_level=noise_level,
        control_points=control_points,
        smoothing_factor=smoothing_factor,
        plot_indices=plot_indices,
        plot_initial=plot_initial
    )
    return results


def run_batch_test(tester, noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2]):
    # Create a parent directory for all noise levels
    parent_dir = f"test_results/batch_test_{tester.sqp_variant}"
    os.makedirs(parent_dir, exist_ok=True)
    
    for noise_level in noise_levels:
        # Use a path that includes the parent directory
        batch_name = f"batch_test_{tester.sqp_variant}/noise_{noise_level}"
        results = tester.run_test(
            batch_name=batch_name,
            num_tests=10,
            num_noise_starts=3,
            noise_level=noise_level,
            control_points=3,
            smoothing_factor=0.9,
            plot_indices=[], #[0, 1, 2, 3, 4, 5, 6, 7],
            plot_initial=False
        )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run motion planning tests')
    
    # SQP variant
    parser.add_argument('--sqp_variant', type=str, default='vanilla',
                        choices=['vanilla', 'line_search', 'trust_region', 'vanilla_BFGS', 'line_search_maratos'],
                        help='SQP variant to use')
    
    # Test parameters
    parser.add_argument('--batch_name', type=str, default='single_test',
                        help='Name of the test batch (used for file naming)')
    parser.add_argument('--num_tests', type=int, default=10,
                        help='Number of test cases to run')
    parser.add_argument('--num_noise_starts', type=int, default=1,
                        help='Number of noisy initializations per test case')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Amount of noise to add to initial trajectories')
    parser.add_argument('--control_points', type=int, default=3,
                        help='Number of control points for noise generation')
    parser.add_argument('--smoothing_factor', type=float, default=0.9,
                        help='Smoothing factor for noise generation')
    parser.add_argument('--plot_indices', type=str, default='',
                        help='Comma-separated list of test indices to generate plots for')
    parser.add_argument('--plot_initial', action='store_true',
                        help='Whether to plot initial trajectories')
    
    # Optimization parameters
    parser.add_argument('--base_set', type=str, default='dynamic',
                        choices=['dynamic', 'obstacle'],
                        help='Base set to use')
    parser.add_argument('--convergence_tol', type=float, default=1e-4,
                        help='Convergence tolerance')
    parser.add_argument('--qp_tol', type=float, default=1.0,
                        help='QP tolerance')
    parser.add_argument('--eq_tol', type=float, default=1e-3,
                        help='Equality constraint tolerance')
    parser.add_argument('--ineq_tol', type=float, default=1e-4,
                        help='Inequality constraint tolerance')
    parser.add_argument('--time_scaling', type=float, default=0.15,
                        help='Time scaling factor')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--initial_sigma', type=float, default=100.0,
                        help='Initial sigma value')
    parser.add_argument('--initial_trust_radius', type=float, default=0.01,
                        help='Initial trust radius')
    parser.add_argument('--initial_hessian_scale', type=float, default=5.0,
                        help='Initial Hessian scale')
    
    args = parser.parse_args()
    
    # Parse plot indices
    plot_indices = [int(idx) for idx in args.plot_indices.split(',') if idx]
    
    # Create tester instance with specified parameters
    tester = MotionPlanningTest(
        sqp_variant=args.sqp_variant,
        base_set=args.base_set,
        convergence_tol=args.convergence_tol,
        qp_tol=args.qp_tol,
        eq_tol=args.eq_tol,
        ineq_tol=args.ineq_tol,
        time_scaling=args.time_scaling,
        max_iter=args.max_iter,
        initial_sigma=args.initial_sigma,
        initial_trust_radius=args.initial_trust_radius,
        initial_hessian_scale=args.initial_hessian_scale
    )
    
    # Run a single test
    run_singular_test(
        tester,
        batch_name=args.batch_name,
        num_tests=args.num_tests,
        num_noise_starts=args.num_noise_starts,
        noise_level=args.noise_level,
        control_points=args.control_points,
        smoothing_factor=args.smoothing_factor,
        plot_indices=plot_indices,
        plot_initial=args.plot_initial
    )


