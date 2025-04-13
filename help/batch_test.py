#!/usr/bin/env python3
import os
import argparse
import subprocess
import json
import pandas as pd
from datetime import datetime
import sys
import time

def run_batch_test(sqp_variant, noise_levels, num_tests, num_noise_starts, 
                   control_points=3, smoothing_factor=0.9, plot_indices=None, 
                   plot_initial=False, base_set='dynamic', convergence_tol=1e-4, 
                   qp_tol=1.0, eq_tol=1e-3, ineq_tol=1e-4, time_scaling=0.15,
                   max_iter=100, initial_sigma=100.0, initial_trust_radius=0.01, 
                   initial_hessian_scale=5.0):
    """
    Run a batch of tests by calling the test.py script for each noise level.
    
    Parameters:
        sqp_variant (str): The SQP variant to use ('vanilla', 'line_search', 'trust_region', etc.)
        noise_levels (list): List of noise levels to test
        num_tests (int): Number of test cases to run for each noise level
        num_noise_starts (int): Number of noisy initializations per test case
        control_points (int): Number of control points for noise generation
        smoothing_factor (float): Smoothing factor for noise generation
        plot_indices (list): List of test indices to generate plots for
        plot_initial (bool): Whether to plot initial trajectories
        base_set (str): Base set to use ('dynamic' or 'obstacle')
        convergence_tol (float): Convergence tolerance
        qp_tol (float): QP tolerance
        eq_tol (float): Equality constraint tolerance
        ineq_tol (float): Inequality constraint tolerance
        time_scaling (float): Time scaling factor
        max_iter (int): Maximum number of iterations
        initial_sigma (float): Initial sigma value
        initial_trust_radius (float): Initial trust radius
        initial_hessian_scale (float): Initial Hessian scale
    """
    # Create a parent directory for all noise levels
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = f"test_results/batch_test_{sqp_variant}_{timestamp}"
    os.makedirs(parent_dir, exist_ok=True)
    
    # Save batch configuration
    config = {
        'sqp_variant': sqp_variant,
        'noise_levels': noise_levels,
        'num_tests': num_tests,
        'num_noise_starts': num_noise_starts,
        'control_points': control_points,
        'smoothing_factor': smoothing_factor,
        'plot_indices': plot_indices,
        'plot_initial': plot_initial,
        'base_set': base_set,
        'convergence_tol': convergence_tol,
        'qp_tol': qp_tol,
        'eq_tol': eq_tol,
        'ineq_tol': ineq_tol,
        'time_scaling': time_scaling,
        'max_iter': max_iter,
        'initial_sigma': initial_sigma,
        'initial_trust_radius': initial_trust_radius,
        'initial_hessian_scale': initial_hessian_scale,
        'timestamp': timestamp
    }
    
    with open(f"{parent_dir}/batch_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create a summary DataFrame to store results from all noise levels
    all_results = []
    
    # Run tests for each noise level
    for noise_level in noise_levels:
        print(f"\n{'='*80}")
        print(f"Running tests for noise level: {noise_level}")
        print(f"{'='*80}")
        
        # Create batch name for this noise level
        batch_name = f"{parent_dir}/noise_{noise_level}"
        
        # Prepare command to run test.py
        cmd = [
            sys.executable,  # Use the current Python interpreter
            "help/test.py",  # Path to test.py
            "--sqp_variant", sqp_variant,
            "--base_set", base_set,
            "--convergence_tol", str(convergence_tol),
            "--qp_tol", str(qp_tol),
            "--eq_tol", str(eq_tol),
            "--ineq_tol", str(ineq_tol),
            "--time_scaling", str(time_scaling),
            "--max_iter", str(max_iter),
            "--initial_sigma", str(initial_sigma),
            "--initial_trust_radius", str(initial_trust_radius),
            "--initial_hessian_scale", str(initial_hessian_scale),
            "--batch_name", batch_name,
            "--num_tests", str(num_tests),
            "--num_noise_starts", str(num_noise_starts),
            "--noise_level", str(noise_level),
            "--control_points", str(control_points),
            "--smoothing_factor", str(smoothing_factor),
        ]
        
        # Add optional arguments
        if plot_indices:
            cmd.extend(["--plot_indices", ",".join(map(str, plot_indices))])
        
        if plot_initial:
            cmd.append("--plot_initial")
        
        # Run the command
        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Get any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                print(stdout.strip())
            
            if process.returncode != 0:
                print(f"Error running test for noise level {noise_level}:")
                print(stderr)
            else:
                print(f"Successfully completed tests for noise level {noise_level}")
                
                # Try to load results from CSV - Fix the path to match what test.py creates
                try:
                    # The correct path structure based on the test.py output
                    results_file = f"{batch_name}/{batch_name.split('/')[-1]}_results.csv"
                    
                    # Check if file exists at the expected path
                    if not os.path.exists(results_file):
                        # Try alternative paths
                        alt_path_1 = f"test_results/{batch_name}/{batch_name.split('/')[-1]}_results.csv"
                        alt_path_2 = f"{batch_name}_results.csv"
                        
                        if os.path.exists(alt_path_1):
                            results_file = alt_path_1
                        elif os.path.exists(alt_path_2):
                            results_file = alt_path_2
                        else:
                            # Search for any CSV file in the directory
                            csv_files = []
                            for root, dirs, files in os.walk(batch_name):
                                for file in files:
                                    if file.endswith('.csv'):
                                        csv_files.append(os.path.join(root, file))
                            
                            if csv_files:
                                results_file = csv_files[0]
                                print(f"Found results at: {results_file}")
                            else:
                                print(f"Warning: No results CSV file found for noise level {noise_level}")
                                continue
                    
                    print(f"Loading results from: {results_file}")
                    results_df = pd.read_csv(results_file)
                    results_df['noise_level'] = noise_level  # Ensure noise level is in the results
                    all_results.append(results_df)
                except Exception as e:
                    print(f"Error loading results: {str(e)}")
                    # Print the directory contents to help debug
                    print(f"Directory contents of {batch_name}:")
                    try:
                        if os.path.exists(batch_name):
                            print(os.listdir(batch_name))
                        else:
                            print(f"Directory {batch_name} does not exist")
                    except Exception as dir_error:
                        print(f"Error listing directory: {str(dir_error)}")
        
        except Exception as e:
            print(f"Error executing test.py: {str(e)}")
        
        duration = time.time() - start_time
        print(f"Completed noise level {noise_level} in {duration:.2f} seconds")
    
    # Combine all results and save to a single CSV
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{parent_dir}/all_results.csv", index=False)
        print(f"Combined results saved to {parent_dir}/all_results.csv")
    else:
        print("No results were collected from any test run.")
    
    print("\nBatch testing completed!")
    return parent_dir

def main():
    parser = argparse.ArgumentParser(description='Run batch tests for motion planning')
    
    # SQP variant
    parser.add_argument('--sqp_variant', type=str, default='vanilla',
                        choices=['vanilla', 'line_search', 'trust_region', 'vanilla_BFGS', 'line_search_maratos'],
                        help='SQP variant to use')
    
    # Noise levels
    parser.add_argument('--noise_levels', type=str, default='0.0,0.01,0.05,0.1,0.2',
                        help='Comma-separated list of noise levels')
    
    # Test parameters
    parser.add_argument('--num_tests', type=int, default=10,
                        help='Number of test cases to run for each noise level')
    parser.add_argument('--num_noise_starts', type=int, default=3,
                        help='Number of noisy initializations per test case')
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
    
    # Parse noise levels
    noise_levels = [float(level) for level in args.noise_levels.split(',') if level]
    
    # Parse plot indices
    plot_indices = [int(idx) for idx in args.plot_indices.split(',') if idx]
    
    # Run batch tests
    run_batch_test(
        sqp_variant=args.sqp_variant,
        noise_levels=noise_levels,
        num_tests=args.num_tests,
        num_noise_starts=args.num_noise_starts,
        control_points=args.control_points,
        smoothing_factor=args.smoothing_factor,
        plot_indices=plot_indices,
        plot_initial=args.plot_initial,
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

if __name__ == "__main__":
    main() 