#!/usr/bin/env python3

"""
Simple script to plot trajectories one at a time using a separate process for each world ID.
This avoids any potential memory issues or state conflicts between plots.
"""

import os
import sys
import subprocess
import pandas as pd
import ast
import argparse
import time
import random

def plot_single_world(world_id, trajectory, output_file):
    """
    Create a temporary Python script to plot a single world and run it as a separate process.
    This ensures a clean environment for each plot.
    """
    # Create a temporary script with the trajectory and world_id
    script_content = f"""
import sys
import os
import numpy as np

# Add the workspace to Python's path
sys.path.insert(0, "/workspace")

from help.plot_path import plot_pasted_trajectory

# The trajectory data
trajectory = {trajectory}

# Plot the trajectory
plot_pasted_trajectory(
    trajectory,
    n_dof=4,
    world_id={world_id},
    save_path="{output_file}",
    max_intermediate_steps=10
)
"""
    
    # Write the script to a temporary file
    temp_script = f"/tmp/plot_world_{world_id}.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    # Run the script as a separate process
    process = subprocess.Popen(
        [sys.executable, temp_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for it to complete
    stdout, stderr = process.communicate()
    
    # Print any output
    if stdout:
        print(f"Output for world {world_id}:")
        print(stdout.decode())
    
    # Print any errors
    if process.returncode != 0 and stderr:
        print(f"Error for world {world_id}:")
        print(stderr.decode())
        return False
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot trajectories for selected worlds.')
    parser.add_argument('--num', type=int, default=100, help='Number of worlds to plot')
    parser.add_argument('--start', type=int, default=0, help='Start index in the dataset')
    parser.add_argument('--csv', type=str, default="/workspace/test_results/batch_test_trust_region_20250317_075712/noise_0.0/noise_0.0_results.csv", help='Path to the results CSV file')
    args = parser.parse_args()
    
    # Load the CSV file
    print(f"Loading CSV file: {args.csv}")
    results_df = pd.read_csv(args.csv)
    print(f"CSV loaded with {len(results_df)} rows")
    
    # Randomly select rows if needed
    if len(results_df) > args.num:
        print(f"Selecting {args.num} random examples")
        selected_rows = results_df.sample(n=args.num).reset_index()
    else:
        print(f"Using all {len(results_df)} examples")
        selected_rows = results_df.reset_index()
    
    # Process each example separately
    successful = 0
    for idx in range(args.start, min(args.start + args.num, len(selected_rows))):
        row = selected_rows.iloc[idx]
        print(f"\nProcessing example {idx} (row index: {row['index']})")
        
        try:
            # Parse the trajectory
            trajectory_str = row['initial_trajectory']
            trajectory = ast.literal_eval(trajectory_str)
            
            # Get the world ID
            world_id = int(row['world_id'])
            print(f"World ID: {world_id}")
            
            # Create output filename
            output_file = f"y=zworld_{world_id}_example_{idx}.png"
            
            # Plot the trajectory in a separate process
            success = plot_single_world(world_id, trajectory, output_file)
            if success:
                print(f"Successfully plotted world {world_id}, saved to {output_file}")
                successful += 1
            
            # Sleep briefly to allow resources to be freed
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    print(f"\nCompleted plotting {successful} examples successfully out of {min(args.num, len(selected_rows))}")

if __name__ == "__main__":
    main() 