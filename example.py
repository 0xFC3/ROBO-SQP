from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", False)
config.update('jax_platform_name', 'gpu')  # Change to 'cpu' if you don't have a GPU

import os
import numpy as np
from rokin import robots
from wzk import sql2
import jax
import jax.numpy as jnp
from chompax.structures import WorldInfo, OptimInfo, RobotInfo, Spheres_Jax
from chompax.obs import obstacle_img2dist_img
from chompax.twod_plotly import RobotVis2DPlotly
from sqp_solver import SQP

def main():
    # Check if ROBDATA_PATH environment variable is set
    if os.getenv('ROBDATA_PATH') is None:
        print("Please set the ROBDATA_PATH environment variable")
        return
    
    # Load database
    db_path = os.getenv('ROBDATA_PATH') + '/StaticArm04_2.db'
    
    # Constants
    n_waypoints = 20
    n_dof = 4
    test_index = 5  # Choose a test case index
    
    # Create mask for optimization (fixed start and end points)
    mask = jnp.vstack([
        jnp.zeros((1, 1)),
        jnp.ones((n_waypoints-2, 1)),
        jnp.zeros((1, 1))
    ])
    
    # Load worlds and paths
    print("Loading world and path data...")
    worlds = sql2.get_values_sql(file=db_path, table='worlds', columns="img_cmp")
    worlds = sql2.compressed2img(img_cmp=worlds, shape=(64, 64), dtype='bool')
    
    i_world, q = sql2.get_values_sql(file=db_path, table='paths', 
                                    columns=["world_i32", "q_f32"], rows=[test_index])
    q = q.reshape(-1, n_waypoints, n_dof)
    
    # Handle i_world whether it's a scalar or array
    if np.isscalar(i_world):
        world_idx = i_world
    else:
        world_idx = i_world[0]
    
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
    
    # Create robot info
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
    
    # Get the world and trajectory
    world = worlds[world_idx]
    original_traj = q[0]
    
    # Compute signed distance field
    print("Computing signed distance field...")
    sdf = obstacle_img2dist_img(world, world_info)
    
    # Create OptimInfo for optimization and visualization
    optim_info = OptimInfo(
        num_points=n_waypoints,
        mask=mask,
        convergence_tol=1e-4,
        qp_tol=1.0,
        eq_tol=1e-3,
        ineq_tol=1e-4,
        time_scaling=0.15
    )
    
    # Display the initial trajectory
    print("Plotting initial trajectory...")
    viz_initial = RobotVis2DPlotly(original_traj, world, world_info, robot_info, optim_info)
    viz_initial.plot(100, show_plot=False)
    viz_initial.save_fig('initial_trajectory.html')
    print("Plot saved to initial_trajectory.html")

    # Create SQP solver
    solver = SQP(   
        method="line_search",
        max_iter=30,
        optim_info=optim_info,
        initial_sigma=10.0,
        initial_hessian_scale=5.0
    )
    
    # Prepare the initial trajectory for optimization
    init_traj = original_traj.reshape(-1)  # Flatten the trajectory
    
    # Run optimization
    result = solver.solve(
        world_info=world_info,
        robot_info=robot_info,
        sdf=sdf,
        x0=init_traj
    )
    
    # Reshape the optimized trajectory
    optimized_traj = result.x.reshape(n_waypoints, n_dof)
    
    # Print results
    print(f"Optimization completed in {result.n_iter} iterations")
    print(f"Final objective value: {result.f_val:.4e}")
    print(f"Equality constraint violation: {result.eq_violation:.4e}")
    print(f"Inequality constraint violation: {result.ineq_violation:.4e}")
    print(f"Success: {result.success}")
    
    # Display the optimized trajectory
    print(f"Displaying optimized trajectory ...")
    viz_optimized = RobotVis2DPlotly(optimized_traj, world, world_info, robot_info, optim_info)
    viz_optimized.plot(100, show_plot=False)
    viz_optimized.save_fig('optimized_trajectory.html')
    print("Plot saved to optimized_trajectory.html")
        

if __name__ == "__main__":
    main()
