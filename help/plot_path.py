import os
# Force matplotlib to use 'Agg' backend through environment variable
# This needs to be set before any other imports that might use matplotlib
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from functools import partial

# Monkey patch matplotlib.use to prevent other libraries from changing the backend
import matplotlib
original_use = matplotlib.use
def use_wrapper(backend, **kwargs):
    # Silently ignore attempts to change the backend
    pass
matplotlib.use = use_wrapper

# Now import the libraries that might try to change the backend
from rokin import robots
from wzk import sql2

# Import necessary functions from chompax
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chompax.structures import WorldInfo, RobotInfo, Spheres_Jax
from chompax.rob import forward_kinematic_StaticArm, get_sphere_pos
from chompax.utils import construct_kinematic_chains

# Create plots directory if it doesn't exist
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_world_and_trajectory(world_id, test_index=5):

    # Check if ROBDATA_PATH environment variable is set
    if os.getenv('ROBDATA_PATH') is None:
        raise ValueError("Please set the ROBDATA_PATH environment variable")
    
    # Load database
    db_path = os.getenv('ROBDATA_PATH') + '/StaticArm04_2.db'
    
    # Load worlds
    worlds = sql2.get_values_sql(file=db_path, table='worlds', columns="img_cmp")
    worlds = sql2.compressed2img(img_cmp=worlds, shape=(64, 64), dtype='bool')
    
    # Load paths
    i_world, q = sql2.get_values_sql(file=db_path, table='paths', 
                                    columns=["world_i32", "q_f32"], rows=[test_index])
    
    # Handle i_world whether it's a scalar or array
    if np.isscalar(i_world):
        world_idx = i_world
    else:
        world_idx = i_world[0]
    
    # If world_id is specified, use it instead
    if world_id is not None:
        world_idx = world_id
    
    # Get the world
    try:
        world = worlds[world_idx].copy()  # Create a explicit copy to avoid references
    except IndexError:
        print(f"Warning: World ID {world_idx} not found. Using empty world instead.")
        world = create_empty_world()
    
    # Setup robot and world parameters
    n_dof = 4  # Assuming 4 DOF robot
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
        link_masses=np.array([5.0, 5.0, 5.0, 5.0]),  
        max_joint_forces=np.array([1.0, 0.5, 0.2, 0.1])
    )
    
    # Reshape trajectory
    n_waypoints = q.shape[0] // n_dof
    trajectory = q.reshape(1, n_waypoints, n_dof)[0]
    
    # Clear references to large objects to free memory
    worlds = None
    
    return world, trajectory, robot_info, world_info


def create_empty_world(shape=(64, 64)):

    return np.zeros(shape, dtype=bool)


def create_robot_info(n_dof=4, link_length=0.15, sphere_width=0.05):

    robot = robots.StaticArm(n_dof, link_length, widths=sphere_width)
    sph = Spheres_Jax(robot.spheres.x, robot.spheres.r, robot.spheres.f_idx)
    
    robot_info = RobotInfo(
        dim=robot.n_dof,
        limits=robot.limits,
        link_length=robot.lengths,
        spheres=sph,
        type="staticarm",
        safety_margin=0.04,
        next_frame_idx=robot.chain.next_f_idx,
        link_masses=np.array([5.0] * n_dof),
        max_joint_forces=np.array([1.0, 0.5, 0.2, 0.1] + [0.1] * (n_dof - 4) if n_dof > 4 else [1.0, 0.5, 0.2, 0.1][:n_dof])
    )
    
    return robot_info


def create_world_info(shape=(64, 64), limits=np.array([[-1, 1], [-1, 1]])):

    voxel_size = (limits[:, 1] - limits[:, 0]) / np.array(shape)
    voxel_size = np.min(voxel_size)
    
    world_info = WorldInfo(
        limits=limits,
        voxel_size=voxel_size,
        num_vox=shape[0]
    )
    
    return world_info


def plot_world(ax, world, world_info):

    # Calculate real-world coordinates for voxels
    x_real = np.linspace(
        world_info.limits[0, 0],
        world_info.limits[0, 1],
        world.shape[0],
        endpoint=False
    )
    y_real = np.linspace(
        world_info.limits[1, 0],
        world_info.limits[1, 1],
        world.shape[1],
        endpoint=False
    )
    
    # Plot obstacles
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            if world[i, j]:
                rect = plt.Rectangle(
                    (x_real[i], y_real[j]),
                    world_info.voxel_size,
                    world_info.voxel_size,
                    color='gray',
                    alpha=0.7
                )
                ax.add_patch(rect)


def plot_robot(ax, q, robot_info, color='blue', alpha=0.7, is_start_or_end=False):
    try:
        # Make sure q is a numpy array
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        
        # Get joint positions
        try:
            fk = partial(forward_kinematic_StaticArm, return_all=True, rob_info=robot_info)
            joint_positions = fk(q)
            
            # Convert from JAX array to numpy array if needed
            if hasattr(joint_positions, 'device_buffer'):
                joint_positions = np.array(joint_positions)
                
            # Increase linewidth and opacity for start/end configurations to make the arm links more visible
            if is_start_or_end:
                linewidth = 6  # Much thicker lines for start/end
                line_alpha = 0.9  # Higher opacity
                
                # For the arm links, use black instead of the robot color
                line_color = 'black'
                
                # Plot spheres first (so they appear behind the links)
                try:
                    get_spheres_fn = partial(get_sphere_pos, rob_info=robot_info)
                    sphere_positions = get_spheres_fn(q)
                    
                    # Convert from JAX array to numpy array if needed
                    if hasattr(sphere_positions, 'device_buffer'):
                        sphere_positions = np.array(sphere_positions)
                    
                    for i, (rad, pos) in enumerate(zip(robot_info.spheres.r, sphere_positions)):
                        circle = Circle(
                            (float(pos[0]), float(pos[1])),
                            float(rad),
                            color=color,
                            alpha=alpha,
                            zorder=3  # High enough to be above background but below lines
                        )
                        ax.add_patch(circle)
                except Exception as e:
                    print(f"Warning: Error plotting robot spheres: {e}")
            else:
                linewidth = 3
                line_alpha = alpha
                line_color = color  # Use regular color for non-start/end
                
                # For non-start/end positions, plot spheres normally
                try:
                    get_spheres_fn = partial(get_sphere_pos, rob_info=robot_info)
                    sphere_positions = get_spheres_fn(q)
                    
                    # Convert from JAX array to numpy array if needed
                    if hasattr(sphere_positions, 'device_buffer'):
                        sphere_positions = np.array(sphere_positions)
                    
                    for i, (rad, pos) in enumerate(zip(robot_info.spheres.r, sphere_positions)):
                        circle = Circle(
                            (float(pos[0]), float(pos[1])),
                            float(rad),
                            color=color,
                            alpha=alpha,
                            zorder=2
                        )
                        ax.add_patch(circle)
                except Exception as e:
                    print(f"Warning: Error plotting robot spheres: {e}")
            
            # Plot links - these are the arm segments connecting joints
            # Place them on top with black color for start/end positions
            for i in range(len(joint_positions) - 1):
                ax.plot(
                    [float(joint_positions[i, 0]), float(joint_positions[i+1, 0])],
                    [float(joint_positions[i, 1]), float(joint_positions[i+1, 1])],
                    color=line_color,
                    linewidth=linewidth,
                    alpha=line_alpha,
                    zorder=4 if is_start_or_end else 2  # Higher zorder to appear on top of spheres
                )
            
            # Plot joints with lighter color and smaller size
            # Use a lighter version of the joint color or a light gray
            joint_color = '#D3D3D3'  # Light gray
            joint_size = 4  # Smaller joint markers
            
            for pos in joint_positions:
                ax.plot(float(pos[0]), float(pos[1]), 'o', color=joint_color, 
                       markersize=joint_size, alpha=0.7, zorder=5 if is_start_or_end else 3)
        except Exception as e:
            print(f"Warning: Error plotting robot links: {e}")
            # Draw a placeholder for the robot if forward kinematics fails
            ax.text(0, 0, "Robot (Error)", color='red', fontsize=12, ha='center', va='center')
        
    except Exception as e:
        print(f"Critical error in plot_robot: {e}")
        # Draw a placeholder for the robot if everything fails
        ax.text(0, 0, "Robot (Error)", color='red', fontsize=12, ha='center', va='center')


def calculate_optimal_boundaries(trajectory, robot_info, padding_factor=0.2):
    try:
        # Get joint positions for all configurations
        fk = partial(forward_kinematic_StaticArm, return_all=True, rob_info=robot_info)
        all_positions = []
        
        # Convert trajectory to numpy array if it's not already
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        # Process each configuration
        for q in trajectory:
            try:
                # Get all joint positions for this configuration
                joint_positions = fk(q)
                
                # Convert from JAX array to numpy array if needed
                if hasattr(joint_positions, 'device_buffer'):
                    joint_positions = np.array(joint_positions)
                
                # Append to our list
                for pos in joint_positions:
                    all_positions.append(pos)
                
                # Get sphere positions for this configuration
                try:
                    get_spheres_fn = partial(get_sphere_pos, rob_info=robot_info)
                    sphere_positions = get_spheres_fn(q)
                    
                    # Convert from JAX array to numpy array if needed
                    if hasattr(sphere_positions, 'device_buffer'):
                        sphere_positions = np.array(sphere_positions)
                    
                    # Add sphere positions with their radii
                    for i, (rad, pos) in enumerate(zip(robot_info.spheres.r, sphere_positions)):
                        # Add points at the extremes of each sphere
                        all_positions.append(np.array([pos[0] + rad, pos[1]]))
                        all_positions.append(np.array([pos[0] - rad, pos[1]]))
                        all_positions.append(np.array([pos[0], pos[1] + rad]))
                        all_positions.append(np.array([pos[0], pos[1] - rad]))
                except Exception as e:
                    print(f"Warning: Error getting sphere positions: {e}. Continuing without them.")
            except Exception as e:
                print(f"Warning: Error processing configuration: {e}. Skipping this configuration.")
                continue
        
        # Convert to a 2D numpy array
        all_positions = np.array(all_positions)
        
        # Find min and max coordinates
        x_min, y_min = np.min(all_positions, axis=0)[:2]
        x_max, y_max = np.max(all_positions, axis=0)[:2]
        
        # Calculate center
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min
        
        # Use the larger dimension to ensure square aspect ratio
        max_dimension = max(width, height)
        
        # Add padding
        padded_dimension = max_dimension * (1 + padding_factor)
        
        # Calculate new boundaries centered on the trajectory
        x_min_new = x_center - padded_dimension / 2
        x_max_new = x_center + padded_dimension / 2
        y_min_new = y_center - padded_dimension / 2
        y_max_new = y_center + padded_dimension / 2
        
        return [x_min_new, x_max_new, y_min_new, y_max_new]
        
    except Exception as e:
        print(f"Warning: Error calculating optimal boundaries: {e}. Using default boundaries.")
        # Return some default boundaries
        return [-1.5, 1.5, -1.5, 1.5]


def plot_trajectory(trajectory, world_id=None, test_index=5, save_path=None, 
                   use_database=True, n_dof=None, world=None, robot_info=None, world_info=None,
                   max_intermediate_steps=None, zoom_region=None, auto_zoom=True,
                   show_legend=True, show_axis_labels=True, show_title=True, text_size=10):
    # Close any existing figures to ensure a clean state
    plt.close('all')
    
    if use_database:
        # Load world and trajectory data from database
        world, traj, robot_info, world_info = load_world_and_trajectory(world_id, test_index)
        
        # If trajectory is provided, use it instead
        if trajectory is not None:
            traj = trajectory
    else:
        # Use provided data or create default data
        traj = trajectory
        
        if n_dof is None:
            n_dof = traj.shape[1] if len(traj.shape) > 1 else 4
        
        if robot_info is None:
            robot_info = create_robot_info(n_dof)
        
        if world is None:
            world = create_empty_world()
        
        if world_info is None:
            world_info = create_world_info(shape=world.shape)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set axis limits based on zoom_region, auto-calculated boundaries, or world_info
    if zoom_region is not None:
        ax.set_xlim(zoom_region[0], zoom_region[1])
        ax.set_ylim(zoom_region[2], zoom_region[3])
    elif auto_zoom:
        # Calculate optimal boundaries
        boundaries = calculate_optimal_boundaries(traj, robot_info)
        ax.set_xlim(boundaries[0], boundaries[1])
        ax.set_ylim(boundaries[2], boundaries[3])
    else:
        ax.set_xlim(world_info.limits[0, 0], world_info.limits[0, 1])
        ax.set_ylim(world_info.limits[1, 0], world_info.limits[1, 1])
    
    ax.set_aspect('equal')
    
    # Add title if show_title is True
    if show_title:
        title = f'Robot Arm Trajectory'
        if world_id is not None:
            title += f' (World {world_id})'
        ax.set_title(title, fontsize=text_size * 1.2)  # Title slightly larger than other text
    
    # Add axis labels if show_axis_labels is True
    if show_axis_labels:
        ax.set_xlabel('X', fontsize=text_size)
        ax.set_ylabel('Y', fontsize=text_size)
    else:
        # Hide axis ticks and labels if we're not showing axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # Plot world
    plot_world(ax, world, world_info)
    
    # Select intermediate steps to display
    if len(traj) > 2:  # Only if there are intermediate steps
        if max_intermediate_steps is not None and max_intermediate_steps < len(traj) - 2:
            # Select evenly spaced intermediate steps
            if max_intermediate_steps > 0:
                step_indices = np.linspace(1, len(traj) - 2, max_intermediate_steps, dtype=int)
            else:
                step_indices = []  # No intermediate steps
        else:
            # Display all intermediate steps
            step_indices = range(1, len(traj) - 1)
        
        # Plot selected intermediate configurations in gray
        for i in step_indices:
            plot_robot(ax, traj[i], robot_info, color='gray', alpha=0.3)
    
    # Plot start configuration in green
    plot_robot(ax, traj[0], robot_info, color='green', alpha=0.7, is_start_or_end=True)
    
    # Plot end configuration in a slightly toned-down red
    plot_robot(ax, traj[-1], robot_info, color='red', alpha=0.7, is_start_or_end=True)  # Slightly less aggressive red
    
    # Get end effector positions for the trajectory
    try:
        fk = partial(forward_kinematic_StaticArm, return_all=True, rob_info=robot_info)
        positions = []
        
        for q in traj:
            try:
                # Get forward kinematics result
                result = fk(q)
                
                # Convert from JAX array to numpy array if needed
                if hasattr(result, 'device_buffer'):
                    result = np.array(result)
                
                # Get the last point (end effector)
                end_effector = result[-1]
                positions.append([float(end_effector[0]), float(end_effector[1])])
            except Exception as e:
                print(f"Warning: Error calculating forward kinematics: {e}")
                # Skip this point
        
        # Convert to numpy array
        if positions:
            positions = np.array(positions)
            
            # Plot trajectory line
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                'b-',
                linewidth=3,
                alpha=0.7,
                label='End Effector Path'
            )
    except Exception as e:
        print(f"Warning: Error plotting trajectory line: {e}")
    
    # Add legend with custom entries only if show_legend is True
    if show_legend:
        ax.plot([], [], color='green', linewidth=3, label='Start Configuration')
        ax.plot([], [], color='#D43A2F', linewidth=3, label='End Configuration')
        ax.plot([], [], color='gray', linewidth=3, label='Intermediate Steps')
        ax.legend(fontsize=text_size)
    
    plt.tight_layout()
    
    # Save image if requested or save to default location
    if save_path is None:
        # Generate a default filename based on world_id
        world_str = f"world{world_id}" if world_id is not None else "empty_world"
        save_path = os.path.join(PLOTS_DIR, f"trajectory_{world_str}.png")
    elif not os.path.isabs(save_path):
        # If save_path is not an absolute path, save to plots directory
        save_path = os.path.join(PLOTS_DIR, save_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Close the figure to prevent memory leaks and interference with subsequent plots
    plt.close(fig)


def plot_pasted_trajectory(trajectory_array, n_dof=4, save_path=None, world_id=None,
                          max_intermediate_steps=None, zoom_region=None, auto_zoom=True,
                          show_legend=True, show_axis_labels=True, show_title=True, text_size=10):
    # Make sure matplotlib is in a clean state
    plt.close('all')
    
    try:
        # Convert to numpy array if it's not already
        if not isinstance(trajectory_array, np.ndarray):
            trajectory = np.array(trajectory_array)
        else:
            trajectory = trajectory_array.copy()  # Make a copy to avoid modifying the original
        
        # Reshape if necessary (if it's a flat array)
        if len(trajectory.shape) == 1:
            n_waypoints = trajectory.shape[0] // n_dof
            trajectory = trajectory.reshape(n_waypoints, n_dof)
        
        if world_id is not None:
            # Load world data from database but use the pasted trajectory
            try:
                # Load world and dummy trajectory data
                world, _, robot_info, world_info = load_world_and_trajectory(world_id)
                
                # Plot with the loaded world but pasted trajectory
                plot_trajectory(
                    trajectory=trajectory,
                    use_database=False,
                    n_dof=n_dof,
                    save_path=save_path,
                    world=world,
                    robot_info=robot_info,
                    world_info=world_info,
                    max_intermediate_steps=max_intermediate_steps,
                    zoom_region=zoom_region,
                    auto_zoom=auto_zoom,
                    show_legend=show_legend,
                    show_axis_labels=show_axis_labels,
                    show_title=show_title,
                    text_size=text_size
                )
            except Exception as e:
                print(f"Error loading world {world_id}: {e}")
                print("Falling back to empty world...")
                # Create default robot and world info
                robot_info = create_robot_info(n_dof)
                world = create_empty_world()
                world_info = create_world_info(shape=world.shape)
                
                plot_trajectory(
                    trajectory=trajectory,
                    use_database=False,
                    n_dof=n_dof,
                    save_path=save_path,
                    world=world,
                    robot_info=robot_info,
                    world_info=world_info,
                    max_intermediate_steps=max_intermediate_steps,
                    zoom_region=zoom_region,
                    auto_zoom=auto_zoom,
                    show_legend=show_legend,
                    show_axis_labels=show_axis_labels,
                    show_title=show_title,
                    text_size=text_size
                )
        else:
            # Plot with an empty world
            plot_trajectory(
                trajectory=trajectory,
                use_database=False,
                n_dof=n_dof,
                save_path=save_path,
                max_intermediate_steps=max_intermediate_steps,
                zoom_region=zoom_region,
                auto_zoom=auto_zoom,
                show_legend=show_legend,
                show_axis_labels=show_axis_labels,
                show_title=show_title,
                text_size=text_size
            )
    except Exception as e:
        print(f"Critical error plotting trajectory: {e}")
        import traceback
        traceback.print_exc()
        # Give up and return without plotting


def main():

    pasted_trajectory = [[-0.4012773931026459, 2.743856906890869, -1.7976619005203247, -1.6813536882400513], [-0.35669422149658203, 2.5935726165771484, -1.5538815259933472, -1.4899976253509521], [-0.31105148792266846, 2.444079875946045, -1.3095812797546387, -1.298395037651062], [-0.2665391266345978, 2.293745994567871, -1.0658284425735474, -1.1070470809936523], [-0.22147811949253082, 2.1438193321228027, -0.8218122124671936, -0.9155793786048889], [-0.17648090422153473, 1.993844985961914, -0.5778271555900574, -0.7241256833076477], [-0.13156521320343018, 1.8438103199005127, -0.3338809609413147, -0.5326892137527466], [-0.0867137461900711, 1.693727731704712, -0.08996617794036865, -0.34126701951026917], [-0.0412534661591053, 1.5440921783447266, 0.15423086285591125, -0.1497228890657425], [0.003224033862352371, 1.3937369585037231, 0.3979761004447937, 0.04162740707397461], [0.04865371435880661, 1.2440818548202515, 0.6421650648117065, 0.23317117989063263], [0.09311813861131668, 1.0937144756317139, 0.885899007320404, 0.4245142638683319], [0.1380775421857834, 0.9437124729156494, 1.129866123199463, 0.6159600019454956], [0.18350042402744293, 0.794054388999939, 1.3740555047988892, 0.8075050711631775], [0.22847776114940643, 0.6440649628639221, 1.6180297136306763, 0.9989534020423889], [0.27297094464302063, 0.49371829628944397, 1.8617761135101318, 1.1903023719787598], [0.3179071247577667, 0.34369757771492004, 2.1057286262512207, 1.381739854812622], [0.3633575439453125, 0.19406120479106903, 2.3499343395233154, 1.5732959508895874], [0.4076560437679291, 0.04356497526168823, 2.5935754776000977, 1.7645856142044067], [0.4530715346336365, -0.10609523952007294, 2.8377692699432373, 1.9561399221420288]]

    




    noise_level = 0.0

    # Generate noisy trajectory
    from help.noise_generator import generate_noise_trajectory
    noisy_trajectory = generate_noise_trajectory(np.array(pasted_trajectory), noise_level=noise_level, control_points=3, smoothing_factor=0.9)

    # Convert the noisy trajectory from numpy array back to a list of lists
    noisy_trajectory = noisy_trajectory.tolist()


    # Plot the pasted trajectory with world ID 1153 and auto-zoom enabled
    plot_pasted_trajectory(
        noisy_trajectory, 
        n_dof=4, 
        world_id=81, 
        save_path=f"dynamic_only_3_final.png",
        max_intermediate_steps=10,
        show_legend=False,
        show_axis_labels=False,
        show_title=False,
        text_size=18
        # No zoom_region specified, auto_zoom will be used by default
    )
    

if __name__ == "__main__":
    main()
