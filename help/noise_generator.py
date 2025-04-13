import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def generate_noise_trajectory(original_trajectory, noise_level=0.1, control_points=5, smoothing_factor=0.7):


    if noise_level == 0.0:
        return original_trajectory

    num_points, num_dof = original_trajectory.shape
    
    # Keep start and end points fixed
    noisy_trajectory = original_trajectory.copy()
    
    if num_points <= 2:
        return noisy_trajectory  # Not enough points to add noise
    
    # Generate control points (excluding start and end)
    control_indices = np.linspace(0, num_points-1, control_points+2).astype(int)
    control_indices = control_indices[1:-1]  # Remove start and end
    
    # Generate coherent noise directions for control points
    # This creates more natural perturbations by pulling in consistent directions
    noise_directions = np.random.randn(len(control_indices), num_dof)
    
    # Normalize the noise directions
    norms = np.linalg.norm(noise_directions, axis=1, keepdims=True)
    noise_directions = noise_directions / (norms + 1e-8)
    
    # Scale the noise by the noise level and the distance between start and end
    path_scale = np.linalg.norm(original_trajectory[-1] - original_trajectory[0])
    noise_scale = noise_level * path_scale
    
    # Apply noise to control points
    for i, idx in enumerate(control_indices):
        # Scale noise by position (less noise near endpoints)
        position_factor = 4 * (idx / num_points) * (1 - idx / num_points)
        noisy_trajectory[idx] += noise_directions[i] * noise_scale * position_factor
    
    # Interpolate between control points for smooth transitions
    if smoothing_factor > 0:
        # Create interpolation for each DOF
        for dof in range(num_dof):
            # Points we know (control points + start/end)
            known_indices = np.concatenate(([0], control_indices, [num_points-1]))
            known_values = noisy_trajectory[known_indices, dof]
            
            # Create interpolation function
            interp_func = interp1d(known_indices, known_values, 
                                  kind='cubic' if len(known_indices) > 3 else 'quadratic',
                                  bounds_error=False, fill_value="extrapolate")
            
            # Indices to interpolate (all except known points)
            interp_indices = np.setdiff1d(np.arange(num_points), known_indices)
            
            # Apply interpolation
            interp_values = interp_func(interp_indices)
            
            # Blend with original trajectory based on smoothing factor
            original_values = original_trajectory[interp_indices, dof]
            blended_values = smoothing_factor * interp_values + (1 - smoothing_factor) * original_values
            
            # Update trajectory
            noisy_trajectory[interp_indices, dof] = blended_values
    
    return noisy_trajectory


if __name__ == "__main__":
    # Example usage
    original_trajectory = np.array([[0.5970741510391235, -3.1363589763641357, -1.8916205167770386, -0.9250635504722595], [0.5584820508956909, -2.8821873664855957, -1.866169810295105, -0.9650081992149353], [0.5215433239936829, -2.6289241313934326, -1.8392655849456787, -1.006387710571289], [0.48828020691871643, -2.380310535430908, -1.806977391242981, -1.0497066974639893], [0.45733511447906494, -2.1335880756378174, -1.7756867408752441, -1.093467354774475], [0.4240669012069702, -1.890419363975525, -1.7458961009979248, -1.1365654468536377], [0.3741281032562256, -1.6722055673599243, -1.6869508028030396, -1.1451709270477295], [0.30881547927856445, -1.4879573583602905, -1.5967129468917847, -1.1064437627792358], [0.2331269532442093, -1.3416930437088013, -1.4813257455825806, -1.0178253650665283], [0.13033844530582428, -1.191689372062683, -1.3552747964859009, -0.918560802936554], [-0.0015765544958412647, -1.0434575080871582, -1.2145847082138062, -0.7859249114990234], [-0.11385922133922577, -0.9504215121269226, -1.0778906345367432, -0.6234029531478882], [-0.23594127595424652, -0.8656495809555054, -0.945124089717865, -0.4590621292591095], [-0.3678518235683441, -0.7884621024131775, -0.8161633610725403, -0.29573360085487366], [-0.501541018486023, -0.6992918252944946, -0.7110530734062195, -0.15387950837612152], [-0.6366199254989624, -0.6129736304283142, -0.6070504188537598, -0.011886810883879662], [-0.7695083618164062, -0.5270305275917053, -0.5020086765289307, 0.13073761761188507], [-0.9032516479492188, -0.44306161999702454, -0.3974625766277313, 0.27331113815307617], [-1.0363266468048096, -0.3587486147880554, -0.29241570830345154, 0.41785377264022827], [-1.1694592237472534, -0.2739656865596771, -0.1873716562986374, 0.5631995797157288]])
    noisy_trajectory = generate_noise_trajectory(original_trajectory, noise_level=0.1, control_points=3, smoothing_factor=0.9)
    print(noisy_trajectory)
