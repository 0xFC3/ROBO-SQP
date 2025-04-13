from .structures import RobotInfo, WorldInfo, Spheres_Jax
import jax.numpy as jnp
from rokin import robots
robot = robots.StaticArm(6, 0.15, widths=0.05)
sph = Spheres_Jax(robot.spheres.x, robot.spheres.r, robot.spheres.f_idx)
rob_type = "staticarm"
robot_info = RobotInfo(
    dim=robot.n_dof,
    limits=robot.limits,
    link_length=robot.lengths,
    spheres=sph,
    type=rob_type,
    safety_margin=0.005,
    next_frame_idx=robot.chain.next_f_idx
    )
world_info = WorldInfo(limits=jnp.array([[-1, 1], [-1, 1]]), voxel_size=0.03125, num_vox=64)