import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Callable, Literal, Optional, Union, Dict, Any
from functools import partial
from chompax.structures import WorldInfo, OptimInfo, RobotInfo
import sqp.sqp_line_search as line_search
import sqp.sqp_trust_region as trust_region
import sqp.sqp_vanilla_BFGS as vanilla_BFGS
import sqp.sqp_vanilla as vanilla
import sqp.sqp_line_search_maratos as line_search_maratos

class SQP:
    
    def __init__(
        self,
        method: Literal["line_search", "trust_region", "vanilla_BFGS", "vanilla", "line_search_maratos"] = "line_search",
        max_iter: int = 100,
        optim_info: OptimInfo = None,
        initial_sigma: float = 100.0,
        initial_trust_radius: float = 0.01,
        initial_hessian_scale: float = 5.0,
        **kwargs
    ):

        self.method = method
        self.max_iter = max_iter
        self.optim_info = optim_info
        self.initial_sigma = initial_sigma
        self.initial_trust_radius = initial_trust_radius
        self.initial_hessian_scale = initial_hessian_scale
        self.additional_params = kwargs
        
        # Validate method choice
        if method not in ["line_search", "trust_region", "vanilla_BFGS", "vanilla", "line_search_maratos"]:
            raise ValueError(f"Method must be 'line_search' or 'trust_region' or 'vanilla_BFGS' or 'vanilla' or 'line_search_maratos', got {method}")
    
    
    def solve(
        self,
        world_info: WorldInfo,
        robot_info: RobotInfo,
        sdf: jnp.ndarray,
        x0: jnp.ndarray
    ) -> Union[line_search.OptResult, trust_region.OptResult]:

        
        if self.method == "line_search":
            return line_search.sqp_solve(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_sigma=self.initial_sigma,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
        elif self.method == "trust_region":
            return trust_region.sqp_solve(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_trust_radius=self.initial_trust_radius,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
        elif self.method == "vanilla_BFGS":
            return vanilla_BFGS.sqp_solve(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
        elif self.method == "vanilla":
            return vanilla.sqp_solve(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                max_iter=self.max_iter
            )
        elif self.method == "line_search_maratos":
            return line_search_maratos.sqp_solve(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_sigma=self.initial_sigma,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
    def solve_batch(
        self,
        world_info: WorldInfo,
        robot_info: RobotInfo,
        sdf: jnp.ndarray,
        x0: jnp.ndarray
    ) -> Union[line_search.OptResult, trust_region.OptResult]:

        
        if self.method == "line_search":
            return line_search.sqp_solve_batch(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_sigma=self.initial_sigma,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
        elif self.method == "trust_region":
            return trust_region.sqp_solve_batch(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_trust_radius=self.initial_trust_radius,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
        elif self.method == "vanilla_BFGS":
            return vanilla_BFGS.sqp_solve_batch(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )
        elif self.method == "vanilla":
            return vanilla.sqp_solve_batch(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                max_iter=self.max_iter
            )
        elif self.method == "line_search_maratos":
            return line_search_maratos.sqp_solve_batch(
                world_info=world_info,
                optim_info=self.optim_info,
                robot_info=robot_info,
                sdf=sdf,
                x0=x0,
                initial_sigma=self.initial_sigma,
                initial_hessian_scale=self.initial_hessian_scale,
                max_iter=self.max_iter
            )