import plotly.graph_objects as go
import numpy as np
import functools
from yaspin import yaspin
from .obs import obstacle_img2dist_img
from .structures import *
from .rob import get_sphere_pos, get_frames_SingleSphere, get_frames_StaticArm
from .utils import construct_kinematic_chains
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sqp.problem_definition import compute_joint_forces


class RobotVis2DPlotly:
    def __init__(
        self,
        q: ArrayLike,
        world: ArrayLike,
        world_info: WorldInfo,
        robot_info: RobotInfo,
        optim_info: OptimInfo
    ):
        self.robot_info = robot_info
        self.world_info = world_info
        self.optim_info = optim_info
        self.voxels = world
        self.dist_img = None
        self.q = q
        self.animate = True if len(q.shape) > 1 and q.shape[1] > 1 else False
        # for convenience get some robot properties
        self.chains = (
            construct_kinematic_chains(robot_info.next_frame_idx)
            if robot_info.type.lower() == "staticarm"
            else [[]]
        )
        self.limb_length = (
            robot_info.link_length if robot_info.type.lower() == "staticarm" else 0.0
        )
        self.n_dof = robot_info.dim
        self.radii = robot_info.spheres.r
        self.spheres_frame_ids = robot_info.spheres.f_idx
        self.current_q = q if not self.animate else None
        self.obstacle_shapes = None
        self.get_frames = (
            get_frames_StaticArm
            if robot_info.type.lower() == "staticarm"
            else get_frames_SingleSphere
        )
        self.get_frames = partial(self.get_frames, rob_info=robot_info)
        self.get_spheres = partial(get_sphere_pos, rob_info=robot_info)
        # register trace and shape getters
        self.trace_getters = [
            func
            for name, func in vars(self.__class__).items()
            if callable(func) and getattr(func, "_tracegetter", False)
        ]
        self.shapes_getters = [
            func
            for name, func in vars(self.__class__).items()
            if callable(func) and getattr(func, "_shapegetter", False)
        ]

        self.bgtrace_getters = [
            func
            for name, func in vars(self.__class__).items()
            if callable(func) and getattr(func, "_bgtracegetter", False)
        ]

        # create figure
        self.fig = go.Figure()
        self.fig.update_layout(
            xaxis=dict(
                range=[self.world_info.limits[0, 0], self.world_info.limits[0, 1]]
            ),
            yaxis=dict(
                range=[self.world_info.limits[1, 0], self.world_info.limits[1, 1]]
            ),
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            showlegend=True,
            width=1000,
            height=800,
            paper_bgcolor="white",
            plot_bgcolor="white",
            # Fix legend position and width
            legend=dict(
                x=-0.5,  # Move legend closer to plot (was -0.3)
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10),  # Smaller font to keep text more compact
                itemwidth=30,  # Fixed width for legend items
            ),
            # Adjust margins to make room for legend but move plot right
            margin=dict(l=200, r=50, t=50, b=50)  # Reduced left margin from 250 to 200
        )
        # add robot information to figure layout
        info_box = (
            f"<br>Robot: {self.robot_info.type}<br>DOF: {self.robot_info.dim}<br>Link lengths: {self.robot_info.link_length}"
            if robot_info.type.lower() == "staticarm"
            else f"<br>Robot: {self.robot_info.type}<br>DOF: {self.robot_info.dim}<br>Radius: {self.robot_info.spheres.r[0]}"
        )
        self.fig.add_annotation(
            text=info_box,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=-0.1,
            y=1.05,
            align="left",
            xanchor="left",
            yanchor="bottom",
        )

        # preconfigure spinner
        self.spinner = yaspin(text="Plotting robot...", timer=True)

    @staticmethod
    def tracegetter(func):
        """
        Decorator to mark a function as a trace getter. Trace getters are functions that return plotly traces.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tracegetter = True
        return wrapper

    @staticmethod
    def shapegetter(func):
        """
        Decorator to mark a function as a shape getter. Shape getters are functions that return plotly shapes.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._shapegetter = True
        return wrapper

    @staticmethod
    def bgtracegetter(func):
        """
        Decorator to mark a function as a background trace getter. They return Plotly traces that are only plotted once
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._bgtracegetter = True
        return wrapper

    def plot(self, duration=100, show_plot=True):
        """
        Plot the robot.
        Args:
            duration (int): The duration of the animation in milliseconds. Only used if the robot is animated.
            show_plot (bool): Whether to show the plot in the browser. Defaults to True.
        """
        self.spinner.text = "Plotting robot..."
        self.spinner.start()
        try:
            if self.animate:
                frames = []
                for i in range(self.q.shape[0]):
                    frame = go.Frame(data=self.__get_frame_data(i), name=str(i))
                    frames.append(frame)
                fig = go.Figure(data=self.__get_frame_data(0), frames=frames)
                # copy layout from self.fig
                fig.layout = self.fig.layout
                self.fig = fig
                self.fig.frames = frames
                self.fig.update_layout(
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [
                                        None,
                                        {
                                            "frame": {
                                                "duration": duration,
                                                "redraw": True,
                                            },
                                            "fromcurrent": True,
                                        },
                                    ],
                                    "label": "Play",
                                    "method": "animate",
                                },
                                {
                                    "args": [
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    "label": "Pause",
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 10, "t": 87},
                            "showactive": True,
                            "type": "buttons",
                            "x": 1.025,
                            "xanchor": "left",
                            "y": 0.4,
                            "yanchor": "top",
                        }
                    ],
                    sliders=[
                        {
                            "steps": [
                                {
                                    "args": [
                                        [str(i)],
                                        {
                                            "frame": {
                                                "duration": duration,
                                                "redraw": True,
                                            },
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    "label": str(i),
                                    "method": "animate",
                                }
                                for i in range(len(frames))
                            ],
                            "transition": {"duration": 0},
                            "x": 0.5,
                            "len": 0.9,
                            "xanchor": "center",
                            "y": -0.0,
                            "yanchor": "top",
                            "currentvalue": {
                                "prefix": "Frame:",
                                "visible": True,
                                "xanchor": "right",
                            },
                            "pad": {"b": 10, "t": 50},
                        }
                    ],
                )
            else:
                self.fig.data = []  # clear all traces
                self.fig.layout.shapes = []  # clear all shapes
                for getter in self.trace_getters:
                    self.fig.add_traces(getter(self))
            for getter in self.shapes_getters:
                for shape in getter(self):  # shapes getter returns a list of shapes
                    self.fig.add_shape(shape)
            self.fig.add_shape(
                type="rect",
                x0=self.world_info.limits[0, 0],
                y0=self.world_info.limits[1, 0],
                x1=self.world_info.limits[0, 1],
                y1=self.world_info.limits[1, 1],
                line=dict(width=3, color="gray"),
            )
            for getter in self.bgtrace_getters:
                self.fig.add_traces(getter(self))
            if show_plot:
                self.fig.show()
        except Exception as e:
            self.spinner.fail()
            self.spinner.write(f"Failed to plot robot")
            raise e
        self.spinner.stop()

    def __get_frame_data(self, frame_idx):
        """
        Get the data for a single frame when animating the robot.
        Args:
            frame_idx (int): The index of the frame to get.
        Returns:
            List[go.Scatter]: A list of plotly traces.
        """
        self.current_q = self.q[frame_idx, :]
        self.current_frame_idx = frame_idx  # Store current frame index
        data = []
        for getter in self.trace_getters:
            data.extend(getter(self))
        return data

    @tracegetter
    def __get_links(self):
        """
        Get the links of the robot.
        Returns:
            List[go.Scatter]: A list of plotly traces representing the links of the robot.
        """
        frames = self.get_frames(self.current_q)
        links = []
        prev_origin = frames[0][:-1, -1]  # usually 0,0
        already_plotted = []
        for chain in self.chains:
            for frame_id in chain:
                if frame_id in already_plotted:
                    continue
                frame = frames[frame_id]
                link_origin = frame[:-1, -1]
                link = go.Scatter(
                    x=[prev_origin[0], link_origin[0]],
                    y=[prev_origin[1], link_origin[1]],
                    mode="lines + markers",
                    name=f"Link {frame_id}",
                    line=dict(width=5),
                    showlegend=False,  # Don't show links in legend
                )
                links.append(link)
                already_plotted.append(frame_id)
                prev_origin = link_origin
        return links

    def group_voxels(self, voxels):
        """
        Greedily group adjacent voxels into larger blocks.
        Args:
            voxels (np.ndarray): The voxel grid.
        Returns:
            List[dict]: A list of Plotly shape dictionaries representing the grouped voxels.
        """
        self.spinner.text = "Grouping voxels..."
        self.spinner.start()
        shapes = []
        visited = np.zeros_like(voxels, dtype=bool)
        # TODO: For now we assume bottom left corner is (0,0). Subject to change
        x_real = np.arange(
            self.world_info.limits[0, 0],
            self.world_info.limits[0, 1],
            self.world_info.voxel_size,
        )
        y_real = np.arange(
            self.world_info.limits[1, 0],
            self.world_info.limits[1, 1],
            self.world_info.voxel_size,
        )
        init_legend = True
        for i in range(voxels.shape[0]):  # TODO iterate over True voxels only
            for j in range(voxels.shape[1]):
                if voxels[i, j] and not visited[i, j]:
                    x0, y0 = i, j
                    x1, y1 = i, j
                    while x1 + 1 < voxels.shape[0] and all(
                        voxels[x1 + 1, y] and not visited[x1 + 1, y]
                        for y in range(y0, y1 + 1)
                    ):
                        x1 += 1
                    while y1 + 1 < voxels.shape[1] and all(
                        voxels[x, y1 + 1] and not visited[x, y1 + 1]
                        for x in range(x0, x1 + 1)
                    ):
                        y1 += 1
                    for x in range(x0, x1 + 1):
                        for y in range(y0, y1 + 1):
                            visited[x, y] = True
                    shapes.append(
                        {
                            "type": "rect",
                            "x0": x_real[x0],
                            "y0": y_real[y0],
                            "x1": x_real[x1] + self.world_info.voxel_size,
                            "y1": y_real[y1] + self.world_info.voxel_size,
                            "fillcolor": "rgba(128, 128, 128, 1)",
                            "line": {"width": 1.0, "color": "rgba(128, 128, 128, 1)"},
                            "name": "Obstacle",
                            "legendgroup": "obstacle",
                            "showlegend": init_legend,
                        }
                    )
                    init_legend = False
        self.spinner.stop()
        # self.spinner.write(
        #    f"> Reduced {np.sum(voxels)} voxels to {len(shapes)} shapes")
        return shapes

    @shapegetter
    def __get_voxel_shapes(self):
        """
        Get the shapes of the obstacles.
        Returns:
            List[dict]: A list of Plotly shape dictionaries representing the obstacles.
        """

        if self.obstacle_shapes is None:
            # we only need to group the voxels once
            self.obstacle_shapes = self.group_voxels(self.voxels)
        return self.obstacle_shapes

    @tracegetter
    def __get_frames(self):
        """
        Get the frames of the robot.
        Returns:
            List[go.Scatter]: A list of plotly traces representing the frames of the robot.
        """
        frames = self.get_frames(self.current_q)
        frame_plots = []
        init_legend = True
        for frame in frames:
            x_vec = frame[:-1, 0]
            y_vec = frame[:-1, 1]
            origin = frame[:-1, -1]
            # plot x axis
            x_axis = go.Scatter(
                x=[origin[0], origin[0] + x_vec[0] * 0.1],
                y=[origin[1], origin[1] + x_vec[1] * 0.1],
                mode="lines",
                name="x-axis",
                line=dict(color="red"),
                showlegend=init_legend,
                legendgroup="frame",
            )
            # plot y axis
            y_axis = go.Scatter(
                x=[origin[0], origin[0] + y_vec[0] * 0.1],
                y=[origin[1], origin[1] + y_vec[1] * 0.1],
                mode="lines",
                name="y-axis",
                line=dict(color="blue"),
                showlegend=init_legend,
                legendgroup="frame",
            )
            init_legend = False
            frame_plots.append(x_axis)
            frame_plots.append(y_axis)
        return frame_plots

    def get_force_color(self, force, max_force):
        """
        Get color for sphere based on force value.
        Green (0 force) -> Red (max force) -> Black (>max force)
        """
        if force > max_force + 0.01:
            return "rgb(0,0,0)"  # Black for forces exceeding max
        
        ratio = force / (max_force + 0.01)
        # Linear interpolation between green and red
        red = int(255 * ratio)
        green = int(255 * (1 - ratio))
        return f"rgb({red},{green},0)"

    def compute_current_forces(self, q_current):
        """
        Compute forces for current configuration
        Returns forces in shape (num_frames, num_joints)
        """
        # We need at least 3 points to compute forces
        if len(q_current.shape) == 1:
            # For single configuration, we can't compute forces
            return np.zeros(self.robot_info.dim)
        else:
            forces = compute_joint_forces(q_current, self.optim_info.time_scaling, self.robot_info)
            return forces

    @tracegetter
    def __get_spheres(self):
        """
        Get the (approximated) spheres of the robot with colors based on joint forces.
        """
        spheres = []
        # Keep track of which joints we've already added to the legend
        legend_added = set()
        
        # Compute forces for current configuration
        if self.animate:
            # For animation, compute forces for the whole trajectory once
            if not hasattr(self, 'trajectory_forces'):
                self.trajectory_forces = self.compute_current_forces(self.q)
            
            # Get forces for current frame
            if hasattr(self, 'current_frame_idx'):
                current_forces = self.trajectory_forces[self.current_frame_idx]
            else:
                current_forces = np.zeros(self.robot_info.dim)
        else:
            # For static view, just show initial position with green coloring
            current_forces = np.zeros(self.robot_info.dim)
            
        for i, (rad, f_idx, center) in enumerate(
            zip(
                self.robot_info.spheres.r,
                self.robot_info.spheres.f_idx,
                self.get_spheres(self.current_q),
            )
        ):
            # Get force color for this joint
            force = abs(current_forces[f_idx]) if f_idx < len(current_forces) else 0
            max_force = self.robot_info.max_joint_forces[f_idx]
            
            # Always use force-based coloring
            color = self.get_force_color(force, max_force)
            opacity = 0.6
            force_text = f"Force: {force:.2f}/{max_force:.2f}"
            
            # Only show in legend if we haven't seen this joint before
            show_in_legend = f_idx not in legend_added
            if show_in_legend:
                legend_added.add(f_idx)
            
            circle = self.draw_filled_circle(
                center,
                rad,
                fillcolor=color,
                opacity=opacity,
                name=f"Joint {f_idx} ({force_text})",
                legendgroup=f"joint_{f_idx}",  # Unique group per joint
                showlegend=show_in_legend,
            )
            spheres.append(circle)
        return spheres

    @bgtracegetter
    def get_sdf(self):
        """
        Get the signed distance field of the robot.
        Returns:
            List[go.Scatter]: A list of plotly traces representing the signed distance field of the robot.
        """

        if self.dist_img is None:
            dist_img = obstacle_img2dist_img(
                self.voxels, self.world_info, add_boundary=True
            )
            self.dist_img = dist_img
        else:
            dist_img = self.dist_img

        # make a 2d heatmap
        # Plotly heatmaps plot patches centered in the middle. We need to add an offset to the x and y values

        x_real = np.linspace(
            self.world_info.limits[0, 0],
            self.world_info.limits[0, 1],
            self.voxels.shape[0],
            endpoint=False,
        )
        y_real = np.linspace(
            self.world_info.limits[1, 0],
            self.world_info.limits[1, 1],
            self.voxels.shape[1],
            endpoint=False,
        )
        x_real += self.world_info.voxel_size / 2
        y_real += self.world_info.voxel_size / 2
        return [
            go.Heatmap(
                z=dist_img.T,
                x=x_real,
                y=y_real,
                colorscale="Viridis",
                showscale=False,
                name="Signed Distance Field",
                legendgroup="sdf",
                showlegend=True,
            )
        ]  # Transpose make sense here for correct indexing

    def save_fig(self, file_path):
        """
        Save the current figure to a file.
        Args:
            file_path (str): The path to save the figure to.
        """
        self.fig.write_html(file_path, auto_play=False)
        self.spinner.write(f"Figure saved to {file_path}")

    @staticmethod
    def draw_filled_circle(center, radius, n_points=10, **kwargs):
        """
        Draw a filled circle.
        Args:
            center (Tuple[float, float]): The center of the circle.
            radius (float): The radius of the circle.
            n_points (int): The number of points to use for the circle approximation.
            **kwargs: Additional arguments to pass to the plotly Scatter objects.
        Returns:
            go.Scatter: A plotly trace representing the circle.
        """
        # we use a filled scatter plot to draw a circle
        # this is not ideal, but it works with animations
        circle = np.linspace(0, 2 * np.pi, num=n_points)
        x = center[0] + radius * np.cos(circle)
        y = center[1] + radius * np.sin(circle)
        return go.Scatter(
            x=x, y=y, mode="lines", fill="toself", line=dict(width=0), **kwargs
        )


class RobotVizOptim(RobotVis2DPlotly):
    # overwrite plot function to show the optimization instead.

    def __get_path_trace(self):
        # consider that q is batched like [num_traj, num_steps, num_dof]
        # we want one scatter trace per batch element
        traces = []
        for i, q in enumerate(self.q):
            x = q[:, 0]
            y = q[:, 1]
            traces.append(go.Scatter(x=x, y=y, mode="lines+markers", name=f"Traj {i}"))
        return traces

    def plot(self, duration=100, show_plot=True):
        """
        Plot the robot.
        Args:
            duration (int): The duration of the animation in milliseconds. Only used if the robot is animated.
            show_plot (bool): Whether to show the plot in the browser. Defaults to True.
        """
        self.spinner.text = "Plotting robot..."
        self.spinner.start()
        frames = self.__get_path_trace()
        base_data = frames[0]
        frames = [go.Frame(data=[dat], name=f"{i}") for i, dat in enumerate(frames)]
        fig = go.Figure(data=base_data, frames=frames)
        # copy layout from self.fig
        fig.layout = self.fig.layout
        self.fig = fig
        self.fig.frames = frames
        self.fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 1.025,
                    "xanchor": "left",
                    "y": 0.7,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": duration, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i in range(len(frames))
                    ],
                    "transition": {"duration": 0},
                    "x": 0.5,
                    "len": 0.9,
                    "xanchor": "center",
                    "y": -0.0,
                    "yanchor": "top",
                    "currentvalue": {
                        "prefix": "Frame:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "pad": {"b": 10, "t": 50},
                }
            ],
        )
        for getter in self.shapes_getters:
            for shape in getter(self):  # shapes getter returns a list of shapes
                self.fig.add_shape(shape)
        self.fig.add_shape(
            type="rect",
            x0=self.world_info.limits[0, 0],
            y0=self.world_info.limits[1, 0],
            x1=self.world_info.limits[0, 1],
            y1=self.world_info.limits[1, 1],
            line=dict(width=3, color="gray"),
        )
        obst = self.group_voxels(self.voxels)
        for shape in obst:
            self.fig.add_shape(shape)
        for getter in self.bgtrace_getters:
            self.fig.add_traces(getter(self))
        if show_plot:
            self.fig.show()
        self.spinner.stop()
