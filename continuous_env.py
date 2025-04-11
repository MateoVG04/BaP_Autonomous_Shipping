import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk

import os
import copy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Polygon
from continuous_tools import create_checkpoints_from_simple_path, check_collision_ship


def add_noise(action, noise_level=0.1) -> np.ndarray:
    """Add noise to the action for more realism."""
    return np.clip(action + np.random.normal(0, noise_level, action.shape), -1, 1)


def calculate_perpendicular_lines(checkpoints, line_length=100.0):
    """
    Calculate the perpendicular lines at each checkpoint using the smoothed tangent direction.

    Args:
        checkpoints: List of dictionaries containing checkpoint positions and radii.
        line_length: Length of the perpendicular lines.

    Returns:
        List of tuples containing start and end points of perpendicular lines.
    """

    def smooth_tangent(check_points, index):
        """Calculate the tangent at checkpoint `i` by averaging vectors to neighbors."""
        if index == 0:  # Start of the path
            tangent = check_points[index + 1]['pos'] - check_points[index]['pos']
        elif index == len(check_points) - 1:  # End of the path
            tangent = check_points[index]['pos'] - check_points[index - 1]['pos']
        else:  # Middle of the path
            to_next = check_points[index + 1]['pos'] - check_points[index]['pos']
            to_prev = check_points[index]['pos'] - check_points[index - 1]['pos']
            tangent = to_next + to_prev  # Average direction

        if np.all(tangent == 0):
            return tangent

        return tangent / np.linalg.norm(tangent)


    lines = []  # to store start and end points of perpendicular lines
    for i in range(len(checkpoints)):
        # Get the perpendicular direction using the smoothed tangent at the current checkpoint
        smoothed_tangent = smooth_tangent(checkpoints, i)
        perpendicular_direction = np.array([-smoothed_tangent[1], smoothed_tangent[0]])

        # Calculate the start and end points of the perpendicular line at the checkpoint
        midpoint = np.array(checkpoints[i]['pos'])
        start_point = midpoint + perpendicular_direction * (line_length / 2)
        end_point = midpoint - perpendicular_direction * (line_length / 2)

        lines.append((start_point, end_point))

    return lines


class Continuous2DEnv(gym.Env):
    """Improved custom Python Simulator environment for ship navigation."""

    TIME_STEP = 0.1
    MIN_SURGE_VELOCITY = 0.0
    MIN_SWAY_VELOCITY = -2.0
    MIN_YAW_RATE = -0.5
    MAX_SURGE_VELOCITY = 5.0
    MAX_SWAY_VELOCITY = 2.0
    MAX_YAW_RATE = 0.5
    CHECKPOINTS_DISTANCE = 350

    MIN_GRID_POS = -11700
    MAX_GRID_POS = 14500

    # Class-level constants
    REWARD_DISTANCE_SCALE = 2.0
    CROSS_TRACK_ERROR_PENALTY_SCALE = 50.0

    def __init__(self, render_mode=None, max_steps=200, verbose=None, target_pos=None, ship_pos=None, wind=False, current=False):
        super().__init__()

        # Environment parameters
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_action = [0.0, 0.0]
        self.verbose = verbose
        self.step_count = 0
        self.stuck_steps = 0
        self.wind = wind
        self.current = current
        self.cross_error = 0

        # Environmental effects 
        self.radians_current = np.radians(180)
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)])
        self.current_strength = 0.35
        self.radians_wind = np.radians(90)
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)])
        self.wind_strength = .35

        # Ship state initialization
        self.initial_ship_pos = np.array(ship_pos, dtype=np.float32) if ship_pos else np.array([5.0, 5.0], dtype=np.float32)
        self.ship_pos = copy.deepcopy(self.initial_ship_pos)
        self.previous_ship_pos = [0.0, 0.0]
        self.previous_heading = 0.0
        self.ship_angle = 0.0
        self.ship_velocity = 0.0
        self.momentum = np.zeros(2)
        self.randomization_scale = 1    # Scale for randomization

        self.max_dist = np.sqrt(2) * self.MAX_GRID_POS

        # Environment setup
        self.checkpoints = []
        self.current_checkpoint = 1
        # self.target_pos = np.array(target_pos, dtype=np.float32)

        csv_input_dir = os.path.dirname(os.path.abspath(__file__))
        self._load_obstacles_and_paths(csv_input_dir)

        # Path and checkpoints initialization
        try:
            path = np.loadtxt(os.path.join(csv_input_dir, 'trajectory_points_no_scale.csv'), delimiter=',',
                              skiprows=1)
        except FileNotFoundError:
            raise FileNotFoundError("The file 'trajectory_points_no_scale.csv' could not be found.")

        path = create_checkpoints_from_simple_path(path, self.CHECKPOINTS_DISTANCE)
        checkpoints = [{'pos': np.array(point, dtype=np.float32), 'radius': 1.0} for point in path]

        # Calculate perpendicular lines
        lines = calculate_perpendicular_lines(checkpoints, 10)
        self.checkpoints = [
            {**checkpoint, 'perpendicular_line': line} for checkpoint, line in zip(checkpoints, lines)
        ]
        self.target_pos = np.array(self.checkpoints[-1]['pos'], dtype=np.float32)

        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                                           high=np.array([1, 1], dtype=np.float32),
                                           dtype=np.float32)
        self.observation_space = self._initialize_observation_space()

        # Rendering setup
        self.initialize_plots = True

        # Hydrodynamic coefficients
        self.xu = -0.02  # Surge damping
        self.yv = -0.4   # Sway damping
        self.yv_r = -0.09  # Default sway-to-yaw coupling coefficient
        self.nr = -0.26   # Yaw damping
        self.l = 50.0    # Ship length

        # Dynamic coefficients
        self.k_t = 0.05  # Thrust coefficient
        self.k_r = 0.039   # Rudder coefficient
        self.k_v = 0.03 # Sway coefficient

        # Ship state [x, y, heading, surge_velocity, sway_velocity, yaw_rate]
        self.state = np.array([self.ship_pos[0], self.ship_pos[1], 0.0, 0.0, 0.0, 0.0])


    def _load_obstacles_and_paths(self, csv_input_dir):
        """Load obstacle and path data from CSV files."""
        try:
            self.obstacles = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_250cm_no_scale.csv'), delimiter=',',
                                        skiprows=1).reshape(-1, 2)
            self.polygon_shape = Polygon(self.obstacles)

        except FileNotFoundError:
            raise FileNotFoundError("The file 'env_Sche_250cm_no_scale.csv' could not be found.")

        try:
            self.overall = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_no_scale.csv'), delimiter=',',
                                      skiprows=1).reshape(-1, 2)
        except FileNotFoundError:
            raise FileNotFoundError("The file 'env_Sche_no_scale.csv' could not be found.")


    def _initialize_observation_space(self):
        """
        Observation space for path-following:
            - Ship position (x, y): [0, MAX_GRID_POS]
            - Ship heading: [-pi, pi]
            - Surge velocity (u): [0, 5 m/s]
            - Sway velocity (v): [-2, 2 m/s]
            - Yaw rate (r): [-0.5, 0.5 m/s]
            - Distance to current checkpoint: [0, MAX_GRID_POS]
            - Distance to checkpoint+1 (if available): [0, MAX_GRID_POS]
            - Distance to checkpoint+2 (if available): [0, MAX_GRID_POS]
            - Cross-track error: [0, MAX_GRID_POS]
            - Heading error: [-pi, pi]
            - Rudder angle: [-1, 1]
            - Thrust: [-1, 1]
        """
        base_low=np.array([
            self.MIN_GRID_POS,          # Ship position x
            self.MIN_GRID_POS,          # Ship position y
            -np.pi,                     # Ship heading
            self.MIN_SURGE_VELOCITY,    # Surge velocity
            self.MIN_SWAY_VELOCITY,     # Sway velocity
            self.MIN_YAW_RATE,          # Yaw rate
            0.0,                        # Distance to current checkpoint
            0.0,                        # Distance to checkpoint+1
            0.0,                        # Distance to checkpoint+2
            0.0,                        # Cross-track error
            -np.pi,                     # Heading error
            -1.0,                       # Rudder angle
            -1.0,                       # Thrust
        ], dtype=np.float32)

        base_high=np.array([
            self.MAX_GRID_POS,          # Ship position x
            self.MAX_GRID_POS,          # Ship position y
            np.pi,                      # Ship heading
            self.MAX_SURGE_VELOCITY,    # Surge velocity
            self.MAX_SWAY_VELOCITY,     # Sway velocity
            self.MAX_YAW_RATE,          # Yaw rate
            self.MAX_GRID_POS,          # Distance to current checkpoint
            self.MAX_GRID_POS,          # Distance to checkpoint+1
            self.MAX_GRID_POS,          # Distance to checkpoint+2
            self.MAX_GRID_POS,          # Cross-track error
            np.pi,                      # Heading error
            1.0,                        # Rudder angle
            1.0,                        # Thrust
        ], dtype=np.float32)

        if self.wind:
            # Add wind parameters: direction and strength
            wind_low = np.array([-1.0, -1.0], dtype=np.float32)  # Normalized wind direction components
            wind_high = np.array([1.0, 1.0], dtype=np.float32)   # Normalized wind direction components
            base_low = np.concatenate([base_low, wind_low])
            base_high = np.concatenate([base_high, wind_high])

        if self.current:
            # Add current parameters: direction and strength
            current_low = np.array([-1.0, -1.0], dtype=np.float32)  # Normalized current direction components
            current_high = np.array([1.0, 1.0], dtype=np.float32)   # Normalized current direction components
            base_low = np.concatenate([base_low, current_low])
            base_high = np.concatenate([base_high, current_high])

        return gym.spaces.Box(
            low=base_low,
            high=base_high,
            dtype=np.float32
        )


    def _initialize_rendering(self):
        """Set up the rendering elements for visualization."""
        self.initialize_plots = False

        # self.fig, self.ax = plt.subplots(figsize=(18,15))
        # Create a temporary Tkinter root window to get screen dimensions
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()  # Close the temporary Tkinter window

        # Define figure dimensions (ensure it fits within screen)
        fig_width, fig_height = min(1200, screen_width), min(900, screen_height)
        dpi = 100  # Adjust as needed

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Get figure manager
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend()

        # Set window position based on the backend
        try:
            if backend in {"TkAgg"} and hasattr(manager, "window"):
                manager.window.wm_geometry(f"+0+0")  # Move to top-left
            elif backend in {"QtAgg", "Qt5Agg"} and hasattr(manager, "window"):
                manager.window.setGeometry(0, 0, fig_width, fig_height)
            elif backend == "GTK3Agg" and hasattr(manager, "window"):
                manager.window.move(0, 0)
            else:
                print(f"Backend {backend} does not support direct window positioning.")
        except Exception as e:
            print(f"Error setting window position: {e}")

        self.ship_plotC, = plt.plot([], [], 'bo', markersize=10, label='ShipC')
        self.target_plot, = plt.plot([], [], 'ro', markersize=10, label='Target')
        self.heading_line, = plt.plot([], [], color='black', linewidth=2, label='Heading')

        self.ax.set_xlim(self.MIN_GRID_POS, self.MAX_GRID_POS)
        self.ax.set_ylim(self.MIN_GRID_POS, self.MAX_GRID_POS)

        # Update plot title and legend
        self.ax.set_title('Ship Navigation in a Path Following Environment')
        self.ax.legend()


    def randomize(self, randomization_scale=None):
        """
        Randomize the initial conditions of the environment.
        :param randomization_scale: Scale of the randomization (optional).
        """
        if randomization_scale is not None:
            self.randomization_scale = randomization_scale

        # Apply random perturbation to the ship position
        self.ship_pos += np.random.uniform(
            low=-self.randomization_scale,
            high=self.randomization_scale,
            size=self.ship_pos.shape,
        )


    def env_specific_reset(self):
        pass


    def reset(self, seed=None, **kwargs):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)

        self.env_specific_reset()

        self.step_count = 0
        self.current_checkpoint = 1

        self.ship_pos = copy.deepcopy(self.initial_ship_pos)
        self.previous_ship_pos = [0.0, 0.0]
        self.ship_angle = 0.0
        self.ship_velocity = 0.0

        direction_vector = self.checkpoints[1]['pos'] - self.ship_pos
        self.ship_angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle in radians
        # Set the initial state
        self.state = np.array([self.ship_pos[0], self.ship_pos[1], self.ship_angle, 0.0, 0.0, 0.0])

        return self._get_obs(), {}


    def _get_obs(self):
        """Return normalized observation for the current state."""
        # Normalize positions to [-1, 1]
        norm_pos = 2 * (self.ship_pos - self.MIN_GRID_POS) / (self.MAX_GRID_POS - self.MIN_GRID_POS) - 1

        # Normalize velocities
        norm_velocities = np.array([np.clip(self.state[3] / self.MAX_SURGE_VELOCITY, -1, 1), np.clip(self.state[4] / (self.MAX_SWAY_VELOCITY/2), -1, 1), np.clip(self.state[5] / self.MAX_YAW_RATE, -1, 1)])

        # Calculate distances and normalize
        current_checkpoint_pos = self.checkpoints[self.current_checkpoint]['pos']
        distance_to_checkpoint = np.linalg.norm(self.ship_pos - current_checkpoint_pos)
        norm_distance = distance_to_checkpoint / self.max_dist

        # Calculate and normalize distances to next checkpoints
        norm_next_distances = np.zeros(2)
        if self.current_checkpoint + 1 < len(self.checkpoints):
            next_checkpoint_pos = self.checkpoints[self.current_checkpoint + 1]['pos']
            norm_next_distances[0] = np.linalg.norm(self.ship_pos - next_checkpoint_pos) / self.max_dist
        if self.current_checkpoint + 2 < len(self.checkpoints):
            next_next_checkpoint_pos = self.checkpoints[self.current_checkpoint + 2]['pos']
            norm_next_distances[1] = np.linalg.norm(self.ship_pos - next_next_checkpoint_pos) / self.max_dist

        # Calculate cross-track error and heading error
        previous_checkpoint_pos = self.checkpoints[self.current_checkpoint - 1]['pos']
        cross_track_error = self._distance_from_point_to_line(
            self.ship_pos, previous_checkpoint_pos, current_checkpoint_pos)
        norm_cross_error = cross_track_error / (self.CHECKPOINTS_DISTANCE / 2)

        direction_to_checkpoint = current_checkpoint_pos - self.ship_pos
        desired_heading = np.arctan2(direction_to_checkpoint[1], direction_to_checkpoint[0])
        heading_error = (desired_heading - self.state[2] + np.pi) % (2 * np.pi) - np.pi

        # Concatenate normalized observations
        obs = np.concatenate([
            norm_pos,                               # Normalized position
            [self.state[2] / np.pi],                # Normalized heading
            norm_velocities,                        # Normalized velocities
            [norm_distance],                        # Distance to current checkpoint
            norm_next_distances,                    # Distances to next checkpoints
            [norm_cross_error],                     # Normalized cross-track error
            [heading_error / np.pi],                # Normalized heading error
            self.current_action,                    # Current action
        ], dtype=np.float32)

        # Add wind observations if enabled
        if self.wind:
            wind_obs = np.array([
                self.wind_direction[0] * self.wind_strength,
                self.wind_direction[1] * self.wind_strength
            ], dtype=np.float32)
            obs = np.concatenate([obs, wind_obs])

        # Add current observations if enabled
        if self.current:
            current_obs = np.array([
                self.current_direction[0] * self.current_strength,
                self.current_direction[1] * self.current_strength
            ], dtype=np.float32)
            obs = np.concatenate([obs, current_obs])

        return obs


    def step(self, action):
        """Execute one timestep within the environment."""
        action = np.array(action) if not np.isscalar(action) else np.array([action, 0.0])
        # action = add_noise(action)

        # Update ship dynamics and calculate the reward
        self._update_ship_dynamics(action)
        reward, done = self._calculate_reward()

        # Increment step count and check for maximum steps
        self.step_count += 1
        if self.step_count >= self.max_steps:
            reward = -5
            done = True

        if not check_collision_ship(self.ship_pos, self.polygon_shape):
            reward = -10
            done = True

        return self._get_obs(), reward, done, False, {}


    def _update_ship_dynamics(self, action, alpha=0.2):
        """Update ship dynamics with improved physics and environmental effects."""
        # Smooth action application
        turning_smooth = alpha * action[0] + (1 - alpha) * self.current_action[0]
        thrust_smooth = alpha * abs(action[1]) + (1 - alpha) * self.current_action[1]
        self.current_action = [turning_smooth, thrust_smooth]

        # Convert actions to forces
        delta_r = np.radians(turning_smooth * 40)
        t = thrust_smooth * 60

        # Current state
        x, y, psi, u, v, r = self.state

        # Transform wind effects into ship's coordinate system
        relative_wind_angle = self.radians_wind - psi
        wind_effect = np.array([
            self.wind_strength * np.cos(relative_wind_angle),  # Longitudinal component
            self.wind_strength * np.sin(relative_wind_angle)   # Lateral component
        ])

        # Transform current effects into ship's coordinate system
        relative_current_angle = self.radians_current - psi
        current_effect = np.array([
            self.current_strength * np.cos(relative_current_angle),  # Longitudinal component
            self.current_strength * np.sin(relative_current_angle)   # Lateral component
        ])

        # Precompute reusable values
        sin_delta_r = np.sin(delta_r)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Update momentum
        self.momentum = 0.95 * self.momentum + 0.05 * np.array([u, v])

        # Update dynamics with environmental effects
        # Wind
        tmp = 1
        if self.wind and not self.current or tmp == 1:
            du = self.k_t * t + self.xu * u + wind_effect[0]
            dv = self.k_v * sin_delta_r + self.yv * v + wind_effect[1]
        # Current
        elif self.current and not self.wind:
            du = self.k_t * t + self.xu * u + current_effect[0]
            dv = self.k_v * sin_delta_r + self.yv * v + current_effect[1]
        # Both
        elif self.current and self.wind:
            du = self.k_t * t + self.xu * u + current_effect[0] + wind_effect[0]
            dv = self.k_v * sin_delta_r + self.yv * v + current_effect[1] + wind_effect[1]
        # Neither
        else:
            du = self.k_t * t + self.xu * u
            dv = self.k_v * sin_delta_r + self.yv * v
        #dr = self.k_r * delta_r + self.nr * v / self.l
        dr = self.k_r * delta_r + self.nr * r + self.yv_r * v + v * u / self.l - 0.1 * r

        u = np.clip(u + du * self.TIME_STEP, self.MIN_SURGE_VELOCITY, self.MAX_SURGE_VELOCITY)
        v = np.clip(v + dv * self.TIME_STEP, self.MIN_SWAY_VELOCITY, self.MAX_SWAY_VELOCITY)
        r = np.clip(r + dr * self.TIME_STEP, self.MIN_YAW_RATE, self.MAX_YAW_RATE)  # Update yaw rate with limits

        # Update position and heading
        dx = u * cos_psi - v * sin_psi
        dy = u * sin_psi + v * cos_psi
        dpsi = r

        # Store previous state and update
        self.previous_ship_pos = copy.deepcopy(self.ship_pos)
        self.previous_heading = self.state[2]

        # Update state
        self.state[0] = np.clip(self.state[0] + dx * self.TIME_STEP, self.MIN_GRID_POS, self.MAX_GRID_POS)
        self.state[1] = np.clip(self.state[1] + dy * self.TIME_STEP, self.MIN_GRID_POS, self.MAX_GRID_POS)
        self.state[2] += dpsi * self.TIME_STEP
        self.state[3] = u
        self.state[4] = v
        self.state[5] = r

        # Update ship position
        self.ship_pos = self.state[:2]


    @staticmethod
    def _distance_from_point_to_line(point, line_seg_start, line_seg_end):
        """    Calculate the perpendicular distance from point P to the line segment defined by A and B.    """
        line_vec = line_seg_end - line_seg_start
        point_vec = point - line_seg_start

        # Line magnitude squared (to avoid division by zero)
        line_mag_squared = np.dot(line_vec, line_vec)
        if line_mag_squared == 0:
           # If the two points defining the line are identical, return the distance to this point
           return np.linalg.norm(point - line_seg_start)

        # Projection of the point onto the line
        projection_scalar = np.dot(point_vec, line_vec) / line_mag_squared
        projected_point = line_seg_start + projection_scalar * line_vec

        # Distance from the point to the projected point on the infinite line
        return np.linalg.norm(point - projected_point)


    def _is_object_within_distance_of_line(self, point, line_seg_start, line_seg_end, threshold):
        """
        Checks if the object's path from P_prev to P_curr comes within 'distance' of the line segment A-B.
        """
        # If the distance is less than or equal to the threshold, return True
        return self._distance_from_point_to_line(point, line_seg_start, line_seg_end) <= threshold


    def _calculate_reward(self):
        """Calculate reward with improved shaping and penalties."""
        reward = 0.0
        done = False

        # Distance reward with better scaling
        current_distance = np.linalg.norm(self.ship_pos - self.target_pos)
        prev_distance = np.linalg.norm(self.previous_ship_pos - self.target_pos)
        distance_delta = prev_distance - current_distance
        distance_reward = self.REWARD_DISTANCE_SCALE * np.tanh(distance_delta)
        reward += distance_reward

        # Heading alignment reward
        desired_heading = np.arctan2(
            self.checkpoints[self.current_checkpoint]['pos'][1] - self.ship_pos[1],
            self.checkpoints[self.current_checkpoint]['pos'][0] - self.ship_pos[0]
        )
        heading_diff = abs(self.state[2] - desired_heading) % (2 * np.pi)
        heading_reward = np.cos(heading_diff)
        reward += heading_reward

        # Cross-track error penalty
        line_start = self.checkpoints[self.current_checkpoint - 1]['pos']
        line_end = self.checkpoints[self.current_checkpoint]['pos']
        cross_error = self._distance_from_point_to_line(self.ship_pos, line_start, line_end)
        cross_error_penalty = -np.tanh(cross_error / self.CROSS_TRACK_ERROR_PENALTY_SCALE)  # Normalized and smooth
        reward += cross_error_penalty
        self.cross_error = cross_error

        # Checkpoint reward with progression scaling
        checkpoint_pos = self.checkpoints[self.current_checkpoint]['pos']
        distance_to_checkpoint = np.linalg.norm(self.ship_pos - checkpoint_pos)
        checkpoint_scale = (self.current_checkpoint / len(self.checkpoints)) * 10

        if distance_to_checkpoint < 10.0:
            reward += checkpoint_scale
            self.current_checkpoint += 1
            self.step_count = 0

        # Movement reward to prevent getting stuck
        movement = np.linalg.norm(self.ship_pos - self.previous_ship_pos)
        if movement < 0.1:
            self.stuck_steps += 1
            if self.stuck_steps > 20:
                reward -= 10
                done = True
        else:
            self.stuck_steps = 0

        # Success reward
        if current_distance < 30.0:
            done = True
            reward += 50.0

        # Termination conditions
        if cross_error > self.CHECKPOINTS_DISTANCE * 2:
            done = True
            reward -= 20

        if abs(self.state[2] - self.previous_heading) > np.pi/2:
            done = True
            reward -= 15

        if self.current_checkpoint < len(self.checkpoints)-1:
            if self._distance_from_point_to_line(self.ship_pos, self.checkpoints[self.current_checkpoint]['perpendicular_line'][0], self.checkpoints[self.current_checkpoint]['perpendicular_line'][1]) <= 2:
                self.current_checkpoint += 1
                self.step_count = 0
                if self.current_checkpoint == len(self.checkpoints)-1:
                    done = True

        return reward, done


    def _draw_fixed_landmarks(self):
        # Draw the path as straight lines
        if self.checkpoints:  # Check if there are checkpoints to draw
            # Start with the ship position and end with the target position
            path_points = [self.ship_pos] + [checkpoint['pos'] for checkpoint in self.checkpoints] + [self.target_pos]

            # Loop through the path points and draw straight lines between consecutive points
            for i in range(len(path_points) - 1):
                start_point = path_points[i]
                end_point = path_points[i + 1]
                self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g-', label='Path' if i == 0 else "")  # Green lines for the path
            for i, checkpoint in enumerate(self.checkpoints):
                check = patches.Circle((checkpoint['pos'][0], checkpoint['pos'][1]), 10, color='black', alpha=0.3)
                self.ax.add_patch(check)
                start_point = checkpoint['perpendicular_line'][0]
                end_point = checkpoint['perpendicular_line'][1]
                self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g-', label='Path' if i == 0 else "")  # Green lines for the path

        # Draw the target location (if necessary)
        self.target_plot.set_data(self.target_pos[0:1], self.target_pos[1:2])

        # Add the obstacles as polygons (uncomment if needed)
        # for x, y in self.obstacles:
        #     rect = patches.Circle((x, y), 100, color='black', alpha=0.3)
        #     self.ax.add_patch(rect)

        # Add the polygon to the plot
        polygon_patch = patches.Polygon(self.obstacles, closed=True, edgecolor='r', facecolor='none', lw=2, label='Waterway')
        western_scheldt = patches.Polygon(self.overall, closed=True, edgecolor='brown', facecolor='none', lw=2, label='Western Scheldt')
        self.ax.add_patch(polygon_patch)
        self.ax.add_patch(western_scheldt)


    def render(self):
        """Render the environment and visualize the ship's movement."""
        if self.render_mode == 'human':
            if self.initialize_plots:
                self._initialize_rendering()
                self._draw_fixed_landmarks()

            # Update ship's location
            self.ship_plotC.set_data(self.ship_pos[0:1], self.ship_pos[1:2])
            heading_x = self.ship_pos[0] + np.cos(self.state[2]) * 30  # Length of the heading line
            heading_y = self.ship_pos[1] + np.sin(self.state[2]) * 30
            self.heading_line.set_data([self.ship_pos[0], heading_x], [self.ship_pos[1], heading_y])

            plt.pause(0.001)


    def close(self):
        """Close the environment."""
        if self.render_mode == 'human':
            plt.close()
