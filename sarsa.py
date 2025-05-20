import random
import time

import torch
import math
import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import os
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point


class SARSA:
    def __init__(self, alpha=0.1, gamma=0.97, epsilon=0.2, epochs=50, grid_width=100, grid_height=100, goal=(-1530.0,12010.0), start=(2060.0,-50.0), grid_resolution=50.0, save_map=False, max_steps=1200, curricular_learning=[]):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self._just_switched = False
        self.device = torch.device("cpu")
        if len(curricular_learning) != 0:
            self.curricular_learning = curricular_learning.copy()
            self.original_curricular_learning = curricular_learning.copy()
        else:
            self.curricular_learning = None
            self.original_curricular_learning = None
        if (self.curricular_learning != None and not len(self.curricular_learning) == 0):
            self.isCurricular = True
        else:
            self.isCurricular = False
        self.grid_width, self.grid_height = grid_width, grid_height
        self.goal = goal
        self.start = start
        self.current_position = start
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        """
        More realistic action space (but it currently doesn't fully work, because the direction of
        the ships doesn't turn with the action taken. This makes it so the ship can turn with a
        maximum angle of 45°. But in the future, with the direction changing along the action
        this problem should be solved. 
        """
        #self.actions = [(0, 1), (-1, 1), (1, 1)]

        self.action_size = len(self.actions)
        self.total_distance_real = abs(self.goal[0]-self.start[0])+abs(self.goal[1]-self.start[1])
        self.grid_resolution = grid_resolution
        self.min_reward_seen = float("inf")
        self.max_reward_seen = float("-inf")
        if save_map:
            self.map = self.discretize_map()
        else:
            self.map = self.load_occupancy_grid()
        if self.isCurricular:
            self.total_distance_discrete = abs(self.x_goal - self.x_start) + abs(self.y_goal - self.y_start)
        else:
            self.total_distance_discrete = abs(self.x_goal - self.x_start) + abs(self.y_goal - self.y_start)

        self.show_map()
        self.q_table = np.zeros((self.grid_width, self.grid_height, self.action_size))
        self.previous_distance = 0
        self.wall_distance_map = distance_transform_edt(self.map == 1) * self.grid_resolution
        self.max_steps = max_steps




    def discretize_map(self):
        """
            Discretize the environment contour into a grid occupancy map using self.grid_resolution.

            :param: None
            :return: occupancy_grid (np.ndarray): 2D array of shape (grid_width, grid_height) with 1 for navigable cells and 0 for obstacles
        """
        csv_input_dir = os.path.dirname(os.path.abspath(__file__))
        contour_points = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_250cm_no_scale.csv'), delimiter=',',
                              skiprows=1)

        map_polygon = Polygon(contour_points)
        self.x_min, self.y_min = contour_points.min(axis=0)
        self.x_max, self.y_max = contour_points.max(axis=0)

        x_grid = np.arange(self.x_min, self.x_max, self.grid_resolution)
        y_grid = np.arange(self.y_min, self.y_max, self.grid_resolution)
        self.grid_width = len(x_grid)
        self.grid_height = len(y_grid)
        occupancy_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.uint8)

        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                point = Point(x + self.grid_resolution / 2, y + self.grid_resolution / 2)
                if map_polygon.contains(point):
                    occupancy_grid[i, j] = 1  # 1 = navigable
                else:
                    occupancy_grid[i, j] = 0  # 0 = obstacle or outside

        self.save_occupancy_grid(occupancy_grid, self.grid_resolution)
        self.x_start, self.y_start = self.discretize_position(self.start)
        if self.isCurricular:
            """
                Because at the start of the learning (show_map) it already sets 
                self.x_goal and self.y_goal to the first sub goal of the curricular learning
            """
            first_sg = self.curricular_learning[0]
            self.x_goal, self.y_goal = self.discretize_position(first_sg)
        else:
            self.x_goal, self.y_goal = self.discretize_position(self.goal)
        return occupancy_grid

    def save_occupancy_grid(self, occupancy_grid, resolution):
        """
            Save the occupancy grid to a .npy file named using the given resolution.

            :param occupancy_grid, resolution: occupancy_grid is an np.ndarray of the grid; resolution is the grid cell size for the filename.
            :return: None
        """
        csv_input_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(csv_input_dir, 'occupancy_grid_'+str(resolution)+'.npy')
        np.save(save_path, occupancy_grid)
        print(f"Occupancy grid saved to {save_path}")

    def load_occupancy_grid(self):
        """
            Load the most accurate saved occupancy grid (or create one if none exist).

            :param: None
            :return: occupancy_grid (np.ndarray): 2D array of 1-navigable/0-obstacle cells
        """
        csv_input_dir = os.path.dirname(os.path.abspath(__file__))

        files = [f for f in os.listdir(csv_input_dir) if f.startswith('occupancy_grid_') and f.endswith('.npy')]
        if not files:
            self.discretize_map()
            files = [f for f in os.listdir(csv_input_dir) if f.startswith('occupancy_grid_') and f.endswith('.npy')]
            if not files:
                raise FileNotFoundError("Occupancy grid could not be created.")

        resolutions = []
        for file in files:
            try:
                res = float(file.split('_')[-1].replace('.npy', ''))
                resolutions.append((res, file))
            except ValueError:
                continue  # skip invalid files

        if not resolutions:
            raise FileNotFoundError("No valid occupancy grid files with resolutions found.")

        # Pick the file with the smallest resolution (highest accuracy)
        self.grid_resolution, best_file = min(resolutions)
        contour_points = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_250cm_no_scale.csv'), delimiter=',',
                                    skiprows=1)

        self.x_min, self.y_min = contour_points.min(axis=0)
        self.x_max, self.y_max = contour_points.max(axis=0)
        self.x_start, self.y_start = self.discretize_position(self.start)
        if self.isCurricular:
            """
                Because at the start of the learning (show_map) it already sets 
                self.x_goal and self.y_goal to the first sub goal of the curricular learning
            """
            first_sg = self.curricular_learning[0]
            self.x_goal, self.y_goal = self.discretize_position(first_sg)

        else:
            self.x_goal, self.y_goal = self.discretize_position(self.goal)
        x_grid = np.arange(self.x_min, self.x_max, self.grid_resolution)
        y_grid = np.arange(self.y_min, self.y_max, self.grid_resolution)

        self.grid_width = len(x_grid)
        self.grid_height = len(y_grid)
        load_path = os.path.join(csv_input_dir, best_file)
        occupancy_grid = np.load(load_path)
        print(f"Occupancy grid loaded from {load_path} with resolution {self.grid_resolution}")
        return occupancy_grid

    def discretize_position(self, state):
        """
            Convert a continuous (x, y) position to grid indices based on self.x_min, self.y_min, and self.grid_resolution.

            :param state: tuple of floats (x, y) in world coordinates
            :return: tuple of ints (x_idx, y_idx) corresponding to the grid cell indices
        """
        x_disc = int((state[0] - self.x_min) / self.grid_resolution)
        y_disc = int((state[1] - self.y_min) / self.grid_resolution)
        return (x_disc,y_disc)

    def dediscretize_path(self, discrete_path):
        """
            Convert a sequence of grid index pairs to continuous (x, y) coordinates at each cell’s center.

            :param discrete_path: Iterable of (x_idx, y_idx) tuples in grid indices
            :return: List of (x, y) float tuples representing cell-center positions
        """
        continuous = []
        for x_idx, y_idx in discrete_path:
            x = self.x_min + (x_idx + 0.5) * self.grid_resolution
            y = self.y_min + (y_idx + 0.5) * self.grid_resolution
            continuous.append((x, y))
        return continuous

    def show_map(self):
        """
            Display the discretized grid and continuous environment polygon with start, goal, and sub-goals.

            :param: None
            :return: None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 16))

        ax1.imshow(self.map.T, origin='lower', cmap='Greys')
        ax1.set_title('Discretized Map (1 = Water, 0 = Land), Resolution = ' + str(self.grid_resolution))
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')
        fig.colorbar(ax1.images[0], ax=ax1)  # colorbar must be associated with the figure, not the axis
        if self.isCurricular:
            ax1.scatter(self.x_start, self.y_start, color='blue', s=25, label='Start')
            goal_x, goal_y = self.discretize_position(self.goal)
            ax1.scatter(goal_x, goal_y, color='red', s=25, label='Goal')
            xs, ys = zip(*(self.discretize_position(sg) for sg in self.original_curricular_learning))
            ax1.scatter(xs, ys,
                        color='yellow',
                        s=25,
                        label='Sub-goals')

        else:
            ax1.scatter(self.x_start, self.y_start, color='blue', s=25, label='Start')
            ax1.scatter(self.x_goal, self.y_goal, color='red', s=25, label='Goal')

        ax1.legend()

        csv_input_dir = os.path.dirname(os.path.abspath(__file__))
        map = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_250cm_no_scale.csv'), delimiter=',', skiprows=1)
        map_polygon = Polygon(map)

        patch = MplPolygon(np.array(map_polygon.exterior.coords), closed=True, facecolor='lightgray', edgecolor='black')
        ax2.add_patch(patch)

        # Set the axes limits to fit the polygon
        ax2.set_xlim(min(map[:, 0]), max(map[:, 0]))
        ax2.set_ylim(min(map[:, 1]), max(map[:, 1]))
        ax2.scatter(self.start[0], self.start[1], color='blue', s=25, label='Start')
        ax2.scatter(self.goal[0], self.goal[1], color='red', s=25, label='Goal')
        ax2.legend()

        plt.show()

    def reward_function(self, state, episode=0):
        """
            Compute the reward, termination flag, and goal indicator for a given grid state and episode.

            :param state, episode: state is (x_idx, y_idx) grid indices tuple; episode is the current episode number (int)
            :return: tuple (final_reward (float), done (bool), goal (int))
        """
        x_idx, y_idx = state
        scaling_factor_goal = 3
        time_constraint = 0.5
        reach_goal = scaling_factor_goal*(abs(self.x_goal-self.x_start)+abs(self.y_goal-self.y_start))
        goal = 0
        done = False

        # Check goal
        if (x_idx, y_idx) == (int(self.x_goal), int(self.y_goal)):
            reward = reach_goal+10
            goal = 1
            if self.isCurricular and self.curricular_learning:
                self.next_sub_goal(state, episode)
                done = False
            else:
                done = True
            print(f"Goal reached at ({x_idx}, {y_idx})")
        elif not (0 <= x_idx < self.grid_width and 0 <= y_idx < self.grid_height):
            # Out of bounds → treat like hitting a wall
            reward = self.reward_collision(reach_goal, time_constraint)
            done = True # and terminate!
            print(f"Out of bounds at ({x_idx}, {y_idx})")
        else:
            # Inside bounds
            cell_value = self.map[x_idx, y_idx]

            if cell_value == 0:
                # Hit wall/obstacle
                reward = self.reward_collision(reach_goal, time_constraint)
                done = True # and terminate!
                print(f"Hit obstacle at ({x_idx}, {y_idx})")
            else:
                reward = 0  # Normal move, no big reward
        reward += self.reward_direction(state)
        # This can be added in future work
        #reward += self.reward_proximity_to_wall(state[0],state[1])
        #final_reward = self.normalize_reward(reward - time_constraint) # this is the time constraint punishment. So that the agent finds the fastest path
        final_reward = reward - time_constraint # this is the time constraint punishment. So that the agent finds the fastest path
        print("     final_reward = " + str(final_reward)+", Done: "+str(done))
        return final_reward, done, goal

    def next_sub_goal(self,state, episode=0):
        """
            Advance to and log the next sub-goal when using curricular learning.

            :param state: tuple of ints (x_idx, y_idx) indicating the reached grid cell
            :param episode: int current episode number
            :return: None
        """
        if self._just_switched:
            return
        self._just_switched = True

        if len(self.curricular_learning) > 1:
            upcoming = self.curricular_learning[1]
        else:
            upcoming = self.goal

        reached = (int(state[0]), int(state[1]))
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("subgoal_transitions.log", "a") as f:
            f.write(f"{ts} at episode {episode}| reached {reached}, next = {upcoming}\n")

        if self.isCurricular and self.curricular_learning:
            # pop the one we just reached
            self.curricular_learning.pop(0)
            if self.curricular_learning:
                # set the next one
                self.x_goal, self.y_goal = self.discretize_position(self.curricular_learning[0])
                self._just_switched = False

            else:
                # no more sub-goals, go to the real goal
                self.x_goal, self.y_goal = self.discretize_position(self.goal)
        else:
            self.x_goal, self.y_goal = self.discretize_position(self.goal)
            self.x_start, self.y_start = self.discretize_position(self.start)

        self.previous_distance = None
        self.total_distance_discrete = abs(self.x_goal - self.x_start) + abs(self.y_goal - self.y_start)


    def reward_direction(self,current_position):
        """
            Reward based on change in Manhattan distance to the goal since the last step.

            :param current_position: tuple of ints (x_idx, y_idx) current grid indices
            :return: float positive if closer, negative if farther, zero on first move
        """

        # Other simple option for reward direction
        # if distance <= self.previous_distance:
        #     reward = 1
        # else:
        #     reward = -1
        # self.previous_distance = distance
        #
        # return reward

        x_idx, y_idx = current_position
        current_dist = abs(self.x_goal - x_idx) + abs(self.y_goal - y_idx)

        if not hasattr(self, 'previous_distance') or self.previous_distance is None:
            self.previous_distance = abs(self.x_goal - self.x_start) + abs(self.y_goal - self.y_start)
            # No reward on the first move
            return 0.0

        # Change in distance: positive if we got closer
        delta = self.previous_distance - current_dist
        self.previous_distance = current_dist
        return delta

    def reward_proximity_to_wall(self, x, y):
        """
            Not fully implemented and currently unused; penalizes proximity to walls for safer paths.

            :param x, y: int grid indices to evaluate distance from nearest wall
            :return: float negative penalty based on wall distance, or -5.0 if out-of-bounds
        """
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            dist = self.wall_distance_map[x, y]
            if dist < 2*self.grid_resolution:  # too close
                return -1.60
            elif dist < 4 * self.grid_resolution:
                return -1.25
            elif dist < 8 * self.grid_resolution:
                return -1.1
            else:
                return 0.0
        else:
            return -5.0  # punish unknown/out-of-bounds area just in case

    def reward_collision(self, reach_goal, time_constraint):
        """
            Compute the collision penalty from goal reach estimate and time constraint.

            :param reach_goal, time_constraint: reach_goal (float) estimated reward to reach goal; time_constraint (float) per-step time penalty factor
            :return: float negative penalty applied on collision
        """
        return -(reach_goal + abs(time_constraint*self.total_distance_discrete))

    def normalize_reward(self, r:float)->float:
        """
            Scale the raw reward into a bounded range using hyperbolic tangent.
            Not used, because I this had not the intended result.

            :param r: raw reward value (float)
            :return: normalized reward in (-1, 1) (float)
        """
        return math.tanh(r)


    def epsilon_greedy(self, state):
        """
            Choose an action using ε-greedy policy for exploration vs. exploitation.

            :param state: tuple of ints (x_idx, y_idx) current grid indices
            :return: int selected action index
        """
        x, y = state
        print('current epsilon: '+str(self.epsilon))
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q = self.q_table[x, y]
        best = np.flatnonzero(q == q.max())
        # random between multiple best actions (in case of a tie)
        return np.random.choice(best)

    def update(self,state_idx, action_idx, reward, next_state_idx, next_action_idx):
        """
            Perform SARSA Q-table update using current and next state-action information.

            :param state_idx, action_idx, reward, next_state_idx, next_action_idx: state_idx is (x, y) tuple; action_idx and next_action_idx are int actions; reward is float; next_state_idx is (nx, ny) tuple
            :return: None
        """
        x, y = state_idx
        nx, ny = next_state_idx

        # Q-learning update (SARSA)
        self.q_table[x, y, action_idx] += self.alpha * (
                reward + self.gamma * self.q_table[nx, ny, next_action_idx] - self.q_table[x, y, action_idx]
        )


    def select_action(self, state):
        """
            Choose an action via ε-greedy policy and return its movement vector and index.

            :param state: tuple of ints (x_idx, y_idx) current grid indices
            :return: tuple (np.ndarray of shape (2,), int) movement delta [dx, dy] and selected action index
        """
        action = self.epsilon_greedy(state)
        dx, dy = self.actions[action]
        return np.array([dx, dy], dtype=np.float32), action

    def reset_episode(self):
        """
            Reset episode state: curriculum, start/goal positions, flags, and distance tracker.

            :param: None
            :return: None
        """
        if self.isCurricular and self.original_curricular_learning is not None:
            self.curricular_learning = self.original_curricular_learning.copy()
        self._just_switched = False

        self.x_start, self.y_start = self.discretize_position(self.start)
        if self.isCurricular and self.curricular_learning:
            # point at the first sub‐goal in the list
            self.x_goal, self.y_goal = self.discretize_position(self.curricular_learning[0])
        else:
            # no sub‐goals => point at the real goal
            self.x_goal, self.y_goal = self.discretize_position(self.goal)

        self.previous_distance = None
        self.total_distance_discrete = (
                abs(self.x_goal - self.x_start) + abs(self.y_goal - self.y_start)
        )

def run_episode(agent,episode):
    """
        Execute one SARSA learning episode: reset, step until goal/termination, and record results.

        :param agent, episode: agent is the RL agent instance; episode is the current episode number (int)
        :return: dict with keys 'episode_reward' (float), 'path' (list of (x_idx, y_idx)), and 'goal' (int indicator)
    """
    agent.reset_episode()
    path = []
    state = agent.start  # Start at the agent's starting position
    # agent.current_position = state  # Reset current position
    # agent.previous_distance = None # reset distance for distance rewards
    # if agent.isCurricular and agent.original_curricular_learning is not None:
    #     agent.curricular_learning = agent.original_curricular_learning.copy()
    #     agent.x_goal, agent.y_goal = agent.discretize_position(agent.curricular_learning[0])
    done = False
    episode_reward = 0
    x_idx = int(agent.x_start)
    y_idx = int(agent.y_start)

    if agent.map[x_idx, y_idx] == 0:
        raise ValueError(f"Start position ({x_idx}, {y_idx}) is inside an obstacle!")

    path.append((x_idx,y_idx))
    # Choose initial action
    action_idx = agent.epsilon_greedy((x_idx, y_idx))
    goal = 0
    steps_taken = 0
    while not done and steps_taken < agent.max_steps:
        print('step: '+str(steps_taken+1)+'/'+str(agent.max_steps))
        print("EPISODE: "+str(episode))
        dx, dy = agent.actions[action_idx]
        next_position = (state[0] + dx * agent.grid_resolution,
                         state[1] + dy * agent.grid_resolution)

        nx_idx, ny_idx = agent.discretize_position((next_position[0], next_position[1]))
        reward, done, goal = agent.reward_function((nx_idx, ny_idx),episode)
        path.append((nx_idx, ny_idx))

        # Clip movement if it hits wall or out of bounds (reward_function already handles penalties)
        if not done and (0 <= nx_idx < agent.grid_width) and (0 <= ny_idx < agent.grid_height):
            if agent.map[nx_idx, ny_idx] == 1:
                next_state = next_position
            else:
                next_state = state  # Stay in place if wall
        else:
            next_state = state  # Stay in place

        next_action_idx = agent.epsilon_greedy((nx_idx, ny_idx))
        # SARSA Q-update
        agent.update((x_idx, y_idx), action_idx, reward, (nx_idx, ny_idx), next_action_idx)
        # Move to next step
        state = next_state
        x_idx, y_idx = nx_idx, ny_idx
        action_idx = next_action_idx
        episode_reward += reward
        steps_taken+=1
    # returns a dictionary with the result of the episode
    return {
        'episode_reward': episode_reward,
        'path': path,
        'goal': goal
    }

def mean_and_max_cross_error(good_path, my_path):
    """
        Calculates the mean and maximum cross-track error between a reference path and a given path.

        Parameters:
        good_path (list of tuples): Coordinates defining the reference path.
        my_path (list of tuples): Coordinates defining the user-provided path.

        Returns:
        tuple: Mean and maximum cross-track error, along with a list of individual errors.
    """
    ref_line = LineString(good_path)
    errors = [ref_line.distance(Point(x, y)) for x, y in my_path]
    print("Mean cross-track error: {:.1f} m".format(np.mean(errors)))
    print("Max  cross-track error: {:.1f} m".format(np.max(errors)))
    return np.mean(errors), np.max(errors), errors

def benchmark(good_path, my_path, agent):
    """
        Plot path comparison and error statistics between reference and test paths.

        :param good_path, my_path, agent: good_path and my_path are sequences of (x, y) coordinates; agent provides start and goal for annotations.
        :return: None
    """
    _,_,errors = mean_and_max_cross_error(good_path,my_path)
    plt.figure()
    plt.plot()

    # 1) Path overlay
    plt.figure(figsize=(8, 8))
    plt.plot(good_path[:, 0], good_path[:, 1], '-k', lw=2, label='Reference')
    plt.plot(my_path[:, 0], my_path[:, 1], '-r', lw=1, label='SARSA')
    plt.scatter([agent.start[0]], [agent.start[1]], c='blue', label='Start')
    plt.scatter([agent.goal[0]], [agent.goal[1]], c='green', label='Goal')
    plt.legend()
    plt.axis('equal')
    plt.title("Path Comparison")
    plt.show()

    # 2) Error vs. step index
    plt.figure(figsize=(6, 3))
    plt.plot(errors, '-o', markersize=3)
    plt.xlabel("Step index")
    plt.ylabel("Cross-track error (m)")
    plt.title("Cross-track error over trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Histogram of errors
    plt.figure(figsize=(6, 3))
    plt.hist(errors, bins=30, edgecolor='k')
    plt.xlabel("Cross-track error (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of cross-track errors")
    plt.tight_layout()
    plt.show()

def linear_epsilon(episode, num_episodes, max_eps=0.2, min_eps=0.0, anneal_episodes=0.9):
    """
        Linearly anneal ε from max_eps to min_eps over initial episodes.

        :param episode, num_episodes, max_eps, min_eps, anneal_episodes: episode index (int), total episodes (int), start ε (float), end ε (float), fraction of episodes to anneal (float)
        :return: ε value for the current episode (float)
    """
    # drop ε from max_eps → min_eps over the first `anneal_episodes * num_episodes` episodes
    frac = min(1.0, episode/(anneal_episodes*num_episodes))
    epsilon = 0
    if episode == num_episodes - 1:
        epsilon = 0
    else:
        epsilon =max_eps + frac * (min_eps - max_eps)
    return epsilon

def create_path_sarsa(agent:SARSA, num_episodes):
    """
        Train the SARSA agent over multiple episodes and return the final continuous path.

        :param agent, num_episodes: agent is a SARSA instance; num_episodes is the total episodes to run (int)
        :return: np.ndarray of continuous (x, y) coordinates for the path taken in the last episode
    """
    eval_frequency = 1
    rewards = []
    goals = []
    best_reward = -float('inf')
    best_path = None
    last_path = None

    print("Starting training...")

    for episode in range(num_episodes):
        agent.epsilon = linear_epsilon(episode, num_episodes, max_eps=0.2, min_eps=0.0, anneal_episodes=0.8)
        result = run_episode(agent, episode)

        # Store metrics
        rewards.append(result['episode_reward'])
        rewards.append(result['episode_reward'])
        if result['goal'] == 1 and result['episode_reward'] > best_reward:
            best_reward, best_path = result['episode_reward'], result['path']
        if result['goal'] == 1:
            goals.append(1)

        if episode == int(num_episodes * 0.75):
            agent.alpha = 0.02

        if episode == num_episodes - 1:
            last_path = result['path']
        print('Episode: ' + str(episode))
        if (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(rewards[-eval_frequency:])
            print(f"Episode {episode + 1}/{num_episodes} | Average Reward (last {eval_frequency}): {avg_reward:.3f}")

    print("\nTraining completed!")

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('SARSA Training Reward Curve')
    plt.grid(True)
    plt.show()

    # Optionally, save final Q-table
    np.save('trained_q_table.npy', agent.q_table)
    print("Final Q-table saved to trained_q_table.npy")
    print('Reached ' + str(sum(goals)) + ' time(s) the goal state')

    plt.figure()
    plt.imshow(agent.map.T, origin='lower', cmap='Greys')
    if best_path is None:
        print("Warning: no episode reached the goal → skipping best‐path plot")
    else:
        bx, by = zip(*best_path)
        plt.plot(
            bx, by,
            marker='o', markersize=3, linestyle='-',
            linewidth=1.25, color='m',
            label='best path to goal'
        )
    lx, ly = zip(*last_path)

    plt.plot(lx, ly, marker='o', markersize=3, linestyle='-', linewidth=1.25, color='green', label='(last) ideal path')
    plt.scatter(agent.x_start, agent.y_start, color='blue', s=80, label='Start')
    gx, gy = agent.discretize_position(agent.goal)
    plt.scatter(gx, gy, color='red', s=80, label='Goal')
    if agent.original_curricular_learning != None and len(agent.original_curricular_learning) != 0:
        xs, ys = zip(*(agent.discretize_position(sg) for sg in agent.original_curricular_learning))
        plt.scatter(xs, ys,
                    color='yellow',
                    s=25,
                    label='Sub-goals')
    plt.title('Path followed in last episode')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.legend()
    plt.show()
    continuous_final_path = agent.dediscretize_path(last_path)
    return np.array(continuous_final_path)

def main():
    num_episodes = 3000
    eval_frequency = 1

    """
    Different goals, to test different setups with different curriculum learning
    """
    # Original goal and start state
    #ship_pos = [2060.0, -50.0]
    #target_pos = [-1530.0, 12010.0]
    #curricular_learning = [(2520,350),(2850,840),(3070,1420),(3600,1510),(3570,3370),(3380,4790),(2990,6040),(1460,6930),(770,8500),(320,10090),(-540,11550)]

    # Very close goal and start state
    #ship_pos = [2200.0, 30.0]
    #target_pos = [2330.0, 60.0]

    # Goal state is around the corner
    ship_pos = [2200.0, 30.0]
    target_pos = [2435.0, 1556]
    curricular_learning = [(2520,350),(2850,840),(3070,1420)]

    agent = SARSA(
        alpha=0.1,
        epsilon=0.2,
        grid_resolution=10,
        goal=(target_pos[0],target_pos[1]),
        start=(ship_pos[0],ship_pos[1]),
        save_map=False, # if True -> makes new grid with resoultion
        max_steps= 24000,
        curricular_learning=[]
    )

    rewards = []
    paths = []
    goals = []

    print("Starting training...")

    for episode in range(num_episodes):
        # Run one full episode
        if episode == int(num_episodes/2):
            agent.epsilon = 0.1
        if episode == int(num_episodes*0.75):
            agent.alpha = 0.02
        if episode == num_episodes-1:
            agent.epsilon = 0
        result = run_episode(agent,episode)

        # Store metrics
        rewards.append(result['episode_reward'])
        paths.append((result['path'],result['goal']))
        goals.append(result['goal'])
        print('Episode: '+str(episode))
        # Optionally print progress
        if (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(rewards[-eval_frequency:])
            print(f"Episode {episode + 1}/{num_episodes} | Average Reward (last {eval_frequency}): {avg_reward:.3f}")

    print("\nTraining completed!")

    # Optionally plot final reward curve
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('SARSA Training Reward Curve')
    plt.grid(True)
    plt.show()

    # Optionally, save final Q-table
    np.save('trained_q_table.npy', agent.q_table)
    print("Final Q-table saved to trained_q_table.npy")
    print('Reached '+str(sum(goals))+' time(s) the goal state')

    # You could also plot some paths if you want (optional)
    # Example: plot last path
    final_path = paths[-1][0]
    shortest_path_to_goal = paths[0][0]
    max_steps = 0
    min_steps_goal = float('inf')
    largst_path = paths[0][0]
    for path in paths:
        count = len(path)
        if path[1] == 1:
            if count <= min_steps_goal:
                shortest_path_to_goal = path[0]
                min_steps_goal = count
        if path[1] == 1 and count >= max_steps:
            largst_path = path[0]
            max_steps = count

    xs, ys = zip(*final_path)
    xls, yls = zip(*largst_path)
    short_x, short_y = zip(*shortest_path_to_goal)
    plt.figure()
    plt.imshow(agent.map.T, origin='lower', cmap='Greys')
    plt.plot(xls, yls, marker='o', markersize=3, linestyle='-', linewidth=0.75, color='yellow',label='longest path')
    plt.plot(short_x, short_y, marker='o', markersize=3, linestyle='-', linewidth=1.25, color='m', label='shortest path to goal')
    plt.plot(xs, ys, marker='o', markersize=3, linestyle='-', linewidth=1.25, color='green',label='(last) ideal path')
    plt.scatter(agent.x_start, agent.y_start, color='blue', s=80, label='Start')
    gx, gy = agent.discretize_position(agent.goal)
    plt.scatter(gx, gy, color='red', s=80, label='Goal')
    if agent.original_curricular_learning != None and len(agent.original_curricular_learning) != 0:
        xs, ys = zip(*(agent.discretize_position(sg) for sg in agent.original_curricular_learning))
        # plot ze in één keer, met één label
        plt.scatter(xs, ys,
                    color='yellow',
                    s=25,
                    label='Sub-goals')
    plt.title('Path followed in last episode')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #main()

    #Uncomment for evaluation
    #checkpoint_path = "ship_ppo_checkpoint_1900.pt"  # Adjust to your checkpoint file
    #evaluate_trained_agent(checkpoint_path)
    main()