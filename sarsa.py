import random
import torch
import math
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import os

from sympy.physics.quantum.operatorset import state_mapping

from continuous_env import Continuous2DEnv


class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.2, epochs=50, grid_width=100, grid_height=100, goal=(-1530.0,12010.0), start=(2060.0,-50.0), grid_resolution=50.0, save_map=False):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.grid_width, self.grid_height = grid_width, grid_height
        self.goal = goal
        self.start = start
        self.current_position = start
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.action_size = len(self.actions)
        self.total_distance = abs(self.goal[0]-self.start[0])+abs(self.goal[1]-self.start[1])
        self.q_table = np.zeros((self.grid_width, self.grid_height, self.action_size))
        self.grid_resolution = grid_resolution
        if save_map:
            self.map = self.discretize_map()
        else:
            self.map = self.load_occupancy_grid()
            self.show_map()

    def discretize_map(self):
        csv_input_dir = os.path.dirname(os.path.abspath(__file__))
        contour_points = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_250cm_no_scale.csv'), delimiter=',',
                              skiprows=1)

        # 2. Create a Polygon object from contour
        map_polygon = Polygon(contour_points)

        # 3. Define grid
        self.x_min, self.y_min = contour_points.min(axis=0)
        self.x_max, self.y_max = contour_points.max(axis=0)

        x_grid = np.arange(self.x_min, self.x_max, self.grid_resolution)
        y_grid = np.arange(self.y_min, self.y_max, self.grid_resolution)

        grid_width = len(x_grid)
        grid_height = len(y_grid)

        # 4. Initialize occupancy grid
        occupancy_grid = np.zeros((grid_width, grid_height), dtype=np.uint8)

        # 5. Check each grid cell center
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                point = Point(x + self.grid_resolution / 2, y + self.grid_resolution / 2)
                if map_polygon.contains(point):
                    occupancy_grid[i, j] = 1  # 1 = navigable
                else:
                    occupancy_grid[i, j] = 0  # 0 = obstacle or outside

        self.save_occupancy_grid(occupancy_grid, self.grid_resolution)
        self.show_map()
        return occupancy_grid

    def save_occupancy_grid(self, occupancy_grid, resolution):
        csv_input_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(csv_input_dir, 'occupancy_grid_'+str(resolution)+'.npy')
        np.save(save_path, occupancy_grid)
        print(f"Occupancy grid saved to {save_path}")

    def load_occupancy_grid(self):
        csv_input_dir = os.path.dirname(os.path.abspath(__file__))

        # Find all occupancy grid files
        files = [f for f in os.listdir(csv_input_dir) if f.startswith('occupancy_grid_') and f.endswith('.npy')]
        if not files:
            self.discretize_map()
            files = [f for f in os.listdir(csv_input_dir) if f.startswith('occupancy_grid_') and f.endswith('.npy')]
            if not files:
                raise FileNotFoundError("Occupancy grid could not be created.")

        # Extract resolution values
        resolutions = []
        for file in files:
            try:
                res = float(file.split('_')[-1].replace('.npy', ''))
                resolutions.append((res, file))
            except ValueError:
                continue  # skip invalid files

        if not resolutions:
            raise FileNotFoundError("No valid occupancy grid files with resolutions found.")

        # Pick the file with the **smallest resolution** (highest accuracy)
        self.grid_resolution, best_file = min(resolutions)
        contour_points = np.loadtxt(os.path.join(csv_input_dir, 'env_Sche_250cm_no_scale.csv'), delimiter=',',
                                    skiprows=1)

        # 2. Create a Polygon object from contour
        map_polygon = Polygon(contour_points)

        # 3. Define grid
        self.x_min, self.y_min = contour_points.min(axis=0)
        self.x_max, self.y_max = contour_points.max(axis=0)

        load_path = os.path.join(csv_input_dir, best_file)
        occupancy_grid = np.load(load_path)
        print(f"Occupancy grid loaded from {load_path} with resolution {self.grid_resolution}")
        return occupancy_grid

    def show_map(self):
        plt.imshow(self.map.T, origin='lower', cmap='Greys')
        plt.title('Discretized Map (1 = Water, 0 = Land), Resolution = ' + str(self.grid_resolution))
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.colorbar()

        self.x_start = (self.start[0] - self.x_min) / self.grid_resolution
        self.y_start = (self.start[1] - self.y_min) / self.grid_resolution
        plt.scatter(self.x_start, self.y_start, color='blue', s=25, label='Start')

        self.x_goal = (self.goal[0] - self.x_min) / self.grid_resolution
        self.y_goal = (self.goal[1] - self.y_min) / self.grid_resolution
        plt.scatter(self.x_goal, self.y_goal, color='red', s=25, label='Goal')
        plt.legend()

        plt.show()

    def reward_function(self, state):
        print("reward_function():")
        scaling_factor_goal = 2
        time_constraint = 0.1
        reach_goal = scaling_factor_goal*(abs(self.goal[0]-self.start[0])+abs(self.goal[1]-self.start[0]))
        print("     reach_goal = "+str(reach_goal))

        if state == self.goal:
            reward = reach_goal
        elif state == self.wall or state == self.obstacle:
            reward = self.reward_collision(reach_goal, time_constraint)
            print("     reward_collision = " + str(reward))
        elif state == self.shallow_water:
            reward = self.reward_collision(reach_goal, time_constraint)/2
            print("     reward_shallow_water = " + str(reward))

        reward += self.reward_direction()
        print("     final_reward = " + str(self.normalize_reward(reward - time_constraint)))
        return self.normalize_reward(reward - time_constraint) # this is the time constraint punishment. So that the agent finds the fastest path

    def reward_direction(self):
        """
        A reward/punishment based on the distance of the agent to the goal
        :return: the reward/punishment based on the distance
        """
        distance = abs(self.goal[0] - self.current_position[0]) + abs(self.goal[1] - self.current_position[0])
        reward = 1/math.exp(distance)
        if distance > self.total_distance:
            reward -= (self.total_distance-distance)**2
        elif distance <= 0.05:
            reward += 0.05
        print("     reward_direction = " + str(reward))
        return reward

    def reward_collision(self, reach_goal, time_constraint):
        return -(reach_goal + abs(time_constraint*self.total_distance))

    def normalize_reward(self, reward):
        """
        dynamic reward normalization because we don't know the exact reward bounds.
        :param reward:
        :return: normalized reward
        """
        self.min_reward_seen = float("inf")
        self.max_reward_seen = float("inf")

        self.min_reward_seen = min(self.min_reward_seen, reward)
        self.max_reward_seen = max(self.max_reward_seen, reward)

        if self.max_reward_seen == self.min_reward_seen:
            return 0
        else:
            return 2*(reward-self.min_reward_seen)/(self.max_reward_seen-self.min_reward_seen) - 1

    def epsilon_greedy(self, state):
        """
        Balance between exploration/exploitation
        :param state:
        :return:
        """
        x, y = state
        if random.random() < self.epsilon:
            # takes a random action
            return random.randint(0, self.action_size - 1)
        # argmax finds the action with the highest q-value
        return np.argmax(self.q_table[x, y])

    def update(self, state, action):
        # Get the next action and corresponding index from epsilon-greedy policy
        next_action, next_action_index = self.select_action(state)

        # Discretize state into grid indices for Q-table lookup
        x, y = int(self.current_position[0]), int(self.current_position[1])
        nx, ny = int(state[0]), int(state[1])  # next state's discrete position

        # Clip to make sure we stay within bounds of the Q-table
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height and
                0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
            return  # Skip update if state is outside grid bounds

        # Reward: use raw reward from environment or use custom shaping
        reward = self.reward_function(state)

        # Perform SARSA update
        self.q_table[x, y, action] += self.alpha * (
                reward + self.gamma * self.q_table[nx, ny, next_action_index]
                - self.q_table[x, y, action]
        )

        # Track current position (needed for reward shaping)
        self.current_position = state


    def select_action(self, state):
        action = self.epsilon_greedy(state)
        dx, dy = self.actions[action]
        return np.array([dx, dy], dtype=np.float32), action

def run_episode(env, agent, start_state):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    path_points = []
    while not done:
        dx, dy, position, discrete_action = agent.select_action(state)
        new_point = [position[0] + dx, position[1] + dy]
        next_state, reward, terminated, truncated, _ = env.step(discrete_action)
        episode_reward += reward
        agent.update(state, discrete_action)
        state = next_state
        done = terminated or truncated
        state = next_state
    return {'episode_reward': episode_reward}

def main():
    num_iterations = 500  # Number of policy updates
    num_episodes_per_update = 5  # Number of episodes to collect before updating
    eval_frequency = 100

    # Environment setup
    ship_pos = [2060.0, -50.0]
    target_pos = [11530.0, 13000.0]
    env = Continuous2DEnv(
        render_mode='human',
        max_steps=1200,  # This is now just a safety limit
        ship_pos=ship_pos,
        target_pos=target_pos,
    )
    agent = SARSA(env, alpha=3e-4, grid_resolution=20, save_map=False)

    # Training metrics
    # metrics = {
    #     'loss': [],
    #     'reward': []
    # }
    #
    # print("Starting training...")
    #
    # for iteration in range(num_iterations):
    #     episode_data = []
    #     total_reward = 0
    #
    #     # Collect episodes with different initial conditions
    #     for episode in range(num_episodes_per_update):
    #         # Optionally modify initial conditions here
    #         env.ship_pos = [2060.0 + np.random.uniform(-1, 1),
    #                         -50.0 + np.random.uniform(-1, 1)]
    #
    #         # Run episode
    #         episode_result = run_episode(env, agent)
    #         episode_data.append(episode_result)
    #         total_reward += episode_result['episode_reward']
    #
    #     # Update policy using all collected episodes
    #     loss = agent.update(episode_data)
    #
    #     # Store metrics
    #     avg_reward = total_reward / num_episodes_per_update
    #     metrics['loss'].append(loss)
    #     metrics['reward'].append(avg_reward)
    #
    #     if iteration % eval_frequency == 0:
    #         print(f"Iteration {iteration}, Average Reward: {avg_reward:.3f}, Loss: {loss:.3f}")
    #
    #         # Save checkpoint
    #         if iteration > 0 and iteration % 100 == 0:
    #             torch.save({
    #                 'iteration': iteration,
    #                 'model_state_dict': agent.network.state_dict(),
    #                 'optimizer_state_dict': agent.optimizer.state_dict(),
    #             }, f'results/ship_ppo_checkpoint_{iteration}.pt')
    #
    # print("\nTraining completed!")
    # env.close()

if __name__ == "__main__":
    #main()

    #Uncomment for evaluation
    #checkpoint_path = "ship_ppo_checkpoint_1900.pt"  # Adjust to your checkpoint file
    #evaluate_trained_agent(checkpoint_path)
    main()