import numpy as np
import random
import torch
import math

class SARSA:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.2, epochs=50, grid_width=100, grid_height=100, goal=(10,10), start=(5,5)):
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
