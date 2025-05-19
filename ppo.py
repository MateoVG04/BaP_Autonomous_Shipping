import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from continuous_env import Continuous2DEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from shapely.geometry import Polygon, MultiPolygon

import csv
import os
from datetime import datetime

def save_metrics_to_csv(filename, metrics_dict):
    """Save metrics to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write metrics to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['iteration', 'loss', 'average_reward'])
        # Write data
        for i in range(len(metrics_dict['loss'])):
            writer.writerow([i, 
                           metrics_dict['loss'][i], 
                           metrics_dict['reward'][i]])

class ShipActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, prev_nn):
        super().__init__()
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_mean = nn.Linear(128, output_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(output_dim))
        
        # Critic head
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        shared_features = self.shared_network(state)
        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))  # Clamp to avoid numerical issues
        state_value = self.critic(shared_features)
        return action_mean, action_std, state_value
    
    def sample_action(self, state):
        action_mean, action_std, _ = self(state)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample and bound action
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        
        # Compute log probability with adjustment for tanh squashing
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob

class ShipPPOAgent:
    def __init__(self, env, learning_rate=3e-4, clip_range=0.2,
                 value_loss_coef=0.5, max_grad_norm=0.5, prev_nn=False):
        self.device = torch.device("cpu")
        
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.shape[0]
        
        self.network = ShipActorCriticNetwork(self.input_dim, self.output_dim, prev_nn).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.network.sample_action(state)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def update(self, all_episodes_data):
        """
        Update policy using data from multiple episodes
        
        Args:
            all_episodes_data: list of dictionaries, each containing:
                - states: numpy array of states
                - actions: numpy array of actions
                - advantages: numpy array of advantages
                - old_log_probs: numpy array of old log probabilities
                - returns: numpy array of returns
        """
        # Concatenate all episodes data
        all_states = np.concatenate([ep['states'] for ep in all_episodes_data])
        all_actions = np.concatenate([ep['actions'] for ep in all_episodes_data])
        all_advantages = np.concatenate([ep['advantages'] for ep in all_episodes_data])
        all_old_log_probs = np.concatenate([ep['log_probs'] for ep in all_episodes_data])
        all_returns = np.concatenate([ep['returns'] for ep in all_episodes_data])
        
        # Convert to tensors
        states = torch.FloatTensor(all_states).to(self.device)
        actions = torch.FloatTensor(all_actions).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(10):  # Multiple epochs of training
            # Get current policy distribution
            action_mean, action_std, state_values = self.network(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # Get log probabilities with tanh adjustment
            raw_actions = torch.atanh(actions.clamp(-0.99, 0.99))
            log_probs = dist.log_prob(raw_actions)
            log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
            log_probs = log_probs.sum(dim=-1)
            
            # Calculate ratios and surrogate objectives
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - state_values.squeeze()).pow(2).mean()
            
            # Add entropy bonus to encourage exploration
            entropy = dist.entropy().mean()
            loss = policy_loss + self.value_loss_coef * value_loss - 0.01 * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / 10

def run_episode(env, agent):
    """Run a single episode and return the collected data"""
    state, _ = env.reset()
    done = False
    
    # Lists to store episode data
    states, actions, rewards, values, log_probs = [], [], [], [], []
    episode_reward = 0
    
    while not done:
        # Store state
        states.append(state)
        
        # Select action
        action, log_prob = agent.select_action(state)
        
        # Get value prediction
        with torch.no_grad():
            _, _, value = agent.network(torch.FloatTensor(state).unsqueeze(0))
        
        # Store info
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value.item())
        
        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        episode_reward += reward
        
        done = terminated or truncated
        state = next_state
    
    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    values = np.array(values)
    log_probs = np.array(log_probs)
    
    # Compute advantages and returns
    with torch.no_grad():
        _, _, next_value = agent.network(torch.FloatTensor(next_state).unsqueeze(0))
    
    advantages = compute_gae(rewards, values, np.array([False] * len(rewards) + [done]), 
                           next_value.item())
    returns = advantages + values
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'advantages': advantages,
        'log_probs': log_probs,
        'returns': returns,
        'episode_reward': episode_reward
    }

def main():
    # Training parameters
    num_iterations = 500  # Number of policy updates
    num_episodes_per_update = 5  # Number of episodes to collect before updating
    eval_frequency = 100
    
    # Environment setup
    ship_pos = [2060.0, -50.0]
    target_pos = [11530.0, 13000.0]

    # corner
    #target_pos = [2435.0, 1556]


    env = Continuous2DEnv(
        render_mode='human',
        max_steps=1200,  # This is now just a safety limit
        ship_pos=ship_pos,
        target_pos=target_pos,
    )
    
    # Initialize agent
    agent = ShipPPOAgent(env, learning_rate=3e-4)
    
    # Training metrics
    metrics = {
        'loss': [],
        'reward': []
    }
    
    print("Starting training...")
    
    for iteration in range(num_iterations):
        episode_data = []
        total_reward = 0
        
        # Collect episodes with different initial conditions
        for episode in range(num_episodes_per_update):
            # Optionally modify initial conditions here
            env.ship_pos = [2060.0 + np.random.uniform(-1, 1),
                          -50.0 + np.random.uniform(-1, 1)]
            
            # Run episode
            episode_result = run_episode(env, agent)
            episode_data.append(episode_result)
            total_reward += episode_result['episode_reward']
        
        # Update policy using all collected episodes
        loss = agent.update(episode_data)
        
        # Store metrics
        avg_reward = total_reward / num_episodes_per_update
        metrics['loss'].append(loss)
        metrics['reward'].append(avg_reward)
        
        if iteration % eval_frequency == 0:
            print(f"Iteration {iteration}, Average Reward: {avg_reward:.3f}, Loss: {loss:.3f}")
            
            # Save checkpoint
            if iteration > 0 and iteration % 100 == 0:
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, f'results/ship_ppo_checkpoint_{iteration}.pt')
    
    print("\nTraining completed!")
    env.close()

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    rewards: array of rewards for the batch
    values: array of value estimates
    dones: array of done flags
    next_value: value estimate for the state after the batch
    gamma: discount factor
    lambda_: GAE parameter
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    # Reverse iteration for GAE calculation
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae
    
    return advantages

def evaluate_trained_agent(checkpoint_path, num_episodes=3):
    """
    Evaluate a trained agent and plot both the paths taken and cross-track errors.
    
    Args:
        checkpoint_path (str): Path to the .pt checkpoint file
        num_episodes (int): Number of episodes to run evaluation
    """
    # Set up environment without rendering
    ship_pos = [2060.0, -50.0]
    target_pos = [-1530.0, 12010.0]

    # corner
    # target_pos = [2435.0, 1556]

    """
                Custom start and goal states
                """
    # Original goal and start state
    # ship_pos = [2060.0, -50.0]
    # target_pos = [-1530.0, 12010.0]
    # curricular_learning = [(2520,350),(2850,840),(3070,1420),(3600,1510),(3570,3370),(3380,4790),(2990,6040),(1460,6930),(770,8500),(320,10090),(-540,11550)]

    # Very close goal and start state
    # ship_pos = [2200.0, 30.0]
    # target_pos = [2330.0, 60.0]

    # Goal state is around the corner
    # ship_pos = [2200.0, 30.0]
    # target_pos = [2435.0, 1556]
    # curricular_learning = [(2520, 350), (2850, 840), (3070, 1420)]

    env = Continuous2DEnv(
        render_mode=None,  # Disable rendering
        max_steps=1200,
        ship_pos=ship_pos,
        target_pos=target_pos,
    )
    
    # Initialize agent
    agent = ShipPPOAgent(env)
    
    # Load checkpoint
    # checkpoint = torch.load(checkpoint_path) # This line of code doesn't work, because I don't have a GPU
    # and the checkpoint_path was made on a device with a GPU

    # The map_location=torch.device('cpu') tells PyTorch to move all model weights to the CPU even if they
    # were originally stored on a GPU.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    agent.network.load_state_dict(checkpoint['model_state_dict'])
    agent.network.eval()
    
    print(f"\nRunning {num_episodes} episodes...")
    
    # Store results for each episode
    all_paths = []
    total_rewards = []
    all_cross_errors = []  # Store cross-track errors for each episode
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        # Store positions and errors for this episode
        path_positions = []
        cross_errors = []
        path_positions.append(env.ship_pos.copy())  # Store initial position
        cross_errors.append(env.cross_error)  # Store initial cross-track error
        
        while not done:
            # Select action
            action, _ = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store current position and cross-track error
            path_positions.append(env.ship_pos.copy())
            cross_errors.append(env.cross_error)
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                break
        
        all_paths.append(np.array(path_positions))
        all_cross_errors.append(np.array(cross_errors))
        total_rewards.append(episode_reward)
    
    # Print summary statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nEvaluation completed!")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Best episode reward: {max(total_rewards):.2f}")
    print(f"Worst episode reward: {min(total_rewards):.2f}")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot paths in the first subplot
    # Create hashed pattern for the area between obstacles and Western Scheldt
    def create_hashed_area(obstacles, overall):
        obstacle_poly = Polygon(obstacles)
        overall_poly = Polygon(overall)
        difference = overall_poly.difference(obstacle_poly)
        
        if isinstance(difference, MultiPolygon):
            coords = []
            for poly in difference.geoms:
                coords.append(np.array(poly.exterior.coords))
        else:
            coords = [np.array(difference.exterior.coords)]
        
        return coords
    
    # Add hatched pattern to first subplot
    difference_coords = create_hashed_area(env.obstacles, env.overall)
    for coords in difference_coords:
        ax1.add_patch(patches.Polygon(
            coords,
            facecolor='none',
            edgecolor='gray',
            hatch='///',
            alpha=0.3,
            label='Low water level area' if coords is difference_coords[0] else ""
        ))
    
    # Plot checkpoints
    checkpoint_positions = [checkpoint['pos'] for checkpoint in env.checkpoints]
    checkpoint_positions = np.array(checkpoint_positions)
    for checkpoint in env.checkpoints:
        circle = plt.Circle((checkpoint['pos'][0], checkpoint['pos'][1]), 
                          radius=10,
                          color='gray',
                          alpha=0.5,
                          fill=True)
        ax1.add_patch(circle)
    
    # Draw the path between checkpoints
    ax1.plot(checkpoint_positions[:, 0], checkpoint_positions[:, 1], 
            'g--', alpha=0.5, label='Ideal Path')
    
    # Plot obstacles
    polygon_patch = patches.Polygon(env.obstacles, closed=True, 
                                  edgecolor='r', facecolor='none', 
                                  lw=2, label='Waterway')
    ax1.add_patch(polygon_patch)
    
    western_scheldt = patches.Polygon(env.overall, closed=True, 
                                    edgecolor='brown', facecolor='none', 
                                    lw=2, label='Western Scheldt')
    ax1.add_patch(western_scheldt)

    # Plot paths for each episode with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_episodes))
    for i, path in enumerate(all_paths):
        ax1.plot(path[:, 0], path[:, 1], '-', 
                color=colors[i], alpha=0.7, 
                label=f'Episode {i+1} (reward: {total_rewards[i]:.1f})')
    
    # Plot start and target positions
    ax1.scatter([ship_pos[0]], [ship_pos[1]], c='blue', s=100, label='Start')
    ax1.scatter([env.target_pos[0]], [env.target_pos[1]], c='red', s=100, label='Target')
    
    ax1.set_title('Realistic Ship Path In Western Scheldt')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.axis('equal')
    
    errors = all_cross_errors[0]  # Take only the first episode's errors
    timesteps = np.arange(len(errors))
    ax2.plot(timesteps, errors, '-', 
            color=colors[0], alpha=0.7, 
            label='Cross-track Error')    

    ax2.set_title('Cross-Track Error Over Time')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Cross-Track Error')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Adjust layout and save
    plt.show()
    plt.tight_layout()
    plt.savefig('ship_paths_and_errors.png', bbox_inches='tight', dpi=300)
    
    print("\nPath and error visualization saved as 'ship_paths_and_errors.png'")
    
    env.close()
    return all_paths, total_rewards, all_cross_errors


if __name__ == "__main__":
    #main()

    #Uncomment for evaluation
    checkpoint_path = "ship_ppo_checkpoint_1900.pt"  # Adjust to your checkpoint file
    evaluate_trained_agent(checkpoint_path)
