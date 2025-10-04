"""
Training Script for Microgrid EMS RL Agent
Supports PPO and SAC algorithms with curriculum learning
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from collections import deque
import json
from datetime import datetime
from typing import Dict, List, Tuple

from microgrid_env import MicrogridEMSEnv
from env_config import TRAINING, REWARD, STEPS_PER_EPISODE


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.observations[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_observations[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )


class Actor(nn.Module):
    """Actor network for continuous action space"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs):
        features = self.network(obs)
        mean = torch.tanh(self.mean_layer(features))
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, obs_dim: int, action_dim: int = None, hidden_dims=[256, 256]):
        super().__init__()
        
        input_dim = obs_dim if action_dim is None else obs_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs, action=None):
        if action is not None:
            x = torch.cat([obs, action], dim=-1)
        else:
            x = obs
        return self.network(x)


class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, obs_dim: int, action_dim: int, config=TRAINING):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.learning_rate_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.learning_rate_critic)
        
        # Buffers
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mean, log_std = self.actor(obs_tensor)
            
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                
                # Store for training
                value = self.critic(obs_tensor)
                self.observations.append(obs)
                self.actions.append(action.squeeze(0).numpy())
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())
            
            action = torch.clamp(action, -1.0, 1.0)
            return action.squeeze(0).numpy()
    
    def store_transition(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        # Convert to tensors
        observations = torch.FloatTensor(np.array(self.observations))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Random mini-batches
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            for start in range(0, len(observations), self.config.ppo_batch_size):
                end = start + self.config.ppo_batch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor loss
                mean, log_std = self.actor(batch_obs)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 
                                   1 + self.config.ppo_clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                values_pred = self.critic(batch_obs).squeeze()
                critic_loss = nn.MSELoss()(values_pred, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # Clear buffers
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return actor_loss.item(), critic_loss.item()
    
    def _compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


def train(
    env: MicrogridEMSEnv,
    agent: PPOAgent,
    num_episodes: int,
    log_dir: str = "logs",
    model_dir: str = "models",
    eval_interval: int = 50
):
    """
    Main training loop
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Metrics
    episode_returns = []
    episode_costs = []
    episode_emissions = []
    episode_safety_violations = []
    
    best_return = -np.inf
    
    print("="*60)
    print("Starting Training")
    print("="*60)
    print(f"Algorithm: PPO")
    print(f"Episodes: {num_episodes}")
    print(f"Observation dim: {agent.obs_dim}")
    print(f"Action dim: {agent.action_dim}")
    print("="*60)
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(reward, done)
            
            episode_reward += reward
            obs = next_obs
            step += 1
        
        # Update agent after episode
        if len(agent.observations) > 0:
            actor_loss, critic_loss = agent.update()
        
        # Log metrics
        episode_returns.append(episode_reward)
        episode_costs.append(info['episode_metrics']['total_cost'])
        episode_emissions.append(info['episode_metrics']['total_emissions'])
        episode_safety_violations.append(info['episode_metrics']['safety_overrides'])
        
        # Print progress
        if (episode + 1) % TRAINING.log_interval == 0:
            avg_return = np.mean(episode_returns[-TRAINING.log_interval:])
            avg_cost = np.mean(episode_costs[-TRAINING.log_interval:])
            avg_emissions = np.mean(episode_emissions[-TRAINING.log_interval:])
            avg_violations = np.mean(episode_safety_violations[-TRAINING.log_interval:])
            
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Return: {avg_return:.2f}")
            print(f"  Avg Cost: ${avg_cost:.2f}")
            print(f"  Avg Emissions: {avg_emissions:.2f} kg CO2")
            print(f"  Avg Safety Violations: {avg_violations:.1f}")
            print(f"  Unmet Demand Events: {info['episode_metrics']['unmet_demand_events']}")
        
        # Save best model
        if episode_reward > best_return:
            best_return = episode_reward
            agent.save(os.path.join(model_dir, "best_model.pt"))
            print(f"  → New best model saved! Return: {best_return:.2f}")
        
        # Save checkpoint
        if (episode + 1) % TRAINING.save_interval == 0:
            agent.save(os.path.join(model_dir, f"checkpoint_ep{episode+1}.pt"))
            
            # Save training curves
            pd.DataFrame({
                'episode': range(len(episode_returns)),
                'return': episode_returns,
                'cost': episode_costs,
                'emissions': episode_emissions,
                'safety_violations': episode_safety_violations
            }).to_csv(os.path.join(log_dir, "training_metrics.csv"), index=False)
    
    print("="*60)
    print("Training Complete!")
    print(f"Best Return: {best_return:.2f}")
    print("="*60)
    
    return episode_returns, episode_costs


def main():
    """Main training entry point"""
    # Load data
    print("Loading data profiles...")
    pv_profile = pd.read_csv('data/pv_profile_processed.csv')
    wt_profile = pd.read_csv('data/wt_profile_processed.csv')
    load_profile = pd.read_csv('data/load_profile_processed.csv')
    price_profile = pd.read_csv('data/price_profile_processed.csv')
    
    print(f"Data loaded: {len(pv_profile)} timesteps")
    
    # Create environment
    print("\nCreating environment...")
    env = MicrogridEMSEnv(
        pv_profile=pv_profile,
        wt_profile=wt_profile,
        load_profile=load_profile,
        price_profile=price_profile,
        enable_evs=True,
        enable_degradation=True,
        enable_emissions=True,
        forecast_noise_std=0.1,
        random_seed=42
    )
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Create agent
    print("\nCreating PPO agent...")
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=TRAINING
    )
    
    # Train
    print("\nStarting training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ppo_{timestamp}"
    model_dir = f"models/ppo_{timestamp}"
    
    episode_returns, episode_costs = train(
        env=env,
        agent=agent,
        num_episodes=500,  # Start with 500 episodes
        log_dir=log_dir,
        model_dir=model_dir
    )
    
    print(f"\nLogs saved to: {log_dir}")
    print(f"Models saved to: {model_dir}")


if __name__ == "__main__":
    main()
