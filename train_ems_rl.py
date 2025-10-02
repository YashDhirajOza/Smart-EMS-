"""
Smart Energy Management System (EMS) RL Training Script
Uses processed real-world power grid data to train an RL agent for optimal energy dispatch decisions.
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch
import warnings
warnings.filterwarnings('ignore')

class SmartEMSEnv(gym.Env):
    """
    Smart Energy Management System Environment
    
    State space: [SoC, demand, solar_output, wind_output, grid_price, hour_sin, hour_cos]
    Action space: [battery_action, grid_action] (continuous)
        - battery_action: -1 (discharge) to +1 (charge)
        - grid_action: 0 (no grid) to 1 (full grid usage)
    """
    
    def __init__(self, data_df, episode_length=24, max_episodes=None):
        super(SmartEMSEnv, self).__init__()
        
        self.data_df = data_df.copy()
        self.episode_length = episode_length  # 24 hours
        self.max_episodes = max_episodes
        self.current_episode = 0
        
        # EMS parameters
        self.battery_capacity = 100.0  # kWh
        self.battery_efficiency = 0.95
        self.max_charge_rate = 50.0  # kW
        self.max_discharge_rate = 50.0  # kW
        self.grid_connection_cost = 0.02  # $/kWh connection fee
        
        # State space: [SoC, demand, solar_output, wind_output, grid_price, hour_sin, hour_cos]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]),
            high=np.array([1.0, 3000.0, 500.0, 500.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: [battery_action, grid_action]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode tracking
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to start a new episode"""
        super().reset(seed=seed)
        
        if self.max_episodes and self.current_episode >= self.max_episodes:
            self.current_episode = 0
        
        # Select random starting point ensuring we have enough data for full episode
        max_start_idx = len(self.data_df) - self.episode_length
        if max_start_idx <= 0:
            raise ValueError(f"Dataset too small. Need at least {self.episode_length} records.")
        
        self.start_idx = np.random.randint(0, max_start_idx)
        self.current_step = 0
        self.current_episode += 1
        
        # Initialize battery SoC randomly between 20-80%
        self.battery_soc = np.random.uniform(0.2, 0.8)
        
        # Episode tracking for rewards
        self.total_cost = 0.0
        self.total_renewable_used = 0.0
        self.total_demand_met = 0.0
        self.grid_violations = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation"""
        if self.start_idx + self.current_step >= len(self.data_df):
            # Return last available observation if we exceed data
            row = self.data_df.iloc[-1]
            hour = 0
        else:
            row = self.data_df.iloc[self.start_idx + self.current_step]
            hour = row.name.hour if hasattr(row.name, 'hour') else 0
        
        # Time encoding (cyclical)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        obs = np.array([
            self.battery_soc,
            row['demand'],
            row['solar_output'],
            row['wind_output'], 
            row['grid_price'],
            hour_sin,
            hour_cos
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.start_idx + self.current_step >= len(self.data_df):
            # Episode exceeded data, return terminal state
            return self._get_observation(), -100.0, True, True, {"error": "exceeded_data"}
        
        # Get current data
        row = self.data_df.iloc[self.start_idx + self.current_step]
        demand = row['demand']
        solar_output = row['solar_output']
        wind_output = row['wind_output']
        grid_price = row['grid_price']
        
        # Parse actions
        battery_action = np.clip(action[0], -1.0, 1.0)  # -1: discharge, +1: charge
        grid_action = np.clip(action[1], 0.0, 1.0)      # 0: no grid, 1: full grid
        
        # Calculate renewable generation
        renewable_generation = solar_output + wind_output
        
        # Battery dynamics
        if battery_action > 0:  # Charging
            charge_power = battery_action * self.max_charge_rate
            # Can only charge with excess renewable or grid
            available_for_charging = max(0, renewable_generation - demand)
            actual_charge = min(charge_power, available_for_charging)
            actual_charge = min(actual_charge, (1.0 - self.battery_soc) * self.battery_capacity)
            
            self.battery_soc += (actual_charge * self.battery_efficiency) / self.battery_capacity
            battery_power = -actual_charge  # Negative because consuming power
        else:  # Discharging
            discharge_power = abs(battery_action) * self.max_discharge_rate
            max_discharge = self.battery_soc * self.battery_capacity
            actual_discharge = min(discharge_power, max_discharge)
            
            self.battery_soc -= actual_discharge / self.battery_capacity
            battery_power = actual_discharge / self.battery_efficiency  # Positive because supplying power
        
        # Ensure SoC bounds
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
        
        # Energy balance
        renewable_used = min(renewable_generation, demand)
        battery_used = min(battery_power, max(0, demand - renewable_used))
        remaining_demand = max(0, demand - renewable_used - battery_used)
        
        # Grid usage
        grid_demand = remaining_demand
        grid_power = grid_action * grid_demand
        unmet_demand = max(0, remaining_demand - grid_power)
        
        # Calculate costs and rewards
        grid_cost = grid_power * (grid_price + self.grid_connection_cost)
        
        # Reward components
        renewable_reward = (renewable_used / max(demand, 1)) * 10  # Renewable usage bonus
        efficiency_reward = ((demand - unmet_demand) / max(demand, 1)) * 5  # Demand satisfaction
        cost_penalty = -grid_cost / 10  # Grid cost penalty
        battery_health_penalty = -abs(battery_action) * 0.1  # Battery wear penalty
        unmet_penalty = -unmet_demand * 5  # Penalty for unmet demand
        
        # SoC management reward (encourage keeping SoC in healthy range)
        if 0.2 <= self.battery_soc <= 0.8:
            soc_reward = 1.0
        else:
            soc_reward = -abs(self.battery_soc - 0.5) * 2
        
        # Combined reward
        reward = (renewable_reward + efficiency_reward + cost_penalty + 
                 battery_health_penalty + unmet_penalty + soc_reward)
        
        # Update tracking
        self.total_cost += grid_cost
        self.total_renewable_used += renewable_used
        self.total_demand_met += (demand - unmet_demand)
        if unmet_demand > 0.1:
            self.grid_violations += 1
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        truncated = False
        
        # Info dictionary
        info = {
            'demand': demand,
            'renewable_generation': renewable_generation,
            'battery_soc': self.battery_soc,
            'grid_cost': grid_cost,
            'unmet_demand': unmet_demand,
            'renewable_used': renewable_used,
            'grid_power': grid_power,
            'battery_power': battery_power,
            'total_cost': self.total_cost,
            'renewable_efficiency': self.total_renewable_used / max(self.total_demand_met, 1),
            'grid_violations': self.grid_violations
        }
        
        return self._get_observation(), reward, done, truncated, info

def load_and_prepare_data(csv_file='processed_ems_data.csv'):
    """Load and prepare the processed EMS data"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file, index_col='DATE_TIME', parse_dates=True)
    
    # Normalize data for better training stability
    df['demand'] = df['demand'].clip(0, df['demand'].quantile(0.99))
    df['solar_output'] = df['solar_output'].clip(0, df['solar_output'].quantile(0.99))
    df['wind_output'] = df['wind_output'].clip(0, df['wind_output'].quantile(0.99))
    df['grid_price'] = df['grid_price'].clip(0.05, 0.5)  # Reasonable price bounds
    
    print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def create_training_environment(data_df, n_envs=4):
    """Create vectorized training environment"""
    def make_env():
        env = SmartEMSEnv(data_df, episode_length=24)
        env = Monitor(env)
        return env
    
    return make_vec_env(make_env, n_envs=n_envs)

def train_rl_agent(data_df, algorithm='PPO', total_timesteps=100000, model_name='ems_agent'):
    """Train RL agent using the specified algorithm"""
    print(f"\n=== Training {algorithm} Agent ===")
    
    # Create environments
    train_env = create_training_environment(data_df, n_envs=4)
    eval_env = SmartEMSEnv(data_df, episode_length=24)
    eval_env = Monitor(eval_env)
    
    # Model configuration
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU,
    )
    
    # Create model
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto'
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto'
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/',
        log_path=f'./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f'models/{model_name}_{algorithm.lower()}_final')
    print(f"Model saved as: models/{model_name}_{algorithm.lower()}_final")
    
    return model, eval_env

def evaluate_agent(model, eval_env, n_episodes=10):
    """Evaluate trained agent performance"""
    print(f"\n=== Evaluating Agent Performance ===")
    
    episode_rewards = []
    episode_costs = []
    episode_renewable_efficiency = []
    episode_violations = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = done or truncated
        
        episode_rewards.append(episode_reward)
        episode_costs.append(info['total_cost'])
        episode_renewable_efficiency.append(info['renewable_efficiency'])
        episode_violations.append(info['grid_violations'])
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Cost=${info['total_cost']:.2f}, "
              f"Renewable Eff={info['renewable_efficiency']:.3f}, "
              f"Violations={info['grid_violations']}")
    
    # Summary statistics
    print(f"\n=== Performance Summary ===")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Daily Cost: ${np.mean(episode_costs):.2f} ± ${np.std(episode_costs):.2f}")
    print(f"Average Renewable Efficiency: {np.mean(episode_renewable_efficiency):.3f} ± {np.std(episode_renewable_efficiency):.3f}")
    print(f"Average Violations per Day: {np.mean(episode_violations):.1f} ± {np.std(episode_violations):.1f}")
    
    return {
        'rewards': episode_rewards,
        'costs': episode_costs,
        'renewable_efficiency': episode_renewable_efficiency,
        'violations': episode_violations
    }

def plot_training_results(results, save_path='training_results.png'):
    """Plot training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(results['rewards'])
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Daily costs
    ax2.plot(results['costs'])
    ax2.set_title('Daily Grid Costs')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cost ($)')
    ax2.grid(True)
    
    # Renewable efficiency
    ax3.plot(results['renewable_efficiency'])
    ax3.set_title('Renewable Energy Efficiency')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Efficiency Ratio')
    ax3.grid(True)
    
    # Violations
    ax4.plot(results['violations'])
    ax4.set_title('Grid Violations per Day')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of Violations')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Results plot saved as: {save_path}")

def main():
    """Main training and evaluation pipeline"""
    # Create directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    data_df = load_and_prepare_data('processed_ems_data.csv')
    
    # Train PPO agent
    print("Training PPO agent...")
    ppo_model, eval_env = train_rl_agent(
        data_df, 
        algorithm='PPO', 
        total_timesteps=50000,  # Adjust based on your compute resources
        model_name='smart_ems'
    )
    
    # Evaluate PPO agent
    ppo_results = evaluate_agent(ppo_model, eval_env, n_episodes=20)
    
    # Plot results
    plot_training_results(ppo_results, 'ppo_training_results.png')
    
    # Optional: Train SAC agent for comparison
    train_sac = input("\nTrain SAC agent as well? (y/n): ").lower().strip() == 'y'
    if train_sac:
        print("\nTraining SAC agent...")
        sac_model, _ = train_rl_agent(
            data_df,
            algorithm='SAC',
            total_timesteps=50000,
            model_name='smart_ems'
        )
        
        sac_results = evaluate_agent(sac_model, eval_env, n_episodes=20)
        plot_training_results(sac_results, 'sac_training_results.png')
        
        # Compare algorithms
        print(f"\n=== Algorithm Comparison ===")
        print(f"PPO - Avg Reward: {np.mean(ppo_results['rewards']):.2f}, Avg Cost: ${np.mean(ppo_results['costs']):.2f}")
        print(f"SAC - Avg Reward: {np.mean(sac_results['rewards']):.2f}, Avg Cost: ${np.mean(sac_results['costs']):.2f}")
    
    print(f"\n=== Training Complete! ===")
    print(f"Models saved in './models/' directory")
    print(f"Logs saved in './logs/' directory")
    print(f"Use the trained model for your Smart EMS dashboard!")

if __name__ == "__main__":
    main()