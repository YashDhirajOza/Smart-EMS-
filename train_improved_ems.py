"""
Improved Smart EMS RL Training Script with Better Reward Shaping
"""

from train_ems_rl import load_and_prepare_data, SmartEMSEnv, create_training_environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class ImprovedEMSEnv(SmartEMSEnv):
    """Improved EMS Environment with better reward shaping"""
    
    def step(self, action):
        """Execute one step with improved reward calculation"""
        if self.start_idx + self.current_step >= len(self.data_df):
            return self._get_observation(), -1000.0, True, True, {"error": "exceeded_data"}
        
        # Get current data
        row = self.data_df.iloc[self.start_idx + self.current_step]
        demand = row['demand']
        solar_output = row['solar_output']
        wind_output = row['wind_output']
        grid_price = row['grid_price']
        
        # Parse actions
        battery_action = np.clip(action[0], -1.0, 1.0)
        grid_action = np.clip(action[1], 0.0, 1.0)
        
        # Calculate renewable generation
        renewable_generation = solar_output + wind_output
        
        # Battery dynamics (improved)
        old_soc = self.battery_soc
        if battery_action > 0:  # Charging
            charge_power = battery_action * self.max_charge_rate
            available_for_charging = max(0, renewable_generation - demand)
            actual_charge = min(charge_power, available_for_charging)
            actual_charge = min(actual_charge, (1.0 - self.battery_soc) * self.battery_capacity)
            
            self.battery_soc += (actual_charge * self.battery_efficiency) / self.battery_capacity
            battery_power = -actual_charge
        else:  # Discharging
            discharge_power = abs(battery_action) * self.max_discharge_rate
            max_discharge = self.battery_soc * self.battery_capacity
            actual_discharge = min(discharge_power, max_discharge)
            
            self.battery_soc -= actual_discharge / self.battery_capacity
            battery_power = actual_discharge / self.battery_efficiency
        
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
        
        # Energy balance calculation
        renewable_used = min(renewable_generation, demand)
        battery_contribution = min(max(0, battery_power), max(0, demand - renewable_used))
        remaining_demand = max(0, demand - renewable_used - battery_contribution)
        
        # Grid usage
        grid_power = grid_action * max(remaining_demand, 0)
        unmet_demand = max(0, remaining_demand - grid_power)
        
        # Calculate costs
        grid_cost = grid_power * (grid_price + self.grid_connection_cost)
        
        # IMPROVED REWARD SHAPING
        # 1. Demand satisfaction (most important)
        demand_satisfaction = (demand - unmet_demand) / max(demand, 1)
        demand_reward = demand_satisfaction * 100  # High weight
        
        # 2. Renewable utilization
        renewable_ratio = renewable_used / max(renewable_generation, 1)
        renewable_reward = renewable_ratio * 20
        
        # 3. Cost efficiency (normalized by demand)
        normalized_cost = grid_cost / max(demand, 1)
        cost_reward = -normalized_cost * 50
        
        # 4. Battery management
        if 0.2 <= self.battery_soc <= 0.8:
            soc_reward = 5
        else:
            soc_reward = -abs(self.battery_soc - 0.5) * 10
        
        # 5. Operational efficiency
        battery_utilization = abs(battery_power) / max(self.max_discharge_rate, 1)
        if unmet_demand < 0.1:  # No unmet demand
            efficiency_bonus = 10
        else:
            efficiency_bonus = 0
        
        # 6. Penalty for unmet demand (severe)
        unmet_penalty = -unmet_demand * 100
        
        # Combined reward
        reward = (demand_reward + renewable_reward + cost_reward + 
                 soc_reward + efficiency_bonus + unmet_penalty)
        
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
            'grid_violations': self.grid_violations,
            'demand_satisfaction': demand_satisfaction,
            'reward_breakdown': {
                'demand_reward': demand_reward,
                'renewable_reward': renewable_reward,
                'cost_reward': cost_reward,
                'soc_reward': soc_reward,
                'efficiency_bonus': efficiency_bonus,
                'unmet_penalty': unmet_penalty
            }
        }
        
        return self._get_observation(), reward, done, truncated, info

def train_improved_agent(data_df, total_timesteps=50000):
    """Train improved RL agent"""
    print("=== Training Improved Smart EMS Agent ===")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create environments using improved environment
    def make_env():
        env = ImprovedEMSEnv(data_df, episode_length=24)
        env = Monitor(env)
        return env
    
    from stable_baselines3.common.env_util import make_vec_env
    train_env = make_vec_env(make_env, n_envs=4)
    eval_env = make_env()
    
    # Model with tuned hyperparameters
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=512,  # Smaller steps for better learning
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        policy_kwargs={
            'net_arch': [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            'activation_fn': torch.nn.ReLU,
        },
        verbose=1,
        device='auto'
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=2000,
        deterministic=True,
        n_eval_episodes=5
    )
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save('models/improved_ems_agent')
    print("Model saved as: models/improved_ems_agent")
    
    return model, eval_env

def detailed_evaluation(model, eval_env, n_episodes=10):
    """Detailed evaluation with visualizations"""
    print(f"\n=== Detailed Agent Evaluation ===")
    
    all_data = []
    episode_summaries = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_data = []
        
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            # Store step data
            step_data = {
                'episode': episode,
                'step': step,
                'demand': info['demand'],
                'renewable_generation': info['renewable_generation'],
                'battery_soc': info['battery_soc'],
                'grid_cost': info['grid_cost'],
                'unmet_demand': info['unmet_demand'],
                'renewable_used': info['renewable_used'],
                'grid_power': info['grid_power'],
                'battery_power': info['battery_power'],
                'demand_satisfaction': info['demand_satisfaction'],
                'reward': reward,
                'action_battery': action[0],
                'action_grid': action[1]
            }
            episode_data.append(step_data)
            step += 1
            done = done or truncated
        
        # Episode summary
        episode_summary = {
            'episode': episode,
            'total_reward': episode_reward,
            'total_cost': info['total_cost'],
            'renewable_efficiency': info['renewable_efficiency'],
            'violations': info['grid_violations'],
            'avg_demand_satisfaction': np.mean([d['demand_satisfaction'] for d in episode_data])
        }
        episode_summaries.append(episode_summary)
        all_data.extend(episode_data)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, "
              f"Cost=${info['total_cost']:.2f}, "
              f"Renewable Eff={info['renewable_efficiency']:.3f}, "
              f"Violations={info['grid_violations']}, "
              f"Demand Sat={episode_summary['avg_demand_satisfaction']:.3f}")
    
    # Create visualizations
    import pandas as pd
    df = pd.DataFrame(all_data)
    
    # Plot results for first episode
    episode_0 = df[df['episode'] == 0]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Smart EMS Agent Performance - Episode 1', fontsize=16)
    
    # Demand vs Supply
    axes[0,0].plot(episode_0['step'], episode_0['demand'], label='Demand', linewidth=2)
    axes[0,0].plot(episode_0['step'], episode_0['renewable_generation'], label='Renewable Gen', linewidth=2)
    axes[0,0].plot(episode_0['step'], episode_0['grid_power'], label='Grid Power', linewidth=2)
    axes[0,0].set_title('Energy Balance')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Power (kW)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Battery SOC and Actions
    axes[0,1].plot(episode_0['step'], episode_0['battery_soc'], label='Battery SoC', linewidth=2, color='green')
    axes[0,1].set_ylabel('SoC', color='green')
    axes[0,1].tick_params(axis='y', labelcolor='green')
    ax2 = axes[0,1].twinx()
    ax2.plot(episode_0['step'], episode_0['action_battery'], label='Battery Action', linewidth=2, color='red', alpha=0.7)
    ax2.set_ylabel('Battery Action', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    axes[0,1].set_title('Battery Management')
    axes[0,1].set_xlabel('Hour')
    axes[0,1].grid(True)
    
    # Costs and Rewards
    axes[1,0].plot(episode_0['step'], episode_0['grid_cost'], label='Grid Cost', linewidth=2)
    axes[1,0].plot(episode_0['step'], episode_0['reward'], label='Reward', linewidth=2)
    axes[1,0].set_title('Costs and Rewards')
    axes[1,0].set_xlabel('Hour')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Unmet Demand
    axes[1,1].plot(episode_0['step'], episode_0['unmet_demand'], linewidth=2, color='red')
    axes[1,1].set_title('Unmet Demand')
    axes[1,1].set_xlabel('Hour')
    axes[1,1].set_ylabel('Unmet Demand (kW)')
    axes[1,1].grid(True)
    
    # Actions
    axes[2,0].plot(episode_0['step'], episode_0['action_battery'], label='Battery Action', linewidth=2)
    axes[2,0].plot(episode_0['step'], episode_0['action_grid'], label='Grid Action', linewidth=2)
    axes[2,0].set_title('Agent Actions')
    axes[2,0].set_xlabel('Hour')
    axes[2,0].set_ylabel('Action Value')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    # Performance Summary
    summary_df = pd.DataFrame(episode_summaries)
    axes[2,1].bar(range(len(episode_summaries)), summary_df['total_reward'])
    axes[2,1].set_title('Episode Rewards')
    axes[2,1].set_xlabel('Episode')
    axes[2,1].set_ylabel('Total Reward')
    axes[2,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_ems_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== Performance Summary ===")
    print(f"Average Episode Reward: {summary_df['total_reward'].mean():.1f} ± {summary_df['total_reward'].std():.1f}")
    print(f"Average Daily Cost: ${summary_df['total_cost'].mean():.2f} ± ${summary_df['total_cost'].std():.2f}")
    print(f"Average Renewable Efficiency: {summary_df['renewable_efficiency'].mean():.3f} ± {summary_df['renewable_efficiency'].std():.3f}")
    print(f"Average Violations per Day: {summary_df['violations'].mean():.1f} ± {summary_df['violations'].std():.1f}")
    print(f"Average Demand Satisfaction: {summary_df['avg_demand_satisfaction'].mean():.3f} ± {summary_df['avg_demand_satisfaction'].std():.3f}")
    
    return summary_df

def main():
    """Main training pipeline"""
    # Load data
    data_df = load_and_prepare_data('processed_ems_data.csv')
    
    # Train improved agent
    model, eval_env = train_improved_agent(data_df, total_timesteps=25000)
    
    # Detailed evaluation
    results = detailed_evaluation(model, eval_env, n_episodes=10)
    
    print("\n=== Training Complete! ===")
    print("Model saved in './models/improved_ems_agent'")
    print("Performance visualization saved as 'improved_ems_performance.png'")

if __name__ == "__main__":
    main()