"""
Final Optimized Smart EMS RL Training Script
Production-ready version with proper scaling and reward shaping
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch
import os
from train_ems_rl import load_and_prepare_data, SmartEMSEnv

class ProductionEMSEnv(SmartEMSEnv):
    """Production-ready EMS Environment with optimized parameters"""
    
    def __init__(self, data_df, episode_length=24):
        # Initialize parent class first
        super().__init__(data_df, episode_length)
        
        # FIXED: Data-driven scaling based on actual statistics
        self.demand_scale = 1.0 / self.data_df['demand'].max()  # Normalize to [0,1] range
        self.renewable_scale = 1.0 / max(self.data_df['solar_output'].max(), self.data_df['wind_output'].max())
        
        # Realistic microgrid parameters
        self.battery_capacity = 200.0  # kWh - reasonable for small microgrid
        self.max_charge_rate = 100.0   # kW
        self.max_discharge_rate = 100.0  # kW
        self.battery_efficiency = 0.9
        
        # FIXED: Add missing critical parameters
        self.dt = 1.0  # time step in hours (CRITICAL for kW/kWh conversion)
        self.max_grid_power = 500.0  # kW - enforce grid capacity limit
        self.grid_connection_cost = 0.0  # $/kWh additional grid cost
        
        # FIXED: Initialize all counters
        self.total_cost = 0.0
        self.total_renewable_used = 0.0
        self.total_demand_met = 0.0
        self.grid_violations = 0
        
        # Calculate normalization factors for observations
        self.max_demand_kw = self.data_df['demand'].max() * self.demand_scale
        self.max_renewable_kw = max(
            self.data_df['solar_output'].max() * self.renewable_scale,
            self.data_df['wind_output'].max() * self.renewable_scale
        )
        self.max_grid_price = self.data_df['grid_price'].max()
        
        # FIXED: Properly normalized observation space
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # All normalized to [0,1] or [-1,1]
            dtype=np.float32
        )
    
    def _get_observation(self):
        """Get properly normalized observation"""
        if self.start_idx + self.current_step >= len(self.data_df):
            row = self.data_df.iloc[-1]
            hour = 0
        else:
            row = self.data_df.iloc[self.start_idx + self.current_step]
            hour = row.name.hour if hasattr(row.name, 'hour') else 0
        
        # Scale and normalize all values to [0,1] range
        norm_demand = (row['demand'] * self.demand_scale) / max(1e-6, self.max_demand_kw)
        norm_solar = (row['solar_output'] * self.renewable_scale) / max(1e-6, self.max_renewable_kw)
        norm_wind = (row['wind_output'] * self.renewable_scale) / max(1e-6, self.max_renewable_kw)
        norm_grid_price = row['grid_price'] / max(1e-6, self.max_grid_price)
        
        # Time encoding (already normalized to [-1,1])
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        obs = np.array([
            self.battery_soc,      # [0,1]
            norm_demand,           # [0,1]
            norm_solar,            # [0,1]
            norm_wind,             # [0,1]
            norm_grid_price,       # [0,1]
            hour_sin,              # [-1,1]
            hour_cos               # [-1,1]
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Optimized step function"""
        if self.start_idx + self.current_step >= len(self.data_df):
            return self._get_observation(), -50.0, True, True, {"error": "exceeded_data"}
        
        # Get scaled current data
        row = self.data_df.iloc[self.start_idx + self.current_step]
        demand = row['demand'] * self.demand_scale
        solar_output = row['solar_output'] * self.renewable_scale
        wind_output = row['wind_output'] * self.renewable_scale
        grid_price = row['grid_price']
        
        # Parse actions
        battery_action = np.clip(action[0], -1.0, 1.0)
        grid_action = np.clip(action[1], 0.0, 1.0)
        
        # Calculate renewable generation
        renewable_generation = solar_output + wind_output
        
        # FIXED: Battery dynamics with proper kW/kWh unit consistency
        dt = self.dt  # hours
        battery_power_supply_kw = 0.0  # positive = supplies load; negative = consumes (charging)
        
        if battery_action > 0:  # Charging request
            charge_power_kw = battery_action * self.max_charge_rate
            available_excess_kw = max(0.0, renewable_generation - demand)
            # Capacity-limited as power equivalent over dt
            max_charge_capacity_kw = (1.0 - self.battery_soc) * self.battery_capacity / dt
            actual_charge_kw = min(charge_power_kw, available_excess_kw, max_charge_capacity_kw)
            
            # Energy to store (kWh) accounting for efficiency
            energy_stored_kwh = actual_charge_kw * dt * self.battery_efficiency
            self.battery_soc += energy_stored_kwh / self.battery_capacity
            
            battery_power_supply_kw = -actual_charge_kw  # Consumes power to charge
            
        else:  # Discharging request
            discharge_power_kw = abs(battery_action) * self.max_discharge_rate
            # Capacity-limited as power-equivalent over dt
            max_discharge_capacity_kw = (self.battery_soc * self.battery_capacity) / dt
            actual_discharge_kw = min(discharge_power_kw, max_discharge_capacity_kw)
            
            # Energy drawn from battery (kWh)
            energy_drawn_kwh = actual_discharge_kw * dt
            # Energy delivered to load after efficiency losses
            energy_delivered_kwh = energy_drawn_kwh * self.battery_efficiency
            
            # Reduce SoC by energy drawn (pre-efficiency)
            self.battery_soc -= energy_drawn_kwh / self.battery_capacity
            
            # kW delivered to load (since dt=1h, numerically equal)
            battery_power_supply_kw = energy_delivered_kwh / dt
        
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
        
        # Energy balance calculation
        # Priority: 1) Renewables 2) Battery 3) Grid
        renewable_used = min(renewable_generation, demand)
        remaining_after_renewable = max(0, demand - renewable_used)
        
        battery_used = min(max(0, battery_power_supply_kw), remaining_after_renewable)
        remaining_after_battery = max(0, remaining_after_renewable - battery_used)
        
        # FIXED: Grid action maps to absolute power with capacity limit
        requested_grid_kw = grid_action * self.max_grid_power
        grid_power = min(requested_grid_kw, remaining_after_battery)
        unmet_demand = max(0, remaining_after_battery - grid_power)
        
        # Calculate costs
        grid_cost = grid_power * (grid_price + self.grid_connection_cost)
        
        # OPTIMIZED REWARD FUNCTION
        # Scale rewards to be in reasonable range (-100 to +100)
        
        # 1. Demand satisfaction reward (0 to 40 points)
        demand_satisfaction = (demand - unmet_demand) / max(demand, 0.1)
        demand_reward = demand_satisfaction * 40
        
        # 2. Renewable utilization reward (0 to 20 points)
        if renewable_generation > 0.1:
            renewable_efficiency = renewable_used / renewable_generation
            renewable_reward = renewable_efficiency * 20
        else:
            renewable_reward = 0
        
        # 3. Cost efficiency reward (-20 to 0 points)
        # Normalize cost by demand to make it scale-invariant
        if demand > 0.1:
            cost_ratio = grid_cost / demand
            cost_reward = -cost_ratio * 20
        else:
            cost_reward = 0
        
        # 4. Battery health reward (-5 to +5 points)
        if 0.2 <= self.battery_soc <= 0.8:
            soc_reward = 5
        elif 0.1 <= self.battery_soc <= 0.9:
            soc_reward = 2
        else:
            soc_reward = -5
        
        # 5. Unmet demand penalty (-40 to 0 points)
        if demand > 0.1:
            unmet_ratio = unmet_demand / demand
            unmet_penalty = -unmet_ratio * 40
        else:
            unmet_penalty = 0
        
        # 6. Operational bonus (0 to 5 points)
        # Bonus for stable operation
        operational_bonus = 5 if unmet_demand < 0.01 else 0
        
        # Total reward (range approximately -60 to +90)
        reward = (demand_reward + renewable_reward + cost_reward + 
                 soc_reward + unmet_penalty + operational_bonus)
        
        # Update tracking
        self.total_cost += grid_cost
        self.total_renewable_used += renewable_used
        self.total_demand_met += (demand - unmet_demand)
        if unmet_demand > 0.01:  # 10W threshold
            self.grid_violations += 1
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        info = {
            'demand': demand,
            'renewable_generation': renewable_generation,
            'battery_soc': self.battery_soc,
            'grid_cost': grid_cost,
            'unmet_demand': unmet_demand,
            'renewable_used': renewable_used,
            'grid_power': grid_power,
            'battery_power': battery_power_supply_kw,
            'total_cost': self.total_cost,
            'renewable_efficiency': self.total_renewable_used / max(self.total_demand_met, 0.1),
            'grid_violations': self.grid_violations,
            'demand_satisfaction': demand_satisfaction,
            'reward_breakdown': {
                'demand_reward': demand_reward,
                'renewable_reward': renewable_reward,
                'cost_reward': cost_reward,
                'soc_reward': soc_reward,
                'unmet_penalty': unmet_penalty,
                'operational_bonus': operational_bonus
            }
        }
        
        return self._get_observation(), reward, done, False, info

def train_production_model():
    """Train production-ready model"""
    print("=== Training Production Smart EMS Agent ===")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    data_df = load_and_prepare_data('processed_ems_data.csv')
    
    # Create environment
    env = ProductionEMSEnv(data_df, episode_length=24)
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = ProductionEMSEnv(data_df, episode_length=24)
    eval_env = Monitor(eval_env)
    
    print("Training optimized PPO agent...")
    
    # Optimized PPO hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,  # No entropy bonus needed for this problem
        policy_kwargs={
            'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
            'activation_fn': torch.nn.ReLU,
        },
        verbose=1,
        device='auto'
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=10,
        verbose=1
    )
    
    # Train
    model.learn(
        total_timesteps=20000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save('models/production_ems_agent')
    print("Production model saved!")
    
    return model, eval_env

def comprehensive_evaluation(model, eval_env, n_episodes=20):
    """Comprehensive evaluation with detailed analytics"""
    print(f"\n=== Comprehensive Evaluation ({n_episodes} episodes) ===")
    
    all_episodes = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_data = []
        episode_reward = 0
        
        for step in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            episode_data.append({
                'episode': episode,
                'step': step,
                'hour': step,
                'demand': info['demand'],
                'renewable_gen': info['renewable_generation'],
                'battery_soc': info['battery_soc'],
                'grid_power': info['grid_power'],
                'unmet_demand': info['unmet_demand'],
                'grid_cost': info['grid_cost'],
                'renewable_used': info['renewable_used'],
                'battery_power': info['battery_power'],
                'reward': reward,
                'action_battery': action[0],
                'action_grid': action[1]
            })
            
            if done or truncated:
                break
        
        all_episodes.extend(episode_data)
        
        # Episode summary
        total_demand = sum(d['demand'] for d in episode_data)
        total_renewable = sum(d['renewable_gen'] for d in episode_data)
        total_unmet = sum(d['unmet_demand'] for d in episode_data)
        total_cost = info['total_cost']
        
        print(f"Episode {episode+1:2d}: Reward={episode_reward:6.1f}, "
              f"Cost=${total_cost:5.2f}, "
              f"Renewable={total_renewable/total_demand*100:5.1f}%, "
              f"Unmet={total_unmet/total_demand*100:5.1f}%, "
              f"Violations={info['grid_violations']:2d}")
    
    # Create comprehensive visualization
    df = pd.DataFrame(all_episodes)
    
    # Select best episode for detailed view
    episode_rewards = df.groupby('episode')['reward'].sum()
    best_episode = episode_rewards.idxmax()
    best_ep_data = df[df['episode'] == best_episode]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Smart EMS Production Agent - Best Episode Performance', fontsize=16)
    
    # Energy balance
    axes[0,0].plot(best_ep_data['hour'], best_ep_data['demand'], 'b-', linewidth=3, label='Demand', marker='o')
    axes[0,0].plot(best_ep_data['hour'], best_ep_data['renewable_gen'], 'g-', linewidth=3, label='Renewable', marker='s')
    axes[0,0].plot(best_ep_data['hour'], best_ep_data['grid_power'], 'r-', linewidth=2, label='Grid', marker='^')
    axes[0,0].fill_between(best_ep_data['hour'], 0, best_ep_data['unmet_demand'], alpha=0.3, color='orange', label='Unmet')
    axes[0,0].set_title('Energy Balance (Best Episode)')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Power (kW)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Battery management
    axes[0,1].plot(best_ep_data['hour'], best_ep_data['battery_soc'], 'g-', linewidth=4, marker='o')
    axes[0,1].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Min SoC')
    axes[0,1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Max SoC')
    axes[0,1].set_title('Battery State of Charge')
    axes[0,1].set_xlabel('Hour')
    axes[0,1].set_ylabel('SoC')
    axes[0,1].set_ylim(0, 1)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Costs
    axes[0,2].bar(best_ep_data['hour'], best_ep_data['grid_cost'], alpha=0.7, color='red')
    axes[0,2].set_title('Hourly Grid Costs')
    axes[0,2].set_xlabel('Hour')
    axes[0,2].set_ylabel('Cost ($)')
    axes[0,2].grid(True, alpha=0.3)
    
    # Agent actions
    axes[1,0].plot(best_ep_data['hour'], best_ep_data['action_battery'], 'b-', linewidth=2, marker='o', label='Battery Action')
    axes[1,0].plot(best_ep_data['hour'], best_ep_data['action_grid'], 'r-', linewidth=2, marker='s', label='Grid Action')
    axes[1,0].set_title('Agent Actions')
    axes[1,0].set_xlabel('Hour')
    axes[1,0].set_ylabel('Action Value')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Performance distribution
    episode_summaries = df.groupby('episode').agg({
        'reward': 'sum',
        'grid_cost': 'sum',
        'unmet_demand': 'sum',
        'demand': 'sum'
    }).reset_index()
    episode_summaries['unmet_ratio'] = episode_summaries['unmet_demand'] / episode_summaries['demand']
    
    axes[1,1].hist(episode_summaries['reward'], bins=10, alpha=0.7, color='blue')
    axes[1,1].set_title('Episode Reward Distribution')
    axes[1,1].set_xlabel('Total Episode Reward')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    # Cost vs Performance
    axes[1,2].scatter(episode_summaries['grid_cost'], episode_summaries['reward'], alpha=0.7, s=50)
    axes[1,2].set_title('Cost vs Reward')
    axes[1,2].set_xlabel('Total Grid Cost ($)')
    axes[1,2].set_ylabel('Total Reward')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('production_ems_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive statistics
    print(f"\n=== Performance Statistics ===")
    print(f"Average Episode Reward: {episode_summaries['reward'].mean():.1f} ± {episode_summaries['reward'].std():.1f}")
    print(f"Average Daily Grid Cost: ${episode_summaries['grid_cost'].mean():.2f} ± ${episode_summaries['grid_cost'].std():.2f}")
    print(f"Average Unmet Demand Ratio: {episode_summaries['unmet_ratio'].mean():.3f} ± {episode_summaries['unmet_ratio'].std():.3f}")
    print(f"Best Episode Reward: {episode_summaries['reward'].max():.1f}")
    print(f"Worst Episode Reward: {episode_summaries['reward'].min():.1f}")
    
    return df, episode_summaries

def main():
    """Main production training and evaluation"""
    # Train production model
    model, eval_env = train_production_model()
    
    # Comprehensive evaluation
    results_df, summary_df = comprehensive_evaluation(model, eval_env, n_episodes=20)
    
    print(f"\n=== Production Training Complete! ===")
    print(f"Final model saved as: models/production_ems_agent")
    print(f"Evaluation results saved as: production_ems_evaluation.png")
    print(f"Ready for deployment in Smart EMS dashboard!")

if __name__ == "__main__":
    main()