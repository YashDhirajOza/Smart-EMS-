"""
Simplified Smart EMS RL Demo Script
Optimized for quick demonstration and testing
"""

from train_ems_rl import load_and_prepare_data, SmartEMSEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class SimpleEMSEnv(SmartEMSEnv):
    """Simplified EMS Environment for faster training"""
    
    def __init__(self, data_df, episode_length=24):
        super().__init__(data_df, episode_length)
        # Reduce battery capacity for faster learning
        self.battery_capacity = 50.0  # kWh
        self.max_charge_rate = 25.0  # kW
        self.max_discharge_rate = 25.0  # kW
    
    def step(self, action):
        """Simplified step function with clearer rewards"""
        if self.start_idx + self.current_step >= len(self.data_df):
            return self._get_observation(), -100.0, True, True, {"error": "exceeded_data"}
        
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
        
        # Simple battery dynamics
        if battery_action > 0:  # Charging
            charge_power = battery_action * self.max_charge_rate
            available_renewable = max(0, renewable_generation - demand)
            actual_charge = min(charge_power, available_renewable, 
                              (1.0 - self.battery_soc) * self.battery_capacity)
            self.battery_soc += actual_charge / self.battery_capacity
            battery_power = -actual_charge
        else:  # Discharging
            discharge_power = abs(battery_action) * self.max_discharge_rate
            max_discharge = self.battery_soc * self.battery_capacity
            actual_discharge = min(discharge_power, max_discharge)
            self.battery_soc -= actual_discharge / self.battery_capacity
            battery_power = actual_discharge
        
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
        
        # Energy balance
        renewable_used = min(renewable_generation, demand)
        battery_contribution = min(max(0, battery_power), max(0, demand - renewable_used))
        remaining_demand = max(0, demand - renewable_used - battery_contribution)
        
        # Grid usage
        grid_power = grid_action * remaining_demand
        unmet_demand = max(0, remaining_demand - grid_power)
        
        # Calculate cost
        grid_cost = grid_power * grid_price
        
        # SIMPLIFIED REWARD CALCULATION
        # Main goal: Meet demand while minimizing cost and maximizing renewable usage
        
        # 1. Demand satisfaction (0-100 points)
        demand_met_ratio = (demand - unmet_demand) / max(demand, 1)
        demand_reward = demand_met_ratio * 100
        
        # 2. Renewable efficiency (0-50 points)
        renewable_efficiency = renewable_used / max(renewable_generation, 0.1)
        renewable_reward = renewable_efficiency * 20
        
        # 3. Cost penalty (-cost in dollars)
        cost_penalty = -grid_cost
        
        # 4. Battery health (encourage moderate SoC)
        if 0.2 <= self.battery_soc <= 0.8:
            battery_reward = 5
        else:
            battery_reward = -5
        
        # 5. Severe penalty for unmet demand
        unmet_penalty = -unmet_demand * 50
        
        # Total reward
        reward = demand_reward + renewable_reward + cost_penalty + battery_reward + unmet_penalty
        
        # Update tracking
        self.total_cost += grid_cost
        self.total_renewable_used += renewable_used
        self.total_demand_met += (demand - unmet_demand)
        if unmet_demand > 0.1:
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
            'battery_power': battery_power,
            'total_cost': self.total_cost,
            'renewable_efficiency': self.total_renewable_used / max(self.total_demand_met, 1),
            'grid_violations': self.grid_violations,
            'demand_satisfaction': demand_met_ratio
        }
        
        return self._get_observation(), reward, done, False, info

def quick_train_and_demo():
    """Quick training and demonstration"""
    print("=== Smart EMS RL Demo ===")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    data_df = load_and_prepare_data('processed_ems_data.csv')
    
    # Create simple environment
    env = SimpleEMSEnv(data_df, episode_length=24)
    env = Monitor(env)
    
    print("Training PPO agent (demo version)...")
    
    # Simple PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.95,
        policy_kwargs={'net_arch': [128, 128]},
        verbose=1
    )
    
    # Train for limited timesteps
    model.learn(total_timesteps=5000, progress_bar=True)
    
    # Save model
    model.save('models/simple_ems_demo')
    print("Demo model saved!")
    
    # Test the trained agent
    print("\n=== Testing Trained Agent ===")
    
    obs, _ = env.reset()
    episode_data = []
    total_reward = 0
    
    for step in range(24):  # One full day
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        episode_data.append({
            'hour': step,
            'demand': info['demand'],
            'renewable_gen': info['renewable_generation'],
            'battery_soc': info['battery_soc'],
            'grid_power': info['grid_power'],
            'unmet_demand': info['unmet_demand'],
            'reward': reward,
            'battery_action': action[0],
            'grid_action': action[1]
        })
        
        if done or truncated:
            break
    
    # Create visualization
    df = pd.DataFrame(episode_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Smart EMS Agent - 24 Hour Demo', fontsize=14)
    
    # Energy balance
    axes[0,0].plot(df['hour'], df['demand'], label='Demand', linewidth=2, marker='o')
    axes[0,0].plot(df['hour'], df['renewable_gen'], label='Renewable', linewidth=2, marker='s')
    axes[0,0].plot(df['hour'], df['grid_power'], label='Grid', linewidth=2, marker='^')
    axes[0,0].set_title('Energy Balance')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Power (kW)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Battery management
    axes[0,1].plot(df['hour'], df['battery_soc'], linewidth=3, color='green', marker='o')
    axes[0,1].set_title('Battery State of Charge')
    axes[0,1].set_xlabel('Hour')
    axes[0,1].set_ylabel('SoC')
    axes[0,1].set_ylim(0, 1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Unmet demand
    axes[1,0].bar(df['hour'], df['unmet_demand'], color='red', alpha=0.7)
    axes[1,0].set_title('Unmet Demand')
    axes[1,0].set_xlabel('Hour')
    axes[1,0].set_ylabel('Unmet Demand (kW)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Agent actions
    axes[1,1].plot(df['hour'], df['battery_action'], label='Battery Action', linewidth=2, marker='o')
    axes[1,1].plot(df['hour'], df['grid_action'], label='Grid Action', linewidth=2, marker='s')
    axes[1,1].set_title('Agent Actions')
    axes[1,1].set_xlabel('Hour')
    axes[1,1].set_ylabel('Action Value')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ems_demo_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== Demo Results ===")
    print(f"Total Episode Reward: {total_reward:.1f}")
    print(f"Total Grid Cost: ${info['total_cost']:.2f}")
    print(f"Renewable Efficiency: {info['renewable_efficiency']:.3f}")
    print(f"Grid Violations: {info['grid_violations']}")
    print(f"Final Battery SoC: {info['battery_soc']:.3f}")
    print(f"Average Unmet Demand: {df['unmet_demand'].mean():.2f} kW")
    
    # Performance metrics
    total_demand = df['demand'].sum()
    total_renewable = df['renewable_gen'].sum()
    total_grid = df['grid_power'].sum()
    total_unmet = df['unmet_demand'].sum()
    
    print(f"\n=== Energy Balance Summary ===")
    print(f"Total Demand: {total_demand:.1f} kWh")
    print(f"Renewable Supply: {total_renewable:.1f} kWh ({total_renewable/total_demand*100:.1f}%)")
    print(f"Grid Supply: {total_grid:.1f} kWh ({total_grid/total_demand*100:.1f}%)")
    print(f"Unmet: {total_unmet:.1f} kWh ({total_unmet/total_demand*100:.1f}%)")
    
    print(f"\nDemo visualization saved as: ems_demo_results.png")
    return model, episode_data

if __name__ == "__main__":
    model, data = quick_train_and_demo()