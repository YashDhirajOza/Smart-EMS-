import gym
from gym import spaces
import numpy as np
import pandas as pd

class EMSEnv(gym.Env):
    def __init__(self, data_df):
        super(EMSEnv, self).__init__()
        self.data = data_df  # Pandas DF with columns: timestamp, solar, demand, price, etc.
        self.battery_capacity = 100  # kWh
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # State: [SoC, demand, solar, price]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1000, 1000, 10]), dtype=np.float32)
        
        # Action: [battery_action] (-1 discharge to 1 charge)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.soc = 0.5  # Initial SoC
        return self._get_obs()

    def step(self, action):
        battery_action = action[0]  # -1 to 1
        row = self.data.iloc[self.current_step]
        demand = row['demand']
        solar = row['solar']
        price = row['price']
        
        # Simulate dynamics
        available_energy = solar + max(0, -battery_action * self.battery_capacity * 0.1)  # Simplified
        grid_usage = max(0, demand - available_energy)
        self.soc = np.clip(self.soc + battery_action * 0.1, 0, 1)  # Update SoC
        
        # Reward
        cost = price * grid_usage
        emissions = 0.5 * grid_usage  # Assume emission factor
        reliability_penalty = 100 if demand > available_energy + grid_usage else 0
        reward = - (cost + emissions) - reliability_penalty
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        return np.array([self.soc, row['demand'], row['solar'], row['price']], dtype=np.float32)