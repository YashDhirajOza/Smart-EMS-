# Smart Energy Management System (EMS) RL Training Suite

## Overview

This repository contains a comprehensive Reinforcement Learning (RL) training suite for Smart Energy Management Systems. The system uses real-world Indian power grid data to train agents that can optimally manage energy resources in a microgrid scenario.

## 🎯 Key Features

- **Real-world data integration**: Processes 18,984+ hours of Indian power grid data (2021-2023)
- **Multiple RL environments**: From simple demo to production-ready environments
- **Comprehensive training**: PPO-based RL agent with optimized hyperparameters
- **Detailed evaluation**: Performance analytics with visualizations
- **Production-ready**: Scaled for realistic microgrid deployment

## 📁 File Structure

```
├── prepare_data.py              # Data preprocessing script
├── processed_ems_data.csv       # Processed training data (18,984 records)
├── train_ems_rl.py             # Core RL training framework
├── train_improved_ems.py       # Improved environment with better rewards
├── train_production_ems.py     # Production-ready training script
├── ems_demo.py                 # Quick demo script
├── test_rl_quick.py           # Quick testing script
├── requirements.txt            # Python dependencies
├── models/                     # Trained model storage
└── logs/                      # Training logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Data Processing

The processed data file `processed_ems_data.csv` contains:
- **18,984 hourly records** (Nov 2021 - Dec 2023)
- **Real demand patterns** from Indian power grid
- **Solar and wind generation** data
- **Simulated EV charging** and battery SoC patterns
- **Dynamic grid pricing**

### 3. Run Quick Demo

```bash
python ems_demo.py
```

This will:
- Train a simple agent (5,000 timesteps)
- Test on one 24-hour episode
- Generate visualization (`ems_demo_results.png`)

### 4. Train Production Model

```bash
python train_production_ems.py
```

This will:
- Train optimized PPO agent (20,000 timesteps)
- Scale data to realistic microgrid levels
- Evaluate performance across 20 episodes
- Save production model (`models/production_ems_agent`)
- Generate detailed analytics

## 🏗️ Environment Details

### State Space (7 dimensions)
- `battery_soc`: Battery state of charge (0-1)
- `demand`: Current energy demand (kW)
- `solar_output`: Solar generation (kW)
- `wind_output`: Wind generation (kW)
- `grid_price`: Current electricity price ($/kWh)
- `hour_sin`, `hour_cos`: Time encoding

### Action Space (2 dimensions)
- `battery_action`: Battery charge/discharge (-1 to +1)
- `grid_action`: Grid usage factor (0 to 1)

### Reward Function
The production environment uses a balanced reward function:
- **Demand satisfaction** (0-40 points): Rewards meeting energy demand
- **Renewable utilization** (0-20 points): Encourages using solar/wind
- **Cost efficiency** (-20-0 points): Penalizes high grid costs
- **Battery health** (-5-+5 points): Maintains optimal SoC range
- **Unmet demand penalty** (-40-0 points): Heavily penalizes blackouts
- **Operational bonus** (0-5 points): Rewards stable operation

## 📊 Performance Metrics

The system tracks:
- **Average episode reward**: Overall agent performance
- **Daily grid cost**: Economic efficiency
- **Renewable efficiency**: Percentage of renewables used
- **Demand satisfaction**: Percentage of demand met
- **Grid violations**: Number of unmet demand incidents

## 🔧 Environment Configurations

### 1. SimpleEMSEnv (ems_demo.py)
- **Use case**: Quick demonstration
- **Battery**: 50 kWh, 25 kW charge/discharge
- **Training**: 5,000 timesteps
- **Focus**: Basic functionality

### 2. ImprovedEMSEnv (train_improved_ems.py)
- **Use case**: Research and development
- **Battery**: 100 kWh, 50 kW charge/discharge
- **Training**: 25,000 timesteps
- **Focus**: Advanced reward shaping

### 3. ProductionEMSEnv (train_production_ems.py)
- **Use case**: Production deployment
- **Battery**: 200 kWh, 100 kW charge/discharge
- **Training**: 20,000 timesteps
- **Focus**: Realistic microgrid scaling

## 🎯 Training Results

Expected performance after training:
- **Episode Reward**: 800-1200 points
- **Daily Grid Cost**: $5-15
- **Renewable Efficiency**: 85-95%
- **Demand Satisfaction**: 98-100%
- **Grid Violations**: 0-2 per day

## 📈 Usage in Smart EMS Dashboard

### Loading Trained Model

```python
from stable_baselines3 import PPO
from train_production_ems import ProductionEMSEnv
import pandas as pd

# Load trained model
model = PPO.load('models/production_ems_agent')

# Load data
data_df = pd.read_csv('processed_ems_data.csv', 
                     index_col='DATE_TIME', 
                     parse_dates=True)

# Create environment for real-time operation
env = ProductionEMSEnv(data_df, episode_length=24)

# Get recommendation
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)

print(f"Battery Action: {action[0]:.3f} (-1=discharge, +1=charge)")
print(f"Grid Action: {action[1]:.3f} (0=no grid, 1=full grid)")
```

### Real-time Integration

The trained model can provide real-time recommendations:
1. **Input**: Current system state (SoC, demand, generation, price, time)
2. **Output**: Optimal actions for battery and grid usage
3. **Frequency**: Updated every hour or as needed

## 🛠️ Customization

### Scaling for Different Microgrids

Adjust parameters in `ProductionEMSEnv.__init__()`:
```python
self.battery_capacity = 500.0      # Larger battery (kWh)
self.max_charge_rate = 200.0       # Faster charging (kW)
self.demand_scale = 0.01           # Larger microgrid scale
```

### Adding New Features

Extend the state space in `_get_observation()`:
```python
# Add weather, load forecast, etc.
obs = np.array([
    self.battery_soc,
    scaled_demand,
    scaled_solar,
    scaled_wind,
    grid_price,
    weather_condition,  # New feature
    load_forecast,      # New feature
    hour_sin,
    hour_cos
])
```

## 🔍 Troubleshooting

### Common Issues

1. **Low Performance**: Increase training timesteps or adjust reward weights
2. **High Unmet Demand**: Reduce demand scale or increase battery capacity
3. **Poor Battery Management**: Adjust SoC reward bounds
4. **Training Instability**: Reduce learning rate or batch size

### Debugging Tools

```python
# Monitor training progress
from stable_baselines3.common.monitor import Monitor
env = Monitor(env, "training_log.csv")

# Analyze reward components
info = env.step(action)[4]
print(info['reward_breakdown'])
```

## 📚 References

- **Data Source**: Mendeley - Electricity Demand, Solar and Wind Generation Data for India
- **RL Framework**: Stable Baselines3 (PPO algorithm)
- **Environment**: Custom Gymnasium-based EMS environment

## 🤝 Contributing

To extend this system:
1. Modify environment parameters for your use case
2. Adjust reward function for different objectives
3. Add new data sources or features
4. Implement different RL algorithms (SAC, TD3, etc.)

## 📄 License

This project is designed for the IIT Smart Grid Hackathon and demonstrates the application of RL in energy management systems.

---

**Ready for integration into your Smart EMS dashboard!** 🚀