"""
Quick Test Script
Tests all major components to ensure everything is working
"""

import numpy as np
import pandas as pd
from datetime import datetime

print("="*60)
print("Microgrid EMS RL - Component Tests")
print("="*60)

# Test 1: Configuration
print("\n1. Testing Configuration...")
try:
    from env_config import (
        BATTERIES, GRID, EV_FLEET, RENEWABLE, 
        OBS_SPACE, ACTION_SPACE, REWARD, SAFETY,
        print_config_summary
    )
    print_config_summary()
    print("✓ Configuration loaded successfully")
except Exception as e:
    print(f"✗ Configuration error: {e}")

# Test 2: Battery Degradation Model
print("\n2. Testing Battery Degradation Model...")
try:
    from battery_degradation import BatteryDegradationModel, BatteryThermalModel
    from env_config import BATTERY_5
    
    deg_model = BatteryDegradationModel(BATTERY_5)
    thermal_model = BatteryThermalModel(BATTERY_5)
    
    # Simulate one charge-discharge cycle
    soc = 0.5
    for i in range(4):
        power_kw = 500 if i < 2 else -500
        energy_kwh = abs(power_kw) * 0.25
        soc_before = soc
        soc += (power_kw * 0.25) / BATTERY_5.capacity_kwh
        soc = np.clip(soc, 0.1, 0.9)
        
        temp = thermal_model.update(power_kw, soc, 0.25)
        cost, soh = deg_model.update(energy_kwh, soc_before, soc, temp, 0.25)
    
    metrics = deg_model.get_metrics()
    print(f"  SoH after cycle: {metrics['soh_percent']:.2f}%")
    print(f"  Throughput: {metrics['cumulative_throughput_kwh']:.1f} kWh")
    print("✓ Battery models working")
except Exception as e:
    print(f"✗ Battery model error: {e}")

# Test 3: EV Simulator
print("\n3. Testing EV Fleet Simulator...")
try:
    from ev_simulator import EVFleetSimulator
    
    ev_sim = EVFleetSimulator(random_seed=42)
    arrival_pattern = ev_sim.generate_arrival_pattern(96)
    
    for step in range(10):
        ev_sim.step(step, 96, arrival_pattern)
    
    state = ev_sim.get_fleet_state()
    print(f"  EVs connected: {state['num_connected']}")
    print(f"  Total energy needed: {state['total_energy_needed']:.1f} kWh")
    print("✓ EV simulator working")
except Exception as e:
    print(f"✗ EV simulator error: {e}")

# Test 4: Safety Supervisor
print("\n4. Testing Safety Supervisor...")
try:
    from safety_supervisor import SafetySupervisor
    
    supervisor = SafetySupervisor()
    
    raw_actions = {
        'battery_power': [600, 200],
        'grid_power': 1000,
        'ev_charging_power': 100
    }
    
    battery_states = [
        {'soc': 0.88, 'soh': 1.0, 'temperature': 30.0, 
         'max_charge_kw': 600, 'max_discharge_kw': 600},
        {'soc': 0.5, 'soh': 1.0, 'temperature': 28.0,
         'max_charge_kw': 200, 'max_discharge_kw': 200}
    ]
    
    grid_state = {'max_import_kw': 5000, 'max_export_kw': 3000}
    ev_state = {'total_max_charge_rate': 200}
    component_health = {'battery_temperatures': [30.0, 28.0], 'inverter_temperature': 45.0}
    
    safe_actions, penalty = supervisor.check_and_clip_actions(
        raw_actions, battery_states, grid_state, ev_state, 
        component_health, timestep=0
    )
    
    print(f"  Raw battery powers: {raw_actions['battery_power']}")
    print(f"  Safe battery powers: {safe_actions['battery_power']}")
    print(f"  Safety penalty: ${penalty:.2f}")
    print("✓ Safety supervisor working")
except Exception as e:
    print(f"✗ Safety supervisor error: {e}")

# Test 5: Data Preprocessing
print("\n5. Testing Data Preprocessing...")
try:
    from data_preprocessing import SolarDataLoader
    import os
    
    loader = SolarDataLoader()
    
    # Check if plant data exists
    if os.path.exists("Plant_1_Generation_Data.csv"):
        print("  Found Plant_1 generation data")
        gen_df, weather_df = loader.load_plant_data(plant_id=1)
        print(f"  Loaded {len(gen_df)} generation records")
        print(f"  Date range: {gen_df['DATE_TIME'].min()} to {gen_df['DATE_TIME'].max()}")
        print("✓ Data preprocessing working")
    else:
        print("  ⚠ Plant data not found - will use synthetic data")
        # Create synthetic profiles
        pv_profile = pd.DataFrame({
            'timestamp': pd.date_range('2020-05-15', periods=1000, freq='15T'),
            'pv_total': np.random.rand(1000) * 3000,
            'pv3': np.zeros(1000), 'pv4': np.zeros(1000),
            'pv5': np.zeros(1000), 'pv6': np.zeros(1000),
            'pv8': np.zeros(1000), 'pv9': np.zeros(1000),
            'pv10': np.zeros(1000), 'pv11': np.zeros(1000)
        })
        wt_profile = loader.create_wind_profile(1000)
        load_profile = loader.create_load_profile(1000)
        price_profile = loader.create_price_profile(1000)
        
        os.makedirs('data', exist_ok=True)
        pv_profile.to_csv('data/pv_profile_processed.csv', index=False)
        wt_profile.to_csv('data/wt_profile_processed.csv', index=False)
        load_profile.to_csv('data/load_profile_processed.csv', index=False)
        price_profile.to_csv('data/price_profile_processed.csv', index=False)
        print("  ✓ Created synthetic profiles in data/")
except Exception as e:
    print(f"✗ Data preprocessing error: {e}")

# Test 6: Microgrid Environment
print("\n6. Testing Microgrid Environment...")
try:
    from microgrid_env import MicrogridEMSEnv
    
    # Load or create simple profiles
    if not os.path.exists('data/pv_profile_processed.csv'):
        # Create minimal test data
        print("  Creating test data...")
        test_length = 200
        timestamps = pd.date_range('2020-05-15', periods=test_length, freq='15T')
        
        pv_profile = pd.DataFrame({
            'timestamp': timestamps, 'pv_total': np.random.rand(test_length) * 2000,
            'pv3': np.zeros(test_length), 'pv4': np.zeros(test_length),
            'pv5': np.zeros(test_length), 'pv6': np.zeros(test_length),
            'pv8': np.zeros(test_length), 'pv9': np.zeros(test_length),
            'pv10': np.zeros(test_length), 'pv11': np.zeros(test_length)
        })
        wt_profile = pd.DataFrame({
            'timestamp': timestamps, 'wt7': np.random.rand(test_length) * 1500
        })
        load_profile = pd.DataFrame({
            'timestamp': timestamps,
            'load_r1': np.random.rand(test_length) * 1000,
            'Load_r3': np.zeros(test_length), 'Load_r4': np.zeros(test_length),
            'Load_r5': np.zeros(test_length), 'Load_r6': np.zeros(test_length),
            'Load_r8': np.zeros(test_length), 'Load_r10': np.zeros(test_length),
            'Load_r11': np.zeros(test_length)
        })
        price_profile = pd.DataFrame({
            'timestamp': timestamps, 'price': np.random.rand(test_length) * 0.15 + 0.05
        })
        
        os.makedirs('data', exist_ok=True)
        pv_profile.to_csv('data/pv_profile_processed.csv', index=False)
        wt_profile.to_csv('data/wt_profile_processed.csv', index=False)
        load_profile.to_csv('data/load_profile_processed.csv', index=False)
        price_profile.to_csv('data/price_profile_processed.csv', index=False)
    
    pv = pd.read_csv('data/pv_profile_processed.csv')
    wt = pd.read_csv('data/wt_profile_processed.csv')
    load = pd.read_csv('data/load_profile_processed.csv')
    price = pd.read_csv('data/price_profile_processed.csv')
    
    env = MicrogridEMSEnv(
        pv_profile=pv,
        wt_profile=wt,
        load_profile=load,
        price_profile=price,
        enable_evs=True,
        enable_degradation=True,
        enable_emissions=True,
        random_seed=42
    )
    
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    # Test reset and step
    obs = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    
    print(f"  Step reward: {reward:.2f}")
    print(f"  Cost: ${info['cost']:.2f}")
    print(f"  Emissions: {info['emissions']:.2f} kg CO2")
    print("✓ Environment working")
    
except Exception as e:
    print(f"✗ Environment error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: PPO Agent
print("\n7. Testing PPO Agent...")
try:
    from train_ppo import PPOAgent
    
    agent = PPOAgent(obs_dim=100, action_dim=5)
    
    # Test action selection
    test_obs = np.random.randn(100)
    action = agent.select_action(test_obs, deterministic=True)
    
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.2f}, {action.max():.2f}]")
    print("✓ PPO agent working")
except Exception as e:
    print(f"✗ PPO agent error: {e}")

# Summary
print("\n" + "="*60)
print("Component Test Summary")
print("="*60)
print("All major components tested!")
print("\nNext steps:")
print("1. Run: python data_preprocessing.py")
print("2. Run: python train_ppo.py")
print("3. Run: python evaluate.py")
print("="*60)
