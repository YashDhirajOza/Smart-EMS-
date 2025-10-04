# 🇮🇳 Microgrid Energy Management System with Reinforcement Learning & EV Charging

## 🎯 Project Overview

This project implements a comprehensive **Reinforcement Learning (RL)** agent for real-time microgrid energy management with EV charging optimization, **specifically configured for the Indian power market**. The system minimizes operational costs and emissions while ensuring reliability, equipment safety, and battery longevity.

### 🌟 Indian Context Configuration
- **All costs in Indian Rupees (₹)**
- **Indian electricity tariffs** (₹4.50-9.50/kWh ToU)
- **Indian grid emission factors** (0.82 kg CO₂/kWh)
- **Real solar plant data** from Indian location
- **Practical savings**: ₹3-5 lakhs per year

👉 **See [INDIAN_CONTEXT.md](INDIAN_CONTEXT.md) for complete Indian configuration details**

### Key Features

✅ **Multi-Objective Optimization**
- Minimize energy costs (import/export from grid)
- Minimize CO₂ emissions
- Minimize battery degradation
- Guarantee zero unmet demand (hard constraint)

✅ **Comprehensive System Modeling**
- Battery energy storage with degradation (cycle & calendar aging)
- EV fleet charging with realistic arrival/departure patterns
- Renewable generation (solar PV + wind)
- Grid import/export with time-of-use pricing
- Safety constraints enforcement

✅ **Advanced RL Implementation**
- PPO (Proximal Policy Optimization) algorithm
- Continuous action space for fine-grained control
- Safety supervisor for constraint enforcement
- Curriculum learning support
- Explainable AI - action justifications

✅ **Real Data Integration**
- Uses actual solar plant generation data
- 15-minute decision intervals
- 24-hour episodes

---

## 📁 New Project Structure

```
microgrid-ems-drl/
├── env_config.py              # Environment configuration & hyperparameters
├── battery_degradation.py     # Battery degradation & thermal models
├── ev_simulator.py             # EV fleet simulator with arrival patterns
├── safety_supervisor.py        # Safety constraint enforcer
├── microgrid_env.py           # Main Gym environment
├── data_preprocessing.py       # Data loading & preprocessing
├── train_ppo.py               # PPO training script
├── evaluate.py                 # Evaluation & baseline comparison
├── README.md                   # This file
│
├── data/                       # Processed data profiles
│   ├── pv_profile_processed.csv
│   ├── wt_profile_processed.csv
│   ├── load_profile_processed.csv
│   └── price_profile_processed.csv
│
├── models/                     # Trained model checkpoints
├── logs/                       # Training logs & metrics
└── evaluation/                 # Evaluation results & plots
```

---

## 🚀 Quick Start

### 1. Installation

```powershell
pip install numpy pandas matplotlib seaborn torch gym
```

### 2. Data Preprocessing

Process your solar plant data:

```powershell
python data_preprocessing.py
```

### 3. Train RL Agent

```powershell
python train_ppo.py
```

### 4. Evaluate Agent

```powershell
python evaluate.py
```

---

## 🧠 RL Agent Details

### Observation Space (~100 dimensions)
- Temporal features (hour, day-of-week, etc.)
- Renewable generation (current + 2h forecast + 1h history)
- Load demand (current + forecast + history)
- Battery status (SoC, SoH, temperature, limits)
- Grid status (price, limits)
- EV fleet status (count, energy needed, deadlines)
- Component health indices
- Recent actions

### Action Space (5 dimensions, normalized [-1,1])
1. Battery 1 Power
2. Battery 2 Power  
3. Grid Power
4. EV Charging Power
5. Renewable Curtailment

### Reward Function
```
reward = -(cost + α·emissions + β·degradation + γ·reliability_penalty)
```
- α = 0.05 (emission weight)
- β = 0.5 (degradation weight)
- γ = 1000.0 (reliability penalty - ensures no unmet demand)

---

## 📊 Key Components

### 1. Battery Degradation Model
- Cycle aging (DoD-dependent)
- Calendar aging
- Temperature effects
- Throughput cost (kWh^1.1)

### 2. EV Fleet Simulator
- Realistic arrival patterns (morning/evening peaks)
- Variable battery sizes (40-100 kWh)
- Deadline-aware charging
- Success rate tracking

### 3. Safety Supervisor
- Enforces SoC limits (10-90%)
- Rate limits
- Temperature limits
- Clips unsafe actions + penalties

### 4. Explainable AI
Each action includes human-readable explanation:
```
"Discharge Battery_5 at 450 kW to reduce peak import cost 
during 18:00-19:00 high price; prevents unmet demand risk"
```

---

## 📈 Evaluation Metrics

**Operational:**
- Total cost ($)
- Total emissions (kg CO₂)
- Peak import/export (kW)

**Reliability:**
- Unmet demand events (should be 0)
- Unmet demand energy (kWh)

**Safety:**
- Safety overrides count
- Violations by type

**Battery:**
- Depth of Discharge (DoD)
- Cumulative throughput
- SoH degradation

**EV:**
- Charging success rate (%)
- Average final SoC

---

## 🔬 Advanced Features

- **Curriculum Learning**: Progressive difficulty (deterministic → stochastic → EV complexity)
- **Domain Randomization**: Varies demand, weather, prices across training
- **Forecast Errors**: Realistic imperfect forecasts
- **Multi-Battery Coordination**: Optimizes 2 batteries with different capacities

---

## 📚 Original Project (Legacy)

Original thesis work on time series observation and battery management for microgrids:

| File                  | Description         |
|-----------------------|---------------------|
| cigre_mv_microgrid.py | CIGRE MV test grid  |
| data.py               | PJM data conversion |
| main.py               | Original experiments|
| setting.py            | Original settings   |
| controllers/          | TD3/PPO controllers |

---

## 🙏 Acknowledgments

- Original thesis: Time Series & Battery Management in Microgrid RL
- Solar plant data from public datasets
- CIGRE microgrid benchmark
- OpenAI Gym & PyTorch

---

**Enhanced for: EV Charging + Emissions + Degradation + Safety! 🚀⚡🔋**
