"""
Enhanced Environment Configuration for RL-based Microgrid EMS with EV Charging
Includes all requirements: emissions, degradation, EV fleet, safety constraints

INDIAN CONTEXT:
- All costs in Indian Rupees (₹)
- Grid tariffs based on Indian power market rates
- Emission factors for Indian grid
- EV charging parameters for Indian market
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ===== CURRENCY CONFIGURATION =====
CURRENCY = "INR"  # Indian Rupees
USD_TO_INR = 83.0  # Conversion rate (approximate)

# ===== EPISODE & TIMESTEP CONFIGURATION =====
EPISODE_HOURS = 24  # 24-hour episodes
DECISION_INTERVAL_MINUTES = 15  # 15-minute decision intervals
STEPS_PER_EPISODE = int(EPISODE_HOURS * 60 / DECISION_INTERVAL_MINUTES)  # 96 steps
HOURS_PER_STEP = DECISION_INTERVAL_MINUTES / 60.0  # 0.25 hours

# ===== BATTERY CONFIGURATION =====
@dataclass
class BatteryConfig:
    """Battery storage configuration with degradation modeling"""
    name: str
    capacity_kwh: float  # Max energy capacity (kWh)
    max_charge_kw: float  # Max charge power (kW)
    max_discharge_kw: float  # Max discharge power (kW)
    soc_min: float = 0.1  # Minimum SoC (10%)
    soc_max: float = 0.9  # Maximum SoC (90%)
    initial_soc: float = 0.5  # Initial SoC (50%)
    efficiency: float = 0.95  # Round-trip efficiency
    
    # Degradation parameters
    cycle_life: int = 5000  # Nominal cycles to 80% capacity
    calendar_life_years: float = 10.0  # Calendar aging
    temperature_nominal: float = 25.0  # Nominal temperature (°C)
    soh_initial: float = 1.0  # State of Health (100%)
    
    # Cost parameters
    degradation_cost_per_kwh: float = 12.45  # ₹/kWh throughput cost (0.15 USD * 83)

# Battery instances
BATTERY_5 = BatteryConfig(
    name="Battery_5",
    capacity_kwh=3000,  # 3 MWh = 3000 kWh
    max_charge_kw=600,
    max_discharge_kw=600
)

BATTERY_10 = BatteryConfig(
    name="Battery_10",
    capacity_kwh=1000,  # 1 MWh = 1000 kWh
    max_charge_kw=200,
    max_discharge_kw=200
)

BATTERIES = [BATTERY_5, BATTERY_10]

# ===== EV CHARGING CONFIGURATION =====
@dataclass
class EVChargerConfig:
    """EV charger configuration"""
    name: str
    max_power_kw: float  # Max charging power per charger
    efficiency: float = 0.95  # Charging efficiency
    num_ports: int = 1  # Number of charging ports

@dataclass
class EVFleetConfig:
    """EV fleet parameters for simulation"""
    max_concurrent_evs: int = 10  # Max EVs charging simultaneously
    ev_battery_capacity_range: Tuple[float, float] = (40, 100)  # kWh range (small to large EVs)
    ev_arrival_soc_range: Tuple[float, float] = (0.2, 0.5)  # Arrival SoC range
    ev_departure_soc_target: float = 0.9  # Target SoC at departure (90%)
    ev_max_charge_rate_range: Tuple[float, float] = (7, 50)  # kW (Level 2 to DC fast)
    
    # Temporal patterns
    morning_arrival_peak: int = 8  # 8 AM
    evening_arrival_peak: int = 18  # 6 PM
    avg_parking_duration_hours: float = 4.0
    
    # Cost parameters
    ev_demand_charge_premium: float = 1.2  # Premium for peak EV demand

# EV Fleet instance
EV_FLEET = EVFleetConfig()

# Charger instances (can be expanded)
EV_CHARGERS = [
    EVChargerConfig(name="Charger_1", max_power_kw=50, num_ports=4),
    EVChargerConfig(name="Charger_2", max_power_kw=50, num_ports=4),
    EVChargerConfig(name="Charger_3", max_power_kw=22, num_ports=2),
]

# ===== GRID CONFIGURATION =====
@dataclass
class GridConfig:
    """Grid connection parameters"""
    max_import_kw: float = 5000  # Max import from grid
    max_export_kw: float = 3000  # Max export to grid
    availability: float = 0.999  # Grid availability (99.9%)
    
    # Emissions parameters for Indian grid (kg CO2 per kWh)
    # India's grid emission factor is higher due to coal dependency
    emission_factor_base: float = 0.82  # Base grid emission factor (Indian average)
    emission_factor_peak: float = 0.95  # Peak time emission factor
    emission_factor_offpeak: float = 0.70  # Off-peak emission factor
    
    peak_hours: List[int] = field(default_factory=lambda: list(range(17, 21)))  # 5 PM - 9 PM
    offpeak_hours: List[int] = field(default_factory=lambda: list(range(0, 6)))  # 12 AM - 6 AM

GRID = GridConfig()

# ===== RENEWABLE GENERATION =====
@dataclass
class RenewableConfig:
    """Renewable generation configuration"""
    # PV systems
    pv_total_capacity_kw: float = 3200  # Total PV capacity
    pv_num_systems: int = 8
    
    # Wind turbines
    wt_total_capacity_kw: float = 2500  # Total wind capacity
    wt_num_turbines: int = 1
    
    # Forecast error parameters
    pv_forecast_std: float = 0.15  # 15% standard deviation
    wt_forecast_std: float = 0.20  # 20% standard deviation
    forecast_horizon_steps: int = 8  # Forecast next 2 hours (8 * 15min)

RENEWABLE = RenewableConfig()

# ===== LOAD CONFIGURATION =====
@dataclass
class LoadConfig:
    """Load demand configuration"""
    base_load_kw: float = 4000  # Base critical load
    num_load_points: int = 8
    forecast_std: float = 0.10  # 10% forecast error

LOAD = LoadConfig()

# ===== OBSERVATION SPACE CONFIGURATION =====
@dataclass
class ObservationSpaceConfig:
    """Defines the observation space structure"""
    
    # Temporal features (4)
    hour_of_day: int = 1  # [0, 23] normalized
    minute_of_hour: int = 1  # [0, 45] normalized
    day_of_week: int = 1  # [0, 6] normalized
    is_weekend: int = 1  # Binary
    
    # Renewable generation (current + forecast + history)
    pv_current: int = 1
    pv_forecast: int = RENEWABLE.forecast_horizon_steps  # 8 steps ahead
    pv_history: int = 4  # Last 4 steps (1 hour)
    wt_current: int = 1
    wt_forecast: int = RENEWABLE.forecast_horizon_steps
    wt_history: int = 4
    
    # Load demand (current + forecast + history)
    load_current: int = 1
    load_forecast: int = RENEWABLE.forecast_horizon_steps
    load_history: int = 4
    
    # Battery status (per battery: 6 features)
    battery_soc: int = len(BATTERIES)
    battery_soh: int = len(BATTERIES)
    battery_temperature: int = len(BATTERIES)
    battery_max_charge_rate: int = len(BATTERIES)
    battery_max_discharge_rate: int = len(BATTERIES)
    battery_throughput: int = len(BATTERIES)
    
    # Grid status (4)
    grid_price_current: int = 1
    grid_price_forecast: int = RENEWABLE.forecast_horizon_steps
    grid_import_limit: int = 1
    grid_export_limit: int = 1
    
    # EV fleet status (5)
    ev_connected_count: int = 1
    ev_total_energy_needed: int = 1
    ev_avg_deadline_steps: int = 1
    ev_total_max_charge_rate: int = 1
    ev_earliest_deadline: int = 1
    
    # Component health indices (3)
    inverter_temperature: int = 1
    transformer_load_factor: int = 1
    grid_voltage_deviation: int = 1
    
    # Recent actions (optional, 3 actions × 4 history steps)
    recent_battery_actions: int = len(BATTERIES) * 4
    recent_grid_action: int = 4
    recent_ev_action: int = 4
    
    def get_total_dim(self) -> int:
        """Calculate total observation dimension"""
        return sum([
            self.hour_of_day, self.minute_of_hour, self.day_of_week, self.is_weekend,
            self.pv_current, self.pv_forecast, self.pv_history,
            self.wt_current, self.wt_forecast, self.wt_history,
            self.load_current, self.load_forecast, self.load_history,
            self.battery_soc, self.battery_soh, self.battery_temperature,
            self.battery_max_charge_rate, self.battery_max_discharge_rate, self.battery_throughput,
            self.grid_price_current, self.grid_price_forecast, self.grid_import_limit, self.grid_export_limit,
            self.ev_connected_count, self.ev_total_energy_needed, self.ev_avg_deadline_steps,
            self.ev_total_max_charge_rate, self.ev_earliest_deadline,
            self.inverter_temperature, self.transformer_load_factor, self.grid_voltage_deviation,
            self.recent_battery_actions, self.recent_grid_action, self.recent_ev_action
        ])

OBS_SPACE = ObservationSpaceConfig()

# ===== ACTION SPACE CONFIGURATION =====
@dataclass
class ActionSpaceConfig:
    """Defines the action space structure"""
    # Battery power setpoints (per battery, normalized to [-1, 1])
    battery_power: int = len(BATTERIES)  # 2 batteries
    
    # Grid import/export setpoint (normalized to [-1, 1])
    grid_power: int = 1
    
    # EV charging allocation (normalized to [0, 1])
    ev_charging_power: int = 1  # Aggregate EV charging power
    
    # Optional: renewable curtailment (normalized to [0, 1])
    renewable_curtailment: int = 1
    
    def get_total_dim(self) -> int:
        """Calculate total action dimension"""
        return self.battery_power + self.grid_power + self.ev_charging_power + self.renewable_curtailment

ACTION_SPACE = ActionSpaceConfig()

# ===== REWARD FUNCTION CONFIGURATION =====
@dataclass
class RewardConfig:
    """Reward function weights and parameters (Indian Context - All in ₹)"""
    # Composite reward: r_t = -(cost_t + α*emissions_t + β*degradation_t + γ*reliability_penalty_t)
    
    alpha: float = 4.15  # Emission weight (₹/kg CO2) - 0.05 USD * 83
    beta: float = 0.5  # Degradation weight
    gamma: float = 100.0  # Reliability penalty weight (MUST BE LARGE)
    
    # Cost parameters (Indian power market)
    revenue_export_multiplier: float = 0.75  # Export price as fraction of import (lower in India)
    
    # Reliability penalties (in ₹)
    unmet_demand_penalty_per_kwh: float = 830.0  # ₹/kWh for unmet demand (10 USD * 83)
    safety_violation_penalty: float = 8300.0  # Large penalty for safety violations (100 USD * 83)
    
    # Degradation modeling
    degradation_exponent: float = 1.1  # kWh^1.1 for throughput cost
    
    # Normalization (for stable training)
    cost_scale: float = 0.01  # Scale costs to reasonable range
    emission_scale: float = 1.0
    
    def calculate_reward(self, cost: float, emissions: float, degradation: float, 
                        reliability_penalty: float) -> float:
        """Calculate composite reward"""
        return -(cost * self.cost_scale + 
                self.alpha * emissions * self.emission_scale + 
                self.beta * degradation + 
                self.gamma * reliability_penalty)

REWARD = RewardConfig()

# ===== SAFETY CONSTRAINTS =====
@dataclass
class SafetyConfig:
    """Safety constraint parameters"""
    # Battery constraints
    soc_violation_threshold: float = 0.02  # Allow 2% temporary violation
    power_violation_threshold: float = 0.05  # Allow 5% power violation
    
    # Grid constraints
    grid_power_violation_threshold: float = 0.1  # 10% over limit
    
    # EV constraints
    ev_max_current_per_charger: float = 80  # Amps (DC fast charging limit)
    
    # Temperature limits
    battery_temp_max: float = 45.0  # °C
    inverter_temp_max: float = 80.0  # °C
    
    # Override penalty
    safety_override_penalty: float = 100.0  # Penalty when supervisor clips action
    
    # Clipping method
    use_hard_clipping: bool = True  # Hard clip or soft penalty

SAFETY = SafetyConfig()

# ===== TRAINING CONFIGURATION =====
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Algorithm selection
    algorithm: str = "PPO"  # ["PPO", "SAC", "TD3"]
    
    # PPO specific
    ppo_clip_ratio: float = 0.2
    ppo_target_kl: float = 0.01
    ppo_epochs: int = 10
    ppo_batch_size: int = 64
    
    # SAC specific
    sac_alpha: float = 0.2  # Entropy temperature
    sac_tau: float = 0.005  # Soft update parameter
    
    # General
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99  # Discount factor
    buffer_size: int = 100000
    batch_size: int = 256
    
    # Domain randomization
    use_domain_randomization: bool = True
    demand_variation: float = 0.2  # ±20% demand variation
    price_variation: float = 0.15  # ±15% price variation
    weather_variation: float = 0.25  # ±25% renewable variation
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"name": "Stage 1: No EVs, deterministic", "episodes": 500, "enable_evs": False, "forecast_noise": 0.0},
        {"name": "Stage 2: Add forecast noise", "episodes": 500, "enable_evs": False, "forecast_noise": 0.1},
        {"name": "Stage 3: Add EVs, low complexity", "episodes": 1000, "enable_evs": True, "max_evs": 3, "forecast_noise": 0.1},
        {"name": "Stage 4: Full complexity", "episodes": 2000, "enable_evs": True, "max_evs": 10, "forecast_noise": 0.15},
    ])
    
    # Logging
    log_interval: int = 10  # Log every N episodes
    save_interval: int = 100  # Save model every N episodes
    eval_interval: int = 50  # Evaluate every N episodes
    eval_episodes: int = 10  # Number of evaluation episodes

TRAINING = TrainingConfig()

# ===== EVALUATION METRICS =====
@dataclass
class EvaluationMetrics:
    """Metrics to track during evaluation"""
    total_cost: float = 0.0
    total_emissions: float = 0.0
    unmet_demand_events: int = 0
    unmet_demand_duration_steps: int = 0
    unmet_demand_energy_kwh: float = 0.0
    safety_overrides: int = 0
    
    # Battery metrics
    avg_depth_of_discharge: List[float] = field(default_factory=list)
    cumulative_throughput: List[float] = field(default_factory=list)
    estimated_cycle_life_used: List[float] = field(default_factory=list)
    
    # EV metrics
    ev_charging_success_rate: float = 0.0
    ev_avg_final_soc: float = 0.0
    
    # Grid metrics
    peak_import_kw: float = 0.0
    peak_export_kw: float = 0.0
    grid_energy_imported_kwh: float = 0.0
    grid_energy_exported_kwh: float = 0.0

# ===== EXPLAINABILITY CONFIGURATION =====
@dataclass
class ExplainabilityConfig:
    """Configuration for explainable AI features"""
    enable_explanations: bool = True
    explanation_detail_level: str = "medium"  # ["low", "medium", "high"]
    
    # Explanation templates
    action_explanation_template: str = (
        "{action_type} {value:.1f} kW to {reason} during {time_context}; "
        "{impact_description} given {forecast_context}"
    )
    
    # Impact estimation
    estimate_cost_delta: bool = True
    estimate_emission_delta: bool = True
    compare_to_baseline: bool = True

EXPLAINABILITY = ExplainabilityConfig()

# ===== SUMMARY =====
def print_config_summary():
    """Print configuration summary"""
    print("="*60)
    print("MICROGRID EMS RL ENVIRONMENT CONFIGURATION")
    print("="*60)
    print(f"Episode Duration: {EPISODE_HOURS} hours")
    print(f"Decision Interval: {DECISION_INTERVAL_MINUTES} minutes")
    print(f"Steps per Episode: {STEPS_PER_EPISODE}")
    print(f"\nObservation Space Dimension: {OBS_SPACE.get_total_dim()}")
    print(f"Action Space Dimension: {ACTION_SPACE.get_total_dim()}")
    print(f"\nBatteries: {len(BATTERIES)}")
    for bat in BATTERIES:
        print(f"  - {bat.name}: {bat.capacity_kwh} kWh, ±{bat.max_charge_kw} kW")
    print(f"\nEV Chargers: {len(EV_CHARGERS)}")
    print(f"Max Concurrent EVs: {EV_FLEET.max_concurrent_evs}")
    print(f"\nRenewable Capacity: {RENEWABLE.pv_total_capacity_kw} kW PV + {RENEWABLE.wt_total_capacity_kw} kW Wind")
    print(f"Grid Limits: Import {GRID.max_import_kw} kW, Export {GRID.max_export_kw} kW")
    print(f"\nReward Weights: alpha={REWARD.alpha}, beta={REWARD.beta}, gamma={REWARD.gamma}")
    print(f"Training Algorithm: {TRAINING.algorithm}")
    print("="*60)

if __name__ == "__main__":
    print_config_summary()
