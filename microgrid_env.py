"""
Microgrid Energy Management System Gym Environment (INDIAN CONTEXT)
Comprehensive RL environment with:
- Battery degradation modeling
- EV fleet charging
- Emissions tracking (Indian grid emission factors)
- Safety constraints
- Composite reward function (All costs in Indian Rupees ₹)
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import asdict

from env_config import (
    EPISODE_HOURS, STEPS_PER_EPISODE, HOURS_PER_STEP,
    BATTERIES, GRID, RENEWABLE, LOAD, EV_FLEET, EV_CHARGERS,
    OBS_SPACE, ACTION_SPACE, REWARD, SAFETY, EXPLAINABILITY
)
from battery_degradation import BatteryDegradationModel, BatteryThermalModel
from ev_simulator import EVFleetSimulator
from safety_supervisor import SafetySupervisor


class MicrogridEMSEnv(gym.Env):
    """
    Microgrid Energy Management System Environment
    
    Observation Space:
        - Temporal features (time-of-day, day-of-week, etc.)
        - Renewable generation (current + forecast + history)
        - Load demand (current + forecast + history)
        - Battery status (SoC, SoH, temperature, limits)
        - Grid status (price, limits, availability)
        - EV fleet status (count, energy needed, deadlines)
        - Component health indices
        - Recent actions
    
    Action Space:
        - Battery power setpoints (continuous, per battery)
        - Grid import/export setpoint (continuous)
        - EV charging power (continuous)
        - Renewable curtailment (continuous, optional)
    
    Reward:
        Composite: -(cost + α*emissions + β*degradation + γ*reliability_penalty)
    """
    
    metadata = {'render.modes': ['human', 'text']}
    
    def __init__(
        self,
        pv_profile: pd.DataFrame,
        wt_profile: pd.DataFrame,
        load_profile: pd.DataFrame,
        price_profile: pd.DataFrame,
        enable_evs: bool = True,
        enable_degradation: bool = True,
        enable_emissions: bool = True,
        forecast_noise_std: float = 0.1,
        random_seed: Optional[int] = None
    ):
        super().__init__()
        
        # Data profiles
        self.pv_profile = pv_profile
        self.wt_profile = wt_profile
        self.load_profile = load_profile
        self.price_profile = price_profile
        
        # Configuration flags
        self.enable_evs = enable_evs
        self.enable_degradation = enable_degradation
        self.enable_emissions = enable_emissions
        self.forecast_noise_std = forecast_noise_std
        
        # Random state
        self.rng = np.random.RandomState(random_seed)
        
        # Define observation and action spaces
        obs_dim = OBS_SPACE.get_total_dim()
        action_dim = ACTION_SPACE.get_total_dim()
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Initialize subsystems
        self.battery_models = [
            BatteryDegradationModel(bat_config) for bat_config in BATTERIES
        ]
        self.thermal_models = [
            BatteryThermalModel(bat_config) for bat_config in BATTERIES
        ]
        self.ev_simulator = EVFleetSimulator(config=EV_FLEET, random_seed=random_seed)
        self.safety_supervisor = SafetySupervisor(config=SAFETY)
        
        # Episode state
        self.current_step = 0
        self.episode_start_idx = 0
        self.battery_socs = [bat.initial_soc for bat in BATTERIES]
        self.battery_soh = [bat.soh_initial for bat in BATTERIES]
        self.battery_temps = [bat.temperature_nominal for bat in BATTERIES]
        self.action_history = []
        
        # Metrics tracking
        self.episode_metrics = self._init_episode_metrics()
        
        # Explanation buffer
        self.last_explanation = ""
        
    def reset(self, episode_start_idx: Optional[int] = None) -> np.ndarray:
        """Reset environment for new episode"""
        # Determine episode start
        if episode_start_idx is not None:
            self.episode_start_idx = episode_start_idx
        else:
            # Random start (ensure we have enough data)
            max_start = len(self.pv_profile) - STEPS_PER_EPISODE - RENEWABLE.forecast_horizon_steps
            self.episode_start_idx = self.rng.randint(0, max_start)
        
        self.current_step = 0
        
        # Reset batteries
        for i, (model, thermal, config) in enumerate(zip(
            self.battery_models, self.thermal_models, BATTERIES
        )):
            model.reset()
            thermal.reset()
            self.battery_socs[i] = config.initial_soc
            self.battery_soh[i] = config.soh_initial
            self.battery_temps[i] = config.temperature_nominal
        
        # Reset EV simulator
        if self.enable_evs:
            self.ev_simulator.reset(random_seed=self.rng.randint(0, 10000))
        
        # Reset safety supervisor
        self.safety_supervisor.reset()
        
        # Reset action history
        self.action_history = []
        
        # Reset metrics
        self.episode_metrics = self._init_episode_metrics()
        
        # Generate EV arrival pattern for this episode
        if self.enable_evs:
            self.ev_arrival_pattern = self.ev_simulator.generate_arrival_pattern(STEPS_PER_EPISODE)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one timestep"""
        # Parse and denormalize actions
        raw_actions = self._parse_actions(action)
        
        # Get current system state
        battery_states = self._get_battery_states()
        grid_state = self._get_grid_state()
        ev_state = self._get_ev_state()
        component_health = self._get_component_health()
        
        # Safety check and clip actions
        safe_actions, safety_penalty = self.safety_supervisor.check_and_clip_actions(
            raw_actions, battery_states, grid_state, ev_state, component_health, self.current_step
        )
        
        # Execute actions and update system
        step_info = self._execute_actions(safe_actions)
        
        # Calculate reward
        reward = self._calculate_reward(step_info, safety_penalty)
        
        # Update metrics
        self._update_metrics(step_info, reward, safety_penalty)
        
        # Store action history
        self.action_history.append(safe_actions)
        if len(self.action_history) > 4:
            self.action_history.pop(0)
        
        # Update EV fleet
        if self.enable_evs:
            self.ev_simulator.step(self.current_step, STEPS_PER_EPISODE, self.ev_arrival_pattern)
        
        # Generate explanation
        if EXPLAINABILITY.enable_explanations:
            self.last_explanation = self._generate_explanation(safe_actions, step_info)
        
        # Advance time
        self.current_step += 1
        done = (self.current_step >= STEPS_PER_EPISODE)
        
        # Get next observation
        obs = self._get_observation()
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'cost': step_info['cost'],
            'emissions': step_info['emissions'],
            'degradation_cost': step_info['degradation_cost'],
            'reliability_penalty': step_info['reliability_penalty'],
            'safety_penalty': safety_penalty,
            'explanation': self.last_explanation,
            'battery_socs': self.battery_socs.copy(),
            'battery_soh': self.battery_soh.copy(),
            'unmet_demand': step_info.get('unmet_demand', 0.0),
            'safety_overrides': self.safety_supervisor.total_overrides
        }
        
        if done:
            info['episode_metrics'] = self.episode_metrics
            info['safety_report'] = self.safety_supervisor.get_violation_report()
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        obs = []
        
        # Get absolute timestep in data
        abs_step = self.episode_start_idx + self.current_step
        
        # Temporal features
        hour = (abs_step * HOURS_PER_STEP) % 24
        minute = ((abs_step * HOURS_PER_STEP * 60) % 60)
        # Assuming we have datetime info in profile
        day_of_week = 0  # Placeholder - extract from data if available
        is_weekend = 0  # Placeholder
        
        obs.extend([hour / 24.0, minute / 60.0, day_of_week / 7.0, is_weekend])
        
        # Renewable generation (PV + Wind)
        pv_current = self._get_pv_generation(abs_step)
        pv_forecast = [self._get_pv_generation(abs_step + i + 1, add_noise=True) 
                      for i in range(RENEWABLE.forecast_horizon_steps)]
        pv_history = [self._get_pv_generation(abs_step - i) 
                     for i in range(1, 5)] if self.current_step >= 4 else [pv_current] * 4
        
        wt_current = self._get_wind_generation(abs_step)
        wt_forecast = [self._get_wind_generation(abs_step + i + 1, add_noise=True)
                      for i in range(RENEWABLE.forecast_horizon_steps)]
        wt_history = [self._get_wind_generation(abs_step - i)
                     for i in range(1, 5)] if self.current_step >= 4 else [wt_current] * 4
        
        obs.extend([pv_current] + pv_forecast + pv_history)
        obs.extend([wt_current] + wt_forecast + wt_history)
        
        # Load demand
        load_current = self._get_load_demand(abs_step)
        load_forecast = [self._get_load_demand(abs_step + i + 1, add_noise=True)
                        for i in range(RENEWABLE.forecast_horizon_steps)]
        load_history = [self._get_load_demand(abs_step - i)
                       for i in range(1, 5)] if self.current_step >= 4 else [load_current] * 4
        
        obs.extend([load_current] + load_forecast + load_history)
        
        # Battery status
        for i, config in enumerate(BATTERIES):
            max_charge, max_discharge = self.battery_models[i].get_adjusted_power_limits()
            obs.extend([
                self.battery_socs[i],
                self.battery_soh[i],
                self.battery_temps[i] / 50.0,  # Normalize temperature
                max_charge / config.max_charge_kw,
                max_discharge / config.max_discharge_kw,
                self.battery_models[i].cumulative_throughput_kwh / (config.capacity_kwh * 1000)
            ])
        
        # Grid status
        price_current = self._get_grid_price(abs_step)
        price_forecast = [self._get_grid_price(abs_step + i + 1)
                         for i in range(RENEWABLE.forecast_horizon_steps)]
        
        obs.extend([price_current / 100.0])  # Normalize price
        obs.extend([p / 100.0 for p in price_forecast])
        obs.extend([
            GRID.max_import_kw / 10000.0,  # Normalize
            GRID.max_export_kw / 10000.0
        ])
        
        # EV fleet status
        if self.enable_evs:
            ev_state = self.ev_simulator.get_fleet_state()
            obs.extend([
                ev_state['num_connected'] / EV_FLEET.max_concurrent_evs,
                ev_state['total_energy_needed'] / 1000.0,  # Normalize to MWh
                ev_state['avg_deadline_steps'] / STEPS_PER_EPISODE,
                ev_state['total_max_charge_rate'] / 500.0,  # Normalize
                ev_state['earliest_deadline'] / STEPS_PER_EPISODE
            ])
        else:
            obs.extend([0.0] * 5)
        
        # Component health (simplified)
        obs.extend([
            np.mean(self.battery_temps) / 50.0,  # Inverter temp proxy
            0.5,  # Transformer load factor placeholder
            0.0   # Grid voltage deviation placeholder
        ])
        
        # Recent actions (last 4 steps)
        recent_actions_flat = []
        for _ in range(4):
            if len(self.action_history) >= _+1:
                act = self.action_history[-(1+_)]
                recent_actions_flat.extend([
                    act['battery_power'][0] / BATTERIES[0].max_charge_kw,
                    act['battery_power'][1] / BATTERIES[1].max_charge_kw,
                    act['grid_power'] / GRID.max_import_kw,
                    act.get('ev_charging_power', 0.0) / 500.0
                ])
            else:
                recent_actions_flat.extend([0.0, 0.0, 0.0, 0.0])
        
        obs.extend(recent_actions_flat)
        
        return np.array(obs, dtype=np.float32)
    
    def _parse_actions(self, action: np.ndarray) -> Dict:
        """Parse and denormalize raw actions"""
        idx = 0
        parsed = {}
        
        # Battery powers (normalize from [-1, 1] to [min, max] power)
        battery_powers = []
        for bat_config in BATTERIES:
            normalized = action[idx]
            if normalized >= 0:  # Charging
                power = normalized * bat_config.max_charge_kw
            else:  # Discharging
                power = normalized * bat_config.max_discharge_kw
            battery_powers.append(power)
            idx += 1
        parsed['battery_power'] = battery_powers
        
        # Grid power (normalize from [-1, 1] to [-max_export, max_import])
        normalized = action[idx]
        if normalized >= 0:  # Import
            parsed['grid_power'] = normalized * GRID.max_import_kw
        else:  # Export
            parsed['grid_power'] = -normalized * GRID.max_export_kw
        idx += 1
        
        # EV charging power (normalize from [-1, 1] to [0, max_rate])
        # Take absolute value and clip to [0, 1]
        normalized = np.clip(action[idx], 0, 1)
        parsed['ev_charging_power'] = normalized * sum(c.max_power_kw * c.num_ports for c in EV_CHARGERS)
        idx += 1
        
        # Renewable curtailment (normalize from [-1, 1] to [0, 1])
        parsed['renewable_curtailment'] = (action[idx] + 1) / 2.0
        
        return parsed
    
    def _execute_actions(self, actions: Dict) -> Dict:
        """Execute actions and return step information"""
        abs_step = self.episode_start_idx + self.current_step
        
        # Get renewable generation
        pv_gen_kw = self._get_pv_generation(abs_step)
        wt_gen_kw = self._get_wind_generation(abs_step)
        renewable_curtailment = actions.get('renewable_curtailment', 0.0)
        available_renewable_kw = (pv_gen_kw + wt_gen_kw) * (1 - renewable_curtailment)
        
        # Get load demand
        load_demand_kw = self._get_load_demand(abs_step)
        
        # EV charging
        ev_charging_kw = 0.0
        if self.enable_evs and self.ev_simulator.connected_evs:
            allocations = self.ev_simulator.allocate_charging_power(
                actions['ev_charging_power'],
                strategy='earliest_deadline'
            )
            ev_energy_from_grid = self.ev_simulator.charge_fleet(allocations, HOURS_PER_STEP)
            ev_charging_kw = ev_energy_from_grid / HOURS_PER_STEP
        
        # Total demand
        total_demand_kw = load_demand_kw + ev_charging_kw
        
        # Battery actions
        battery_power_kw = actions['battery_power']  # Positive = charging
        total_battery_charge_kw = sum(max(0, p) for p in battery_power_kw)
        total_battery_discharge_kw = sum(max(0, -p) for p in battery_power_kw)
        
        # Power balance
        # Supply: renewable + battery discharge + grid import
        # Demand: load + EV + battery charge
        supply_kw = available_renewable_kw + total_battery_discharge_kw
        demand_kw = total_demand_kw + total_battery_charge_kw
        
        # Grid power (positive = import, negative = export)
        grid_power_kw = demand_kw - supply_kw
        grid_import_kw = max(0, grid_power_kw)
        grid_export_kw = max(0, -grid_power_kw)
        
        # Check for unmet demand
        unmet_demand_kw = 0.0
        if grid_import_kw > GRID.max_import_kw:
            unmet_demand_kw = grid_import_kw - GRID.max_import_kw
            grid_import_kw = GRID.max_import_kw
        
        # Update battery states
        degradation_costs = []
        for i, power_kw in enumerate(battery_power_kw):
            soc_before = self.battery_socs[i]
            
            # Update SoC
            energy_kwh = power_kw * HOURS_PER_STEP
            capacity_kwh = BATTERIES[i].capacity_kwh * self.battery_soh[i]
            soc_delta = energy_kwh / capacity_kwh
            self.battery_socs[i] = np.clip(self.battery_socs[i] + soc_delta, 
                                          BATTERIES[i].soc_min, BATTERIES[i].soc_max)
            
            # Update temperature
            self.battery_temps[i] = self.thermal_models[i].update(
                power_kw, self.battery_socs[i], HOURS_PER_STEP
            )
            
            # Update degradation
            if self.enable_degradation:
                deg_cost, new_soh = self.battery_models[i].update(
                    abs(energy_kwh), soc_before, self.battery_socs[i], 
                    self.battery_temps[i], HOURS_PER_STEP
                )
                self.battery_soh[i] = new_soh
                degradation_costs.append(deg_cost)
            else:
                degradation_costs.append(0.0)
        
        # Calculate cost
        price = self._get_grid_price(abs_step)
        import_cost = grid_import_kw * HOURS_PER_STEP * price
        export_revenue = grid_export_kw * HOURS_PER_STEP * price * REWARD.revenue_export_multiplier
        energy_cost = import_cost - export_revenue
        
        # Calculate emissions
        if self.enable_emissions:
            hour = int((abs_step * HOURS_PER_STEP) % 24)
            if hour in GRID.peak_hours:
                emission_factor = GRID.emission_factor_peak
            elif hour in GRID.offpeak_hours:
                emission_factor = GRID.emission_factor_offpeak
            else:
                emission_factor = GRID.emission_factor_base
            
            emissions_kg = grid_import_kw * HOURS_PER_STEP * emission_factor
        else:
            emissions_kg = 0.0
        
        # Calculate reliability penalty
        reliability_penalty = 0.0
        if unmet_demand_kw > 0:
            reliability_penalty = unmet_demand_kw * HOURS_PER_STEP * REWARD.unmet_demand_penalty_per_kwh
        
        return {
            'cost': energy_cost,
            'emissions': emissions_kg,
            'degradation_cost': sum(degradation_costs),
            'reliability_penalty': reliability_penalty,
            'unmet_demand': unmet_demand_kw,
            'grid_import_kw': grid_import_kw,
            'grid_export_kw': grid_export_kw,
            'renewable_kw': available_renewable_kw,
            'load_kw': load_demand_kw,
            'ev_charging_kw': ev_charging_kw,
            'battery_power_kw': battery_power_kw,
            'price': price
        }
    
    def _calculate_reward(self, step_info: Dict, safety_penalty: float) -> float:
        """Calculate composite reward"""
        reward = REWARD.calculate_reward(
            cost=step_info['cost'],
            emissions=step_info['emissions'],
            degradation=step_info['degradation_cost'],
            reliability_penalty=step_info['reliability_penalty']
        )
        
        # Subtract safety penalty
        reward -= safety_penalty
        
        return reward
    
    def _get_pv_generation(self, abs_step: int, add_noise: bool = False) -> float:
        """Get PV generation at timestep (kW)"""
        if abs_step < 0 or abs_step >= len(self.pv_profile):
            return 0.0
        
        # Sum all PV columns
        pv_cols = [col for col in self.pv_profile.columns if 'pv' in col.lower()]
        pv_total = self.pv_profile.iloc[abs_step][pv_cols].sum() if pv_cols else 0.0
        
        if add_noise:
            noise = self.rng.normal(0, self.forecast_noise_std * pv_total)
            pv_total = max(0, pv_total + noise)
        
        return pv_total
    
    def _get_wind_generation(self, abs_step: int, add_noise: bool = False) -> float:
        """Get wind generation at timestep (kW)"""
        if abs_step < 0 or abs_step >= len(self.wt_profile):
            return 0.0
        
        wt_cols = [col for col in self.wt_profile.columns if 'wt' in col.lower()]
        wt_total = self.wt_profile.iloc[abs_step][wt_cols].sum() if wt_cols else 0.0
        
        if add_noise:
            noise = self.rng.normal(0, self.forecast_noise_std * wt_total)
            wt_total = max(0, wt_total + noise)
        
        return wt_total
    
    def _get_load_demand(self, abs_step: int, add_noise: bool = False) -> float:
        """Get load demand at timestep (kW)"""
        if abs_step < 0 or abs_step >= len(self.load_profile):
            return 0.0
        
        load_cols = [col for col in self.load_profile.columns if 'load' in col.lower()]
        load_total = self.load_profile.iloc[abs_step][load_cols].sum() if load_cols else 0.0
        
        if add_noise:
            noise = self.rng.normal(0, self.forecast_noise_std * load_total)
            load_total = max(0, load_total + noise)
        
        return load_total
    
    def _get_grid_price(self, abs_step: int) -> float:
        """Get grid price at timestep ($/kWh)"""
        if abs_step < 0 or abs_step >= len(self.price_profile):
            return 0.1  # Default price
        
        price_col = [col for col in self.price_profile.columns if 'price' in col.lower()][0]
        return self.price_profile.iloc[abs_step][price_col]
    
    def _get_battery_states(self) -> List[Dict]:
        """Get current battery states"""
        states = []
        for i, config in enumerate(BATTERIES):
            max_charge, max_discharge = self.battery_models[i].get_adjusted_power_limits()
            states.append({
                'soc': self.battery_socs[i],
                'soh': self.battery_soh[i],
                'temperature': self.battery_temps[i],
                'max_charge_kw': max_charge,
                'max_discharge_kw': max_discharge
            })
        return states
    
    def _get_grid_state(self) -> Dict:
        """Get current grid state"""
        return {
            'max_import_kw': GRID.max_import_kw,
            'max_export_kw': GRID.max_export_kw,
            'available': True
        }
    
    def _get_ev_state(self) -> Dict:
        """Get current EV fleet state"""
        if self.enable_evs:
            return self.ev_simulator.get_fleet_state()
        return {
            'num_connected': 0,
            'total_energy_needed': 0.0,
            'total_max_charge_rate': 0.0
        }
    
    def _get_component_health(self) -> Dict:
        """Get component health indices"""
        return {
            'battery_temperatures': self.battery_temps.copy(),
            'inverter_temperature': np.mean(self.battery_temps) + 10,  # Simplified
            'grid_voltage_deviation': 0.0
        }
    
    def _generate_explanation(self, actions: Dict, step_info: Dict) -> str:
        """Generate human-readable explanation of actions"""
        if not EXPLAINABILITY.enable_explanations:
            return ""
        
        abs_step = self.episode_start_idx + self.current_step
        hour = int((abs_step * HOURS_PER_STEP) % 24)
        
        explanations = []
        
        # Battery actions
        for i, power in enumerate(actions['battery_power']):
            if abs(power) > 10:  # Only explain significant actions
                action_type = "Charge" if power > 0 else "Discharge"
                bat_name = BATTERIES[i].name
                reason = self._infer_action_reason(power, step_info, i)
                explanations.append(
                    f"{action_type} {bat_name} at {abs(power):.0f} kW to {reason} "
                    f"at hour {hour}:00"
                )
        
        # Grid action
        if step_info['grid_import_kw'] > 100:
            explanations.append(
                f"Import {step_info['grid_import_kw']:.0f} kW from grid "
                f"at price ${step_info['price']:.3f}/kWh"
            )
        elif step_info['grid_export_kw'] > 100:
            explanations.append(
                f"Export {step_info['grid_export_kw']:.0f} kW to grid "
                f"for revenue"
            )
        
        # EV charging
        if step_info['ev_charging_kw'] > 10:
            ev_state = self._get_ev_state()
            explanations.append(
                f"Charge {ev_state['num_connected']} EVs with "
                f"{step_info['ev_charging_kw']:.0f} kW"
            )
        
        return "; ".join(explanations) if explanations else "Maintain current state"
    
    def _infer_action_reason(self, power: float, step_info: Dict, battery_idx: int) -> str:
        """Infer reason for battery action"""
        if power > 0:  # Charging
            if step_info['renewable_kw'] > step_info['load_kw']:
                return "store excess renewable energy"
            elif step_info['price'] < 0.08:
                return "take advantage of low grid price"
            else:
                return "prepare for peak demand period"
        else:  # Discharging
            if step_info['price'] > 0.15:
                return "reduce peak import cost"
            elif step_info['load_kw'] > step_info['renewable_kw']:
                return "meet demand deficit"
            else:
                return "provide grid support"
    
    def _init_episode_metrics(self) -> Dict:
        """Initialize episode metrics"""
        from env_config import EvaluationMetrics
        return asdict(EvaluationMetrics())
    
    def _update_metrics(self, step_info: Dict, reward: float, safety_penalty: float):
        """Update episode metrics"""
        self.episode_metrics['total_cost'] += step_info['cost']
        self.episode_metrics['total_emissions'] += step_info['emissions']
        
        if step_info['unmet_demand'] > 0:
            self.episode_metrics['unmet_demand_events'] += 1
            self.episode_metrics['unmet_demand_duration_steps'] += 1
            self.episode_metrics['unmet_demand_energy_kwh'] += step_info['unmet_demand'] * HOURS_PER_STEP
        
        self.episode_metrics['safety_overrides'] = self.safety_supervisor.total_overrides
        self.episode_metrics['grid_energy_imported_kwh'] += step_info['grid_import_kw'] * HOURS_PER_STEP
        self.episode_metrics['grid_energy_exported_kwh'] += step_info['grid_export_kw'] * HOURS_PER_STEP
        self.episode_metrics['peak_import_kw'] = max(
            self.episode_metrics['peak_import_kw'], step_info['grid_import_kw']
        )
        self.episode_metrics['peak_export_kw'] = max(
            self.episode_metrics['peak_export_kw'], step_info['grid_export_kw']
        )
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'text':
            print(f"Step {self.current_step}/{STEPS_PER_EPISODE}")
            print(f"Battery SoCs: {[f'{soc:.2f}' for soc in self.battery_socs]}")
            print(f"Battery SoH: {[f'{soh:.2f}' for soh in self.battery_soh]}")
            if self.enable_evs:
                ev_state = self.ev_simulator.get_fleet_state()
                print(f"EVs connected: {ev_state['num_connected']}")
            print(f"Last explanation: {self.last_explanation}")


if __name__ == "__main__":
    print("Microgrid EMS Environment initialized")
    print(f"Observation space: {OBS_SPACE.get_total_dim()} dimensions")
    print(f"Action space: {ACTION_SPACE.get_total_dim()} dimensions")
