"""
Safety Supervisor
Enforces hard constraints and clips unsafe actions
Logs violations and applies penalties
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass, field
from env_config import (
    BATTERIES, GRID, SAFETY, EV_CHARGERS,
    BatteryConfig, SafetyConfig
)


@dataclass
class SafetyViolation:
    """Records a safety violation"""
    timestep: int
    violation_type: str
    component: str
    value: float
    limit: float
    severity: str  # 'warning', 'critical'
    action_taken: str


class SafetySupervisor:
    """
    Enforces safety constraints on actions
    - Battery SoC limits
    - Battery power rate limits
    - Grid import/export limits
    - EV charger current limits
    - Temperature limits
    """
    
    def __init__(self, config: SafetyConfig = SAFETY):
        self.config = config
        
        # Violation tracking
        self.violations: List[SafetyViolation] = []
        self.violation_counts = {
            'battery_soc': 0,
            'battery_power': 0,
            'grid_power': 0,
            'ev_current': 0,
            'temperature': 0
        }
        
        # Statistics
        self.total_overrides = 0
        self.total_penalty = 0.0
        
    def reset(self):
        """Reset supervisor for new episode"""
        self.violations = []
        self.violation_counts = {k: 0 for k in self.violation_counts}
        self.total_overrides = 0
        self.total_penalty = 0.0
    
    def check_and_clip_actions(
        self,
        raw_actions: Dict[str, float],
        battery_states: List[Dict],
        grid_state: Dict,
        ev_state: Dict,
        component_health: Dict,
        timestep: int
    ) -> Tuple[Dict[str, float], float]:
        """
        Check all safety constraints and clip actions if needed
        
        Args:
            raw_actions: Dictionary of raw actions from agent
                - 'battery_power': List of power setpoints per battery (kW)
                - 'grid_power': Grid import/export setpoint (kW)
                - 'ev_charging_power': Total EV charging power (kW)
                - 'renewable_curtailment': Curtailment fraction [0, 1]
            battery_states: List of battery state dicts
            grid_state: Grid state dict
            ev_state: EV fleet state dict
            component_health: Component health indices
            timestep: Current timestep
            
        Returns:
            (safe_actions, penalty)
        """
        safe_actions = raw_actions.copy()
        total_penalty = 0.0
        
        # Check battery actions
        if 'battery_power' in raw_actions:
            safe_battery_actions, battery_penalty = self._check_battery_actions(
                raw_actions['battery_power'],
                battery_states,
                timestep
            )
            safe_actions['battery_power'] = safe_battery_actions
            total_penalty += battery_penalty
        
        # Check grid action
        if 'grid_power' in raw_actions:
            safe_grid_action, grid_penalty = self._check_grid_action(
                raw_actions['grid_power'],
                grid_state,
                timestep
            )
            safe_actions['grid_power'] = safe_grid_action
            total_penalty += grid_penalty
        
        # Check EV charging action
        if 'ev_charging_power' in raw_actions:
            safe_ev_action, ev_penalty = self._check_ev_action(
                raw_actions['ev_charging_power'],
                ev_state,
                timestep
            )
            safe_actions['ev_charging_power'] = safe_ev_action
            total_penalty += ev_penalty
        
        # Check temperature constraints
        temp_penalty = self._check_temperature_constraints(
            component_health,
            timestep
        )
        total_penalty += temp_penalty
        
        # Clip renewable curtailment
        if 'renewable_curtailment' in raw_actions:
            safe_actions['renewable_curtailment'] = np.clip(
                raw_actions['renewable_curtailment'], 0.0, 1.0
            )
        
        # Update statistics
        if total_penalty > 0:
            self.total_overrides += 1
            self.total_penalty += total_penalty
        
        return safe_actions, total_penalty
    
    def _check_battery_actions(
        self,
        battery_powers: List[float],
        battery_states: List[Dict],
        timestep: int
    ) -> Tuple[List[float], float]:
        """Check and clip battery power actions"""
        safe_powers = []
        penalty = 0.0
        
        for i, (power, state, config) in enumerate(zip(battery_powers, battery_states, BATTERIES)):
            soc = state['soc']
            soh = state.get('soh', 1.0)
            temperature = state.get('temperature', 25.0)
            
            # Get current power limits (may be reduced due to SoH/temperature)
            max_charge = state.get('max_charge_kw', config.max_charge_kw)
            max_discharge = state.get('max_discharge_kw', config.max_discharge_kw)
            
            # Clip power to rate limits
            safe_power = np.clip(power, -max_discharge, max_charge)
            
            # Check if power would violate SoC limits
            # Estimate next SoC
            from env_config import HOURS_PER_STEP
            energy_delta_kwh = safe_power * HOURS_PER_STEP
            capacity_kwh = config.capacity_kwh * soh
            soc_delta = energy_delta_kwh / capacity_kwh
            next_soc = soc + soc_delta
            
            # If next SoC would violate limits, reduce power
            if next_soc > config.soc_max + self.config.soc_violation_threshold:
                # Reduce charging power
                max_safe_soc_delta = config.soc_max - soc
                max_safe_energy = max_safe_soc_delta * capacity_kwh
                max_safe_power = max_safe_energy / HOURS_PER_STEP
                safe_power = min(safe_power, max_safe_power)
                
                # Log violation
                self._log_violation(
                    timestep=timestep,
                    violation_type='battery_soc_high',
                    component=f'Battery_{i}',
                    value=next_soc,
                    limit=config.soc_max,
                    severity='warning',
                    action_taken=f'Reduced charge power to {safe_power:.1f} kW'
                )
                penalty += self.config.safety_override_penalty * 0.5
            
            elif next_soc < config.soc_min - self.config.soc_violation_threshold:
                # Reduce discharging power
                max_safe_soc_delta = soc - config.soc_min
                max_safe_energy = max_safe_soc_delta * capacity_kwh
                max_safe_power = -max_safe_energy / HOURS_PER_STEP
                safe_power = max(safe_power, max_safe_power)
                
                # Log violation
                self._log_violation(
                    timestep=timestep,
                    violation_type='battery_soc_low',
                    component=f'Battery_{i}',
                    value=next_soc,
                    limit=config.soc_min,
                    severity='warning',
                    action_taken=f'Reduced discharge power to {safe_power:.1f} kW'
                )
                penalty += self.config.safety_override_penalty * 0.5
            
            # Check if we had to clip power
            if abs(safe_power - power) > 0.1:  # More than 0.1 kW difference
                if abs(safe_power) < abs(power):  # Power was reduced
                    self.violation_counts['battery_power'] += 1
                    penalty += self.config.safety_override_penalty * 0.3
            
            safe_powers.append(safe_power)
        
        return safe_powers, penalty
    
    def _check_grid_action(
        self,
        grid_power: float,
        grid_state: Dict,
        timestep: int
    ) -> Tuple[float, float]:
        """Check and clip grid power action"""
        penalty = 0.0
        
        # Get current grid limits (may change dynamically)
        max_import = grid_state.get('max_import_kw', GRID.max_import_kw)
        max_export = grid_state.get('max_export_kw', GRID.max_export_kw)
        
        # Clip to limits (positive = import, negative = export)
        safe_power = np.clip(grid_power, -max_export, max_import)
        
        # Check if we had to clip
        if abs(safe_power - grid_power) > max_import * self.config.grid_power_violation_threshold:
            self._log_violation(
                timestep=timestep,
                violation_type='grid_power_limit',
                component='Grid',
                value=grid_power,
                limit=max_import if grid_power > 0 else -max_export,
                severity='warning',
                action_taken=f'Clipped to {safe_power:.1f} kW'
            )
            self.violation_counts['grid_power'] += 1
            penalty = self.config.safety_override_penalty * 0.5
        
        return safe_power, penalty
    
    def _check_ev_action(
        self,
        ev_charging_power: float,
        ev_state: Dict,
        timestep: int
    ) -> Tuple[float, float]:
        """Check and clip EV charging action"""
        penalty = 0.0
        
        # Get maximum possible EV charging power
        max_ev_power = ev_state.get('total_max_charge_rate', 0.0)
        
        # Clip to limits (must be non-negative)
        safe_power = np.clip(ev_charging_power, 0.0, max_ev_power)
        
        # Check charger current limits (simplified)
        # In reality, would check per-charger limits
        total_charger_capacity = sum(c.max_power_kw * c.num_ports for c in EV_CHARGERS)
        safe_power = min(safe_power, total_charger_capacity)
        
        # Check if we had to clip
        if abs(safe_power - ev_charging_power) > 0.1:
            self._log_violation(
                timestep=timestep,
                violation_type='ev_power_limit',
                component='EV_Chargers',
                value=ev_charging_power,
                limit=max_ev_power,
                severity='info',
                action_taken=f'Clipped to {safe_power:.1f} kW'
            )
            self.violation_counts['ev_current'] += 1
            penalty = self.config.safety_override_penalty * 0.2
        
        return safe_power, penalty
    
    def _check_temperature_constraints(
        self,
        component_health: Dict,
        timestep: int
    ) -> float:
        """Check temperature constraints"""
        penalty = 0.0
        
        # Check battery temperatures
        battery_temps = component_health.get('battery_temperatures', [])
        for i, temp in enumerate(battery_temps):
            if temp > self.config.battery_temp_max:
                self._log_violation(
                    timestep=timestep,
                    violation_type='battery_temperature',
                    component=f'Battery_{i}',
                    value=temp,
                    limit=self.config.battery_temp_max,
                    severity='critical',
                    action_taken='Temperature too high - reduce power'
                )
                self.violation_counts['temperature'] += 1
                penalty += self.config.safety_override_penalty
        
        # Check inverter temperature
        inverter_temp = component_health.get('inverter_temperature', 25.0)
        if inverter_temp > self.config.inverter_temp_max:
            self._log_violation(
                timestep=timestep,
                violation_type='inverter_temperature',
                component='Inverter',
                value=inverter_temp,
                limit=self.config.inverter_temp_max,
                severity='critical',
                action_taken='Temperature too high - reduce power'
            )
            self.violation_counts['temperature'] += 1
            penalty += self.config.safety_override_penalty
        
        return penalty
    
    def _log_violation(self, timestep: int, violation_type: str, component: str,
                      value: float, limit: float, severity: str, action_taken: str):
        """Log a safety violation"""
        violation = SafetyViolation(
            timestep=timestep,
            violation_type=violation_type,
            component=component,
            value=value,
            limit=limit,
            severity=severity,
            action_taken=action_taken
        )
        self.violations.append(violation)
    
    def get_statistics(self) -> Dict:
        """Get safety statistics"""
        return {
            'total_overrides': self.total_overrides,
            'total_penalty': self.total_penalty,
            'violation_counts': self.violation_counts.copy(),
            'num_violations': len(self.violations),
            'critical_violations': sum(1 for v in self.violations if v.severity == 'critical')
        }
    
    def get_violation_report(self) -> str:
        """Generate a human-readable violation report"""
        if not self.violations:
            return "No safety violations recorded."
        
        report = f"\nSafety Violation Report\n{'='*60}\n"
        report += f"Total Violations: {len(self.violations)}\n"
        report += f"Total Overrides: {self.total_overrides}\n"
        report += f"Total Penalty: ${self.total_penalty:.2f}\n\n"
        
        report += "Violation Counts by Type:\n"
        for vtype, count in self.violation_counts.items():
            report += f"  {vtype}: {count}\n"
        
        report += f"\n{'='*60}\n"
        report += "Recent Violations (last 10):\n"
        for v in self.violations[-10:]:
            report += f"  Step {v.timestep}: [{v.severity.upper()}] {v.component} - {v.violation_type}\n"
            report += f"    Value: {v.value:.2f}, Limit: {v.limit:.2f}\n"
            report += f"    Action: {v.action_taken}\n"
        
        return report


# Test the safety supervisor
if __name__ == "__main__":
    print("Testing Safety Supervisor...")
    
    supervisor = SafetySupervisor()
    
    # Test case 1: Battery SoC violation
    print("\nTest 1: Battery charging near SoC limit")
    raw_actions = {
        'battery_power': [500, 150],  # Try to charge batteries
        'grid_power': 1000,
        'ev_charging_power': 100
    }
    
    battery_states = [
        {'soc': 0.88, 'soh': 1.0, 'temperature': 30.0, 'max_charge_kw': 600, 'max_discharge_kw': 600},
        {'soc': 0.85, 'soh': 0.95, 'temperature': 28.0, 'max_charge_kw': 200, 'max_discharge_kw': 200}
    ]
    
    grid_state = {'max_import_kw': 5000, 'max_export_kw': 3000}
    ev_state = {'total_max_charge_rate': 200}
    component_health = {'battery_temperatures': [30.0, 28.0], 'inverter_temperature': 45.0}
    
    safe_actions, penalty = supervisor.check_and_clip_actions(
        raw_actions, battery_states, grid_state, ev_state, component_health, timestep=10
    )
    
    print(f"Raw battery powers: {raw_actions['battery_power']}")
    print(f"Safe battery powers: {safe_actions['battery_power']}")
    print(f"Penalty: ${penalty:.2f}")
    
    # Test case 2: Grid power violation
    print("\nTest 2: Grid power exceeds limit")
    raw_actions = {
        'battery_power': [100, 50],
        'grid_power': 6000,  # Exceeds max import
        'ev_charging_power': 100
    }
    
    safe_actions, penalty = supervisor.check_and_clip_actions(
        raw_actions, battery_states, grid_state, ev_state, component_health, timestep=20
    )
    
    print(f"Raw grid power: {raw_actions['grid_power']} kW")
    print(f"Safe grid power: {safe_actions['grid_power']} kW")
    print(f"Penalty: ${penalty:.2f}")
    
    # Print report
    print(supervisor.get_violation_report())
    print(f"\nStatistics: {supervisor.get_statistics()}")
