"""
Battery Degradation Model
Models calendar aging, cycle aging, and temperature effects
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from env_config import HOURS_PER_STEP, BatteryConfig


class BatteryDegradationModel:
    """
    Models battery degradation based on:
    1. Cycle aging (depth of discharge, throughput)
    2. Calendar aging (time-based degradation)
    3. Temperature effects
    """
    
    def __init__(self, config: BatteryConfig):
        self.config = config
        
        # State variables
        self.soh = config.soh_initial  # State of Health (1.0 = 100%)
        self.cumulative_throughput_kwh = 0.0
        self.cumulative_cycles = 0.0  # Equivalent full cycles
        self.age_days = 0.0
        self.temperature = config.temperature_nominal
        
        # Tracking for DoD calculation
        self.soc_history = []
        self.max_soc_since_min = config.initial_soc
        self.min_soc_since_max = config.initial_soc
        
        # Degradation parameters
        self.calendar_degradation_rate = (1.0 - 0.8) / (config.calendar_life_years * 365)  # 20% over lifetime
        self.temperature_acceleration_factor = 2.0  # Degradation doubles per 10°C increase
        
    def reset(self):
        """Reset degradation model"""
        self.soh = self.config.soh_initial
        self.cumulative_throughput_kwh = 0.0
        self.cumulative_cycles = 0.0
        self.age_days = 0.0
        self.temperature = self.config.temperature_nominal
        self.soc_history = []
        self.max_soc_since_min = self.config.initial_soc
        self.min_soc_since_max = self.config.initial_soc
    
    def update(self, energy_kwh: float, soc_before: float, soc_after: float, 
               temperature: float = None, time_hours: float = HOURS_PER_STEP) -> Tuple[float, float]:
        """
        Update degradation based on energy flow
        
        Args:
            energy_kwh: Absolute energy throughput this step (always positive)
            soc_before: SoC before this step
            soc_after: SoC after this step
            temperature: Battery temperature (°C)
            time_hours: Duration of this step
            
        Returns:
            (degradation_cost, new_soh)
        """
        # Update temperature
        if temperature is not None:
            self.temperature = temperature
        
        # Update age
        self.age_days += time_hours / 24.0
        
        # Update throughput
        self.cumulative_throughput_kwh += energy_kwh
        
        # Calculate DoD for this step
        dod = self._calculate_dod(soc_before, soc_after)
        
        # Update cycles
        cycle_increment = dod  # Each 100% DoD = 1 cycle
        self.cumulative_cycles += cycle_increment
        
        # Calculate degradation components
        cycle_degradation = self._calculate_cycle_degradation(energy_kwh, dod)
        calendar_degradation = self._calculate_calendar_degradation(time_hours)
        temperature_factor = self._calculate_temperature_factor()
        
        # Total degradation
        total_degradation = (cycle_degradation + calendar_degradation) * temperature_factor
        
        # Update SoH (bounded at 0.5 = 50%)
        self.soh = max(0.5, self.soh - total_degradation)
        
        # Calculate degradation cost
        degradation_cost = self._calculate_degradation_cost(energy_kwh, dod)
        
        return degradation_cost, self.soh
    
    def _calculate_dod(self, soc_before: float, soc_after: float) -> float:
        """
        Calculate depth of discharge using rainflow counting approximation
        Tracks peaks and valleys in SoC to identify cycles
        """
        self.soc_history.append(soc_after)
        
        # Simple approach: track swings
        if soc_after > soc_before:  # Charging
            self.max_soc_since_min = max(self.max_soc_since_min, soc_after)
        else:  # Discharging
            self.min_soc_since_max = min(self.min_soc_since_max, soc_after)
        
        # Detect full swing (discharge then charge, or vice versa)
        swing = abs(self.max_soc_since_min - self.min_soc_since_max)
        
        # Reset tracking if we've completed a half-cycle
        if soc_after < soc_before and soc_after < self.min_soc_since_max + 0.05:
            # Local minimum
            dod = swing
            self.min_soc_since_max = soc_after
            self.max_soc_since_min = soc_after
            return dod
        elif soc_after > soc_before and soc_after > self.max_soc_since_min - 0.05:
            # Local maximum
            dod = swing
            self.min_soc_since_max = soc_after
            self.max_soc_since_min = soc_after
            return dod
        
        # Return incremental swing
        return abs(soc_after - soc_before)
    
    def _calculate_cycle_degradation(self, energy_kwh: float, dod: float) -> float:
        """
        Calculate cycle-based degradation
        Higher DoD causes more degradation
        """
        if energy_kwh == 0:
            return 0.0
        
        # Degradation per cycle increases with DoD (typically follows power law)
        # At nominal DoD (80%), battery lasts cycle_life cycles to 80% SoH
        # So per-cycle degradation = 0.2 / cycle_life
        base_degradation_per_cycle = 0.2 / self.config.cycle_life
        
        # DoD stress factor (deeper cycles cause more damage)
        # Follows empirical relationship: stress ∝ DoD^2
        dod_stress_factor = (dod / 0.8) ** 2 if dod > 0 else 0
        
        degradation = base_degradation_per_cycle * dod_stress_factor * dod
        
        return degradation
    
    def _calculate_calendar_degradation(self, time_hours: float) -> float:
        """Calculate time-based degradation"""
        days = time_hours / 24.0
        return self.calendar_degradation_rate * days
    
    def _calculate_temperature_factor(self) -> float:
        """
        Calculate temperature acceleration factor
        Degradation approximately doubles for every 10°C increase
        """
        temp_diff = self.temperature - self.config.temperature_nominal
        factor = self.temperature_acceleration_factor ** (temp_diff / 10.0)
        return factor
    
    def _calculate_degradation_cost(self, energy_kwh: float, dod: float) -> float:
        """
        Calculate monetary cost of degradation
        Uses throughput-based model with DoD penalty
        """
        # Base cost per kWh throughput
        base_cost = energy_kwh * self.config.degradation_cost_per_kwh
        
        # Apply DoD penalty (deeper cycles cost more)
        # Using power law as specified: cost ∝ kWh^1.1
        dod_penalty = 1.0 + dod  # Higher DoD increases cost
        
        total_cost = base_cost * dod_penalty
        
        return total_cost
    
    def get_adjusted_capacity(self) -> float:
        """Get current usable capacity based on SoH"""
        return self.config.capacity_kwh * self.soh
    
    def get_adjusted_power_limits(self) -> Tuple[float, float]:
        """Get current power limits based on SoH and temperature"""
        # SoH reduces available power
        soh_factor = self.soh
        
        # Temperature affects power capability
        if self.temperature > 40:  # High temperature
            temp_factor = 0.8
        elif self.temperature < 0:  # Cold temperature
            temp_factor = 0.7
        else:
            temp_factor = 1.0
        
        max_charge = self.config.max_charge_kw * soh_factor * temp_factor
        max_discharge = self.config.max_discharge_kw * soh_factor * temp_factor
        
        return max_charge, max_discharge
    
    def get_metrics(self) -> dict:
        """Get degradation metrics"""
        return {
            'soh': self.soh,
            'soh_percent': self.soh * 100,
            'cumulative_throughput_kwh': self.cumulative_throughput_kwh,
            'cumulative_cycles': self.cumulative_cycles,
            'equivalent_full_cycles': self.cumulative_throughput_kwh / (2 * self.config.capacity_kwh),
            'age_days': self.age_days,
            'age_years': self.age_days / 365.0,
            'remaining_capacity_kwh': self.get_adjusted_capacity(),
            'estimated_remaining_cycles': (self.config.cycle_life * self.soh),
            'cycle_life_used_percent': (self.cumulative_cycles / self.config.cycle_life) * 100
        }


class BatteryThermalModel:
    """
    Simple thermal model for battery temperature
    Models heat generation from I²R losses and cooling
    """
    
    def __init__(self, config: BatteryConfig, 
                 thermal_mass_kwh_per_C: float = 10.0,
                 cooling_rate_C_per_hour: float = 2.0,
                 ambient_temperature: float = 25.0):
        self.config = config
        self.thermal_mass = thermal_mass_kwh_per_C  # kWh per °C
        self.cooling_rate = cooling_rate_C_per_hour  # °C per hour of cooling
        self.ambient_temp = ambient_temperature
        
        # State
        self.temperature = ambient_temperature
        
    def reset(self, ambient_temperature: float = 25.0):
        """Reset thermal model"""
        self.ambient_temp = ambient_temperature
        self.temperature = ambient_temperature
    
    def update(self, power_kw: float, soc: float, time_hours: float = HOURS_PER_STEP) -> float:
        """
        Update battery temperature based on power flow
        
        Args:
            power_kw: Battery power (positive = charge, negative = discharge)
            soc: Current state of charge
            time_hours: Duration of this step
            
        Returns:
            New temperature (°C)
        """
        # Heat generation from losses
        # Loss power = I²R, approximated as power² / capacity
        # Higher losses at extreme SoC
        soc_efficiency_factor = 1.0 + 0.5 * (abs(soc - 0.5) / 0.5)  # Higher losses at extremes
        
        loss_power_kw = (abs(power_kw) ** 2 / self.config.capacity_kwh) * soc_efficiency_factor * 0.05
        
        # Heat energy generated
        heat_energy_kwh = loss_power_kw * time_hours
        
        # Temperature increase from heat generation
        temp_increase = heat_energy_kwh / self.thermal_mass
        
        # Cooling towards ambient
        temp_diff = self.temperature - self.ambient_temp
        cooling = self.cooling_rate * time_hours * (temp_diff / 10.0)  # Proportional to difference
        
        # Update temperature
        self.temperature = self.temperature + temp_increase - cooling
        
        # Bound temperature
        self.temperature = np.clip(self.temperature, 0, 60)
        
        return self.temperature


# Test the degradation model
if __name__ == "__main__":
    from env_config import BATTERY_5
    
    print("Testing Battery Degradation Model...")
    print(f"Battery: {BATTERY_5.name}")
    print(f"Capacity: {BATTERY_5.capacity_kwh} kWh")
    print(f"Cycle life: {BATTERY_5.cycle_life} cycles")
    
    # Initialize models
    degradation = BatteryDegradationModel(BATTERY_5)
    thermal = BatteryThermalModel(BATTERY_5)
    
    # Simulate one full charge-discharge cycle
    print("\nSimulating charge-discharge cycle...")
    
    soc = 0.5
    for step in range(8):  # 8 steps = 2 hours (charge phase)
        power_kw = 500  # Charging at 500 kW
        energy_kwh = abs(power_kw) * HOURS_PER_STEP
        soc_before = soc
        soc += (power_kw * HOURS_PER_STEP) / BATTERY_5.capacity_kwh
        soc = min(0.9, soc)
        
        temp = thermal.update(power_kw, soc)
        cost, soh = degradation.update(energy_kwh, soc_before, soc, temp)
        
        print(f"Step {step}: Charging, SoC={soc:.3f}, Temp={temp:.1f}°C, Cost=${cost:.4f}")
    
    for step in range(8, 16):  # 8 steps = 2 hours (discharge phase)
        power_kw = -500  # Discharging at 500 kW
        energy_kwh = abs(power_kw) * HOURS_PER_STEP
        soc_before = soc
        soc += (power_kw * HOURS_PER_STEP) / BATTERY_5.capacity_kwh
        soc = max(0.1, soc)
        
        temp = thermal.update(power_kw, soc)
        cost, soh = degradation.update(energy_kwh, soc_before, soc, temp)
        
        print(f"Step {step}: Discharging, SoC={soc:.3f}, Temp={temp:.1f}°C, Cost=${cost:.4f}")
    
    metrics = degradation.get_metrics()
    print(f"\nDegradation Metrics:")
    print(f"  SoH: {metrics['soh_percent']:.2f}%")
    print(f"  Throughput: {metrics['cumulative_throughput_kwh']:.1f} kWh")
    print(f"  Cycles: {metrics['cumulative_cycles']:.3f}")
    print(f"  Cycle life used: {metrics['cycle_life_used_percent']:.3f}%")
