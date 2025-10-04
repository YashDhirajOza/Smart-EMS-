"""
EV Fleet Simulator
Simulates EV arrivals, departures, charging requirements, and deadlines
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from env_config import EV_FLEET, HOURS_PER_STEP


@dataclass
class ElectricVehicle:
    """Represents a single electric vehicle"""
    id: int
    arrival_step: int
    departure_step: int
    battery_capacity_kwh: float
    arrival_soc: float
    target_soc: float
    max_charge_rate_kw: float
    current_soc: float = 0.0
    energy_delivered_kwh: float = 0.0
    is_connected: bool = True
    charger_id: Optional[int] = None
    
    def __post_init__(self):
        self.current_soc = self.arrival_soc
        
    @property
    def energy_needed_kwh(self) -> float:
        """Calculate remaining energy needed to reach target SoC"""
        return max(0, (self.target_soc - self.current_soc) * self.battery_capacity_kwh)
    
    @property
    def steps_until_departure(self) -> int:
        """Steps remaining until departure"""
        return self.departure_step - self.arrival_step
    
    @property
    def is_charging_complete(self) -> bool:
        """Check if EV has reached target SoC"""
        return self.current_soc >= self.target_soc - 0.01  # 1% tolerance
    
    def charge(self, power_kw: float, duration_hours: float, efficiency: float = 0.95) -> float:
        """
        Charge the EV with given power for given duration
        Returns actual energy delivered (kWh)
        """
        # Limit power to max charge rate
        actual_power = min(power_kw, self.max_charge_rate_kw)
        
        # Calculate energy that could be delivered
        energy_available = actual_power * duration_hours * efficiency
        
        # Limit to what's needed
        energy_delivered = min(energy_available, self.energy_needed_kwh)
        
        # Update SoC
        soc_increase = energy_delivered / self.battery_capacity_kwh
        self.current_soc = min(1.0, self.current_soc + soc_increase)
        self.energy_delivered_kwh += energy_delivered
        
        # Return actual energy taken from grid (before efficiency loss)
        return energy_delivered / efficiency if efficiency > 0 else 0.0


class EVFleetSimulator:
    """Simulates a fleet of EVs with realistic arrival/departure patterns"""
    
    def __init__(self, config=EV_FLEET, random_seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(random_seed)
        
        # Fleet state
        self.connected_evs: List[ElectricVehicle] = []
        self.departed_evs: List[ElectricVehicle] = []
        self.ev_id_counter = 0
        
        # Statistics
        self.total_evs_served = 0
        self.total_energy_delivered = 0.0
        
    def reset(self, random_seed: Optional[int] = None):
        """Reset simulator for new episode"""
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        
        self.connected_evs = []
        self.departed_evs = []
        self.ev_id_counter = 0
        self.total_evs_served = 0
        self.total_energy_delivered = 0.0
        
    def generate_arrival_pattern(self, num_steps: int) -> np.ndarray:
        """
        Generate EV arrival probabilities for each timestep
        Uses bimodal distribution for morning and evening peaks
        """
        steps_per_hour = int(1.0 / HOURS_PER_STEP)
        hours = np.arange(num_steps) * HOURS_PER_STEP
        
        # Morning peak (Gaussian around morning_arrival_peak)
        morning_peak = self.config.morning_arrival_peak
        morning_std = 1.5
        morning_prob = np.exp(-0.5 * ((hours - morning_peak) / morning_std) ** 2)
        
        # Evening peak (Gaussian around evening_arrival_peak)
        evening_peak = self.config.evening_arrival_peak
        evening_std = 2.0
        evening_prob = np.exp(-0.5 * ((hours - evening_peak) / evening_std) ** 2)
        
        # Combine with weights (evening peak typically larger)
        arrival_prob = 0.3 * morning_prob + 0.7 * evening_prob
        
        # Add small baseline probability
        arrival_prob += 0.05
        
        # Normalize so expected arrivals match max_concurrent_evs over episode
        target_total = self.config.max_concurrent_evs * 1.5  # Some turnover
        arrival_prob = arrival_prob * target_total / arrival_prob.sum()
        
        return arrival_prob
    
    def step(self, current_step: int, num_steps: int, arrival_prob: Optional[np.ndarray] = None) -> None:
        """
        Update fleet state for current timestep
        - Generate new arrivals
        - Remove departures
        """
        # Handle departures
        self._process_departures(current_step)
        
        # Generate arrivals
        if arrival_prob is not None and current_step < len(arrival_prob):
            prob = arrival_prob[current_step]
        else:
            # Default: small constant probability
            prob = 0.1
        
        # Poisson-like arrivals
        if len(self.connected_evs) < self.config.max_concurrent_evs:
            num_arrivals = self.rng.poisson(prob)
            num_arrivals = min(num_arrivals, 
                             self.config.max_concurrent_evs - len(self.connected_evs))
            
            for _ in range(num_arrivals):
                ev = self._generate_ev(current_step, num_steps)
                self.connected_evs.append(ev)
    
    def _process_departures(self, current_step: int) -> None:
        """Remove EVs that have reached their departure time"""
        remaining_evs = []
        for ev in self.connected_evs:
            if current_step >= ev.departure_step:
                ev.is_connected = False
                self.departed_evs.append(ev)
                self.total_evs_served += 1
                self.total_energy_delivered += ev.energy_delivered_kwh
            else:
                remaining_evs.append(ev)
        
        self.connected_evs = remaining_evs
    
    def _generate_ev(self, arrival_step: int, max_steps: int) -> ElectricVehicle:
        """Generate a random EV with realistic parameters"""
        # Battery capacity (uniformly distributed)
        capacity = self.rng.uniform(*self.config.ev_battery_capacity_range)
        
        # Arrival SoC (typically low - people charge when needed)
        arrival_soc = self.rng.uniform(*self.config.ev_arrival_soc_range)
        
        # Target SoC
        target_soc = self.config.ev_departure_soc_target
        
        # Max charge rate (correlated with battery size)
        if capacity > 70:  # Large battery (e.g., Tesla Model S)
            max_rate = self.rng.uniform(40, 50)  # DC fast charging
        elif capacity > 55:  # Medium battery (e.g., Chevy Bolt)
            max_rate = self.rng.uniform(25, 40)
        else:  # Small battery (e.g., Nissan Leaf)
            max_rate = self.rng.uniform(7, 25)  # Level 2 charging
        
        # Parking duration (in steps)
        duration_hours = self.rng.exponential(self.config.avg_parking_duration_hours)
        duration_hours = np.clip(duration_hours, 1.0, 8.0)  # Between 1-8 hours
        duration_steps = int(duration_hours / HOURS_PER_STEP)
        
        departure_step = min(arrival_step + duration_steps, max_steps - 1)
        
        ev = ElectricVehicle(
            id=self.ev_id_counter,
            arrival_step=arrival_step,
            departure_step=departure_step,
            battery_capacity_kwh=capacity,
            arrival_soc=arrival_soc,
            target_soc=target_soc,
            max_charge_rate_kw=max_rate
        )
        
        self.ev_id_counter += 1
        return ev
    
    def get_fleet_state(self) -> dict:
        """Get current fleet state as dictionary"""
        if not self.connected_evs:
            return {
                'num_connected': 0,
                'total_energy_needed': 0.0,
                'avg_deadline_steps': 0.0,
                'total_max_charge_rate': 0.0,
                'earliest_deadline': 999,
                'ev_list': []
            }
        
        total_energy = sum(ev.energy_needed_kwh for ev in self.connected_evs)
        avg_deadline = np.mean([ev.steps_until_departure for ev in self.connected_evs])
        total_rate = sum(ev.max_charge_rate_kw for ev in self.connected_evs)
        earliest = min(ev.steps_until_departure for ev in self.connected_evs)
        
        return {
            'num_connected': len(self.connected_evs),
            'total_energy_needed': total_energy,
            'avg_deadline_steps': avg_deadline,
            'total_max_charge_rate': total_rate,
            'earliest_deadline': earliest,
            'ev_list': self.connected_evs.copy()
        }
    
    def allocate_charging_power(self, total_power_available_kw: float, 
                               strategy: str = 'proportional') -> dict:
        """
        Allocate available charging power to connected EVs
        
        Strategies:
        - 'proportional': Allocate proportional to energy needed
        - 'earliest_deadline': Prioritize EVs leaving soonest
        - 'equal': Split equally among all EVs
        """
        if not self.connected_evs or total_power_available_kw <= 0:
            return {ev.id: 0.0 for ev in self.connected_evs}
        
        allocations = {}
        
        if strategy == 'earliest_deadline':
            # Sort by departure time (earliest first)
            sorted_evs = sorted(self.connected_evs, key=lambda e: e.steps_until_departure)
            
            remaining_power = total_power_available_kw
            for ev in sorted_evs:
                if remaining_power <= 0:
                    allocations[ev.id] = 0.0
                    continue
                
                # Allocate up to EV's max rate or remaining power
                allocated = min(ev.max_charge_rate_kw, remaining_power, 
                              ev.energy_needed_kwh / HOURS_PER_STEP)
                allocations[ev.id] = allocated
                remaining_power -= allocated
        
        elif strategy == 'proportional':
            # Allocate proportional to energy needed
            total_needed = sum(ev.energy_needed_kwh for ev in self.connected_evs)
            
            if total_needed > 0:
                for ev in self.connected_evs:
                    fraction = ev.energy_needed_kwh / total_needed
                    allocated = min(
                        fraction * total_power_available_kw,
                        ev.max_charge_rate_kw,
                        ev.energy_needed_kwh / HOURS_PER_STEP
                    )
                    allocations[ev.id] = allocated
            else:
                allocations = {ev.id: 0.0 for ev in self.connected_evs}
        
        elif strategy == 'equal':
            # Split equally
            power_per_ev = total_power_available_kw / len(self.connected_evs)
            for ev in self.connected_evs:
                allocated = min(power_per_ev, ev.max_charge_rate_kw,
                              ev.energy_needed_kwh / HOURS_PER_STEP)
                allocations[ev.id] = allocated
        
        return allocations
    
    def charge_fleet(self, power_allocations: dict, duration_hours: float = HOURS_PER_STEP) -> float:
        """
        Charge all connected EVs with given power allocations
        Returns total energy drawn from grid (kWh)
        """
        total_energy_from_grid = 0.0
        
        for ev in self.connected_evs:
            if ev.id in power_allocations:
                power = power_allocations[ev.id]
                energy = ev.charge(power, duration_hours)
                total_energy_from_grid += energy
        
        return total_energy_from_grid
    
    def get_charging_success_rate(self) -> float:
        """Calculate fraction of departed EVs that reached target SoC"""
        if not self.departed_evs:
            return 0.0
        
        successful = sum(1 for ev in self.departed_evs if ev.is_charging_complete)
        return successful / len(self.departed_evs)
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics"""
        return {
            'total_evs_served': self.total_evs_served,
            'currently_connected': len(self.connected_evs),
            'total_energy_delivered': self.total_energy_delivered,
            'charging_success_rate': self.get_charging_success_rate(),
            'avg_final_soc': np.mean([ev.current_soc for ev in self.departed_evs]) if self.departed_evs else 0.0
        }


# Test the simulator
if __name__ == "__main__":
    print("Testing EV Fleet Simulator...")
    
    simulator = EVFleetSimulator(random_seed=42)
    num_steps = 96  # 24 hours with 15-minute intervals
    
    # Generate arrival pattern
    arrival_pattern = simulator.generate_arrival_pattern(num_steps)
    
    print(f"\nSimulating {num_steps} timesteps...")
    
    for step in range(num_steps):
        simulator.step(step, num_steps, arrival_pattern)
        
        if step % 16 == 0:  # Every 4 hours
            state = simulator.get_fleet_state()
            hour = step * HOURS_PER_STEP
            print(f"\nHour {hour:.1f}: {state['num_connected']} EVs connected, "
                  f"{state['total_energy_needed']:.1f} kWh needed")
        
        # Simulate charging with 100 kW available
        if simulator.connected_evs:
            allocations = simulator.allocate_charging_power(100, strategy='earliest_deadline')
            energy_used = simulator.charge_fleet(allocations)
    
    stats = simulator.get_statistics()
    print(f"\n{'='*60}")
    print("Final Statistics:")
    print(f"Total EVs served: {stats['total_evs_served']}")
    print(f"Total energy delivered: {stats['total_energy_delivered']:.1f} kWh")
    print(f"Charging success rate: {stats['charging_success_rate']*100:.1f}%")
    print(f"Average final SoC: {stats['avg_final_soc']*100:.1f}%")
    print(f"{'='*60}")
