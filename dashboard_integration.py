"""
Smart EMS RL Model Integration Example
Demonstrates how to use the trained RL model in a real-time dashboard
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from train_production_ems import ProductionEMSEnv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class EMSController:
    """
    Production EMS Controller using trained RL model
    Integrates with real-time data feeds and provides energy dispatch recommendations
    """
    
    def __init__(self, model_path='models/production_ems_agent', data_path='processed_ems_data.csv'):
        """Initialize the EMS controller"""
        print("🔄 Initializing Smart EMS Controller...")
        
        # Load trained model
        try:
            self.model = PPO.load(model_path)
            print("✅ RL Model loaded successfully")
        except:
            print("❌ Error loading model. Using random policy.")
            self.model = None
        
        # Load historical data for environment
        self.data_df = pd.read_csv(data_path, index_col='DATE_TIME', parse_dates=True)
        
        # Create environment instance
        self.env = ProductionEMSEnv(self.data_df, episode_length=24)
        
        # Initialize system state
        self.current_state = None
        self.battery_soc = 0.5  # Start at 50% SoC
        
        print("✅ EMS Controller ready!")
    
    def get_recommendation(self, current_data):
        """
        Get RL-based recommendation for current system state
        
        Args:
            current_data (dict): Current system data
                - demand: Current demand (kW)
                - solar_output: Solar generation (kW)  
                - wind_output: Wind generation (kW)
                - grid_price: Current grid price ($/kWh)
                - timestamp: Current datetime
        
        Returns:
            dict: Recommended actions and analysis
        """
        
        # Extract current hour for time encoding
        hour = current_data['timestamp'].hour if 'timestamp' in current_data else 12
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Scale data to environment levels (microgrid scale)
        scaled_demand = current_data['demand'] * 0.001
        scaled_solar = current_data['solar_output'] * 0.001
        scaled_wind = current_data['wind_output'] * 0.001
        
        # Create observation
        obs = np.array([
            self.battery_soc,
            scaled_demand,
            scaled_solar,
            scaled_wind,
            current_data['grid_price'],
            hour_sin,
            hour_cos
        ], dtype=np.float32)
        
        # Get RL recommendation
        if self.model:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            # Fallback: simple rule-based policy
            action = self._rule_based_policy(current_data)
        
        # Interpret actions
        battery_action = action[0]  # -1 to +1
        grid_action = action[1]     # 0 to 1
        
        # Calculate energy flows
        renewable_generation = current_data['solar_output'] + current_data['wind_output']
        energy_balance = self._calculate_energy_balance(
            current_data['demand'], renewable_generation, battery_action, grid_action
        )
        
        # Create recommendation response
        recommendation = {
            'timestamp': current_data.get('timestamp', datetime.now()),
            'actions': {
                'battery_action': float(battery_action),
                'battery_recommendation': self._interpret_battery_action(battery_action),
                'grid_action': float(grid_action),
                'grid_recommendation': self._interpret_grid_action(grid_action)
            },
            'energy_analysis': {
                'demand': current_data['demand'],
                'renewable_available': renewable_generation,
                'renewable_used': energy_balance['renewable_used'],
                'battery_power': energy_balance['battery_power'],
                'grid_power': energy_balance['grid_power'],
                'unmet_demand': energy_balance['unmet_demand']
            },
            'system_status': {
                'battery_soc': float(self.battery_soc),
                'grid_cost_estimate': energy_balance['grid_cost'],
                'renewable_efficiency': energy_balance['renewable_used'] / max(renewable_generation, 0.1),
                'demand_satisfaction': (current_data['demand'] - energy_balance['unmet_demand']) / current_data['demand']
            },
            'alerts': self._generate_alerts(energy_balance, current_data)
        }
        
        return recommendation
    
    def _rule_based_policy(self, data):
        """Simple rule-based fallback policy"""
        # Charge battery during high renewable generation
        if (data['solar_output'] + data['wind_output']) > data['demand']:
            battery_action = 0.5  # Charge
        elif self.battery_soc > 0.3:
            battery_action = -0.3  # Discharge
        else:
            battery_action = 0.0  # No action
        
        # Use grid if renewable + battery insufficient
        grid_action = 0.8
        
        return np.array([battery_action, grid_action])
    
    def _calculate_energy_balance(self, demand, renewable_gen, battery_action, grid_action):
        """Calculate energy flows based on actions"""
        # Battery parameters (should match environment)
        battery_capacity = 200.0  # kWh
        max_charge_rate = 100.0   # kW
        max_discharge_rate = 100.0  # kW
        battery_efficiency = 0.9
        
        # Scale to microgrid level
        demand_scaled = demand * 0.001
        renewable_scaled = renewable_gen * 0.001
        
        # Battery dynamics
        if battery_action > 0:  # Charging
            charge_power = battery_action * max_charge_rate
            available_excess = max(0, renewable_scaled - demand_scaled)
            actual_charge = min(charge_power, available_excess, 
                              (1.0 - self.battery_soc) * battery_capacity)
            
            self.battery_soc += (actual_charge * battery_efficiency) / battery_capacity
            battery_power = -actual_charge * 1000  # Back to kW scale
        else:  # Discharging
            discharge_power = abs(battery_action) * max_discharge_rate
            available_discharge = self.battery_soc * battery_capacity
            actual_discharge = min(discharge_power, available_discharge)
            
            self.battery_soc -= actual_discharge / battery_capacity
            battery_power = (actual_discharge / battery_efficiency) * 1000  # Back to kW scale
        
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
        
        # Energy balance
        renewable_used = min(renewable_gen, demand)
        remaining_after_renewable = max(0, demand - renewable_used)
        
        battery_used = min(max(0, battery_power), remaining_after_renewable)
        remaining_after_battery = max(0, remaining_after_renewable - battery_used)
        
        grid_power = grid_action * remaining_after_battery
        unmet_demand = max(0, remaining_after_battery - grid_power)
        
        # Cost calculation
        grid_cost = (grid_power * 1.0) * 0.12  # Simplified cost
        
        return {
            'renewable_used': renewable_used,
            'battery_power': battery_power,
            'grid_power': grid_power,
            'unmet_demand': unmet_demand,
            'grid_cost': grid_cost
        }
    
    def _interpret_battery_action(self, action):
        """Convert battery action to human-readable recommendation"""
        if action > 0.3:
            return f"CHARGE battery at {action*100:.1f}% rate"
        elif action < -0.3:
            return f"DISCHARGE battery at {abs(action)*100:.1f}% rate"
        else:
            return "MAINTAIN current battery state"
    
    def _interpret_grid_action(self, action):
        """Convert grid action to human-readable recommendation"""
        if action > 0.7:
            return f"HIGH grid usage ({action*100:.1f}%)"
        elif action > 0.3:
            return f"MODERATE grid usage ({action*100:.1f}%)"
        else:
            return f"MINIMAL grid usage ({action*100:.1f}%)"
    
    def _generate_alerts(self, energy_balance, data):
        """Generate system alerts and warnings"""
        alerts = []
        
        # Battery alerts
        if self.battery_soc < 0.2:
            alerts.append({
                'level': 'WARNING',
                'message': f'Low battery SoC: {self.battery_soc*100:.1f}%',
                'recommendation': 'Consider charging from grid or reducing load'
            })
        elif self.battery_soc > 0.9:
            alerts.append({
                'level': 'INFO',
                'message': f'High battery SoC: {self.battery_soc*100:.1f}%',
                'recommendation': 'Good for meeting evening demand'
            })
        
        # Unmet demand alerts
        if energy_balance['unmet_demand'] > 0.1:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Unmet demand: {energy_balance["unmet_demand"]:.1f} kW',
                'recommendation': 'Increase grid usage or reduce non-critical loads'
            })
        
        # Cost alerts
        if energy_balance['grid_cost'] > 5.0:
            alerts.append({
                'level': 'WARNING',
                'message': f'High grid cost: ${energy_balance["grid_cost"]:.2f}',
                'recommendation': 'Consider load shifting or battery discharge'
            })
        
        # Renewable efficiency
        renewable_total = data['solar_output'] + data['wind_output']
        if renewable_total > 0 and energy_balance['renewable_used'] / renewable_total < 0.8:
            alerts.append({
                'level': 'INFO',
                'message': 'Low renewable utilization',
                'recommendation': 'Consider charging battery or shifting flexible loads'
            })
        
        return alerts

def demo_dashboard_integration():
    """Demonstrate how to integrate EMS controller with dashboard"""
    print("🎯 Smart EMS Dashboard Integration Demo")
    print("=" * 50)
    
    # Initialize controller
    controller = EMSController()
    
    # Simulate real-time data for 24 hours
    start_time = datetime(2023, 6, 15, 0, 0)  # Summer day
    
    results = []
    
    for hour in range(24):
        current_time = start_time + timedelta(hours=hour)
        
        # Simulate realistic data patterns
        solar_pattern = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        wind_pattern = 0.3 + 0.4 * np.random.random()
        demand_pattern = 15 + 10 * (1 + 0.5 * np.sin(2 * np.pi * (hour - 8) / 24))
        
        # Peak pricing during evening hours
        if 18 <= hour <= 22:
            grid_price = 0.15
        elif 22 <= hour or hour <= 6:
            grid_price = 0.08
        else:
            grid_price = 0.12
        
        current_data = {
            'timestamp': current_time,
            'demand': demand_pattern + np.random.normal(0, 2),
            'solar_output': solar_pattern * 25 + np.random.normal(0, 1),
            'wind_output': wind_pattern * 15 + np.random.normal(0, 1),
            'grid_price': grid_price
        }
        
        # Get recommendation
        recommendation = controller.get_recommendation(current_data)
        results.append(recommendation)
        
        # Display key information
        print(f"\n⏰ Hour {hour:2d} ({current_time.strftime('%H:%M')})")
        print(f"   📊 Demand: {current_data['demand']:.1f} kW")
        print(f"   🌞 Solar: {current_data['solar_output']:.1f} kW")
        print(f"   💨 Wind: {current_data['wind_output']:.1f} kW")
        print(f"   🔋 Battery SoC: {recommendation['system_status']['battery_soc']*100:.1f}%")
        print(f"   ⚡ {recommendation['actions']['battery_recommendation']}")
        print(f"   🏭 {recommendation['actions']['grid_recommendation']}")
        
        # Show alerts
        for alert in recommendation['alerts']:
            emoji = "🚨" if alert['level'] == 'CRITICAL' else "⚠️" if alert['level'] == 'WARNING' else "ℹ️"
            print(f"   {emoji} {alert['message']}")
    
    # Create summary visualization
    hours = list(range(24))
    demands = [r['energy_analysis']['demand'] for r in results]
    solar = [r['energy_analysis']['renewable_available'] - r['energy_analysis']['renewable_used'] + r['energy_analysis']['renewable_used'] for r in results]
    battery_soc = [r['system_status']['battery_soc'] for r in results]
    grid_costs = [r['system_status']['grid_cost_estimate'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Smart EMS Dashboard - 24 Hour Operation', fontsize=16)
    
    # Energy balance
    ax1.plot(hours, demands, 'b-', linewidth=2, label='Demand', marker='o')
    renewable_avail = [r['energy_analysis']['renewable_available'] for r in results]
    ax1.plot(hours, renewable_avail, 'g-', linewidth=2, label='Renewable Available', marker='s')
    ax1.set_title('Energy Demand vs Renewable Generation')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Battery management
    ax2.plot(hours, [soc*100 for soc in battery_soc], 'purple', linewidth=3, marker='o')
    ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Min SoC')
    ax2.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Max SoC')
    ax2.set_title('Battery State of Charge')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('SoC (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cost analysis
    ax3.bar(hours, grid_costs, alpha=0.7, color='red')
    ax3.set_title('Hourly Grid Costs')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Cost ($)')
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics
    renewable_eff = [r['system_status']['renewable_efficiency'] for r in results]
    demand_sat = [r['system_status']['demand_satisfaction'] for r in results]
    ax4.plot(hours, [e*100 for e in renewable_eff], 'g-', linewidth=2, label='Renewable Efficiency', marker='o')
    ax4.plot(hours, [d*100 for d in demand_sat], 'b-', linewidth=2, label='Demand Satisfaction', marker='s')
    ax4.set_title('System Performance')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Percentage (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dashboard_integration_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    total_cost = sum(grid_costs)
    avg_renewable_eff = np.mean(renewable_eff)
    avg_demand_sat = np.mean(demand_sat)
    final_soc = battery_soc[-1]
    
    print(f"\n📈 24-Hour Summary:")
    print(f"   💰 Total Grid Cost: ${total_cost:.2f}")
    print(f"   🌱 Average Renewable Efficiency: {avg_renewable_eff*100:.1f}%")
    print(f"   ⚡ Average Demand Satisfaction: {avg_demand_sat*100:.1f}%")
    print(f"   🔋 Final Battery SoC: {final_soc*100:.1f}%")
    
    return results

def export_api_format(recommendation):
    """Export recommendation in API-friendly JSON format"""
    return {
        "timestamp": recommendation['timestamp'].isoformat(),
        "battery": {
            "action": recommendation['actions']['battery_action'],
            "recommendation": recommendation['actions']['battery_recommendation'],
            "current_soc": recommendation['system_status']['battery_soc']
        },
        "grid": {
            "action": recommendation['actions']['grid_action'],
            "recommendation": recommendation['actions']['grid_recommendation'],
            "estimated_cost": recommendation['system_status']['grid_cost_estimate']
        },
        "energy": {
            "demand": recommendation['energy_analysis']['demand'],
            "renewable_available": recommendation['energy_analysis']['renewable_available'],
            "renewable_used": recommendation['energy_analysis']['renewable_used'],
            "unmet_demand": recommendation['energy_analysis']['unmet_demand']
        },
        "performance": {
            "renewable_efficiency": recommendation['system_status']['renewable_efficiency'],
            "demand_satisfaction": recommendation['system_status']['demand_satisfaction']
        },
        "alerts": recommendation['alerts']
    }

if __name__ == "__main__":
    # Run the dashboard integration demo
    results = demo_dashboard_integration()
    
    # Example of API format export
    print(f"\n🔌 API Format Example:")
    api_format = export_api_format(results[12])  # Noon example
    print(json.dumps(api_format, indent=2))