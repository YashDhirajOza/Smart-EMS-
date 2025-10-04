"""
Data Preprocessing for Solar Plant Data
Loads and processes Plant_1 and Plant_2 data for use in RL environment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class SolarDataLoader:
    """Load and preprocess solar plant generation and weather data"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        
    def load_plant_data(self, plant_id: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load generation and weather data for specified plant
        
        Returns:
            (generation_df, weather_df)
        """
        gen_file = self.data_dir / f"Plant_{plant_id}_Generation_Data.csv"
        weather_file = self.data_dir / f"Plant_{plant_id}_Weather_Sensor_Data.csv"
        
        print(f"Loading Plant {plant_id} data...")
        print(f"  Generation: {gen_file}")
        print(f"  Weather: {weather_file}")
        
        # Load generation data
        gen_df = pd.read_csv(gen_file)
        gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
        gen_df = gen_df.sort_values('DATE_TIME').reset_index(drop=True)
        
        # Load weather data
        weather_df = pd.read_csv(weather_file)
        weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
        weather_df = weather_df.sort_values('DATE_TIME').reset_index(drop=True)
        
        print(f"  Generation data: {len(gen_df)} rows")
        print(f"  Weather data: {len(weather_df)} rows")
        print(f"  Date range: {gen_df['DATE_TIME'].min()} to {gen_df['DATE_TIME'].max()}")
        
        return gen_df, weather_df
    
    def aggregate_to_15min(self, gen_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to 15-minute intervals (data appears to already be at 15-min)
        Combine generation and weather into single dataframe
        """
        # Group generation by timestamp and source
        gen_pivot = gen_df.pivot_table(
            index='DATE_TIME',
            values=['DC_POWER', 'AC_POWER'],
            aggfunc='mean'  # Average across inverters
        )
        
        # Group weather by timestamp
        weather_pivot = weather_df.pivot_table(
            index='DATE_TIME',
            values=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION'],
            aggfunc='mean'
        )
        
        # Merge
        combined = gen_pivot.join(weather_pivot, how='inner')
        combined = combined.reset_index()
        
        return combined
    
    def create_pv_profile(self, combined_df: pd.DataFrame, capacity_kw: float = 3200) -> pd.DataFrame:
        """
        Create PV generation profile normalized to microgrid capacity
        
        Args:
            combined_df: Combined generation and weather data
            capacity_kw: Target PV capacity for microgrid (kW)
        """
        # Use AC_POWER as the actual generation (kW)
        # Data is typically in kW already, but verify and scale if needed
        
        # Calculate capacity factor
        max_power = combined_df['AC_POWER'].quantile(0.99)  # 99th percentile as "capacity"
        
        if max_power > 0:
            scaling_factor = capacity_kw / max_power
        else:
            scaling_factor = 1.0
        
        pv_profile = pd.DataFrame({
            'timestamp': combined_df['DATE_TIME'],
            'pv_total': combined_df['AC_POWER'] * scaling_factor,
            'irradiation': combined_df['IRRADIATION'],
            'module_temp': combined_df['MODULE_TEMPERATURE']
        })
        
        # Distribute across 8 PV systems (as per CIGRE microgrid)
        pv_capacities = [0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.3]  # MW
        pv_capacities_kw = [c * 1000 for c in pv_capacities]
        total_capacity_kw = sum(pv_capacities_kw)
        
        for i, pv_cap in enumerate(pv_capacities_kw):
            fraction = pv_cap / total_capacity_kw
            pv_profile[f'pv{i+3}'] = pv_profile['pv_total'] * fraction  # pv3-pv11 (skipping 7, 10, 11)
        
        # Adjust column names to match CIGRE convention: pv3, pv4, pv5, pv6, pv8, pv9, pv10, pv11
        pv_profile = pv_profile.rename(columns={
            'pv3': 'pv3', 'pv4': 'pv4', 'pv5': 'pv5', 'pv6': 'pv6',
            'pv7': 'pv8', 'pv8': 'pv9', 'pv9': 'pv10', 'pv10': 'pv11'
        })
        
        return pv_profile
    
    def create_wind_profile(self, length: int, capacity_kw: float = 2500, 
                           seed: int = 42) -> pd.DataFrame:
        """
        Create synthetic wind profile (since we don't have wind data)
        Uses realistic wind generation patterns
        """
        np.random.seed(seed)
        
        timestamps = pd.date_range('2020-05-15', periods=length, freq='15T')
        
        # Generate wind speed with diurnal pattern and stochasticity
        hours = np.arange(length) * 0.25  # 15-minute intervals
        
        # Base wind speed with diurnal variation
        base_wind = 8 + 3 * np.sin(2 * np.pi * hours / 24 - np.pi/2)  # m/s
        
        # Add temporal correlation (AR process)
        noise = np.zeros(length)
        for i in range(1, length):
            noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 2)
        
        wind_speed = np.maximum(0, base_wind + noise)
        
        # Convert wind speed to power using simplified power curve
        # Typical: cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s
        power = np.zeros(length)
        for i, ws in enumerate(wind_speed):
            if ws < 3:
                power[i] = 0
            elif ws < 12:
                # Cubic relationship below rated
                power[i] = capacity_kw * ((ws - 3) / 9) ** 3
            elif ws < 25:
                power[i] = capacity_kw
            else:
                power[i] = 0  # Cut-out
        
        wt_profile = pd.DataFrame({
            'timestamp': timestamps,
            'wt7': power,  # Single wind turbine at bus 7
            'wind_speed': wind_speed
        })
        
        return wt_profile
    
    def create_load_profile(self, length: int, base_load_kw: float = 4000,
                           seed: int = 42) -> pd.DataFrame:
        """
        Create synthetic load profile with realistic patterns
        - Daily variation (morning/evening peaks)
        - Weekly variation (weekday/weekend)
        - Random fluctuations
        """
        np.random.seed(seed)
        
        timestamps = pd.date_range('2020-05-15', periods=length, freq='15T')
        
        # Time-based features
        hours = timestamps.hour + timestamps.minute / 60.0
        day_of_week = timestamps.dayofweek
        is_weekend = (day_of_week >= 5).astype(float)
        
        # Daily pattern (bimodal: morning and evening peaks)
        morning_peak = np.exp(-((hours - 8) ** 2) / 8)
        evening_peak = np.exp(-((hours - 19) ** 2) / 8)
        daily_pattern = 0.6 + 0.2 * morning_peak + 0.3 * evening_peak
        
        # Weekend reduction
        weekend_factor = 1.0 - 0.2 * is_weekend
        
        # Base load with patterns
        load = base_load_kw * daily_pattern * weekend_factor
        
        # Add noise
        noise = np.random.normal(0, base_load_kw * 0.05, length)
        load = np.maximum(base_load_kw * 0.3, load + noise)  # Min 30% of base load
        
        # Distribute across 8 load points
        load_fractions = [0.85, 0.285, 0.245, 0.65, 0.565, 0.605, 0.49, 0.34]
        load_fractions = np.array(load_fractions) / sum(load_fractions)  # Normalize
        
        load_profile = pd.DataFrame({'timestamp': timestamps})
        load_names = ['load_r1', 'Load_r3', 'Load_r4', 'Load_r5', 
                     'Load_r6', 'Load_r8', 'Load_r10', 'Load_r11']
        
        for i, (name, frac) in enumerate(zip(load_names, load_fractions)):
            load_profile[name] = load * frac
        
        return load_profile
    
    def create_price_profile(self, length: int, seed: int = 42) -> pd.DataFrame:
        """
        Create electricity price profile with time-of-use structure (Indian Tariffs)
        Based on typical Indian commercial/industrial tariffs
        """
        np.random.seed(seed)
        
        timestamps = pd.date_range('2020-05-15', periods=length, freq='15min')
        hours = timestamps.hour
        
        # Time-of-use pricing (in ₹/kWh - Indian Rupees)
        # Off-peak (0-6, 22-24): ₹4.50/kWh
        # Normal (6-9, 12-18): ₹7.50/kWh  
        # Peak (9-12, 18-22): ₹9.50/kWh
        
        price = np.zeros(length)
        for i, h in enumerate(hours):
            if h < 6 or h >= 22:  # Off-peak
                price[i] = 4.50
            elif (9 <= h < 12) or (18 <= h < 22):  # Peak
                price[i] = 9.50
            else:  # Normal
                price[i] = 7.50
        
        # Add small random variations (±₹0.40)
        price += np.random.normal(0, 0.40, length)
        price = np.maximum(3.50, price)  # Minimum price ₹3.50/kWh
        
        price_profile = pd.DataFrame({
            'timestamp': timestamps,
            'price': price
        })
        
        return price_profile
    
    def plot_profiles(self, pv_profile: pd.DataFrame, wt_profile: pd.DataFrame,
                     load_profile: pd.DataFrame, price_profile: pd.DataFrame,
                     num_days: int = 3):
        """Plot generated profiles for visualization"""
        steps_per_day = 96  # 24 hours * 4 (15-min intervals)
        steps = steps_per_day * num_days
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        hours = np.arange(steps) * 0.25
        
        # PV generation
        axes[0].plot(hours, pv_profile['pv_total'].iloc[:steps], label='Total PV', linewidth=2)
        axes[0].fill_between(hours, 0, pv_profile['pv_total'].iloc[:steps], alpha=0.3)
        axes[0].set_ylabel('PV Power (kW)')
        axes[0].set_title('Solar PV Generation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Wind generation
        axes[1].plot(hours, wt_profile['wt7'].iloc[:steps], label='Wind Turbine', 
                    color='green', linewidth=2)
        axes[1].fill_between(hours, 0, wt_profile['wt7'].iloc[:steps], alpha=0.3, color='green')
        axes[1].set_ylabel('Wind Power (kW)')
        axes[1].set_title('Wind Generation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Load demand
        total_load = load_profile[[col for col in load_profile.columns if 'load' in col.lower()]].sum(axis=1)
        axes[2].plot(hours, total_load.iloc[:steps], label='Total Load', 
                    color='red', linewidth=2)
        axes[2].fill_between(hours, 0, total_load.iloc[:steps], alpha=0.3, color='red')
        axes[2].set_ylabel('Load (kW)')
        axes[2].set_title('Load Demand')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Price (Indian Rupees)
        axes[3].plot(hours, price_profile['price'].iloc[:steps], label='Electricity Price',
                    color='purple', linewidth=2, drawstyle='steps-post')
        axes[3].set_ylabel('Price (₹/kWh)')
        axes[3].set_xlabel('Hour')
        axes[3].set_title('Electricity Price (Indian Tariff)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('profiles_preview.png', dpi=150)
        print("Saved profiles visualization to 'profiles_preview.png'")
        plt.close()


def main():
    """Main data preprocessing pipeline"""
    print("="*60)
    print("Solar Plant Data Preprocessing")
    print("="*60)
    
    # Look for CSV files in parent directory first, then current directory
    parent_dir = Path(__file__).parent.parent
    if (parent_dir / "Plant_1_Generation_Data.csv").exists():
        loader = SolarDataLoader(data_dir=str(parent_dir))
        print(f"Using data from: {parent_dir}")
    else:
        loader = SolarDataLoader()
        print(f"Using data from current directory")
    
    # Load Plant 1 data
    gen_df, weather_df = loader.load_plant_data(plant_id=1)
    
    # Aggregate to 15-minute intervals
    print("\nAggregating to 15-minute intervals...")
    combined_df = loader.aggregate_to_15min(gen_df, weather_df)
    print(f"Combined data: {len(combined_df)} timesteps")
    
    # Create PV profile
    print("\nCreating PV profile...")
    pv_profile = loader.create_pv_profile(combined_df, capacity_kw=3200)
    print(f"PV profile shape: {pv_profile.shape}")
    print(f"PV power range: {pv_profile['pv_total'].min():.1f} - {pv_profile['pv_total'].max():.1f} kW")
    
    # Create wind profile (synthetic)
    print("\nCreating wind profile...")
    wt_profile = loader.create_wind_profile(len(pv_profile), capacity_kw=2500)
    print(f"Wind profile shape: {wt_profile.shape}")
    print(f"Wind power range: {wt_profile['wt7'].min():.1f} - {wt_profile['wt7'].max():.1f} kW")
    
    # Create load profile (synthetic)
    print("\nCreating load profile...")
    load_profile = loader.create_load_profile(len(pv_profile), base_load_kw=4000)
    print(f"Load profile shape: {load_profile.shape}")
    total_load = load_profile[[col for col in load_profile.columns if 'load' in col.lower()]].sum(axis=1)
    print(f"Load range: {total_load.min():.1f} - {total_load.max():.1f} kW")
    
    # Create price profile
    print("\nCreating price profile...")
    price_profile = loader.create_price_profile(len(pv_profile))
    print(f"Price profile shape: {price_profile.shape}")
    print(f"Price range: ${price_profile['price'].min():.3f} - ${price_profile['price'].max():.3f}/kWh")
    
    # Save processed profiles
    print("\nSaving processed profiles...")
    pv_profile.to_csv('data/pv_profile_processed.csv', index=False)
    wt_profile.to_csv('data/wt_profile_processed.csv', index=False)
    load_profile.to_csv('data/load_profile_processed.csv', index=False)
    price_profile.to_csv('data/price_profile_processed.csv', index=False)
    print("Saved to data/ directory")
    
    # Plot profiles
    print("\nGenerating visualization...")
    loader.plot_profiles(pv_profile, wt_profile, load_profile, price_profile, num_days=7)
    
    # Statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Total timesteps: {len(pv_profile)}")
    print(f"Total days: {len(pv_profile) / 96:.1f}")
    print(f"\nPV Generation:")
    print(f"  Mean: {pv_profile['pv_total'].mean():.1f} kW")
    print(f"  Peak: {pv_profile['pv_total'].max():.1f} kW")
    print(f"  Capacity Factor: {pv_profile['pv_total'].mean() / 3200 * 100:.1f}%")
    print(f"\nWind Generation:")
    print(f"  Mean: {wt_profile['wt7'].mean():.1f} kW")
    print(f"  Peak: {wt_profile['wt7'].max():.1f} kW")
    print(f"  Capacity Factor: {wt_profile['wt7'].mean() / 2500 * 100:.1f}%")
    print(f"\nLoad Demand:")
    print(f"  Mean: {total_load.mean():.1f} kW")
    print(f"  Peak: {total_load.max():.1f} kW")
    print(f"\nAverage Price: ₹{price_profile['price'].mean():.2f}/kWh (Indian Tariff)")
    print(f"Price Range: ₹{price_profile['price'].min():.2f} - ₹{price_profile['price'].max():.2f}/kWh")
    print("="*60)
    
    return pv_profile, wt_profile, load_profile, price_profile


if __name__ == "__main__":
    pv, wt, load, price = main()
