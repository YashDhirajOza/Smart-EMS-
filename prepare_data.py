import pandas as pd
import numpy as np

# Step 1: Load all Mendeley dataset files (monthly Excel files)
import glob
import os

# Get all Excel files in the directory (excluding any backup or temporary files)
all_excel_files = glob.glob(r'D:\IIT_GAN\*.xlsx')
mendeley_files = [f for f in all_excel_files if not os.path.basename(f).startswith('~')]

print(f"Found {len(mendeley_files)} Excel files to process:")
for file in sorted(mendeley_files):
    print(f"  - {os.path.basename(file)}")

# Load and concatenate Mendeley data
mendeley_dfs = []
for file in mendeley_files:
    df = pd.read_excel(file)  # Assumes data in first sheet; specify sheet_name if needed
    mendeley_dfs.append(df)
mendeley_df = pd.concat(mendeley_dfs, ignore_index=True)

# Step 2: Process Mendeley timestamps and extract relevant columns
# Read from Sheet1 and handle header rows properly  
mendeley_processed_dfs = []
for file in mendeley_files:
    print(f"Processing file: {os.path.basename(file)}")
    try:
        # First check if Sheet1 exists
        xl_file = pd.ExcelFile(file)
        if 'Sheet1' not in xl_file.sheet_names:
            print(f"  - Skipping: No 'Sheet1' found. Available sheets: {xl_file.sheet_names}")
            continue
            
        df = pd.read_excel(file, sheet_name='Sheet1')
        
        # Check if required columns exist
        required_cols = ['Time', 'SCADA/ANALOG/044MQ067/0', 'SCADA/ANALOG/044MQ206/0', 'SCADA/ANALOG/044MQ070/0']
        if not all(col in df.columns for col in required_cols):
            print(f"  - Skipping: Missing required columns. Available: {df.columns.tolist()}")
            continue
        
        # Check if first row contains descriptions (NaT in Time column)
        if pd.isna(df['Time'].iloc[0]):
            # Skip the first row (header descriptions)
            df = df.iloc[1:].reset_index(drop=True)
        
        # Convert Time column to datetime and set as index
        df['DATE_TIME'] = pd.to_datetime(df['Time'])
        df.set_index('DATE_TIME', inplace=True)
        
        # Extract relevant columns and rename for RL
        # SCADA/ANALOG/044MQ067/0 = NLDC_DEMAND (MW)
        # SCADA/ANALOG/044MQ206/0 = ALL_IND_SOLAR (MW) 
        # SCADA/ANALOG/044MQ070/0 = ALL_INDIA_WIND (MW)
        df_processed = df[['SCADA/ANALOG/044MQ067/0', 'SCADA/ANALOG/044MQ206/0', 'SCADA/ANALOG/044MQ070/0']].copy()
        df_processed = df_processed.rename(columns={
            'SCADA/ANALOG/044MQ067/0': 'base_demand',  # MW
            'SCADA/ANALOG/044MQ206/0': 'solar_output',  # MW
            'SCADA/ANALOG/044MQ070/0': 'wind_output'    # MW
        })
        
        # Remove any rows with missing timestamps or data
        df_processed = df_processed.dropna()
        
        if len(df_processed) > 0:
            mendeley_processed_dfs.append(df_processed)
            print(f"  - Loaded {len(df_processed)} records from {df_processed.index.min()} to {df_processed.index.max()}")
        else:
            print(f"  - Skipping: No valid data after processing")
            
    except Exception as e:
        print(f"  - Error processing file: {e}")
        continue

# Concatenate all Mendeley data
if mendeley_processed_dfs:
    mendeley_df = pd.concat(mendeley_processed_dfs, ignore_index=False)
    print(f"\nTotal combined records: {len(mendeley_df)}")
    print(f"Date range: {mendeley_df.index.min()} to {mendeley_df.index.max()}")
else:
    print("\nNo valid data files found!")
    exit(1)

# Step 3: Check if Kaggle CSV files exist and load them if available
import os
kaggle_files_exist = (os.path.exists('Plant_1_Generation_Data.csv') and 
                     os.path.exists('Plant_1_Weather_Sensor_Data.csv'))

if kaggle_files_exist:
    print("Loading Kaggle data for additional weather features...")
    gen_df = pd.read_csv('Plant_1_Generation_Data.csv')
    weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

    # Process Kaggle timestamps
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    gen_df.set_index('DATE_TIME', inplace=True)
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
    weather_df.set_index('DATE_TIME', inplace=True)

    # Aggregate Kaggle generation
    gen_aggregated = gen_df.groupby(gen_df.index).agg({
        'DC_POWER': 'sum',
        'AC_POWER': 'sum',
        'DAILY_YIELD': 'mean',
        'TOTAL_YIELD': 'mean'
    }).rename(columns={'DC_POWER': 'solar_output_kaggle'})

    # Merge Kaggle weather
    kaggle_merged = gen_aggregated.join(weather_df[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']], how='outer')
    kaggle_merged = kaggle_merged.ffill()

    # Resample Kaggle to hourly
    kaggle_hourly = kaggle_merged.resample('h').mean()
    
    # Simulate weather features for Mendeley data based on Kaggle patterns
    avg_temp = kaggle_hourly['AMBIENT_TEMPERATURE'].mean()
    avg_module_temp = kaggle_hourly['MODULE_TEMPERATURE'].mean()
    avg_irradiation = kaggle_hourly['IRRADIATION'].mean()
else:
    print("Kaggle CSV files not found. Using Mendeley data only with simulated weather features.")
    avg_temp = 28.0  # Default average ambient temperature
    avg_module_temp = 35.0  # Default module temperature
    avg_irradiation = 0.4  # Default irradiation

# Step 4: Use Mendeley as primary data and resample to hourly
combined_hourly = mendeley_df.resample('h').mean()

# Step 5: Add simulated/enhanced columns for RL (SoC, EV peaks on real demand, costs, weather)
# Convert MW to kW for microgrid scale (divide by 1000 and scale down further for demo)
combined_hourly['base_demand'] = combined_hourly['base_demand'] / 100  # Scale down for microgrid
combined_hourly['solar_output'] = combined_hourly['solar_output'] / 100  # Scale down for microgrid  
combined_hourly['wind_output'] = combined_hourly['wind_output'] / 100   # Scale down for microgrid

# Enhance demand with EV peaks (add to real base_demand)
np.random.seed(42)  # For reproducible results
# Create evening peak mask (6 PM to 10 PM)
evening_mask = (combined_hourly.index.hour >= 18) & (combined_hourly.index.hour <= 22)
ev_peaks = np.random.normal(50, 15, len(combined_hourly)) * evening_mask
combined_hourly['demand'] = combined_hourly['base_demand'] + ev_peaks

# Add weather features (simulated based on solar patterns)
hour_sin = np.sin(2 * np.pi * combined_hourly.index.hour / 24)
combined_hourly['ambient_temperature'] = avg_temp + 5 * hour_sin + np.random.normal(0, 2, len(combined_hourly))
combined_hourly['module_temperature'] = combined_hourly['ambient_temperature'] + 10 + np.random.normal(0, 3, len(combined_hourly))
combined_hourly['irradiation'] = np.maximum(0, combined_hourly['solar_output'] * 0.01 + np.random.normal(0, 0.1, len(combined_hourly)))

# SoC: Simulated cumulative random walk (clip 0-1)
combined_hourly['SoC'] = 0.5 + np.cumsum(np.random.normal(0, 0.01, len(combined_hourly)))
combined_hourly['SoC'] = np.clip(combined_hourly['SoC'], 0, 1)

# Costs: Grid price varying by time (higher during peak hours)
peak_hours_mask = (combined_hourly.index.hour >= 18) & (combined_hourly.index.hour <= 22)
combined_hourly['grid_price'] = 0.1 + 0.05 * peak_hours_mask  # $/kWh

# Step 6: Handle remaining NaNs and save to CSV
# Fill any remaining NaNs
combined_hourly = combined_hourly.ffill()
combined_hourly = combined_hourly.fillna(0)  # Fill any remaining NaNs with 0

# Ensure all values are positive (clamp negative renewables to 0)
combined_hourly['solar_output'] = combined_hourly['solar_output'].clip(lower=0)
combined_hourly['wind_output'] = combined_hourly['wind_output'].clip(lower=0)
combined_hourly['demand'] = combined_hourly['demand'].clip(lower=0)

# Select final columns for RL training
final_columns = ['demand', 'solar_output', 'wind_output', 'SoC', 'grid_price', 
                'ambient_temperature', 'module_temperature', 'irradiation']
combined_hourly = combined_hourly[final_columns]

# Save to CSV
combined_hourly.to_csv('processed_ems_data.csv')

# Print summary statistics
print("=== Processed EMS Data Summary ===")
print(f"Total records: {len(combined_hourly)}")
print(f"Date range: {combined_hourly.index.min()} to {combined_hourly.index.max()}")
print("\nColumn statistics:")
print(combined_hourly.describe())
print(f"\nData saved to: processed_ems_data.csv")
print("\nFirst few rows:")
print(combined_hourly.head())