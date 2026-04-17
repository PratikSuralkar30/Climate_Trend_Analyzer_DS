import pandas as pd
import numpy as np
import os

def generate_synthetic_climate_data(output_path="../data/raw/climate_data.csv"):
    """
    Generates synthetic climate data for multiple regions from 1990 to 2025.
    """
    print("Generating synthetic climate data for multiple regions (1990-2025)...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_date = pd.to_datetime('1990-01-01')
    end_date = pd.to_datetime('2025-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    regions = ['North_India', 'Sub_Saharan', 'East_Asia', 'South_India', 'North_Europe']
    
    data_list = []
    
    # Global trends
    co2_base = 350.0
    co2_trend = np.linspace(0, 70, len(dates)) ** 1.05
    global_co2 = co2_base + co2_trend + np.random.normal(0, 0.5, len(dates))
    
    sea_level_trend = np.linspace(0, 110, len(dates)) + np.random.normal(0, 2, len(dates))
    
    for region in regions:
        if region == 'North_India':
            base_temp = 25.0
            warming_rate = 1.5 / len(dates) # Moderate warming
        elif region == 'Sub_Saharan':
            base_temp = 32.0
            warming_rate = 2.0 / len(dates) # Fast warming
        elif region == 'East_Asia':
            base_temp = 18.0
            warming_rate = 1.2 / len(dates)
        elif region == 'South_India':
            base_temp = 28.0
            warming_rate = 1.4 / len(dates)
        else: # North_Europe
            base_temp = 10.0
            warming_rate = 1.8 / len(dates)

        warming_trend = np.arange(len(dates)) * warming_rate
        seasonal_variation = 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        
        # Add some random spikes to make anomalies more realistic
        noise = np.random.normal(0, 3, len(dates))
        temperature = base_temp + warming_trend + seasonal_variation + noise
        
        # Rainfall and Humidity
        rainfall = np.random.exponential(scale=4, size=len(dates))
        humidity = np.random.uniform(30, 90, size=len(dates))
        
        # Monsoons / rainy seasons
        if 'India' in region or 'Asia' in region:
            rainfall[dates.month.isin([6, 7, 8, 9])] += np.random.exponential(scale=8, size=len(dates[dates.month.isin([6, 7, 8, 9])]))
            humidity[dates.month.isin([6, 7, 8, 9])] = np.random.uniform(70, 100, size=len(dates[dates.month.isin([6, 7, 8, 9])]))
            
        wind_speed = np.random.normal(15, 5, len(dates))
        wind_speed = np.clip(wind_speed, 0, None)
        
        df_region = pd.DataFrame({
            'Date': dates,
            'Region': region,
            'Temperature_C': temperature,
            'Rainfall_mm': rainfall,
            'Humidity_%': humidity,
            'WindSpeed_kmh': wind_speed,
            'CO2_ppm': global_co2,
            'SeaLevel_mm': sea_level_trend
        })
        
        data_list.append(df_region)
        
    df = pd.concat(data_list, ignore_index=True)
    
    # Introduce some missing values (1%)
    mask_missing = np.random.rand(len(df)) < 0.01
    df.loc[mask_missing, ['Temperature_C', 'Rainfall_mm', 'Humidity_%', 'WindSpeed_kmh']] = np.nan
    
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_climate_data()
