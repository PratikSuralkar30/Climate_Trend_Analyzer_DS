import pandas as pd
import os

def preprocess_data(input_path="../data/raw/climate_data.csv", output_path="../data/processed/cleaned_climate_data.csv"):
    """
    Cleans raw climate data and extracts date components.
    """
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Region', 'Date'])
    
    # Interpolate missing values within each region
    df['Temperature_C'] = df.groupby('Region')['Temperature_C'].transform(lambda x: x.interpolate(method='linear'))
    df['Rainfall_mm'] = df['Rainfall_mm'].fillna(0)
    df['Humidity_%'] = df.groupby('Region')['Humidity_%'].transform(lambda x: x.interpolate(method='linear'))
    df['WindSpeed_kmh'] = df.groupby('Region')['WindSpeed_kmh'].transform(lambda x: x.interpolate(method='linear'))
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Season mapping
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'
    
    df['Season'] = df['Month'].apply(get_season)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    preprocess_data()
