import pandas as pd

def calculate_rolling_averages(df, window_size=365):
    """
    Calculates rolling averages per region.
    """
    df_rolling = df.copy()
    df_rolling = df_rolling.sort_values(by=['Region', 'Date'])
    
    df_rolling['Temp_Rolling_Avg'] = df_rolling.groupby('Region')['Temperature_C'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    return df_rolling

def get_yearly_aggregates(df):
    """
    Returns yearly aggregated data.
    """
    yearly_data = df.groupby(['Region', 'Year']).agg({
        'Temperature_C': 'mean',
        'Rainfall_mm': 'sum',
        'Humidity_%': 'mean',
        'CO2_ppm': 'max',
        'SeaLevel_mm': 'max'
    }).reset_index()
    return yearly_data

def get_monthly_aggregates(df):
    return df.groupby(['Region', 'Year', 'Month']).agg({
        'Temperature_C': 'mean',
        'Rainfall_mm': 'sum'
    }).reset_index()

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_climate_data.csv")
    print("Analysis ready.")
