import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def forecast_temperature_arima(df, region, forecast_months=36):
    """
    ARIMA forecast for a specific region's monthly temperature.
    """
    print(f"Running ARIMA forecast for {region}...")
    region_df = df[df['Region'] == region].copy()
    
    # Aggregate to monthly mean to reduce noise for ARIMA
    region_df['YearMonth'] = pd.to_datetime(region_df['Year'].astype(str) + '-' + region_df['Month'].astype(str) + '-01')
    monthly_data = region_df.groupby('YearMonth')['Temperature_C'].mean().reset_index()
    monthly_data.set_index('YearMonth', inplace=True)
    
    # Train ARIMA model
    # (p,d,q) = (2, 1, 2) based on reference image
    model = ARIMA(monthly_data['Temperature_C'], order=(2, 1, 2))
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.get_forecast(steps=forecast_months)
    forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecast': forecast.predicted_mean.values,
        'Lower_CI': forecast.conf_int().iloc[:, 0].values,
        'Upper_CI': forecast.conf_int().iloc[:, 1].values
    })
    
    return monthly_data.reset_index(), forecast_df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_climate_data.csv")
    hist, forecast = forecast_temperature_arima(df, 'North_India')
    print(forecast.head())
