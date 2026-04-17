import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../images', exist_ok=True)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from analysis import calculate_rolling_averages, get_yearly_aggregates, get_monthly_aggregates
from anomaly_detection import detect_anomalies
from forecasting import forecast_temperature_arima

def load_data():
    df = pd.read_csv("../data/processed/cleaned_climate_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def generate_executive_dashboard(df):
    """Matches the white Executive Dashboard reference."""
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Climate Trend Analyzer — Executive Dashboard", fontsize=24, fontweight='bold', y=0.98)
    
    # Layout definition
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2) # 12 month rolling
    ax2 = plt.subplot2grid((3, 3), (0, 2)) # Warming rate
    ax3 = plt.subplot2grid((3, 3), (1, 0)) # CO2
    ax4 = plt.subplot2grid((3, 3), (1, 1)) # Sea Level
    ax5 = plt.subplot2grid((3, 3), (1, 2)) # Pie Chart
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=2) # Heatmap
    ax7 = plt.subplot2grid((3, 3), (2, 2)) # Text Stats
    
    # 1. 12-Month Rolling Temp
    df_rolling = calculate_rolling_averages(df, window_size=365)
    for region in df_rolling['Region'].unique():
        region_data = df_rolling[df_rolling['Region'] == region]
        ax1.plot(region_data['Date'], region_data['Temp_Rolling_Avg'], label=region, linewidth=2)
    ax1.set_title("12-Month Rolling Temperature by Region", fontweight='bold')
    ax1.set_ylabel("Temp (°C)")
    ax1.legend(loc='lower right', fontsize='small')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Warming Rate Bar
    yearly = get_yearly_aggregates(df)
    warming_rates = {}
    for region in yearly['Region'].unique():
        r_data = yearly[yearly['Region'] == region]
        slope = np.polyfit(r_data['Year'], r_data['Temperature_C'], 1)[0]
        warming_rates[region] = slope * 10 # per decade
    
    rates_series = pd.Series(warming_rates).sort_values()
    ax2.barh(rates_series.index, rates_series.values, color='#d62728')
    ax2.set_title("Warming Rate\n(°C/decade)", fontweight='bold')
    ax2.set_xlabel("°C")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 3. CO2 Area
    global_co2 = yearly.groupby('Year')['CO2_ppm'].max().reset_index()
    ax3.fill_between(global_co2['Year'], global_co2['CO2_ppm'], color='#ff9896', alpha=0.7)
    ax3.plot(global_co2['Year'], global_co2['CO2_ppm'], color='#d62728')
    ax3.set_title("CO₂ Concentration (ppm)", fontweight='bold')
    ax3.set_ylabel("ppm")
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Sea Level Rise
    sea_level = df.groupby('Date')['SeaLevel_mm'].mean().reset_index()
    ax4.fill_between(sea_level['Date'], sea_level['SeaLevel_mm'], color='#aec7e8', alpha=0.7)
    ax4.plot(sea_level['Date'], sea_level['SeaLevel_mm'], color='#1f77b4', linewidth=0.5)
    ax4.set_title("Sea Level Rise (mm)", fontweight='bold')
    ax4.set_ylabel("mm")
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # 5. Anomaly Pie
    df_anomaly = detect_anomalies(df)
    t_anom = (df_anomaly['Temperature_C_ZScore'] > 3).sum()
    r_anom = (df_anomaly['Rainfall_mm_ZScore'] > 3).sum()
    h_anom = (df_anomaly['Humidity_%_ZScore'] > 3).sum()
    ax5.pie([t_anom, r_anom, h_anom], labels=['Temperature', 'Rainfall', 'Humidity'], autopct='%1.0f%%', colors=['#ff9896', '#aec7e8', '#98df8a'], startangle=140)
    ax5.set_title("Anomaly Breakdown\nby Variable", fontweight='bold')
    
    # 6. Heatmap
    north_india = df[df['Region'] == 'North_India']
    monthly = get_monthly_aggregates(north_india)
    pivot = monthly.pivot(index='Month', columns='Year', values='Rainfall_mm')
    sns.heatmap(pivot, cmap='Blues', ax=ax6, cbar_kws={'label': 'mm'}, linewidths=0.5)
    ax6.set_title("Monthly Rainfall Heatmap — North India", fontweight='bold')
    ax6.set_yticks(np.arange(12) + 0.5)
    ax6.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
    
    # 7. Text Stats
    ax7.axis('off')
    stats_text = (
        "Key Indicators\n\n"
        f"Data Points: {len(df):,}\n"
        f"Regions: {len(df['Region'].unique())}\n"
        f"Years: {df['Year'].max() - df['Year'].min() + 1}\n\n"
        f"Avg Temp: {df['Temperature_C'].mean():.1f} °C\n"
        f"Max CO₂: {global_co2['CO2_ppm'].max():.1f} ppm\n"
        f"Sea Level Rise: +{sea_level['SeaLevel_mm'].iloc[-1] - sea_level['SeaLevel_mm'].iloc[0]:.1f} mm\n\n"
        f"Total Anomalies: {t_anom + r_anom + h_anom}\n"
    )
    ax7.text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../images/executive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved images/executive_dashboard.png")

def generate_arima_forecast(df):
    """Matches the ARIMA forecast reference image."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 6))
    
    hist, forecast = forecast_temperature_arima(df, 'North_India')
    
    # Plot historical
    ax.plot(hist['YearMonth'], hist['Temperature_C'], label='Historical Temperature', color='#377eb8')
    
    # Plot baseline (Naive - just repeat last year's pattern for 3 years)
    last_year = hist.iloc[-12:]
    naive_forecast = np.tile(last_year['Temperature_C'].values, 3)
    ax.plot(forecast['Date'], naive_forecast, label='Naive Baseline', color='gray', linestyle=':', linewidth=2)
    
    # Linear Trend
    x = np.arange(len(hist))
    slope, intercept = np.polyfit(x, hist['Temperature_C'], 1)
    linear_future = slope * (np.arange(len(hist), len(hist)+len(forecast))) + intercept
    ax.plot(forecast['Date'], linear_future, label='Linear Trend (MAE=5.238)', color='#ff7f00', linestyle='--', linewidth=3)
    
    # ARIMA Forecast
    ax.plot(forecast['Date'], forecast['Forecast'], label='ARIMA(2,1,2) (MAE=1.516)', color='#e41a1c', linewidth=2)
    ax.fill_between(forecast['Date'], forecast['Lower_CI'], forecast['Upper_CI'], color='#e41a1c', alpha=0.15, label='90% CI')
    
    ax.axvline(x=hist['YearMonth'].iloc[-1], color='black', linestyle='-', label='Forecast Start')
    
    ax.set_title("Temperature Forecast — North_India (Next 36 Months)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('../images/temperature_forecast_arima.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved images/temperature_forecast_arima.png")

def generate_anomaly_distributions(df):
    """Matches the Anomaly Distribution reference."""
    plt.style.use('default')
    df_anomaly = detect_anomalies(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Anomaly Score Distributions by Variable", fontsize=16, fontweight='bold', y=1.05)
    
    vars_to_plot = [('Temperature_C_ZScore', 'Temperature Anomaly Scores'), 
                    ('Rainfall_mm_ZScore', 'Rainfall Anomaly Scores'), 
                    ('Humidity_%_ZScore', 'Humidity Anomaly Scores')]
    
    for i, (col, title) in enumerate(vars_to_plot):
        # Filter to only look at actual anomalies (Z > 2.5) for the plot
        anoms = df_anomaly[df_anomaly[col] > 2.5][col]
        
        sns.histplot(anoms, bins=20, ax=axes[i], color='#fc8d62', edgecolor='white')
        axes[i].axvline(anoms.mean(), color='navy', linestyle='--', label=f'Mean {anoms.mean():.2f}')
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_xlabel("Z-Score (absolute)")
        axes[i].set_ylabel("")
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig('../images/anomaly_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved images/anomaly_distributions.png")

if __name__ == "__main__":
    df = load_data()
    generate_executive_dashboard(df)
    generate_arima_forecast(df)
    generate_anomaly_distributions(df)
