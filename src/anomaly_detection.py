import pandas as pd
from scipy.stats import zscore

def detect_anomalies(df):
    """
    Calculates absolute Z-scores for multiple variables to find anomalies.
    """
    print("Calculating Z-scores for anomalies...")
    df_anomaly = df.copy()
    
    variables = ['Temperature_C', 'Rainfall_mm', 'Humidity_%']
    
    for var in variables:
        df_anomaly[f'{var}_ZScore'] = df_anomaly.groupby('Region')[var].transform(lambda x: abs(zscore(x, nan_policy='omit')))
        
    return df_anomaly

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_climate_data.csv")
    df_results = detect_anomalies(df)
    print("Anomaly detection ready.")
