import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from analysis import calculate_rolling_averages, get_yearly_aggregates
from anomaly_detection import detect_anomalies

st.set_page_config(page_title="Climate Trend Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom Dark CSS matching the reference image
st.markdown("""
<style>
    .reportview-container { background: #12141c; color: #FAFAFA; }
    .sidebar .sidebar-content { background: #1e212b; }
    h1, h2, h3 { color: #8cb4f5; }
    .metric-container { background-color: #1e212b; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: #8cb4f5;'>🌍 Climate Trend Analyzer</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>A Data Science project analyzing global climate patterns, anomalies, and future trends (1990-2023)</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/processed/cleaned_climate_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Data not found. Please run data_generator.py and preprocessing.py.")
else:
    # Sidebar
    st.sidebar.header("🎛️ Filters")
    regions = st.sidebar.multiselect("Select Region(s)", df['Region'].unique(), default=df['Region'].unique())
    variable = st.sidebar.selectbox("Climate Variable", ["Temperature_C", "Rainfall_mm", "Humidity_%", "WindSpeed_kmh"])
    years = st.sidebar.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()), (int(df['Year'].min()), int(df['Year'].max())))
    seasons = st.sidebar.multiselect("Season", ["Spring", "Summer", "Autumn", "Winter"], default=["Spring", "Summer", "Autumn", "Winter"])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Run `python main.py` first to generate all reports.")
    
    # Filter Data
    filtered_df = df[(df['Region'].isin(regions)) & 
                     (df['Year'] >= years[0]) & (df['Year'] <= years[1]) &
                     (df['Season'].isin(seasons))]
    
    # Top Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Data Points", f"{len(filtered_df):,}")
    col2.metric("🌡️ Avg Temp", f"{filtered_df['Temperature_C'].mean():.1f} °C")
    col3.metric("🌧️ Avg Rainfall", f"{filtered_df['Rainfall_mm'].mean():.1f} mm")
    col4.metric("💨 Avg CO₂", f"{filtered_df['CO2_ppm'].mean():.1f} ppm")
    col5.metric("🌊 Sea Level", f"{filtered_df['SeaLevel_mm'].mean():.1f} mm")
    
    st.markdown("---")
    
    # Row 1: Time Series & Regional Avg
    col_ts, col_bar = st.columns([2, 1])
    
    with col_ts:
        st.subheader("📈 Temperature Over Time")
        df_rolling = calculate_rolling_averages(filtered_df, window_size=30) # 30 day smooth
        fig_ts = px.line(df_rolling, x='Date', y='Temp_Rolling_Avg', color='Region', template="plotly_dark")
        fig_ts.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with col_bar:
        st.subheader("🗺️ Regional Average")
        reg_avg = filtered_df.groupby('Region')['Temperature_C'].mean().reset_index().sort_values('Temperature_C')
        fig_bar = px.bar(reg_avg, x='Temperature_C', y='Region', orientation='h', color='Temperature_C', color_continuous_scale='Spectral', template="plotly_dark")
        fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # Row 2: Seasonal Distribution & Correlation Matrix
    col_box, col_corr = st.columns([1, 1])
    
    with col_box:
        st.subheader("📅 Seasonal Distribution")
        fig_box = px.box(filtered_df, x='Season', y='Temperature_C', color='Season', template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_corr:
        st.subheader("🔗 Correlation Matrix")
        corr_vars = ['Temperature_C', 'Rainfall_mm', 'Humidity_%', 'CO2_ppm', 'SeaLevel_mm', 'WindSpeed_kmh']
        corr = filtered_df[corr_vars].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='Greens', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # Row 3: Anomalies & YoY
    col_anom, col_yoy = st.columns([1, 1])
    
    with col_anom:
        st.subheader("⚠️ Detected Anomalies")
        df_anomaly = detect_anomalies(filtered_df)
        anoms = df_anomaly[df_anomaly['Temperature_C_ZScore'] > 3]
        fig_anom = px.scatter(anoms, x='Date', y='Temperature_C', color='Region', size='Temperature_C_ZScore', template="plotly_dark")
        st.plotly_chart(fig_anom, use_container_width=True)
        
    with col_yoy:
        st.subheader("📊 Year-over-Year Temperature")
        yoy = filtered_df.groupby(['Year', 'Region'])['Temperature_C'].mean().reset_index()
        fig_yoy = px.line(yoy, x='Year', y='Temperature_C', color='Region', markers=True, template="plotly_dark")
        st.plotly_chart(fig_yoy, use_container_width=True)
