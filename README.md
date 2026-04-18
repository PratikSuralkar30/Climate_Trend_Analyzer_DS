# 🌍 Climate Trend Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

## 📌 Project Overview
The **Climate Trend Analyzer** is an end-to-end data science project designed to ingest historical climate data (Temperature, CO₂, Rainfall), analyze long-term patterns, detect extreme weather anomalies, and forecast future trends. This project demonstrates practical applications of Exploratory Data Analysis (EDA), unsupervised machine learning (Isolation Forest), and Time Series forecasting in the context of sustainability and environmental research.

## 🎯 Problem Statement & Industry Value
**Problem:** Climate change is driving extreme weather events globally. Manually analyzing decades of raw, noisy environmental data to spot trends and anomalies is inefficient.
**Value:** Governments, NGOs, and sustainable tech companies need data-driven models to track global warming and predict future conditions. This tool automates the ingestion, cleaning, and visualization of climate metrics, serving as a prototype for advanced environmental intelligence systems.

## ⚙️ Tech Stack
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Isolation Forest, Linear Regression)
*   **Visualization:** Plotly, Matplotlib, Seaborn
*   **Web App / Dashboard:** Streamlit (Dark Theme configured)

## 📂 Folder Structure
```text
Climate-Trend-Analyzer/
│
├── data/
│   ├── raw/                 # Generated synthetic data goes here
│   └── processed/           # Cleaned data
├── src/
│   ├── data_generator.py    # Generates 50 years of synthetic climate data
│   ├── preprocessing.py     # Cleans data and handles missing values
│   ├── analysis.py          # Calculates rolling averages and yearly aggregates
│   ├── anomaly_detection.py # Uses Isolation Forest to detect extreme events
│   └── forecasting.py       # Predicts future temperatures
├── app/
│   └── dashboard.py         # Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## 🚀 How to Run

1. **Clone the repository and navigate to the folder:**
   ```bash
   git clone <your-repo-link>
   cd Climate-Trend-Analyzer
   ```
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate and process the data:**
   ```bash
   python src/data_generator.py
   python src/preprocessing.py
   ```
4. **Run the Dashboard:**
   ```bash
   streamlit run app/dashboard.py
   ```

## 🧠 Learning Outcomes
- Handling missing data in time-series datasets.
- Smoothing noisy data using rolling averages.
- Implementing Unsupervised ML for anomaly detection.
- Forecasting using Regression.
- Building interactive analytical web applications.

## 👤 Author
*Pratik Suralkar*
