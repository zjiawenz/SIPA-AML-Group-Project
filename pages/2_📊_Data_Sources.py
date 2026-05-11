#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:24:53 2026

@author: jiawenzou
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Sources",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .method-block {
    background: #f5f6ff;
    border-left: 4px solid #3949ab;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
  }
    .insight-box {
        background: #eef2ff;
        border-left: 4px solid #3949ab;
        border-radius: 4px;
        padding: 0.9rem 1.3rem;
        margin-top: 0.8rem;
        margin-bottom: 1.6rem;
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .insight-box strong {
        color: #1a237e;
    }
  .tag {
    display: inline-block;
    background: #3949ab;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 1px 8px;
    border-radius: 20px;
    margin-right: 5px;
    vertical-align: middle;
  }

  /* results table */
  .results-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    margin-bottom: 1rem;
  }
  .results-table th {
    background: #3949ab;
    color: white;
    padding: 8px 14px;
    text-align: right;
    font-weight: 600;
  }
  .results-table th:first-child { text-align: left; }
  .results-table td {
    padding: 7px 14px;
    text-align: right;
    border-bottom: 1px solid #e8eaf6;
  }
  .results-table td:first-child { text-align: left; font-weight: 600; }
  .results-table tr:nth-child(even) { background: #f5f6ff; }
  .results-table tr:hover { background: #e8eaf6; }
  .best { color: #2d6a4f; font-weight: 700; }
  .worst { color: #c62828; }

  hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Data Sources")

st.markdown("---")

# -----------------------------
# Data Sources
# -----------------------------

st.header("1. Databases Used")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ NYISO Energy Market & Operational Data")
    st.markdown("""
    **Source:** New York Independent System Operator  
    
    **Original link:**  
    https://www.nyiso.com/energy-market-operational-data

    NYISO operates New York’s wholesale electricity markets and provides market and operational data 
    related to electricity pricing, load, grid conditions, and market outcomes.

    According to NYISO, its energy markets allow market participants to buy and sell energy and 
    ancillary services through **real-time and day-ahead auctions**. These auctions are designed to 
    meet electricity demand using the lowest-cost available resources while responding to changing 
    operating conditions.

    In this project, NYISO data provides the core electricity market predictors, especially 
    day-ahead market price components and forecast load variables.
    """)

with col2:
    st.subheader("🌦️ PSML Weather and Environmental Data")
    st.markdown("""
    **Source:** PSML Dataset on Zenodo  
    
    **Original link:**  
    https://zenodo.org/records/5130612#.YTIiZI5KiUk

    PSML stands for **A Multi-scale Time-series Dataset for Machine Learning in Decarbonized Energy Grids**. 
    It is an open-access dataset designed to support machine-learning research for future electricity grids.

    The dataset includes electric load, renewable generation, weather, voltage, and current measurements 
    across multiple spatial and temporal scales. It was created to support ML applications such as load 
    and renewable forecasting, disturbance detection, and synthetic power-system time-series generation.

    In this project, PSML provides weather and environmental predictors, including temperature, humidity, 
    wind speed, dew point, and solar zenith angle.
    """)

st.markdown("---")

# -----------------------------
# Data Cleaning Process
# -----------------------------

st.header("2. Data Cleaning Process")

st.markdown("""
### 🔗 Merge datasets
- Load raw NYISO electricity market data
- Load raw PSML weather and environmental data
- Standardize timestamp formats and NYISO zone coding across datasets
- Match NYISO market observations with weather observations by time and NYISO zone

### ⚙️ Transform variables
- Aggregate minute-level observations into hourly observations by taking the average value within each hour
- Construct time-based features, including month, day of the week, and hour of the day, from the original datetime variable
- Encode NYISO zones as categorical variables using one-hot encoding
- Handle missing values and check for abnormal or extreme observations

### 🤖 Prepare data for machine learning models
- Apply one-hot encoding to month, day-of-week, and hour variables (Linear Regression and KNN models only)
- Apply feature scaling (standardization) to all continuous variables (KNN model only)
- Construct the final model-ready feature matrix
- Split the dataset into training and testing sets using rolling window cross-validation
""")

st.markdown(f"""
<div class="insight-box">
<strong> Why are these transformations necessary?</strong>

- One-hot encoding is required for Linear Regression and KNN because these models cannot directly interpret categorical variables such as NYISO zones or calendar indicators.
- Feature scaling is necessary for KNN because distance-based models are highly sensitive to variable magnitudes; scaling prevents variables with larger numerical ranges from dominating the distance calculation.
- Rolling window cross-validation is used instead of random train-test splitting because electricity price data is time-series data. This prevents future information from leaking into past predictions and provides a more realistic evaluation of out-of-sample forecasting performance.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Variables Used
# -----------------------------

st.header("3. Variables Used in the Forecasting Model")

st.markdown("""
The forecasting models use a combination of market, temporal, spatial, and weather-related variables 
to predict NYISO real-time electricity price spread dynamics.
""")

# -----------------------------
# Outcome Variable
# -----------------------------

st.subheader("🎯 Outcome Variable")

outcome = pd.DataFrame({
    "Variable": ["Spread"],
    "Definition": ["Real-time LBMP − Day-ahead Zonal LBMP"],
    "Description": [
        "Measures the deviation between real-time and day-ahead electricity prices. "
        "Positive values indicate real-time prices exceeded day-ahead expectations, "
        "while negative values indicate lower-than-expected real-time prices."
    ]
})

outcome.index = outcome.index + 1

st.table(outcome)

st.markdown("""
The primary objective of the forecasting models is to predict the real-time price spread, 
which reflects short-term market volatility and forecast deviations in the NYISO electricity market.
""")

st.markdown("---")

# -----------------------------
# Predictor Variables
# -----------------------------

st.subheader("📌 Predictor Variables")

st.markdown("""
The following variables are used as model features for prediction.
""")

variables = pd.DataFrame({
    "Category": [
        "Day-Ahead Market",
        "Day-Ahead Market",
        "Day-Ahead Market",
        "Time",
        "Time",
        "Time",
        "Location",
        "Weather / Environmental",
        "Weather / Environmental",
        "Weather / Environmental",
        "Weather / Environmental",
        "Weather / Environmental",
    ],
    "Variable": [
        "DAM Zonal Losses",
        "DAM Zonal Congestion",
        "DAM Forecast Load",
        "Month",
        "Day of Week",
        "Hour of Day",
        "NYISO Zone",
        "Dew Point",
        "Temperature",
        "Solar Zenith Angle",
        "Relative Humidity",
        "Wind Speed",
    ],
    "Description": [
        "Day-ahead market loss component.",
        "Day-ahead market congestion component.",
        "Forecasted electricity load from the day-ahead market.",
        "Month of the observation.",
        "Day of the week.",
        "Hour of the day.",
        "NYISO electricity market zone.",
        "Atmospheric dew point in degrees Celsius.",
        "Air temperature in degrees Celsius.",
        "Angle between the sun’s rays and the vertical direction.",
        "Relative humidity percentage.",
        "Wind speed measured in meters per second.",
    ],
    "Source": [
        "NYISO",
        "NYISO",
        "NYISO",
        "Constructed",
        "Constructed",
        "Constructed",
        "NYISO",
        "PSML",
        "PSML",
        "PSML",
        "PSML",
        "PSML",
    ]
})

# Start table index from 1
variables.index = variables.index + 1

st.table(variables)

st.markdown(f"""
<div class="insight-box">
<strong> Role of the Predictor Variables: </strong>

<strong> Time Variables (Month, Hour of Day, Day of Week): </strong>
These three variables capture the cyclical patterns in electricity demand and price behaviour across different time scales. Month reflects broad seasonal variation in load and generation mix; hour of day captures the intra-day demand cycle with consistent morning and evening peaks; and day of week distinguishes weekday industrial and commercial load from lower weekend demand. Day of week is derived from the input date at prediction time — day of month itself is not used as a model feature.
<br><br>

<strong> Day-Ahead Market Variables (DAM Zonal LBMP, Losses, Congestion, DAM Forecast Load): </strong>
These four variables capture the day-ahead market's expectations for the following day's prices and demand. DAM Zonal LBMP forms the anchor for our spread target and is excluded as a predictor; the remaining three components — losses, congestion, and forecast load — represent the primary market signals available before real-time settlement and are key predictors of where and why real-time prices deviate from day-ahead forecasts.
<br><br>

<strong> Weather & Environmental Variables (Temperature, Dew Point, Relative Humidity, Wind Speed, Solar Zenith Angle): </strong>
These five variables capture the atmospheric conditions that drive electricity demand and renewable generation output. Temperature and dew point together determine cooling and heating load; humidity amplifies heat stress and air conditioning demand; wind speed directly determines wind generation output; and solar zenith angle captures the availability of solar irradiance as a function of time of day and season. Unexpected deviations from forecast weather conditions are a primary source of real-time price divergence from day-ahead expectations.

</div>
""", unsafe_allow_html=True)

st.markdown("---")
