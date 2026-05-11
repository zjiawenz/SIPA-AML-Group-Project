#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:07:20 2026

@author: jiawenzou
"""

import streamlit as st
import os

# --- Page Config ---
st.set_page_config(
    page_title="NYISO Price",
    page_icon="⚡",
    layout="wide"
)

# --- Hero Banner Image ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(CURRENT_DIR, "images", "BannerPic.png")

if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_column_width=True)
else:
    st.warning("Banner image not found. Please check: images/BannerPic.png")

# --- Title Section ---
st.markdown("""
# ⚡ NYISO Real-Time Price Forecast
#### Electricity Market · Machine Learning · Real-Time Price Prediction
""")

st.markdown("---")

# --- Project Description ---
st.markdown("""
## Project Overview

This project applies machine learning to forecast the real-time/day-ahead electricity price spread in the New York Independent System Operator (NYISO) market using 2018 hourly data. 

The spread, defined as the difference between the real-time integrated LBMP and the day-ahead market price, is the quantity traded by virtual/convergence bidders and serves as a signal of market forecast accuracy. 

Four models are developed and evaluated: Linear Regression, K-Nearest Neighbours, Random Forest, and XGBoost. 

The study examines predictive performance, variable importance, and the practical implications of model choice for convergence bidding strategy.
""")

st.markdown("---")

# --- Journey Section ---
st.markdown("## App Navigation")

# --- Journey Section with Buttons ---
st.markdown("This app includes **seven sections**:")

rows = [
    ("🔮 **Price Predictor & Convergence Bid Simulator**", "pages/1_🔮_Make_Predictions.py",
     "Predicts the real-time/day-ahead price spread, reconstructs real-time LBMP, and simulates a convergence bidding decision with estimated P&L."),

    ("📂 **Data Sources**", "pages/2_📊_Data_Sources.py",
     "Describes the NYISO 2018 hourly dataset, feature set, target variable construction, preprocessing steps, and model-specific scaling and encoding."),

    ("📈 **Linear Regression Analysis**", "pages/3_📈_Model-Linear.py",
     "Evaluates a linear OLS model across 9 rolling CV folds, including per-fold RMSE, permutation importance, and Ridge regularisation sensitivity."),

    ("🧭 **KNN Analysis**", "pages/4_🧭_Model-KNN.py",
     "Evaluates K-Nearest Neighbours with one-hot encoded features, reporting per-fold RMSE, variable importance, and k sensitivity."),

    ("🌲 **Random Forest Analysis**", "pages/5_🌲_Model-RF.py",
     "Evaluates a 100-tree Random Forest across 9 rolling CV folds, reporting per-fold RMSE, permutation importance, and tree-number sensitivity."),

    ("🚀 **XGBoost Analysis**", "pages/6_🚀_Model-XGBoost.py",
     "Evaluates gradient-boosted trees across 9 rolling CV folds, reporting per-fold RMSE, permutation importance, and boosting-round sensitivity."),

    ("📊 **Model Comparison**", "pages/7_⚖️_Model_Comparison.py",
     "Compares all four models using MSE, RMSE, MAE, EVS, R², residual diagnostics, and bias analysis.")
]

for label, page, description in rows:
    button_col, desc_col = st.columns([1, 3])
    with button_col:
        if st.button(label, key=label):
            st.switch_page(page)
    with desc_col:
        st.markdown(description)
    st.markdown("<br>", unsafe_allow_html=True)


st.markdown("---")

# --- Team Info ---
st.markdown("""
## Meet the Team

- **Akshay Kalyan** — [ak5309@columbia.edu]
- **Terry Zhang** — [tz2645@columbia.edu]
- **Jiawen Zou** — [jz3687@columbia.edu]
""")

st.markdown("---")

# --- Footer ---
st.caption("© 2026 · School of International and Public Affairs · Columbia University")