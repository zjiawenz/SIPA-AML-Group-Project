#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:23:32 2026

@author: jiawenzou
"""

import os
import numpy as np
import joblib
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS_DIR = "models"   # folder containing the .joblib files

# Zone name -> integer encoding (must match train_and_save_models.py)
ZONE_MAP = {
    "CAPITL": 0,
    "CENTRL": 1,
    "DUNWOD": 2,
    "GENESE": 3,
    "HUD VL": 4,
    "LONGIL": 5,
    "MHK VL": 6,
    "MILLWD": 7,
    "N.Y.C.": 8,
    "NORTH":  9,
    "WEST":   10,
}

FEATURE_ORDER = [
    "DAM Zonal LBMP",
    "DAM Zonal Losses",
    "DAM Zonal Congestion",
    "DAM Forecast Load",
    "month",
    "day",
    "hour",
    "Dew Point",
    "Temperature",
    "Solar Zenith Angle",
    "Relative Humidity",
    "Wind Speed",
    "zone_name",   # integer-encoded
]


# =============================================================================
# LOAD MODELS — cached so Streamlit only loads from disk once per session,
# not on every widget interaction or button press.
# =============================================================================
@st.cache_resource
def load_models():
    """
    Load the pre-trained scaler and all four models from disk.
    @st.cache_resource ensures this runs only once per Streamlit session.
    Returns a dict of {name: model} plus the scaler.
    """
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    models = {
        "Linear Regression": joblib.load(os.path.join(MODELS_DIR, "linear_regression.joblib")),
        "KNN":               joblib.load(os.path.join(MODELS_DIR, "knn.joblib")),
        "Random Forest":     joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "XGBoost":           joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib")),
    }
    return scaler, models


# =============================================================================
# PREDICT — takes raw user input values, scales them, runs all four models.
# =============================================================================
def predict(input_values, scaler, models):
    """
    Scale inputs and return predictions from all four models.

    Parameters
    ----------
    input_values : dict  — {feature_name: value} for all 13 features
    scaler       : fitted StandardScaler
    models       : dict  — {model_name: fitted model}

    Returns
    -------
    predictions : dict  — {model_name: predicted int_LBMP ($/MWh)}
    """
    # Build a 1-row array in the exact feature order the models expect
    row = np.array([[input_values[f] for f in FEATURE_ORDER]])

    # Scale using the training scaler (same statistics used during training)
    row_scaled = scaler.transform(row)

    predictions = {}
    for name, model in models.items():
        predictions[name] = float(model.predict(row_scaled)[0])

    return predictions


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():

    st.title("⚡ NYISO Electricity Price Predictor")
    st.markdown(
        "Enter values for the independent variables below, then click "
        "**Predict** to see the estimated real-time LBMP ($/MWh) from "
        "four machine learning models."
    )

    # ── Load models (cached) ─────────────────────────────────────────────────
    try:
        scaler, models = load_models()
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run `train_and_save_models.py` "
            "first to generate the `models/` folder."
        )
        st.stop()

    # ── Input widgets ─────────────────────────────────────────────────────────
    st.subheader("Input Variables")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Day-Ahead Market**")
        dam_lbmp = st.number_input(
            "DAM Zonal LBMP ($/MWh)",
            value=30.0, min_value=-200.0, max_value=500.0, step=0.1,
            help="Day-ahead market zonal Locational Based Marginal Price",
        )
        dam_losses = st.number_input(
            "DAM Zonal Losses ($/MWh)",
            value=0.5, min_value=-50.0, max_value=50.0, step=0.01,
            help="Day-ahead market marginal cost of losses component",
        )
        dam_congestion = st.number_input(
            "DAM Zonal Congestion ($/MWh)",
            value=0.0, min_value=-200.0, max_value=200.0, step=0.1,
            help="Day-ahead market marginal cost of congestion component",
        )
        dam_load = st.number_input(
            "DAM Forecast Load (MW)",
            value=14000, min_value=5000, max_value=35000, step=100,
            help="Day-ahead forecast of total zonal load",
        )

    with col2:
        st.markdown("**Time**")
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=5,   # default: June
            format_func=lambda m: [
                "Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"
            ][m - 1],
        )
        day = st.slider("Day of Month", min_value=1, max_value=31, value=15)
        hour = st.slider(
            "Hour of Day (0 = midnight, 12 = noon)",
            min_value=0, max_value=23, value=12,
        )
        zone = st.selectbox(
            "NYISO Zone",
            options=sorted(ZONE_MAP.keys()),
            index=sorted(ZONE_MAP.keys()).index("N.Y.C."),
        )

    with col3:
        st.markdown("**Weather**")
        dew_point = st.number_input(
            "Dew Point (°C)",
            value=10.0, min_value=-30.0, max_value=35.0, step=0.1,
        )
        temperature = st.number_input(
            "Temperature (°C)",
            value=20.0, min_value=-30.0, max_value=45.0, step=0.1,
        )
        solar_zenith = st.number_input(
            "Solar Zenith Angle (°)",
            value=45.0, min_value=0.0, max_value=180.0, step=0.1,
            help="Angle of the sun from vertical (0° = directly overhead)",
        )
        humidity = st.number_input(
            "Relative Humidity (%)",
            value=60.0, min_value=0.0, max_value=100.0, step=0.1,
        )
        wind_speed = st.number_input(
            "Wind Speed (m/s)",
            value=5.0, min_value=0.0, max_value=40.0, step=0.1,
        )

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("⚡ Predict Price", type="primary", use_container_width=True):

        # Assemble input dict — zone_name encoded as integer
        input_values = {
            "DAM Zonal LBMP":      dam_lbmp,
            "DAM Zonal Losses":    dam_losses,
            "DAM Zonal Congestion": dam_congestion,
            "DAM Forecast Load":   float(dam_load),
            "month":               float(month),
            "day":                 float(day),
            "hour":                float(hour),
            "Dew Point":           dew_point,
            "Temperature":         temperature,
            "Solar Zenith Angle":  solar_zenith,
            "Relative Humidity":   humidity,
            "Wind Speed":          wind_speed,
            "zone_name":           float(ZONE_MAP[zone]),
        }

        predictions = predict(input_values, scaler, models)

        # ── Display results ───────────────────────────────────────────────────
        st.subheader("Predicted int_LBMP ($/MWh)")

        res_cols = st.columns(4)
        colours  = ["🔵", "🟠", "🟢", "🔴"]  # one per model
        for col, (name, price), colour in zip(res_cols, predictions.items(), colours):
            with col:
                st.metric(
                    label=f"{colour} {name}",
                    value=f"{price:.2f} $/MWh",
                )

        # Ensemble average
        avg = np.mean(list(predictions.values()))
        st.info(f"**Ensemble average (mean of all four models): {avg:.2f} $/MWh**")

        # Expandable detail table
        with st.expander("Show full results table"):
            import pandas as pd
            df_results = pd.DataFrame(
                [(name, f"{price:.4f}") for name, price in predictions.items()],
                columns=["Model", "Predicted int_LBMP ($/MWh)"],
            ).set_index("Model")
            st.dataframe(df_results, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Models trained on NYISO 2018 hourly data (months 1–10) | "
        "Chronological 80/20 train/test split | "
        "13 features including DAM market, time, weather, and zone"
    )


if __name__ == "__main__":
    main()
