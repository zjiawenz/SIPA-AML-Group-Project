#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_Make_Predictions.py
=============================================================================
NYISO Price Predictor + Convergence Bid Simulator  (merged page)

Target : spread = int_LBMP − DAM Zonal LBMP
Displays: predicted spread, reconstructed int_LBMP, and a Convergence Bid
          Simulator that reuses the same predictions without re-entry.

Session state design
─────────────────────
Streamlit re-runs the entire script on every widget interaction. Without
session state, clicking "Run Simulation" clears the prediction results
because predict_clicked is False on that re-run.

Fix: when the user clicks ⚡ Predict, all results are saved to
st.session_state. The simulator reads from session state, so it persists
independently of which button triggered the re-run.

Key session state keys
───────────────────────
  predictions_ready   bool  — True once Predict has been clicked
  all_spreads         dict  — {model: spread} incl. Ensemble
  spread_preds        dict  — {model: spread} excl. Ensemble (for charts)
  lbmp_preds          dict  — {model: reconstructed int_LBMP}
  avg_spread          float — ensemble mean spread
  avg_lbmp            float — ensemble mean int_LBMP
  dam_anchor          float — DAM Zonal LBMP entered by user
  zone                str
  selected_date       date
  hour                int
  dow_names_display   str   — e.g. "Mon (0)"

Model families
──────────────
  RF, XGBoost  : 12 raw features (no encoding, no scaling)
  KNN, Linear  : 8 continuous (scaled) + OHE month/hour/day_of_week/zone
                 → 62 features total
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

# =============================================================================
# Configuration
# =============================================================================
MODELS_DIR               = "models_spread"
TRANSACTION_COST_PER_MWH = 0.10   # $/MWh — approximate NYISO virtual bid fee

ZONE_MAP = {
    "CAPITL": 0, "CENTRL": 1, "DUNWOD": 2, "GENESE": 3,
    "HUD VL": 4, "LONGIL": 5, "MHK VL": 6, "MILLWD": 7,
    "N.Y.C.": 8, "NORTH":  9, "WEST":   10,
}

MODEL_COLORS = {
    "Linear Regression": "#C44E52",
    "KNN":               "#4C72B0",
    "Random Forest":     "#2d6a4f",
    "XGBoost":           "#DD8452",
}
MODEL_ICONS = {
    "Linear Regression": "📈",
    "KNN":               "🧭",
    "Random Forest":     "🌲",
    "XGBoost":           "🚀",
}
SIM_COLOURS = {
    "Linear Regression": "🔴",
    "KNN":               "🔵",
    "Random Forest":     "🟢",
    "XGBoost":           "🟠",
    "Ensemble":          "⚪",
}

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="NYISO Price Predictor",
    page_icon="🔮",
    layout="wide",
)

st.markdown("""
<style>
  .result-box {
    background: #fff0f0;
    border-left: 5px solid #C44E52;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
  }
  .sim-box {
    background: #fffbeb;
    border-left: 5px solid #d97706;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
  }
  .model-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 20px;
    margin-right: 4px;
    color: white;
  }
  hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialise session state keys
# =============================================================================
# Done at module level so keys always exist before any widget references them.
if "predictions_ready" not in st.session_state:
    st.session_state["predictions_ready"] = False


# =============================================================================
# Load models & metadata (cached — runs once per session)
# =============================================================================
@st.cache_resource
def load_spread_models():
    def p(fname):
        return os.path.join(MODELS_DIR, fname)
    models = {
        "Linear Regression": joblib.load(p("linear_regression_spread.joblib")),
        "KNN":               joblib.load(p("knn_spread.joblib")),
        "Random Forest":     joblib.load(p("random_forest_spread.joblib")),
        "XGBoost":           joblib.load(p("xgboost_spread.joblib")),
    }
    scalers = {
        "Linear Regression": joblib.load(p("scaler_lr_spread.joblib")),
        "KNN":               joblib.load(p("scaler_knn_spread.joblib")),
    }
    tree_features    = joblib.load(p("feature_list_tree.joblib"))
    encoded_features = joblib.load(p("feature_list_encoded.joblib"))
    continuous_cols  = joblib.load(p("continuous_cols.joblib"))
    ohe_cols         = joblib.load(p("ohe_cols.joblib"))
    ohe_structure    = joblib.load(p("ohe_structure.joblib"))
    return (models, scalers, tree_features, encoded_features,
            continuous_cols, ohe_cols, ohe_structure)


# =============================================================================
# Feature construction helpers
# =============================================================================
def build_tree_row(raw_inputs, tree_features):
    """12-feature vector for RF and XGBoost — raw values, no scaling."""
    return np.array([[raw_inputs[f] for f in tree_features]], dtype=float)


def build_encoded_row(raw_inputs, continuous_cols, ohe_cols,
                      ohe_structure, encoded_features, scaler):
    """
    62-feature vector for KNN and Linear Regression.
    Continuous columns are scaled using the saved scaler.
    OHE binary columns are NOT scaled — already on [0,1].
    Column order matches encoded_features exactly.
    """
    row_dict = {col: raw_inputs[col] for col in continuous_cols}
    for col in ohe_cols:
        val = int(raw_inputs[col])
        for cat in ohe_structure[col]:
            row_dict[f"{col}_{cat}"] = 1 if val == cat else 0
    row_vec  = np.array([[row_dict[f] for f in encoded_features]], dtype=float)
    cont_idx = [encoded_features.index(c) for c in continuous_cols]
    row_vec[:, cont_idx] = scaler.transform(row_vec[:, cont_idx])
    return row_vec


def predict_all_spreads(raw_inputs, models, scalers, tree_features,
                        encoded_features, continuous_cols, ohe_cols, ohe_structure):
    """
    Run all four models and return a dict of predicted spreads,
    including an Ensemble key (simple mean of the four).
    """
    row_tree = build_tree_row(raw_inputs, tree_features)
    results  = {}
    for name, model in models.items():
        if name in ("Random Forest", "XGBoost"):
            results[name] = float(model.predict(row_tree)[0])
        else:
            row_enc = build_encoded_row(
                raw_inputs, continuous_cols, ohe_cols,
                ohe_structure, encoded_features, scalers[name]
            )
            results[name] = float(model.predict(row_enc)[0])
    results["Ensemble"] = float(np.mean(list(results.values())))
    return results


# =============================================================================
# Simulator helpers
# =============================================================================
def apply_decision_rule(predicted_spread, threshold):
    """
    Virtual bidding decision rule:
      spread >  +threshold → Virtual Load   (buy DAM, sell RTM)  direction=+1
      spread < −threshold  → Virtual Supply  (sell DAM, buy RTM) direction=−1
      |spread| ≤ threshold → No Trade                             direction= 0
    Returns (decision_str, direction_int, explanation_str).
    """
    if predicted_spread > threshold:
        return (
            "Virtual Load", +1,
            f"Predicted spread ({predicted_spread:+.2f} $/MWh) exceeds "
            f"+{threshold:.2f}. Real-time expected HIGHER than day-ahead. "
            "Strategy: buy in day-ahead, sell in real-time."
        )
    elif predicted_spread < -threshold:
        return (
            "Virtual Supply", -1,
            f"Predicted spread ({predicted_spread:+.2f} $/MWh) is below "
            f"−{threshold:.2f}. Real-time expected LOWER than day-ahead. "
            "Strategy: sell in day-ahead, buy in real-time."
        )
    else:
        return (
            "No Trade", 0,
            f"Predicted spread ({predicted_spread:+.2f} $/MWh) is within "
            f"±{threshold:.2f}. Signal too weak — no position taken."
        )


def calculate_pnl(predicted_spread, direction, trade_size_mw):
    """
    Gross P&L = direction × predicted_spread × trade_size_mw
    Net  P&L  = Gross P&L − transaction_cost
    transaction_cost only applied when direction ≠ 0.
    Returns (gross_pnl, transaction_cost, net_pnl).
    """
    gross_pnl        = direction * predicted_spread * trade_size_mw
    transaction_cost = abs(direction) * TRANSACTION_COST_PER_MWH * trade_size_mw
    net_pnl          = gross_pnl - transaction_cost
    return gross_pnl, transaction_cost, net_pnl


# =============================================================================
# Helper: render prediction results from session state
# =============================================================================
def render_prediction_results():
    """
    Render the full prediction results section using values stored in
    st.session_state. Called both immediately after Predict is clicked
    and on subsequent re-runs (e.g. when the user adjusts simulator inputs).
    """
    all_spreads  = st.session_state["all_spreads"]
    spread_preds = st.session_state["spread_preds"]
    lbmp_preds   = st.session_state["lbmp_preds"]
    avg_spread   = st.session_state["avg_spread"]
    avg_lbmp     = st.session_state["avg_lbmp"]
    dam_anchor   = st.session_state["dam_anchor"]
    zone         = st.session_state["zone"]
    selected_date= st.session_state["selected_date"]
    hour         = st.session_state["hour"]

    st.markdown("## Prediction Results")

    # ── Per-model metric cards ────────────────────────────────────────────────
    cols4 = st.columns(4)
    for col, name in zip(cols4, ["Linear Regression", "KNN", "Random Forest", "XGBoost"]):
        sp   = spread_preds[name]
        lbmp = lbmp_preds[name]
        with col:
            st.markdown(
                f'<span class="model-tag" style="background:{MODEL_COLORS[name]};">'
                f'{MODEL_ICONS[name]} {name}</span>',
                unsafe_allow_html=True
            )
            st.metric(label="Spread  ($/MWh)",   value=f"{sp:+.2f}")
            st.metric(label="int_LBMP  ($/MWh)", value=f"{lbmp:.2f}",
                      delta=f"{sp:+.2f} vs DAM")

    # ── Ensemble summary box ──────────────────────────────────────────────────
    spread_sign   = "positive" if avg_spread >= 0 else "negative"
    spread_interp = (
        "Real-time prices are running <strong>above</strong> the day-ahead forecast, "
        "likely driven by higher-than-expected demand, congestion, or weather stress."
        if avg_spread >= 0 else
        "Real-time prices are running <strong>below</strong> the day-ahead forecast — "
        "demand came in lighter than forecast or supply conditions eased."
    )
    st.markdown(f"""
<div class="result-box">
<strong>Ensemble average spread: {avg_spread:+.2f} $/MWh ({spread_sign})</strong><br>
<strong>Reconstructed int_LBMP: {avg_lbmp:.2f} $/MWh</strong>
&nbsp;=&nbsp; DAM {dam_anchor:.2f} + spread {avg_spread:+.2f} $/MWh
<br><br>{spread_interp}
</div>
""", unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Summary Table")
    df_out = pd.DataFrame({
        "Model":                          list(spread_preds.keys()),
        "Predicted Spread ($/MWh)":       [f"{v:+.4f}" for v in spread_preds.values()],
        "Reconstructed int_LBMP ($/MWh)": [f"{v:.4f}"  for v in lbmp_preds.values()],
        "DAM Zonal LBMP ($/MWh)":         [f"{dam_anchor:.4f}"] * 4,
    }).set_index("Model")
    avg_row = pd.DataFrame({
        "Predicted Spread ($/MWh)":       [f"{avg_spread:+.4f}"],
        "Reconstructed int_LBMP ($/MWh)": [f"{avg_lbmp:.4f}"],
        "DAM Zonal LBMP ($/MWh)":         [f"{dam_anchor:.4f}"],
    }, index=["Ensemble average"])
    st.dataframe(pd.concat([df_out, avg_row]), use_container_width=True)

    # ── Spread bar chart ──────────────────────────────────────────────────────
    st.markdown("### Spread Predictions vs DAM Baseline")
    names   = list(spread_preds.keys())
    sp_vals = [spread_preds[n] for n in names]
    colors  = [MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#f7f9ff"); ax.set_facecolor("#f7f9ff")
    bars = ax.bar([f"{MODEL_ICONS[n]}\n{n}" for n in names], sp_vals,
                  color=colors, edgecolor="white", width=0.55)
    ax.axhline(0, color="#333", lw=1.3, ls="--", label="DAM LBMP (spread = 0)")
    for bar, val in zip(bars, sp_vals):
        spread_range = max(abs(v) for v in sp_vals) if any(sp_vals) else 1
        offset = spread_range * 0.06 * (1 if val >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted spread ($/MWh)", fontsize=10)
    ax.set_title(
        f"Predicted Spread  |  Zone: {zone}  |  "
        f"{selected_date.strftime('%Y-%m-%d')}  {hour:02d}:00",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── int_LBMP reconstruction bar chart ────────────────────────────────────
    lbmp_vals = [lbmp_preds[n] for n in names]
    fig2, ax2 = plt.subplots(figsize=(9, 3.8))
    fig2.patch.set_facecolor("#f7f9ff"); ax2.set_facecolor("#f7f9ff")
    bars2 = ax2.bar([f"{MODEL_ICONS[n]}\n{n}" for n in names], lbmp_vals,
                    color=colors, edgecolor="white", width=0.55,
                    label="Predicted int_LBMP")
    ax2.axhline(dam_anchor, color="#3949ab", lw=1.6, ls="--",
                label=f"DAM Zonal LBMP = {dam_anchor:.2f} $/MWh")
    for bar, val in zip(bars2, lbmp_vals):
        lbmp_range = max(abs(v) for v in lbmp_vals) if any(lbmp_vals) else 1
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + lbmp_range * 0.015,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("int_LBMP ($/MWh)", fontsize=10)
    ax2.set_title(
        f"Reconstructed int_LBMP  |  Zone: {zone}  |  "
        f"{selected_date.strftime('%Y-%m-%d')}  {hour:02d}:00",
        fontsize=10, fontweight="bold"
    )
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ── Input summary (expandable) ────────────────────────────────────────────
    with st.expander("Show full input values"):
        si = st.session_state["raw_inputs_display"]
        st.dataframe(
            pd.DataFrame(si, columns=["Variable", "Value"]).set_index("Variable"),
            use_container_width=True
        )
        st.caption(
            "Note: `day_of_month` is used only to derive `day_of_week` and is "
            "not passed to any model as a feature."
        )


# =============================================================================
# Helper: render simulator section from session state
# =============================================================================
def render_simulator():
    """
    Render the Convergence Bid Simulator using predictions already stored
    in session state. The three trading parameter widgets live here and
    trigger re-runs, but because predictions are in session state the
    results panel above is always re-rendered correctly.
    """
    all_spreads = st.session_state["all_spreads"]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## ⚡ Convergence Bid Simulator")
    st.markdown("""
<div class="sim-box">
The spread predictions above are used directly here. Choose a model,
set your trading threshold and position size, then click
<strong>Run Simulation</strong> to see the trading decision and estimated P&amp;L.
<br><br>
<strong>P&amp;L is estimated from the <em>predicted</em> spread</strong> — actual P&amp;L
depends on the real-time market spread, which is only known after settlement.
</div>
""", unsafe_allow_html=True)

    # ── Trading parameter inputs ──────────────────────────────────────────────
    st.markdown("### Trading Parameters")
    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        model_choice = st.selectbox(
            "Model for trading decision",
            options=["Linear Regression", "KNN", "Random Forest", "XGBoost", "Ensemble"],
            index=4,
            key="sim_model_choice",
            help="Ensemble = simple average of all four model predictions.",
        )
    with sim_col2:
        threshold = st.slider(
            "Threshold ($/MWh)",
            min_value=0.0, max_value=20.0, value=2.0, step=0.5,
            key="sim_threshold",
            help="Minimum predicted spread magnitude required to place a trade.",
        )
    with sim_col3:
        trade_size = st.number_input(
            "Trade Size (MW)",
            min_value=1.0, max_value=500.0, value=10.0, step=1.0,
            key="sim_trade_size",
            help="Notional position size in MW. P&L scales linearly with trade size.",
        )

    # ── Run Simulation button ─────────────────────────────────────────────────
    sim_clicked = st.button(
        "📊  Run Simulation", type="secondary",
        use_container_width=True, key="sim_button"
    )

    if sim_clicked:
        chosen_spread = all_spreads[model_choice]
        decision, direction, explanation = apply_decision_rule(chosen_spread, threshold)
        gross_pnl, tx_cost, net_pnl = calculate_pnl(chosen_spread, direction, trade_size)

        # Store simulation results in session state so they survive further
        # widget interactions within this same page visit
        st.session_state["sim_results"] = {
            "model_choice":   model_choice,
            "chosen_spread":  chosen_spread,
            "threshold":      threshold,
            "trade_size":     trade_size,
            "decision":       decision,
            "direction":      direction,
            "explanation":    explanation,
            "gross_pnl":      gross_pnl,
            "tx_cost":        tx_cost,
            "net_pnl":        net_pnl,
        }

    # ── Show simulation output if results exist in session state ──────────────
    # This persists across re-runs until the user clicks Predict again
    # (which clears sim_results via the predictions block below).
    if "sim_results" in st.session_state:
        sr           = st.session_state["sim_results"]
        model_choice = sr["model_choice"]
        chosen_spread= sr["chosen_spread"]
        threshold    = sr["threshold"]
        trade_size   = sr["trade_size"]
        decision     = sr["decision"]
        direction    = sr["direction"]
        explanation  = sr["explanation"]
        gross_pnl    = sr["gross_pnl"]
        tx_cost      = sr["tx_cost"]
        net_pnl      = sr["net_pnl"]

        # All model spread metric cards
        st.markdown("#### Spread predictions used")
        pred_cols = st.columns(5)
        for col, (name, spread) in zip(pred_cols, all_spreads.items()):
            with col:
                st.metric(
                    label=f"{SIM_COLOURS[name]} {name}"
                          + (" ✓" if name == model_choice else ""),
                    value=f"{spread:+.2f} $/MWh",
                )

        st.markdown("---")

        # Decision badge
        st.markdown("#### Trading Decision")
        if decision == "Virtual Load":
            st.success(f"📈 **{decision}**")
        elif decision == "Virtual Supply":
            st.warning(f"📉 **{decision}**")
        else:
            st.info(f"⏸️ **{decision}**")
        st.markdown(f"> {explanation}")
        st.markdown("---")

        # P&L metrics
        st.markdown("#### Estimated P&L")
        pnl_col1, pnl_col2, pnl_col3, pnl_col4 = st.columns(4)
        with pnl_col1:
            st.metric("Model Used", model_choice)
        with pnl_col2:
            st.metric("Gross P&L", f"${gross_pnl:+.2f}",
                      help="direction × predicted_spread × trade_size_MW")
        with pnl_col3:
            st.metric("Transaction Cost", f"-${tx_cost:.2f}",
                      help=f"~${TRANSACTION_COST_PER_MWH}/MWh × {trade_size:.0f} MW")
        with pnl_col4:
            st.metric(
                "Net P&L", f"${net_pnl:+.2f}",
                delta="Profit" if net_pnl > 0 else ("Loss" if net_pnl < 0 else "Break-even"),
                delta_color="normal" if net_pnl >= 0 else "inverse",
            )

        # Full breakdown
        with st.expander("Show full calculation breakdown"):
            st.markdown(f"""
| Item | Value |
|---|---|
| Model | {model_choice} |
| Predicted spread | {chosen_spread:+.4f} $/MWh |
| Threshold | ±{threshold:.2f} $/MWh |
| Decision | {decision} |
| Direction multiplier | {direction:+d} |
| Trade size | {trade_size:.0f} MW |
| Gross P&L | {direction} × {chosen_spread:.4f} × {trade_size:.0f} = **${gross_pnl:+.2f}** |
| Transaction cost | {TRANSACTION_COST_PER_MWH} × {trade_size:.0f} = **-${tx_cost:.2f}** |
| **Net P&L** | **${net_pnl:+.2f}** |
""")
            if direction != 0:
                st.caption(
                    f"Break-even spread: ±{TRANSACTION_COST_PER_MWH:.2f} $/MWh "
                    "(transaction cost per MWh)"
                )

        st.markdown("---")
        st.caption(
            "⚠️ **Disclaimer:** Estimated P&L is based on the *predicted* spread from a "
            "model trained on 2018 NYISO data. Actual P&L depends on the real-time market "
            "spread, which is unknown until after settlement. This tool is for research and "
            "educational purposes only and does not constitute financial or trading advice. "
        )


# =============================================================================
# HEADER
# =============================================================================
st.title("🔮  NYISO Electricity Price Predictor")
st.markdown(
    "Enter time, weather, market, and zone information below, then click "
    "**Predict** to estimate the real-time integrated LBMP using four ML models. "
    "A **Convergence Bid Simulator** will appear below the predicted prices."
)

# =============================================================================
# Load models
# =============================================================================
try:
    (models, scalers, tree_features, encoded_features,
     continuous_cols, ohe_cols, ohe_structure) = load_spread_models()
except FileNotFoundError as e:
    st.error(
        f"Model files not found in `{MODELS_DIR}/`. "
        "Run `train_spread_models.py` first to generate them.\n\n"
        f"Missing: `{e.filename}`"
    )
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# INPUT FORM
# Always rendered — widgets must exist on every re-run for Streamlit to
# track their state, even if the user is just clicking Run Simulation.
# =============================================================================

# ── Zone ─────────────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Zone")
zone_cols = st.columns([1, 2, 1])
with zone_cols[1]:
    zone = st.selectbox(
        "NYISO Zone",
        options=sorted(ZONE_MAP.keys()),
        index=sorted(ZONE_MAP.keys()).index("N.Y.C."),
        key="input_zone",
    )

st.markdown("---")

# ── Time | Weather | Day-Ahead ────────────────────────────────────────────────
col_time, col_weather, col_dam = st.columns(3)

with col_time:
    st.markdown("### ⏰ Time")
    selected_date = st.date_input("Select Date", value=date(2018, 6, 15),
                                  key="input_date")
    hour = st.selectbox(
        "Hour of Day",
        options=list(range(24)), index=12,
        format_func=lambda h: f"{h:02d}:00",
        key="input_hour",
    )
    month        = selected_date.month
    day_of_week  = selected_date.weekday()
    dow_names    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.caption(
        f"{selected_date.strftime('%Y-%m-%d')} &nbsp;·&nbsp; "
        f"Month {month} &nbsp;·&nbsp; {dow_names[day_of_week]}"
    )

with col_weather:
    st.markdown("### 🌤 Weather")
    dew_point    = st.number_input("Dew Point (°C)",         value=10.0,
                                   min_value=-30.0, max_value=35.0,  step=0.1,
                                   format="%.1f", key="input_dew")
    temperature  = st.number_input("Temperature (°C)",       value=20.0,
                                   min_value=-30.0, max_value=45.0,  step=0.1,
                                   format="%.1f", key="input_temp")
    solar_zenith = st.number_input("Solar Zenith Angle (°)", value=45.0,
                                   min_value=0.0,   max_value=180.0, step=0.1,
                                   format="%.1f", key="input_solar",
                                   help="Angle of the sun from vertical.")
    humidity     = st.number_input("Relative Humidity (%)",  value=60.0,
                                   min_value=0.0,   max_value=100.0, step=0.1,
                                   format="%.1f", key="input_humidity")
    wind_speed   = st.number_input("Wind Speed (m/s)",       value=5.0,
                                   min_value=0.0,   max_value=40.0,  step=0.1,
                                   format="%.1f", key="input_wind")

with col_dam:
    st.markdown("### 📋 Day-Ahead Market")
    dam_lbmp       = st.number_input("DAM Zonal LBMP ($/MWh)",      value=30.0,
                                     min_value=-200.0, max_value=500.0,  step=0.1,
                                     format="%.1f", key="input_dam_lbmp",
                                     help="Used to reconstruct int_LBMP = DAM LBMP + spread.")
    dam_losses     = st.number_input("DAM Zonal Losses ($/MWh)",     value=0.5,
                                     min_value=-50.0,  max_value=50.0,   step=0.1,
                                     format="%.1f", key="input_losses")
    dam_congestion = st.number_input("DAM Zonal Congestion ($/MWh)", value=0.0,
                                     min_value=-200.0, max_value=200.0,  step=0.1,
                                     format="%.1f", key="input_congestion")
    dam_load       = st.number_input("DAM Forecast Load (MW)",       value=5000,
                                     min_value=0, max_value=35000, step=1,
                                     key="input_load")

st.markdown("---")

# =============================================================================
# PREDICT BUTTON
# =============================================================================
predict_clicked = st.button(
    "⚡  Predict Price", type="primary",
    use_container_width=True, key="predict_button"
)

if predict_clicked:
    # Build raw input dict
    raw_inputs = {
        "DAM Zonal Losses":     float(dam_losses),
        "DAM Zonal Congestion": float(dam_congestion),
        "DAM Forecast Load":    float(dam_load),
        "Dew Point":            float(dew_point),
        "Temperature":          float(temperature),
        "Solar Zenith Angle":   float(solar_zenith),
        "Relative Humidity":    float(humidity),
        "Wind Speed":           float(wind_speed),
        "month":                int(month),
        "hour":                 int(hour),
        "day_of_week":          int(day_of_week),
        "zone_name":            int(ZONE_MAP[zone]),
    }

    # Run models
    all_spreads = predict_all_spreads(
        raw_inputs, models, scalers, tree_features,
        encoded_features, continuous_cols, ohe_cols, ohe_structure,
    )
    spread_preds = {k: v for k, v in all_spreads.items() if k != "Ensemble"}
    lbmp_preds   = {name: float(dam_lbmp) + sp for name, sp in spread_preds.items()}
    avg_spread   = all_spreads["Ensemble"]
    avg_lbmp     = float(dam_lbmp) + avg_spread

    # ── Save everything to session state ─────────────────────────────────────
    # This is the core of the fix: all prediction outputs are stored here
    # so they survive the re-run triggered by clicking Run Simulation.
    st.session_state["predictions_ready"] = True
    st.session_state["all_spreads"]       = all_spreads
    st.session_state["spread_preds"]      = spread_preds
    st.session_state["lbmp_preds"]        = lbmp_preds
    st.session_state["avg_spread"]        = avg_spread
    st.session_state["avg_lbmp"]          = avg_lbmp
    st.session_state["dam_anchor"]        = float(dam_lbmp)
    st.session_state["zone"]              = zone
    st.session_state["selected_date"]     = selected_date
    st.session_state["hour"]              = hour

    # Build input display table rows for the expander
    st.session_state["raw_inputs_display"] = [
        ("Zone",                        zone),
        ("Date",                        str(selected_date)),
        ("Month",                       month),
        ("Hour",                        f"{hour:02d}:00"),
        ("Day of Week",                 f"{dow_names[day_of_week]} ({day_of_week})"),
        ("DAM Zonal LBMP ($/MWh)",      dam_lbmp),
        ("DAM Zonal Losses ($/MWh)",    dam_losses),
        ("DAM Zonal Congestion ($/MWh)",dam_congestion),
        ("DAM Forecast Load (MW)",      dam_load),
        ("Dew Point (°C)",              dew_point),
        ("Temperature (°C)",            temperature),
        ("Solar Zenith Angle (°)",      solar_zenith),
        ("Relative Humidity (%)",       humidity),
        ("Wind Speed (m/s)",            wind_speed),
    ]

    # Clear any previous simulation results when new predictions are made
    if "sim_results" in st.session_state:
        del st.session_state["sim_results"]

# =============================================================================
# RENDER RESULTS + SIMULATOR
# Rendered whenever predictions_ready is True — i.e. after any Predict click
# AND on all subsequent re-runs (simulator button, threshold slider, etc.)
# =============================================================================
if st.session_state["predictions_ready"]:
    render_prediction_results()
    render_simulator()
    
st.markdown("<hr>", unsafe_allow_html=True)
