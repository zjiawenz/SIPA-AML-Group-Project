#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)
from xgboost import XGBRegressor

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Model Comparison",
    page_icon="⚖️",
    layout="wide",
)

# =============================================================================
# CSS
# =============================================================================
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
  .insight-box strong { color: #1a237e; }
  .formula-block {
    background: #eef2ff;
    border: 1px solid #9fa8da;
    border-radius: 6px;
    padding: 0.7rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #1a237e;
    margin: 0.6rem 0 1rem 0;
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
  .best  { color: #2d6a4f; font-weight: 700; }
  .worst { color: #c62828; }
  hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Colour palette
# =============================================================================
MODEL_COLORS = {
    "Linear Regression": "#4C72B0",
    "KNN":               "#DD8452",
    "Random Forest":     "#2d6a4f",
    "XGBoost":           "#C44E52",
}
BG = "#f5f6ff"


def apply_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35, linewidth=0.7)


# =============================================================================
# Constants
# =============================================================================
DATASET_YEAR = 2018
DATA_PATH    = "data/cleaned_nyiso_electricity_weather.csv"

# Raw columns needed from CSV
RAW_COLS = [
    "DAM Zonal LBMP",        # used for target construction only — not a feature
    "DAM Zonal Losses",
    "DAM Zonal Congestion",
    "DAM Forecast Load",
    "month",
    "day",                   # used to derive day_of_week, then dropped
    "hour",
    "Dew Point",
    "Temperature",
    "Solar Zenith Angle",
    "Relative Humidity",
    "Wind Speed",
    "zone_name",
    "int_LBMP",
]

# Continuous features — scaled for LR and KNN; passed raw for RF and XGBoost
CONTINUOUS_COLS = [
    "DAM Zonal Losses",
    "DAM Zonal Congestion",
    "DAM Forecast Load",
    "Dew Point",
    "Temperature",
    "Solar Zenith Angle",
    "Relative Humidity",
    "Wind Speed",
]

# Categorical features for OHE (KNN and LR only)
OHE_COLS = ["month", "hour", "day_of_week", "zone_name"]

# Tree feature list: 12 raw features (no DAM LBMP, no day-of-month)
TREE_FEATURES = CONTINUOUS_COLS + ["month", "hour", "day_of_week", "zone_name"]


# =============================================================================
# OHE builder helper
# =============================================================================
def build_ohe_matrix(df_input, ohe_structure, drop_first=False):
    """
    Build the OHE portion of the encoded feature matrix.

    Parameters
    ----------
    df_input     : DataFrame containing OHE_COLS as columns
    ohe_structure: dict {col: sorted list of unique values from training}
    drop_first   : bool
        True  for Linear Regression — removes one reference category per
              group to prevent perfect multicollinearity with the intercept.
        False for KNN — keeps all categories for symmetric distance computation.

    Returns
    -------
    ohe_df : DataFrame of binary dummy columns in consistent category order
    """
    parts = []
    for col in OHE_COLS:
        cat    = pd.Categorical(df_input[col], categories=ohe_structure[col])
        dummies = pd.get_dummies(cat, prefix=col, drop_first=drop_first).astype(int)
        parts.append(dummies)
    return pd.concat(parts, axis=1)


# =============================================================================
# Data loading (cached)
# =============================================================================
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    """
    Load NYISO 2018 data, construct spread target, derive day_of_week,
    drop day-of-month, and produce:
      - Raw tree feature matrix (12 cols, no encoding)
      - Encoded feature matrix for KNN  (62 cols, drop_first=False)
      - Encoded feature matrix for LR   (58 cols, drop_first=True)
      - Spread target vector
      - Train/test index split (80/20 chronological)
      - Target distribution stats for discussion text
    """
    df = pd.read_csv(DATA_PATH)

    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[RAW_COLS].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)

    # ── Spread target ─────────────────────────────────────────────────────────
    # spread = int_LBMP (real-time) − DAM Zonal LBMP (day-ahead forecast)
    # DAM Zonal LBMP is NOT used as a feature — it is part of the target.
    df["spread"] = df["int_LBMP"] - df["DAM Zonal LBMP"]

    # ── day_of_week — drop day of month ───────────────────────────────────────
    df["date"]        = pd.to_datetime({"year": DATASET_YEAR,
                                        "month": df["month"],
                                        "day":   df["day"]})
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday … 6=Sunday
    df.drop(columns=["date", "day"], inplace=True)

    # ── OHE structure (fixed from full dataset) ───────────────────────────────
    ohe_structure = {col: sorted(df[col].unique().tolist()) for col in OHE_COLS}

    # ── 80/20 chronological split ─────────────────────────────────────────────
    split = int(len(df) * 0.80)
    y     = df["spread"].values

    # ── Tree feature matrix — raw, no encoding, no scaling ───────────────────
    # RF and XGBoost receive raw values directly. Scale-invariant.
    X_tree       = df[TREE_FEATURES].values.astype(float)
    X_tree_train = X_tree[:split]
    X_tree_test  = X_tree[split:]

    # ── Encoded feature matrices — continuous cols first, then OHE ───────────
    # Continuous cols are scaled here in load_data() where column structure
    # is unambiguous. Scaling inside run_models() risks column misalignment
    # after numpy serialisation through Streamlit's cache.
    #
    # Scaler fitted on training continuous columns only (no leakage).
    # OHE binary columns are NOT scaled — already on [0,1].
    df_cont   = df[CONTINUOUS_COLS].copy()
    ohe_knn   = build_ohe_matrix(df, ohe_structure, drop_first=False)
    ohe_lr    = build_ohe_matrix(df, ohe_structure, drop_first=True)

    # Continuous arrays for train/test
    cont_train = df_cont.values[:split].astype(float)
    cont_test  = df_cont.values[split:].astype(float)

    # Scale continuous columns — fit on train only
    scaler_knn = StandardScaler()
    scaler_lr  = StandardScaler()
    cont_train_knn = scaler_knn.fit_transform(cont_train)
    cont_test_knn  = scaler_knn.transform(cont_test)
    cont_train_lr  = scaler_lr.fit_transform(cont_train)
    cont_test_lr   = scaler_lr.transform(cont_test)

    # OHE arrays
    ohe_knn_arr = ohe_knn.values.astype(float)
    ohe_lr_arr  = ohe_lr.values.astype(float)

    # Final encoded matrices: [scaled continuous | OHE binary]
    X_knn_train = np.concatenate([cont_train_knn, ohe_knn_arr[:split]], axis=1)
    X_knn_test  = np.concatenate([cont_test_knn,  ohe_knn_arr[split:]], axis=1)
    X_lr_train  = np.concatenate([cont_train_lr,  ohe_lr_arr[:split]],  axis=1)
    X_lr_test   = np.concatenate([cont_test_lr,   ohe_lr_arr[split:]],  axis=1)

    y_train      = y[:split]
    y_test       = y[split:]
    train_months = df["month"].values[:split]
    test_months  = df["month"].values[split:]

    # Target stats for discussion text
    tgt_stats = dict(
        mean=df["spread"].mean(), std=df["spread"].std(),
        mn=df["spread"].min(),    mx=df["spread"].max(),
        pct_pos=(df["spread"] > 0).mean() * 100,
        n_total=len(df),
    )

    return (X_tree_train, X_tree_test,
            X_knn_train,  X_knn_test,
            X_lr_train,   X_lr_test,
            y_train, y_test,
            train_months, test_months,
            tgt_stats, ohe_structure)


# =============================================================================
# Model training (cached)
# =============================================================================
@st.cache_data(show_spinner="Training all four models — this takes ~30–60 s…")
def run_models(_X_tree_train, _X_tree_test,
               _X_knn_train,  _X_knn_test,
               _X_lr_train,   _X_lr_test,
               _y_train, _y_test):
    """
    Fit all four models and compute metrics on the test set.

    All feature matrices arrive pre-processed from load_data():
      X_tree_*  : 12 raw features, no scaling — for RF and XGBoost
      X_knn_*   : 62 features, continuous cols scaled, OHE drop_first=False — for KNN
      X_lr_*    : 58 features, continuous cols scaled, OHE drop_first=True  — for LR

    Scaling is done in load_data() rather than here to avoid any column
    misalignment risk when Streamlit deserialises cached numpy arrays.
    No scaling is applied in this function.
    """
    (X_tree_train, X_tree_test,
     X_knn_train,  X_knn_test,
     X_lr_train,   X_lr_test,
     y_train, y_test) = (
        _X_tree_train, _X_tree_test,
        _X_knn_train,  _X_knn_test,
        _X_lr_train,   _X_lr_test,
        _y_train, _y_test,
    )

    def evaluate(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "Model": name,
            "MSE":   mse,
            "RMSE":  np.sqrt(mse),
            "MAE":   mean_absolute_error(y_true, y_pred),
            "EVS":   explained_variance_score(y_true, y_pred),
            "R2":    r2_score(y_true, y_pred),
            "preds": y_pred,
        }

    # ── Linear Regression — 58 pre-scaled+encoded features ───────────────────
    # OHE with drop_first=True prevents perfect multicollinearity with
    # the intercept. Continuous cols already scaled in load_data().
    lr = LinearRegression()
    lr.fit(X_lr_train, y_train)
    lr_m = evaluate(y_test, lr.predict(X_lr_test), "Linear Regression")

    # ── KNN — k=30, Manhattan, 62 pre-scaled+encoded features ────────────────
    # OHE with drop_first=False for symmetric Manhattan distance.
    # Continuous cols already scaled in load_data().
    knn = KNeighborsRegressor(n_neighbors=30, metric="manhattan",
                               weights="uniform", n_jobs=-1)
    knn.fit(X_knn_train, y_train)
    knn_m = evaluate(y_test, knn.predict(X_knn_test), "KNN")

    # ── Random Forest — 12 raw features, no scaling ───────────────────────────
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X_tree_train, y_train)
    rf_m = evaluate(y_test, rf.predict(X_tree_test), "Random Forest")

    # ── XGBoost — 12 raw features, no scaling ────────────────────────────────
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                        random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tree_train, y_train)
    xgb_m = evaluate(y_test, xgb.predict(X_tree_test), "XGBoost")

    rows    = [lr_m, knn_m, rf_m, xgb_m]
    preds   = {r["Model"]: r.pop("preds") for r in rows}
    results = pd.DataFrame(rows).set_index("Model")
    return results, preds, y_test


# =============================================================================
# HEADER
# =============================================================================
st.title("📊 Machine Learning Model Comparison")
st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 1 — METHODOLOGY
# =============================================================================
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
Four models are evaluated head-to-head on the same chronological 80/20 split of NYISO 2018
hourly data. The <strong>target variable is the spread</strong> — the difference between
the real-time integrated LBMP and the day-ahead market forecast for that same hour
(spread = int_LBMP − DAM Zonal LBMP). A positive spread means real-time prices exceeded
the day-ahead forecast; a negative spread means they fell short.
<br><br>
<strong>DAM Zonal LBMP is excluded as a feature</strong> because it is used to construct
the target. The remaining 12 features are the other DAM market components, time variables
(including day_of_week, derived from date), weather, and zone.
<br><br>
The evaluation is <strong>strictly out-of-sample</strong>: all scalers and encodings are
fitted only on the training set and applied to the test set without refitting.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Experimental design")
    st.markdown("""
<div class="formula-block">
Target  : spread = int_LBMP − DAM Zonal LBMP ($/MWh)<br>
Train   : rows 1–80%  → months 1–10  (~76,868 rows)<br>
Test    : rows 81–100% → months 10–12 (~19,217 rows)<br><br>
Split   : chronological (no shuffling)<br>
Features: 12  (DAM market components + time + weather + zone)<br>
          DAM Zonal LBMP excluded — part of the target
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Model Settings and Encoding")
    st.markdown("""
<div class="formula-block">
Linear Regression : OLS, drop_first=True OHE, 58 encoded features<br>
KNN               : k=30, Manhattan, drop_first=False OHE, 62 features<br>
Random Forest     : 100 trees, max_depth=10, sqrt features, seed=42<br>
XGBoost           : 100 rounds, learning rate (lr)=0.1, max_depth=6, seed=42<br><br>
RF and XGBoost (tree-based, scale-invariant):<br>
  12 raw features — no OHE, no scaling<br>
  Trees split on value thresholds; scale/ordinality irrelevant<br><br>
KNN (distance-based):<br>
  8 continuous → StandardScaler (train fit only)<br>
  month/hour/day_of_week/zone → OHE, drop_first=False<br>
  All categories kept: symmetric Manhattan distance<br><br>
Linear Regression (OLS):<br>
  8 continuous → StandardScaler (train fit only)<br>
  month/hour/day_of_week/zone → OHE, drop_first=True<br>
  Reference category dropped per group → prevents<br>
  perfect multicollinearity with the intercept<br>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("#### Metrics")
    st.markdown("""
<div class="formula-block">
MSE  = (1/n) Σ (yᵢ − ŷᵢ)²       [quadratic, spike-sensitive]<br>
RMSE = √MSE                       [$/MWh, same units as spread]<br>
MAE  = (1/n) Σ |yᵢ − ŷᵢ|         [linear, spike-robust]<br>
EVS  = 1 − Var(y−ŷ) / Var(y)     [variance explained, bias-blind]<br>
R²   = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²   [penalises both variance & bias]
</div>
If EVS > R², the model has a systematic prediction bias.
""", unsafe_allow_html=True)

st.markdown("#### Workflow")
steps = [
    ("Step 1", "Load & preprocess — label-encode zone_name, construct spread target (int_LBMP − DAM Zonal LBMP), derive day_of_week, drop day-of-month and DAM Zonal LBMP from features."),
    ("Step 2", "Chronological 80/20 split — no shuffling; training = months 1–10, test = months 10–12. Harder and more realistic than a random split."),
    ("Step 3", "Build two feature matrices: (a) 12-column raw matrix for RF/XGBoost; (b) OHE-encoded matrices for KNN (62 cols) and LR (58 cols)."),
    ("Step 4", "Scale continuous columns only for KNN and LR — StandardScaler fitted on training data only. Binary OHE columns not scaled."),
    ("Steps 5", "Fit each model on its appropriate training matrix; predict on test set; compute all five metrics."),
]
for tag, text in steps:
    st.markdown(f'<span class="tag">{tag}</span> {text}', unsafe_allow_html=True)
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA & RUN MODELS
# =============================================================================
(X_tree_train, X_tree_test,
 X_knn_train,  X_knn_test,
 X_lr_train,   X_lr_test,
 y_train, y_test,
 train_months, test_months,
 tgt_stats, ohe_structure) = load_data()

results, preds, y_test_arr = run_models(
    X_tree_train, X_tree_test,
    X_knn_train,  X_knn_test,
    X_lr_train,   X_lr_test,
    y_train, y_test,
)

MKEYS   = ["Linear Regression", "KNN", "Random Forest", "XGBoost"]
MLABELS = ["Linear\nRegression", "KNN", "Random\nForest", "XGBoost"]
MCOLORS = [MODEL_COLORS[m] for m in MKEYS]

# =============================================================================
# SECTION 2 — RESULTS TABLE
# =============================================================================
st.markdown("## Performance Metrics")
st.markdown(
    f"Target: **spread = int_LBMP − DAM Zonal LBMP**  &nbsp;|&nbsp;  "
    f"Train: months {train_months.min()}–{train_months.max()} "
    f"({len(X_tree_train):,} rows)  &nbsp;|&nbsp;  "
    f"Test: months {test_months.min()}–{test_months.max()} "
    f"({len(X_tree_test):,} rows)"
)

lower_better  = ["MSE", "RMSE", "MAE"]
higher_better = ["EVS", "R2"]
metrics       = ["MSE", "RMSE", "MAE", "EVS", "R2"]
fmt = {"MSE": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}",
       "EVS": "{:.4f}", "R2":  "{:.4f}"}

best_idx  = {m: results[m].idxmin() if m in lower_better else results[m].idxmax()
             for m in metrics}
worst_idx = {m: results[m].idxmax() if m in lower_better else results[m].idxmin()
             for m in metrics}

header_cells = "".join(f"<th>{m}</th>" for m in metrics)
rows_html    = ""
for model in MKEYS:
    dot   = f'<span style="color:{MODEL_COLORS[model]};font-size:1.1rem;">●</span> '
    cells = ""
    for m in metrics:
        val  = results.loc[model, m]
        fval = fmt[m].format(val)
        cls  = "best" if best_idx[m] == model else ("worst" if worst_idx[m] == model else "")
        star = " ★" if best_idx[m] == model else ""
        cells += f'<td class="{cls}">{fval}{star}</td>'
    rows_html += f"<tr><td>{dot}{model}</td>{cells}</tr>"

st.markdown(f"""
<table class="results-table">
  <thead><tr><th>Model</th>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="font-size:0.8rem; color:#666;">
  ★ = best per metric &nbsp;|&nbsp;
  <span style="color:#2d6a4f; font-weight:700;">green = best</span> &nbsp;|&nbsp;
  <span style="color:#c62828;">red = worst</span> &nbsp;|&nbsp;
  Lower is better for MSE/RMSE/MAE · Higher is better for EVS/R²
</p>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 3 — BAR CHARTS  (RMSE · MAE · R²)
# =============================================================================
st.markdown("## Visual Comparison")
st.markdown(
    "Black border marks the best model per metric. "
    "RMSE and MAE are in $/MWh (spread units); "
    "R² is dimensionless (dashed line = naïve mean-prediction baseline at R²=0)."
)

rmse_v = [results.loc[m, "RMSE"] for m in MKEYS]
mae_v  = [results.loc[m, "MAE"]  for m in MKEYS]
r2_v   = [results.loc[m, "R2"]   for m in MKEYS]

fig_bar, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_bar.patch.set_facecolor(BG)

panels = [
    (axes[0], rmse_v, "RMSE  ($/MWh)", "RMSE — lower is better",  np.argmin),
    (axes[1], mae_v,  "MAE  ($/MWh)",  "MAE — lower is better",   np.argmin),
    (axes[2], r2_v,   "R²",            "R² — higher is better",   np.argmax),
]
for ax, vals, ylabel, title, fn in panels:
    bars = ax.bar(MLABELS, vals, color=MCOLORS, edgecolor="white", width=0.55)
    bi   = int(fn(vals))
    bars[bi].set_edgecolor("black")
    bars[bi].set_linewidth(2.5)
    spread = max(abs(v) for v in vals)
    for bar, val in zip(bars, vals):
        offset = spread * 0.04 if val >= 0 else -spread * 0.09
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    apply_style(ax)
    if "R²" in title:
        ymin = min(min(vals) * 1.5, -0.16)
        ax.set_ylim(ymin, max(max(vals) * 1.2, 0.12))
        ax.axhline(0, color="grey", lw=1.3, ls="--", alpha=0.7,
                   label="R²=0  (predict mean)")
        ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
st.pyplot(fig_bar, use_container_width=True)
plt.close(fig_bar)

# Dynamic insight values
best_model  = results["R2"].idxmax()
worst_model = results["R2"].idxmin()
lr_r2    = results.loc["Linear Regression", "R2"]
lr_rmse  = results.loc["Linear Regression", "RMSE"]
lr_mae   = results.loc["Linear Regression", "MAE"]
knn_r2   = results.loc["KNN", "R2"]
knn_evs  = results.loc["KNN", "EVS"]
rf_r2    = results.loc["Random Forest", "R2"]
xgb_r2   = results.loc["XGBoost", "R2"]
xgb_rmse = results.loc["XGBoost", "RMSE"]

st.markdown(f"""
<div class="insight-box">
<strong>Headline result:</strong> All four models produce low R² values
(range: {results["R2"].min():.3f} to {results["R2"].max():.3f}).
<strong>{best_model} achieves the best R² ({results["R2"].max():.4f})</strong>,
while {worst_model} performs worst (R² = {results["R2"].min():.4f}).
<br><br>
<strong>Seasonal regime shift:</strong> The chronological split forces models
trained on spring/summer months to generalise to an autumn/winter test period.
The spread distribution and its relationship to the remaining features may
differ substantially between seasons — months with large congestion events
or unexpected demand surges produce large spreads that may not be anticipated
from warm-weather training data.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 4 — RMSE vs MAE DIVERGENCE
# =============================================================================
st.markdown("## RMSE vs MAE Divergence")
st.markdown(
    "The gap between RMSE and MAE reveals how much extreme spread events "
    "(large real-time deviations from day-ahead) inflate the quadratic "
    "error relative to the typical prediction error."
)

fig_div, ax_div = plt.subplots(figsize=(9, 4.5))
fig_div.patch.set_facecolor(BG)
x_  = np.arange(len(MKEYS))
w_  = 0.3
ax_div.bar(x_ - w_/2, rmse_v, width=w_, label="RMSE",
           color=MCOLORS, edgecolor="white", alpha=0.9)
ax_div.bar(x_ + w_/2, mae_v,  width=w_, label="MAE",
           color=MCOLORS, edgecolor="white", alpha=0.45, hatch="///")
for i, (r, m) in enumerate(zip(rmse_v, mae_v)):
    ax_div.annotate("", xy=(x_[i] + w_/2, r), xytext=(x_[i] + w_/2, m),
                    arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax_div.text(x_[i] + w_/2 + 0.17, (r + m) / 2,
                f"Δ {r-m:.2f}", fontsize=7.5, va="center", color="#333")
ax_div.set_xticks(x_)
ax_div.set_xticklabels(MKEYS, fontsize=10)
ax_div.set_ylabel("$/MWh", fontsize=10)
ax_div.set_title("RMSE vs MAE by Model  (spread spike sensitivity)",
                 fontsize=11, fontweight="bold")
solid_patch = mpatches.Patch(facecolor="#888", label="RMSE (spike-sensitive)")
hatch_patch = mpatches.Patch(facecolor="#888", alpha=0.45, hatch="///",
                              label="MAE (spike-robust)")
ax_div.legend(handles=[solid_patch, hatch_patch], fontsize=9)
apply_style(ax_div)
plt.tight_layout()
st.pyplot(fig_div, use_container_width=True)
plt.close(fig_div)

st.markdown(f"""
<div class="insight-box">
RMSE spans {min(rmse_v):.3f}–{max(rmse_v):.3f} $/MWh across models, while MAE spans only
{min(mae_v):.3f}–{max(mae_v):.3f} $/MWh — a much tighter range.
<br><br>
The RMSE/MAE ratio confirms that all models handle typical (small) spread hours
reasonably well but struggle with the extreme deviations that inflate RMSE through
the quadratic penalty. These extreme spread events — driven by sudden congestion
activation, unexpected demand surges, or weather extremes — are precisely the
hours most consequential to virtual bidders and to grid operators managing
real-time market stress.
<br><br>
The arrows show the spike-driven gap (RMSE − MAE) for each model. A larger gap
indicates the model is more affected by the extreme spread events in the test period,
either because it overfits to the training-period's spike patterns or because it
systematically misses the magnitude of large deviations.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 5 — EVS vs R² (bias diagnostic)
# =============================================================================
st.markdown("## EVS vs R² — Bias Diagnostic")
st.markdown(
    "When EVS > R² for a model, it has a systematic prediction bias — "
    "the mean prediction differs from the mean of the target. "
    "The gap size measures the magnitude of that bias."
)

evs_v = [results.loc[m, "EVS"] for m in MKEYS]

fig_bias, ax_bias = plt.subplots(figsize=(9, 4.5))
fig_bias.patch.set_facecolor(BG)
w_ = 0.3
ax_bias.bar(x_ - w_/2, evs_v, width=w_, color=MCOLORS, edgecolor="white",
            alpha=0.9, label="EVS")
ax_bias.bar(x_ + w_/2, r2_v,  width=w_, color=MCOLORS, edgecolor="white",
            alpha=0.45, hatch="///", label="R²")
for i, (e, r) in enumerate(zip(evs_v, r2_v)):
    gap = e - r
    if abs(gap) > 0.003:
        ax_bias.annotate("", xy=(x_[i] + w_/2, e), xytext=(x_[i] + w_/2, r),
                         arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
        ax_bias.text(x_[i] + w_/2 + 0.17, (e + r) / 2,
                     f"Δ {gap:.3f}", fontsize=7.5, va="center", color="#555")
ax_bias.axhline(0, color="grey", lw=1.2, ls="--", alpha=0.6, label="R²=0")
ax_bias.set_xticks(x_)
ax_bias.set_xticklabels(MKEYS, fontsize=10)
ax_bias.set_ylabel("Score", fontsize=10)
ax_bias.set_title("EVS vs R²  (gap = systematic prediction bias)",
                  fontsize=11, fontweight="bold")
evs_patch = mpatches.Patch(facecolor="#888", label="EVS (bias-blind)")
r2_patch  = mpatches.Patch(facecolor="#888", alpha=0.45, hatch="///",
                            label="R² (penalises bias)")
ax_bias.legend(handles=[evs_patch, r2_patch], fontsize=9)
apply_style(ax_bias)
plt.tight_layout()
st.pyplot(fig_bias, use_container_width=True)
plt.close(fig_bias)

knn_gap = knn_evs - knn_r2
st.markdown(f"""
<div class="insight-box">
<strong>KNN bias signal:</strong> KNN's EVS ({knn_evs:.4f}) exceeds its R²
({knn_r2:.4f}) by {knn_gap:.4f}. This indicates a systematic prediction bias —
KNN is either consistently over- or under-predicting spreads in the test period.
KNN's local averaging over neighbours from the spring/summer training period
pulls forecasts toward the historical spread distribution, which may differ
structurally from the autumn/winter test period if seasonal demand patterns
and congestion dynamics shift.
<br><br>
<strong>Spread-specific insight:</strong> Because the spread target has a mean
close to zero (real-time and day-ahead prices are designed to converge over time),
a systematic bias in either direction has direct trading implications — a model
that consistently underpredicts the spread will generate too many Virtual Supply
signals, while one that overpredicts will generate too many Virtual Load signals.
The EVS−R² gap is therefore a directly actionable diagnostic for the convergence
bidding strategy built on these predictions.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 6 — RESIDUAL DISTRIBUTIONS
# =============================================================================
st.markdown("## Residual Distributions")
st.markdown(
    "Residual = actual spread − predicted spread. A well-calibrated model "
    "should be centred near zero. The tails reveal how each model handles "
    "extreme spread events."
)

fig_res, axes_res = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)
fig_res.patch.set_facecolor(BG)
clip = 100   # clip residuals for readability (spread values smaller than raw price)

for ax, model in zip(axes_res, MKEYS):
    resid         = y_test_arr - preds[model]
    resid_clipped = np.clip(resid, -clip, clip)
    ax.hist(resid_clipped, bins=60, color=MODEL_COLORS[model],
            edgecolor="white", alpha=0.85, linewidth=0.4)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.axvline(np.mean(resid), color="red", lw=1.5, ls="-",
               label=f"mean={np.mean(resid):.2f}")
    ax.set_title(model, fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual ($/MWh)", fontsize=8.5)
    ax.legend(fontsize=7.5)
    apply_style(ax)

axes_res[0].set_ylabel("Count", fontsize=10)
fig_res.suptitle(
    f"Residual Distributions  (clipped to ±{clip} $/MWh for readability)\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
st.pyplot(fig_res, use_container_width=True)
plt.close(fig_res)

means = {m: np.mean(y_test_arr - preds[m]) for m in MKEYS}
stds  = {m: np.std(y_test_arr  - preds[m]) for m in MKEYS}

st.markdown(f"""
<div class="insight-box">
<strong>Centre (mean residual):</strong>
A mean residual near zero indicates the model is well-calibrated on average.
A positive mean residual means the model underestimates the spread (actual spread
is higher than predicted); a negative mean residual means it overestimates.
<br>
Mean residuals: Linear Regression ({means['Linear Regression']:.3f}),
KNN ({means['KNN']:.3f}),
Random Forest ({means['Random Forest']:.3f}),
XGBoost ({means['XGBoost']:.3f}) $/MWh.
<br><br>
<strong>Spread (residual std):</strong> All models produce similar residual
standard deviations
(LR: {stds['Linear Regression']:.2f}, KNN: {stds['KNN']:.2f},
RF: {stds['Random Forest']:.2f}, XGBoost: {stds['XGBoost']:.2f} $/MWh),
suggesting that differences between models lie primarily in systematic bias
rather than random error — consistent with a seasonal regime shift affecting
all models similarly.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 7 — FULL DISCUSSION
# =============================================================================
st.markdown("## Discussion")


st.markdown(f"""
<div class="insight-box">
<strong>Why the spread is harder to predict than raw int_LBMP.</strong><br>
By constructing the spread as the residual between real-time and day-ahead prices,
we remove the dominant predictive signal — the day-ahead price level — from the
target. What remains is the component of real-time prices that the day-ahead market
did not anticipate: congestion-driven deviations, demand forecast errors, and
weather-driven supply/demand mismatches. These are inherently less predictable than
the price level itself, which explains why R² values here are lower than in the
original int_LBMP comparison. The residual for spreads is narrower than what it would be for 
the raw prices,reflecting the tighter distribution of the spread target. However, the spread
can still exhibit extreme values during congestion events, and the tails of
these distributions reveal how well each model captures those events.
<br><br>
<strong>Model ordering: R²</strong><br>
{best_model} achieves the highest R² ({results["R2"].max():.4f}),
followed by {[m for m in MKEYS if m not in [best_model, worst_model]][0]}
({results["R2"].iloc[1]:.4f}), then
{[m for m in MKEYS if m not in [best_model, worst_model]][1]}
({results["R2"].iloc[2]:.4f}), and
{worst_model} ({results["R2"].min():.4f}).
<br><br>
<strong>Seasonal regime shift.</strong><br>
The chronological split forces models trained on spring/summer spread patterns
to generalise to an autumn/winter test period. The spread dynamics in autumn/winter
may differ from summer — winter demand peaks create different congestion patterns
than summer cooling loads, and the accuracy of the day-ahead forecast (and therefore
the distribution of the spread) may shift seasonally. This regime shift is the primary
driver of all models' limited out-of-sample performance, and is consistent with the
finding in the rolling-window analysis on the individual model pages, where fold-to-fold
RMSE variation (reflecting seasonal difficulty) far exceeds the variation across model types.
<br><br>
<strong>RMSE–MAE divergence.</strong><br>
RMSE ranges from {min(rmse_v):.3f} to {max(rmse_v):.3f} $/MWh, while MAE spans only
{min(mae_v):.3f} to {max(mae_v):.3f} $/MWh. This compression confirms that all models
handle typical (small) spread hours reasonably well but are challenged by the extreme
deviation events that inflate RMSE through the quadratic penalty.
<br><br>
<strong>Boosting importance vs RF importance.</strong><br>XGBoost can assign high importance
to a feature that primarily improves predictions in later boosting rounds — i.e. features
that explain the harder residuals after the easy patterns have been captured. This may
give a subtly different importance ranking from RF, even though both models receive the
same 12 raw features. Comparing the two importance charts reveals which features are
consistently important (robust signal) and which are model-specific.</li>
<br><br>
<strong>Model Complexity</strong><br> In the testing for individual models (on other tabs), the n_estimators sweep for XGBoost,
the n_estimators sweep in the RF analysis, the k-sweep in KNN, and the Ridge alpha sweep in
Linear Regression all address the same underlying question: how much model complexity is optimal
for predicting the spread on the month-12 test fold? Together, they show whether the spread
signal is strong enough to justify complex models, or whether simpler configurations generalise
equally well — with direct implications for the reliability of the convergence bidding strategy
that depends on these predictions.</li>
<br><br>
<strong>Trading implications.</strong><br>
For the convergence bidding strategy built on these spread predictions, model
calibration (mean residual near zero) is as important as overall accuracy (low RMSE).
A systematically biased model will generate directionally incorrect trading signals
in the hours that matter most. The EVS−R² diagnostic and the residual distribution
charts above provide the most actionable information for assessing which model is
most appropriate to drive the virtual bidding decision rule.
<br><br>
<strong>Direction for improvement.</strong><br>
The rolling-window analysis on the individual model pages shows that month-to-month
RMSE variation across folds far exceeds the variation across model types seen here.
Addressing the seasonal generalisation problem — through separate models per season,
sliding training windows, or seasonal feature engineering — would likely yield larger
performance gains than further tuning of model hyperparameters.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.caption("This comparative study of different ML models for an electricity price data set aims to mimic the methodology found in - <br>X. Zhao, J. Ding, X. Huang and Y. Z. Gezhi, \"A Comparative Study of Machine Learning Algorithms for Electricity Price Forecasting with LIME-Based Interpretability,\" 2025 IEEE 15th International Conference on Electronics Information and Emergency Communication (ICEIEC), Kunala Lumpur, Malaysia, 2025, pp. 1-5, doi: 10.1109/ICEIEC65904.2025.11273147. ")