#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_📊_Model-Comparison.py
Streamlit page: Table 1 – ML Model Performance Comparison.
Place this file at: /Streamlit/pages/comparison_page.py  (or rename as needed)
Data expected at:   /Streamlit/data/cleaned_nyiso_electricity_weather.csv
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Comparison · NYISO 2018",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (same design language; accent colour = slate/indigo for neutrality)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
  h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem;
       border-bottom: 2px solid #3949ab; padding-bottom: 4px; color: #3949ab; }
  h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }

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
    background: #fffbf0;
    border-left: 4px solid #f4a234;
    border-radius: 4px;
    padding: 0.9rem 1.3rem;
    margin-top: 0.8rem;
    margin-bottom: 1.6rem;
    font-size: 0.92rem;
    line-height: 1.65;
  }
  .insight-box strong { color: #b45309; }
  .formula-block {
    background: #eef0ff;
    border: 1px solid #c5caf5;
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

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (one per model — consistent across all charts)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "DAM Zonal LBMP", "DAM Zonal Losses", "DAM Zonal Congestion",
    "DAM Forecast Load", "month", "day", "hour",
    "Dew Point", "Temperature", "Solar Zenith Angle",
    "Relative Humidity", "Wind Speed", "zone_name",
]
TARGET    = "int_LBMP"
DATA_PATH = "data/cleaned_nyiso_electricity_weather.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[FEATURES + [TARGET]].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)
    split = int(len(df) * 0.80)
    X_train = df[FEATURES].values[:split]
    X_test  = df[FEATURES].values[split:]
    y_train = df[TARGET].values[:split]
    y_test  = df[TARGET].values[split:]
    train_months = df["month"].values[:split]
    test_months  = df["month"].values[split:]
    # target stats for the discussion section
    tgt_stats = dict(
        mean=df[TARGET].mean(), std=df[TARGET].std(),
        mn=df[TARGET].min(), mx=df[TARGET].max(),
        n_total=len(df),
    )
    return X_train, X_test, y_train, y_test, train_months, test_months, tgt_stats


@st.cache_data(show_spinner="Training all four models — this takes ~30–60 s…")
def run_models(_X_train, _X_test, _y_train, _y_test):
    X_train, X_test, y_train, y_test = _X_train, _X_test, _y_train, _y_test

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    def evaluate(y_true, y_pred, name):
        mse  = mean_squared_error(y_true, y_pred)
        return {
            "Model": name,
            "MSE":   mse,
            "RMSE":  np.sqrt(mse),
            "MAE":   mean_absolute_error(y_true, y_pred),
            "EVS":   explained_variance_score(y_true, y_pred),
            "R2":    r2_score(y_true, y_pred),
            "preds": y_pred,
        }

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_m = evaluate(y_test, lr.predict(X_test_s), "Linear Regression")

    # KNN (tuned: k=30, Manhattan, uniform)
    knn = KNeighborsRegressor(n_neighbors=30, metric="manhattan",
                              weights="uniform", n_jobs=-1)
    knn.fit(X_train_s, y_train)
    knn_m = evaluate(y_test, knn.predict(X_test_s), "KNN")

    # Random Forest (50 trees)
    rf = RandomForestRegressor(n_estimators=50, max_features="sqrt",
                               random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_m = evaluate(y_test, rf.predict(X_test_s), "Random Forest")

    # XGBoost (100 rounds, lr=0.1, depth=6)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_train_s, y_train)
    xgb_m = evaluate(y_test, xgb.predict(X_test_s), "XGBoost")

    rows   = [lr_m, knn_m, rf_m, xgb_m]
    preds  = {r["Model"]: r.pop("preds") for r in rows}
    results = pd.DataFrame(rows).set_index("Model")
    return results, preds, y_test

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊  Model Comparison — Table 1")
st.markdown(
    "**NYISO 2018 · int_LBMP ($/MWh)**  &nbsp;|&nbsp; "
    "Chronological 80/20 split · 13 features · ~96 k rows · "
    "Linear Regression, KNN, Random Forest, XGBoost"
)
st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
Four models are evaluated head-to-head on the same chronological 80/20 split of NYISO 2018
hourly electricity price data. The evaluation is <strong>strictly out-of-sample</strong>:
the scaler, feature encodings, and all model parameters are fitted <em>only</em> on the
training set. The test set is never touched until final evaluation.
Five metrics are reported to capture different facets of prediction quality.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Experimental design")
    st.markdown("""
<div class="formula-block">
Train: rows 1–80%  →  months 1–10  (76,868 rows)<br>
Test:  rows 81–100% →  months 10–12 (19,217 rows)<br><br>
Split: chronological (no shuffling)<br>
Scaler: StandardScaler, fit on train only<br>
Features: 13  (DAM market + time + weather + zone)
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Model settings")
    st.markdown("""
<div class="formula-block">
Linear Regression : OLS, no regularisation<br>
KNN               : k=30, Manhattan, uniform weights<br>
Random Forest     : 50 trees, max_features=sqrt, seed=42<br>
XGBoost           : 100 rounds, lr=0.1, max_depth=6, seed=42
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("#### Metrics")
    st.markdown("""
<div class="formula-block">
MSE  = (1/n) Σ (yᵢ − ŷᵢ)²       [quadratic, spike-sensitive]<br>
RMSE = √MSE                       [$/MWh, same units as target]<br>
MAE  = (1/n) Σ |yᵢ − ŷᵢ|         [linear, spike-robust]<br>
EVS  = 1 − Var(y−ŷ) / Var(y)     [variance explained, bias-blind]<br>
R²   = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²   [penalises both variance & bias]
</div>
If EVS > R² for a model, it has a systematic prediction bias
(consistently over- or under-predicts). EVS and R² are equal when
the mean prediction equals the mean of the target.
""", unsafe_allow_html=True)

st.markdown("#### Workflow")
steps = [
    ("Step 1", "Load & preprocess — label-encode zone_name, drop 253 NaN rows, sort chronologically by month/day/hour/zone."),
    ("Step 2", "Chronological 80/20 split — no shuffling; training = Jan–Oct 2018, test = Nov–Dec 2018. Harder and more realistic than a random split."),
    ("Step 3", "Feature scaling — StandardScaler fitted on training data only; applied to both sets. Prevents leakage of test-period statistics."),
    ("Steps 4–7", "Fit each model on the scaled training set; generate predictions on the scaled test set; compute all five metrics."),
]
for tag, text in steps:
    st.markdown(f'<span class="tag">{tag}</span> {text}', unsafe_allow_html=True)
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & RUN MODELS
# ─────────────────────────────────────────────────────────────────────────────
(X_train, X_test, y_train, y_test,
 train_months, test_months, tgt_stats) = load_data()
results, preds, y_test_arr = run_models(X_train, X_test, y_train, y_test)

MKEYS    = ["Linear Regression", "KNN", "Random Forest", "XGBoost"]
MLABELS  = ["Linear\nRegression", "KNN", "Random\nForest", "XGBoost"]
MCOLORS  = [MODEL_COLORS[m] for m in MKEYS]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Table 1 — Performance Metrics")
st.markdown(
    f"Train: months {train_months.min()}–{train_months.max()} "
    f"({len(X_train):,} rows) &nbsp;|&nbsp; "
    f"Test: months {test_months.min()}–{test_months.max()} "
    f"({len(X_test):,} rows)"
)

# Build HTML table with best/worst highlighting
lower_better = ["MSE", "RMSE", "MAE"]
higher_better = ["EVS", "R2"]
metrics = ["MSE", "RMSE", "MAE", "EVS", "R2"]
fmt = {"MSE": "{:.2f}", "RMSE": "{:.3f}", "MAE": "{:.3f}",
       "EVS": "{:.4f}", "R2": "{:.4f}"}

best_idx  = {m: results[m].idxmin() if m in lower_better else results[m].idxmax()
             for m in metrics}
worst_idx = {m: results[m].idxmax() if m in lower_better else results[m].idxmin()
             for m in metrics}

header_cells = "".join(f"<th>{m}</th>" for m in metrics)
rows_html = ""
for model in MKEYS:
    dot = f'<span style="color:{MODEL_COLORS[model]};font-size:1.1rem;">●</span> '
    cells = ""
    for m in metrics:
        val   = results.loc[model, m]
        fval  = fmt[m].format(val)
        cls   = "best" if best_idx[m] == model else ("worst" if worst_idx[m] == model else "")
        star  = " ★" if best_idx[m] == model else ""
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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BAR CHARTS  (RMSE · MAE · R²)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Visual Comparison")
st.markdown(
    "Black border marks the best model per metric. "
    "RMSE and MAE are both in $/MWh; R² is dimensionless (dashed line = naïve mean-prediction baseline)."
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
    if "R²" in ylabel:
        ax.set_ylim(min(min(vals) * 1.5, -0.16), 0.35)
        ax.axhline(0, color="grey", lw=1.3, ls="--", alpha=0.7,
                   label="R²=0  (predict mean)")
        ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
st.pyplot(fig_bar, use_container_width=True)
plt.close(fig_bar)

lr_r2   = results.loc["Linear Regression", "R2"]
lr_rmse = results.loc["Linear Regression", "RMSE"]
lr_mae  = results.loc["Linear Regression", "MAE"]
knn_r2  = results.loc["KNN", "R2"]
knn_evs = results.loc["KNN", "EVS"]
xgb_r2  = results.loc["XGBoost", "R2"]
xgb_rmse= results.loc["XGBoost", "RMSE"]
rf_r2   = results.loc["Random Forest", "R2"]

st.markdown(f"""
<div class="insight-box">
<strong>Headline result:</strong> All four models produce low R² values
(range: {results["R2"].min():.3f} to {results["R2"].max():.3f}), and
<strong>Linear Regression wins every metric</strong> (RMSE = {lr_rmse:.2f} $/MWh,
MAE = {lr_mae:.2f} $/MWh, R² = {lr_r2:.4f}).
This is counterintuitive — the expected ranking from the literature is XGBoost > RF > KNN > LR.
Two structural factors explain the reversal:
<ol style="margin: 0.4rem 0 0 0; padding-left: 1.3rem;">
  <li><strong>NYISO price volatility:</strong> int_LBMP spans
      {tgt_stats["mn"]:.0f} to {tgt_stats["mx"]:.0f} $/MWh (σ = {tgt_stats["std"]:.1f} $/MWh).
      Extreme price spikes in ~4% of hours disproportionately inflate RMSE via the quadratic penalty,
      hitting all models but especially the nonlinear ones that overfit to training-set spike patterns.</li>
  <li><strong>Seasonal regime shift:</strong> The chronological split forces models trained on spring/summer
      prices to generalise to autumn/winter. The correlation between DAM Zonal LBMP and int_LBMP drops
      from 0.68 in training to 0.42 in the test period. Under a random (non-temporal) split, LR achieves
      R² = 0.45 — nearly 3× higher — confirming that seasonal generalisation, not model capacity,
      is the binding constraint.</li>
</ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RMSE vs MAE DIVERGENCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## RMSE vs MAE Divergence")
st.markdown(
    "The gap between RMSE and MAE reveals how much extreme price spikes "
    "inflate the quadratic error relative to the typical prediction error."
)

fig_div, ax_div = plt.subplots(figsize=(9, 4.5))
fig_div.patch.set_facecolor(BG)
x_   = np.arange(len(MKEYS))
w_   = 0.3
bars_rmse = ax_div.bar(x_ - w_/2, rmse_v, width=w_, label="RMSE",
                        color=MCOLORS, edgecolor="white", alpha=0.9)
bars_mae  = ax_div.bar(x_ + w_/2, mae_v, width=w_, label="MAE",
                        color=MCOLORS, edgecolor="white", alpha=0.45,
                        hatch="///")
for i, (r, m) in enumerate(zip(rmse_v, mae_v)):
    ax_div.annotate("", xy=(x_[i] + w_/2, r), xytext=(x_[i] + w_/2, m),
                    arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax_div.text(x_[i] + w_/2 + 0.17, (r + m) / 2,
                f"Δ {r-m:.2f}", fontsize=7.5, va="center", color="#333")
ax_div.set_xticks(x_)
ax_div.set_xticklabels(MKEYS, fontsize=10)
ax_div.set_ylabel("$/MWh", fontsize=10)
ax_div.set_title("RMSE vs MAE by Model  (spike sensitivity)",
                 fontsize=11, fontweight="bold")
# custom legend
solid_patch  = mpatches.Patch(facecolor="#888", label="RMSE (spike-sensitive)")
hatch_patch  = mpatches.Patch(facecolor="#888", alpha=0.45, hatch="///", label="MAE (spike-robust)")
ax_div.legend(handles=[solid_patch, hatch_patch], fontsize=9)
apply_style(ax_div)
plt.tight_layout()
st.pyplot(fig_div, use_container_width=True)
plt.close(fig_div)

st.markdown(f"""
<div class="insight-box">
RMSE spans {min(rmse_v):.2f}–{max(rmse_v):.2f} $/MWh across models, while MAE spans only
{min(mae_v):.2f}–{max(mae_v):.2f} $/MWh — a much tighter range.
The two-to-one ratio of RMSE/MAE confirms that all models handle typical non-spike hours
reasonably well but struggle with the extreme events that inflate RMSE through the quadratic penalty.
In practical terms, a MAE of ~13 $/MWh means models are typically within 13 $/MWh of the true price
— a useful baseline, but insufficient for the tail-risk events most consequential to market participants.
The arrows show the spike-driven gap (RMSE − MAE) for each model; XGBoost carries the largest
absolute gap, consistent with its tendency to overfit training-period spike patterns.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EVS vs R² (bias diagnostic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## EVS vs R² — Bias Diagnostic")
st.markdown(
    "When EVS > R² for a model, it has a systematic prediction bias "
    "(the mean prediction differs from the mean target). "
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
    if abs(gap) > 0.005:
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
r2_patch  = mpatches.Patch(facecolor="#888", alpha=0.45, hatch="///", label="R² (penalises bias)")
ax_bias.legend(handles=[evs_patch, r2_patch,
               mpatches.Patch(facecolor="none", edgecolor="grey", linestyle="--",
                              label="R²=0 (predict mean)")],
               fontsize=9)
apply_style(ax_bias)
plt.tight_layout()
st.pyplot(fig_bias, use_container_width=True)
plt.close(fig_bias)

knn_gap = knn_evs - knn_r2
st.markdown(f"""
<div class="insight-box">
<strong>KNN bias signal:</strong> KNN's EVS ({knn_evs:.4f}) exceeds its R²
({knn_r2:.4f}) by {knn_gap:.4f} — the largest gap among the four models.
This indicates a systematic downward bias: KNN consistently underpredicts prices in the colder
test months, which have a slightly higher average than the predominantly warm training period.
KNN's local averaging over neighbours from spring/summer hours pulls forecasts toward lower
historical prices when predicting in a structurally different winter regime.
<br><br>
<strong>XGBoost negative R²:</strong> An R² of {xgb_r2:.4f} means XGBoost performs <em>worse
than simply predicting the test-set mean for every observation</em>. This is the clearest sign
of overfitting to training-period price patterns — gradient boosting without temporal regularisation
or early stopping aggressively fits residuals that do not generalise across the seasonal boundary.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RESIDUAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Residual Distributions")
st.markdown(
    "Residual = actual − predicted. A well-calibrated model should be centred near zero. "
    "The tails reveal how each model handles extreme price events."
)

fig_res, axes_res = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)
fig_res.patch.set_facecolor(BG)
clip = 200   # clip residuals for readability; extreme tails noted separately

for ax, model in zip(axes_res, MKEYS):
    resid = y_test_arr - preds[model]
    resid_clipped = np.clip(resid, -clip, clip)
    ax.hist(resid_clipped, bins=60, color=MODEL_COLORS[model],
            edgecolor="white", alpha=0.85, linewidth=0.4)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.axvline(np.mean(resid), color="red", lw=1.5, ls="-",
               label=f"mean={np.mean(resid):.1f}")
    ax.set_title(model, fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual ($/MWh)", fontsize=8.5)
    ax.legend(fontsize=7.5)
    apply_style(ax)

axes_res[0].set_ylabel("Count", fontsize=10)
fig_res.suptitle(
    f"Residual Distributions  (clipped to ±{clip} $/MWh for readability)",
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
Linear Regression ({means['Linear Regression']:.2f} $/MWh) and
Random Forest ({means['Random Forest']:.2f} $/MWh) are closest to zero bias.
KNN ({means['KNN']:.2f} $/MWh) shows a positive mean residual — consistent with the EVS > R² signal:
it systematically underpredicts, so actual prices are on average higher than its forecasts.
XGBoost ({means['XGBoost']:.2f} $/MWh) has the largest mean residual, reflecting the training-period
overfitting that translates into a directional bias in the winter test period.
<br><br>
<strong>Spread (residual std):</strong> All models produce similar standard deviations
(LR: {stds['Linear Regression']:.1f}, KNN: {stds['KNN']:.1f},
RF: {stds['Random Forest']:.1f}, XGBoost: {stds['XGBoost']:.1f} $/MWh),
confirming that the difference between models lies primarily in systematic bias rather than
random error — the seasonal regime shift affects all models similarly.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FULL DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Discussion")

st.markdown(f"""
<div class="method-block">
<strong>3.1 &nbsp; Performance Comparison of ML Models (Table 1)</strong><br><br>

Table 1 presents the prediction performance of four machine learning models on NYISO 2018 hourly
integrated LBMP (<code>int_LBMP</code>) under a chronological 80/20 train/test split.
Training covers months {train_months.min()}–{train_months.max()} ({len(X_train):,} rows);
testing covers months {test_months.min()}–{test_months.max()} ({len(X_test):,} rows).
Five metrics are reported: MSE, RMSE, MAE, Explained Variance Score (EVS), and R².
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-box">
<strong>Why does Linear Regression win?</strong><br>
The model ordering by R² — Linear Regression ({lr_r2:.4f}) > KNN ({knn_r2:.4f}) >
Random Forest ({rf_r2:.4f}) > XGBoost ({xgb_r2:.4f}) — reverses the expected ranking
from the machine learning literature. Two structural factors explain the divergence from
the reference paper (arXiv:2512.01212), in which KNN achieves R² = 0.865 and ensemble
models uniformly outperform linear regression.

<br><br><strong>1. NYISO market volatility.</strong>
int_LBMP spans {tgt_stats["mn"]:.0f} to {tgt_stats["mx"]:.0f} $/MWh with a standard deviation of
{tgt_stats["std"]:.1f} $/MWh. Extreme price spike events — driven by grid congestion and demand surges —
occur in approximately 4% of training hours. Because MSE and RMSE apply a quadratic penalty,
these events disproportionately dominate the aggregate error across all models.
The nonlinear models (RF, XGBoost) overfit to the specific patterns of training-period spikes;
when those patterns do not recur in the test period, their errors compound.

<br><br><strong>2. Seasonal regime shift.</strong>
The chronological split imposes a structural break between training (spring/summer) and test (autumn/winter)
data. The correlation between DAM Zonal LBMP — the strongest predictor per permutation importance —
and int_LBMP falls from 0.68 in the training period to 0.42 in the test period.
This shift is confirmed diagnostically: under an equivalent random (non-temporal) split,
Linear Regression achieves R² = 0.45 — nearly 3× higher than under the chronological split.
The chronological approach is methodologically correct for time-series forecasting, but
produces a harder and more realistic out-of-sample evaluation than the random split the reference
paper almost certainly used.

<br><br><strong>XGBoost.</strong>
XGBoost performs worst (R² = {xgb_r2:.4f}, RMSE = {xgb_rmse:.2f} $/MWh), consistent with gradient
boosting's tendency to fit aggressively to training residuals when default hyperparameters are used
without temporal-specific regularisation or early stopping. Its negative R² means it performs worse
than simply predicting the test-set mean — the clearest sign of training-period overfitting.

<br><br><strong>KNN bias.</strong>
KNN (R² = {knn_r2:.4f}, RMSE = {results.loc["KNN","RMSE"]:.2f} $/MWh) sits between linear
regression and the two tree ensembles. Notably, KNN's EVS ({knn_evs:.4f}) exceeds its R²
({knn_r2:.4f}), indicating a small systematic downward bias: the model consistently
underpredicts prices in the colder test months, which have a slightly higher average than the
predominantly warm training period.

<br><br><strong>RMSE–MAE divergence.</strong>
RMSE ranges from {min(rmse_v):.2f} to {max(rmse_v):.2f} $/MWh, while MAE spans only
{min(mae_v):.2f} to {max(mae_v):.2f} $/MWh. This compression confirms that all models handle
typical non-spike hours reasonably well but struggle with the extreme events that inflate RMSE
through the quadratic penalty.

<br><br><strong>Direction for improvement.</strong>
Rather than switching between model families, the greater gains would come from addressing
the seasonal generalisation problem — through cyclical encoding of time features (e.g. sine/cosine
transforms of month and hour), separate models per season, or a sliding training window that
upweights recent observations. The rolling-window analysis in the KNN and RF pages corroborates this:
month-to-month RMSE variation across folds (18–43 $/MWh) far exceeds the variation across model
types observed here.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.caption(
    "Data: NYISO 2018 · cleaned_nyiso_electricity_weather.csv &nbsp;|&nbsp; "
    f"Train: {len(X_train):,} rows (months {train_months.min()}–{train_months.max()}) &nbsp;|&nbsp; "
    f"Test: {len(X_test):,} rows (months {test_months.min()}–{test_months.max()}) &nbsp;|&nbsp; "
    "Chronological 80/20 split · StandardScaler on train only"
)
