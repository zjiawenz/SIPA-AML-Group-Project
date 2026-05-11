#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3___Model-RF-Spread.py
=============================================================================
Random Forest — Spread Analysis  (NYISO 2018)
Target: spread = int_LBMP - DAM Zonal LBMP

Streamlit page. Drop into the pages/ folder of your multi-page app.

Three analysis sections:
  1. Rolling CV — RMSE per fold (baseline 50 trees vs best 100 trees)
  2. Permutation variable importance (raw features, direct permutation)
  3. MSE vs n_estimators sensitivity sweep

Key methodological points
--------------------------
  Target  : spread = int_LBMP - DAM Zonal LBMP ($/MWh)
  Features: 12 raw features (DAM Zonal LBMP excluded — part of target)
            day_of_week added; day of month dropped
  Encoding: NONE — RF is tree-based and scale-invariant
  Scaling : NONE — trees split on thresholds, not distances
  Dataset : Full dataset (no subsampling — RF prediction is O(depth),
            independent of training set size)
  Folds   : 9 expanding-window, test months 4-12
  Baseline: 50 trees, max_depth=10
  Best    : 100 trees, max_depth=10
  Sensitivity: n_estimators sweep 10-100, month-12 fold only
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Random Forest",
    page_icon="🌲",
    layout="wide",
)

# =============================================================================
# Custom CSS  (consistent with KNN and LR pages)
# =============================================================================
st.markdown("""
<style>
  .method-block {
    background: #f4faf6;
    border-left: 4px solid #2d6a4f;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
  }

  .insight-box {
    background: #edf7f0;
    border-left: 4px solid #2d6a4f;
    border-radius: 6px;
    padding: 0.95rem 1.35rem;
    margin-top: 0.8rem;
    margin-bottom: 1.6rem;
    font-size: 0.92rem;
    line-height: 1.7;
    box-shadow: 0 1px 4px rgba(45,106,79,0.06);
  }

  .insight-box strong { 
    color: #1b4332; 
  }

  .formula-block {
    background: #edf7f0;
    border: 1px solid #b7dfca;
    border-radius: 6px;
    padding: 0.7rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #1b4332;
    margin: 0.6rem 0 1rem 0;
  }

  .tag {
    display: inline-block;
    background: #2d6a4f;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 1px 8px;
    border-radius: 20px;
    margin-right: 5px;
    vertical-align: middle;
  }

  hr { 
    border: none; 
    border-top: 1px solid #e0e0e0; 
    margin: 1.8rem 0; 
  }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BLUE   = "#2196a8"
GREEN  = "#2d6a4f"
RED    = "#ea4335"
ORANGE = "#f4a234"
GREY   = "#aaaaaa"
BG     = "#f4faf6"

def apply_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35, linewidth=0.7)


# =============================================================================
# Constants
# =============================================================================
DATASET_YEAR   = 2018
RANDOM_SEED    = 42
N_EST_BASELINE = 50
N_EST_BEST     = 100
MAX_DEPTH      = 10
N_PERM_REPEATS = 5
N_EST_SWEEP    = [10, 20, 30, 50, 75, 100]

RAW_COLS = [
    "DAM Zonal LBMP", "DAM Zonal Losses", "DAM Zonal Congestion",
    "DAM Forecast Load", "month", "day", "hour",
    "Dew Point", "Temperature", "Solar Zenith Angle",
    "Relative Humidity", "Wind Speed", "zone_name", "int_LBMP",
]

FEATURES = [
    "DAM Zonal Losses", "DAM Zonal Congestion", "DAM Forecast Load",
    "month", "hour", "day_of_week",
    "Dew Point", "Temperature", "Solar Zenith Angle",
    "Relative Humidity", "Wind Speed", "zone_name",
]

DATA_PATH = "data/cleaned_nyiso_electricity_weather.csv"

# =============================================================================
# Data loading (cached)
# =============================================================================
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    """
    Load and preprocess the full NYISO 2018 dataset.
    No subsampling — RF uses the full dataset.
    No encoding or scaling — RF is tree-based and scale-invariant.
    Returns raw feature matrix, spread target, and month index.
    """
    df = pd.read_csv(DATA_PATH)

    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[RAW_COLS].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)

    # Spread target
    df["spread"] = df["int_LBMP"] - df["DAM Zonal LBMP"]

    # day_of_week from calendar; drop day of month
    df["date"]        = pd.to_datetime({"year": DATASET_YEAR,
                                        "month": df["month"],
                                        "day":   df["day"]})
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday ... 6=Sunday
    df.drop(columns=["date", "day"], inplace=True)

    X      = df[FEATURES].values
    y      = df["spread"].values
    months = df["month"].values

    return X, y, months


# =============================================================================
# Model pipeline (cached)
# =============================================================================
@st.cache_data(show_spinner="Training Random Forest models across 9 folds…")
def run_analysis(_X, _y, _months):
    """
    Rolling CV, permutation importance, and n_estimators sensitivity sweep.
    No scaling or encoding applied — RF receives raw feature values.
    """
    X, y, months = _X, _y, _months

    TEST_MONTHS = list(range(4, 13))
    folds = [(np.where(months < tm)[0], np.where(months == tm)[0])
             for tm in TEST_MONTHS]

    def fit_fold(tr, te, n_estimators=N_EST_BEST, max_depth=MAX_DEPTH):
        """
        Fit RF on one fold. No scaling or encoding needed — tree models
        split on raw value thresholds and are scale-invariant.
        Returns fitted model and test MSE.
        """
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X[tr], y[tr])
        mse = mean_squared_error(y[te], rf.predict(X[te]))
        return rf, mse

    # ── Baseline rolling CV (50 trees) ───────────────────────────────────────
    base_mses = []
    for tr, te in folds:
        _, mse = fit_fold(tr, te, n_estimators=N_EST_BASELINE)
        base_mses.append(mse)

    # ── Best-config rolling CV (100 trees) ───────────────────────────────────
    best_mses, train_sizes = [], []
    for tr, te in folds:
        _, mse = fit_fold(tr, te, n_estimators=N_EST_BEST)
        best_mses.append(mse)
        train_sizes.append(len(tr))

    # ── Permutation importance (month-12 fold, raw feature permutation) ──────
    # RF receives raw features — no OHE grouping needed (contrast with KNN/LR).
    # Each of the 12 feature columns is permuted independently.
    tr12, te12 = folds[-1]
    rf_vi, base_mse_vi = fit_fold(tr12, te12, n_estimators=N_EST_BEST)
    yte = y[te12]
    rng = np.random.default_rng(RANDOM_SEED)
    imp_results = {}
    for j, feat in enumerate(FEATURES):
        deltas = []
        for _ in range(N_PERM_REPEATS):
            Xp       = X[te12].copy().astype(float)
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(mean_squared_error(yte, rf_vi.predict(Xp)) - base_mse_vi)
        imp_results[feat] = (np.mean(deltas), np.std(deltas))

    # ── n_estimators sensitivity sweep (month-12 fold) ───────────────────────
    n_est_mses = {}
    for n in N_EST_SWEEP:
        _, mse       = fit_fold(tr12, te12, n_estimators=n)
        n_est_mses[n] = mse

    return dict(
        TEST_MONTHS  = TEST_MONTHS,
        base_mses    = base_mses,
        best_mses    = best_mses,
        train_sizes  = train_sizes,
        imp_results  = imp_results,
        n_est_mses   = n_est_mses,
        base_mse_vi  = base_mse_vi,
    )


# =============================================================================
# HEADER
# =============================================================================
st.title("🌲 Random Forest Model Analysis")
st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 1 — METHODOLOGY
# =============================================================================
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
<strong>Random Forest (RF)</strong> builds an ensemble of decision trees, each trained on a
bootstrap resample of the training data with a random feature subset considered at each split.
Predictions are the average across all trees. This bagging approach reduces variance relative
to a single tree and naturally captures nonlinear interactions between features — a key advantage
over Linear Regression for predicting electricity price spreads, which are driven by threshold
effects (e.g. load exceeding capacity, congestion activating on specific corridors).
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Target variable")
    st.markdown("""
<div class="formula-block">
spread = int_LBMP − DAM Zonal LBMP<br><br>
Positive → real-time exceeded day-ahead<br>
Negative → real-time below day-ahead
</div>
""", unsafe_allow_html=True)
    st.markdown("#### Feature set")
    st.markdown("""
<div class="formula-block">
DAM Zonal Losses<br>
DAM Zonal Congestion<br>
DAM Forecast Load<br>
Month, Hour, Day of Week<br>
Dew Point, Temperature<br>
Solar Zenith Angle<br>
Relative Humidity, Wind Speed<br>
Zone Name
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("#### RF settings")
    st.markdown("""
<div class="formula-block">
n_estimators = 100<br>
max_depth    = 10<br>
max_features = sqrt (3-4 per split)<br>
random_state = 42
</div>
""", unsafe_allow_html=True)

st.markdown("#### Workflow overview")
steps = [
    ("Step 1", "Load & preprocess — Encode zone_name to integer, construct spread target, derive day_of_week, drop day of month."),
    ("Step 2", "Full dataset retained — No subsampling."),
    ("Step 3", "Rolling CV — 9 expanding-window folds, test months 4-12."),
    ("Step 4", "Baseline RF — 50 trees."),
    ("Step 5", "Best-configuration RF — 100 trees."),
    ("Step 6", "Permutation importance — 12 raw features permuted directly."),
    ("Step 7", "n_estimators sensitivity sweep — MSE vs tree count on month-12 fold."),
]
for tag, text in steps:
    st.markdown(f'<span class="tag">{tag}</span> {text}', unsafe_allow_html=True)
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA & RUN
# =============================================================================
X, y, months = load_data()
res = run_analysis(X, y, months)

TEST_MONTHS = res["TEST_MONTHS"]
base_mses   = res["base_mses"]
best_mses   = res["best_mses"]
train_sizes = res["train_sizes"]
imp_results = res["imp_results"]
n_est_mses  = res["n_est_mses"]
x_          = np.arange(len(TEST_MONTHS))

# Sort features by importance descending
feat_arr = list(imp_results.keys())
imp_arr  = [imp_results[f][0] for f in feat_arr]
std_arr  = [imp_results[f][1] for f in feat_arr]
sidx     = np.argsort(imp_arr)[::-1]
feat_s   = [feat_arr[i] for i in sidx]
imp_s    = [imp_arr[i]  for i in sidx]
std_s    = [std_arr[i]  for i in sidx]

# =============================================================================
# SECTION 2 — PLOT 1: RMSE per fold
# =============================================================================
st.markdown("## Rolling CV Model Fit")
st.markdown(
    f"Baseline RF ({N_EST_BASELINE} trees) vs best-config RF ({N_EST_BEST} trees) "
    f"across all 9 rolling folds. Both use max_depth={MAX_DEPTH}, sqrt features, "
    "and the full dataset."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Mean RMSE ({N_EST_BEST} trees)",     f"{np.sqrt(np.mean(best_mses)):.3f} $/MWh")
m2.metric(f"Mean RMSE ({N_EST_BASELINE} trees)",  f"{np.sqrt(np.mean(base_mses)):.3f} $/MWh")
m3.metric("Best fold RMSE",                        f"{np.sqrt(min(best_mses)):.3f} $/MWh")
m4.metric("Worst fold RMSE",                       f"{np.sqrt(max(best_mses)):.3f} $/MWh")

fig_fit, ax_fit = plt.subplots(figsize=(11, 4.6))
fig_fit.patch.set_facecolor(BG)
w_ = 0.34
ax_fit.bar(x_ - w_/2, [np.sqrt(m) for m in base_mses], width=w_,
           color=GREY, label=f"Baseline: {N_EST_BASELINE} trees",
           edgecolor="white")
ax_fit.bar(x_ + w_/2, [np.sqrt(m) for m in best_mses], width=w_,
           color=GREEN, label=f"Best: {N_EST_BEST} trees, depth={MAX_DEPTH}",
           edgecolor="white")
ax_fit.set_xticks(x_)
ax_fit.set_xticklabels([f"M{m}" for m in TEST_MONTHS], fontsize=10)
ax_fit.set_ylabel("RMSE  ($/MWh)", fontsize=11)
ax_fit.set_title(
    f"RMSE by Test Month — Baseline ({N_EST_BASELINE} trees) vs Best Config ({N_EST_BEST} trees)\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
ax_fit.legend(fontsize=9)
apply_style(ax_fit)
plt.tight_layout()
st.pyplot(fig_fit, use_container_width=True)
plt.close(fig_fit)

pct = 100 * (np.sqrt(np.mean(base_mses)) - np.sqrt(np.mean(best_mses))) / np.sqrt(np.mean(base_mses))

st.markdown(f"""
<div class="insight-box">
<strong>Model fit:</strong> The best-config RF ({N_EST_BEST} trees) achieves a mean RMSE of
<strong>{np.sqrt(np.mean(best_mses)):.3f} $/MWh</strong> vs
<strong>{np.sqrt(np.mean(base_mses)):.3f} $/MWh</strong> for the {N_EST_BASELINE}-tree baseline
(<strong>{abs(pct):.1f}% {'improvement' if pct > 0 else 'increase'}</strong>).
<br><br>
The seasonal pattern in RMSE across folds reveals which months had the most unpredictable
deviations from day-ahead prices. Months with large congestion events or unexpected demand
surges produce high spread RMSE — regardless of the absolute price level. This is consistent
with the finding in the KNN and LR spread analyses.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 3 — PLOT 2: Permutation importance
# =============================================================================
st.markdown("## Permutation Variable Importance")
st.markdown(
    "Each of the 12 features is shuffled in the month-12 test set and the resulting increase in MSE (ΔMSE) is recorded. ΔMSE > 1 (red) = dominant; ΔMSE > 0 (orange) = positive; negative (grey) = noisy/redundant."
)

fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
fig_imp.patch.set_facecolor(BG)
bar_colors = [
    RED    if v > 1   else
    ORANGE if v > 0   else
    GREY
    for v in imp_s
]
ax_imp.barh(feat_s[::-1], imp_s[::-1], xerr=std_s[::-1],
            color=bar_colors[::-1], edgecolor="white", capsize=3, height=0.65)
ax_imp.axvline(0, color="grey", lw=1.2, ls="--")
ax_imp.set_xlabel("Mean Δ MSE when feature permuted", fontsize=10)
ax_imp.set_title(
    "Permutation Variable Importance\n",
    fontsize=11, fontweight="bold"
)
apply_style(ax_imp)
plt.tight_layout()
st.pyplot(fig_imp, use_container_width=True)
plt.close(fig_imp)

top3 = [(feat_s[i], imp_s[i], std_s[i]) for i in range(min(3, len(feat_s)))]
top3_html = "".join(
    f"<li><strong>{f}</strong> (ΔMSE = {m:.3f} ± {s:.3f})</li>"
    for f, m, s in top3
)

st.markdown(f"""
<div class="insight-box">
<strong>Top features for spread prediction:</strong>
<ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem;">
  {top3_html}
  <li>The model relies primarily on weather-related variables to predict electricity 
  prices, indicating that atmospheric conditions are the dominant drivers of price 
  fluctuations in the RF model.</li>
  <li>The importance ranking should be compared with the LR, KNN, and XGBoost
      spread pages. Features that rank highly across all four methods carry
      the strongest evidence of genuine predictive value for the spread.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 4 — PLOT 3: n_estimators sensitivity
# =============================================================================
st.markdown("## Sensitivity: MSE vs n_estimators")
st.markdown(
    f"MSE on the month-12 test fold across n_estimators from "
    f"{min(N_EST_SWEEP)} to {max(N_EST_SWEEP)}, with max_depth={MAX_DEPTH} "
    "held fixed. It shows how ensemble size affects spread prediction accuracy and where diminishing returns set in."
)

fig_n, ax_n = plt.subplots(figsize=(8, 4))
fig_n.patch.set_facecolor(BG)
ax_n.plot(list(n_est_mses.keys()), list(n_est_mses.values()),
          marker="o", color=GREEN, lw=2.2, zorder=3)
ax_n.axvline(N_EST_BEST, color=RED, ls="--", lw=1.8,
             label=f"Selected n={N_EST_BEST}")
best_n = min(n_est_mses, key=n_est_mses.get)
ax_n.scatter([best_n], [n_est_mses[best_n]], color=RED, s=90, zorder=5,
             label=f"Best n={best_n} (MSE={n_est_mses[best_n]:.3f})")
ax_n.set_xlabel("n_estimators  (number of trees)", fontsize=10)
ax_n.set_ylabel("MSE  (month-12 test fold)", fontsize=10)
ax_n.set_title(
    f"MSE vs n_estimators  (max_depth={MAX_DEPTH}, month-12 fold)\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
ax_n.legend(fontsize=9)
apply_style(ax_n)
plt.tight_layout()
st.pyplot(fig_n, use_container_width=True)
plt.close(fig_n)

ols_mse  = n_est_mses[min(N_EST_SWEEP)]
best_mse = n_est_mses[best_n]
gain     = 100 * (ols_mse - best_mse) / ols_mse if ols_mse != 0 else 0.0

st.markdown(f"""
<div class="insight-box">
<strong>n_estimators sensitivity:</strong> MSE falls from
{ols_mse:.3f} at n={min(N_EST_SWEEP)} trees to {best_mse:.3f} at n={best_n} trees
({gain:.1f}% improvement).
<br><br>
Model performance improves substantially as the number of trees increases from 10 to 50, with the lowest MSE achieved at 50 estimators. Beyond this point, additional trees provide little improvement, indicating diminishing returns in predictive performance.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
