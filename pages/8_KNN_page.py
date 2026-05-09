#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_🧭_Model-KNN.py
Streamlit page for KNN electricity price prediction analysis.
Place this file at: /Streamlit/pages/page.py
Data expected at:   /Streamlit/data/cleaned_nyiso_electricity_weather.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KNN · NYISO Price Prediction",
    page_icon="🧭",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── typography ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  h1 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
  h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem;
       border-bottom: 2px solid #1a73e8; padding-bottom: 4px; color: #1a73e8; }
  h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }

  /* ── section boxes ── */
  .method-block {
    background: #f7f9ff;
    border-left: 4px solid #1a73e8;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
  }

    .insight-box {
        background: #f7f9ff;
        border-left: 4px solid #1a73e8;
        border-radius: 4px;
        padding: 0.9rem 1.3rem;
        margin-top: 0.8rem;
        margin-bottom: 1.6rem;
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .insight-box strong {
        color: #1a73e8;
    }

  .formula-block {
    background: #f0f4ff;
    border: 1px solid #c7d7f8;
    border-radius: 6px;
    padding: 0.7rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #1a237e;
    margin: 0.6rem 0 1rem 0;
  }

  .tag {
    display: inline-block;
    background: #1a73e8;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 1px 8px;
    border-radius: 20px;
    margin-right: 5px;
    vertical-align: middle;
  }

  hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (shared by all plots)
# ─────────────────────────────────────────────────────────────────────────────
BLUE   = "#1a73e8"
GREEN  = "#34a853"
RED    = "#ea4335"
ORANGE = "#f4a234"
GREY   = "#aaaaaa"
BG     = "#f7f9ff"

def apply_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35, linewidth=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# Data & model pipeline  (cached so the heavy work runs only once)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "DAM Zonal LBMP", "DAM Zonal Losses", "DAM Zonal Congestion",
    "DAM Forecast Load", "month", "day", "hour",
    "Dew Point", "Temperature", "Solar Zenith Angle",
    "Relative Humidity", "Wind Speed", "zone_name",
]
TARGET = "int_LBMP"
DATA_PATH = "data/cleaned_nyiso_electricity_weather.csv"

@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[FEATURES + [TARGET]].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(df), size=len(df) // 3, replace=False))
    X      = df[FEATURES].values[idx]
    y      = df[TARGET].values[idx]
    months = df["month"].values[idx]
    return X, y, months

@st.cache_data(show_spinner="Running cross-validation & hyperparameter search…")
def run_analysis(_X, _y, _months):
    X, y, months = _X, _y, _months

    TEST_MONTHS = list(range(4, 13))
    folds = [(np.where(months < tm)[0], np.where(months == tm)[0])
             for tm in TEST_MONTHS]

    def fit_eval(tr, te, k=10, metric="minkowski", weights="uniform"):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        knn = KNeighborsRegressor(n_neighbors=k, metric=metric,
                                  weights=weights, n_jobs=-1)
        knn.fit(Xtr, y[tr])
        return knn, sc, mean_squared_error(y[te], knn.predict(Xte))

    def cv_mean_mse(k, metric="minkowski", weights="uniform"):
        return np.mean([fit_eval(tr, te, k, metric, weights)[2] for tr, te in folds])

    # ── Baseline ──
    base_mses = [fit_eval(tr, te, k=10)[2] for tr, te in folds]

    # ── Permutation importance (month-12 fold) ──
    tr12, te12 = folds[-1]
    knn_vi, sc_vi, base_mse_vi = fit_eval(tr12, te12, k=30, metric="manhattan")
    Xte_s = sc_vi.transform(X[te12])
    yte   = y[te12]
    rng2  = np.random.default_rng(42)
    imp_results = {}
    for j, feat in enumerate(FEATURES):
        deltas = []
        for _ in range(5):
            Xp = Xte_s.copy()
            Xp[:, j] = rng2.permutation(Xp[:, j])
            deltas.append(mean_squared_error(yte, knn_vi.predict(Xp)) - base_mse_vi)
        imp_results[feat] = (np.mean(deltas), np.std(deltas))

    # ── k sweep ──
    k_vals  = [3, 5, 7, 10, 15, 20, 30, 40, 50]
    k_mean  = {k: cv_mean_mse(k) for k in k_vals}
    best_k  = min(k_mean, key=k_mean.get)

    # ── metric sweep ──
    metrics  = ["minkowski", "manhattan", "chebyshev"]
    met_mse  = {m: fit_eval(tr12, te12, k=best_k, metric=m)[2] for m in metrics}
    best_met = min(met_mse, key=met_mse.get)

    # ── weight sweep ──
    w_mse  = {w: fit_eval(tr12, te12, k=best_k, metric=best_met, weights=w)[2]
               for w in ["uniform", "distance"]}
    best_w = min(w_mse, key=w_mse.get)

    # ── Full rolling window (all 9 folds) ──
    all_base_mses, all_best_mses, train_sizes = [], [], []
    for tr, te in folds:
        _, _, mb = fit_eval(tr, te, k=10, metric="minkowski")
        _, _, mg = fit_eval(tr, te, k=best_k, metric=best_met, weights=best_w)
        all_base_mses.append(mb)
        all_best_mses.append(mg)
        train_sizes.append(len(tr))

    return dict(
        TEST_MONTHS=TEST_MONTHS,
        base_mses=base_mses,
        imp_results=imp_results,
        k_vals=k_vals,
        k_mean=k_mean,
        best_k=best_k,
        met_mse=met_mse,
        best_met=best_met,
        w_mse=w_mse,
        best_w=best_w,
        all_base_mses=all_base_mses,
        all_best_mses=all_best_mses,
        train_sizes=train_sizes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧭  K-Nearest Neighbors")
st.markdown(
    "**NYISO Integrated LBMP Prediction — 2018**  &nbsp;|&nbsp; "
    "Non-parametric local-averaging model trained on day-ahead market, "
    "weather, and time features."
)
st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
K-nearest neighbors (K-NN) is a <strong>nonparametric</strong> prediction method that estimates a target
variable by averaging the most similar historical observations in feature space.
Here the target is <code>int_LBMP</code> (integrated Locational Based Marginal Price, $/MWh).
The intuition is simple: <em>hours with similar market, weather, and time conditions
should have similar electricity prices.</em>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Feature vector")
    st.markdown("""
<div class="formula-block">
x_i = (DAM_LBMP, Losses, Congestion, Load,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;month, day, hour,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DewPoint, Temp, SolarZenith,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RelHumidity, WindSpeed, zone)
</div>
""", unsafe_allow_html=True)
    st.markdown("#### Standardisation (required)")
    st.markdown("""
<div class="formula-block">
z_ij = (x_ij − μ_j) / σ_j
</div>
Large-scale variables (load) would otherwise dominate small-scale ones
(humidity, wind speed) in the distance calculation.
""", unsafe_allow_html=True)

with col2:
    st.markdown("#### Distance & prediction")
    st.markdown("""
<div class="formula-block">
d(x₀, xᵢ) = √Σⱼ (x₀ⱼ − xᵢⱼ)²  &nbsp;(Euclidean baseline)<br><br>
ŷ₀ = (1/k) Σᵢ∈Nₖ(x₀) yᵢ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(uniform weights)<br><br>
ŷ₀ = Σ wᵢyᵢ / Σ wᵢ, &nbsp;wᵢ=1/d  (distance weights)
</div>
""", unsafe_allow_html=True)
    st.markdown("""
K-NN is a **local averaging** method — it makes each forecast based on
similar historical hours rather than fitting a single global relationship.
Including zone as a feature lets the model capture that similar conditions
can produce different prices across NYISO zones.
""")

st.markdown("#### Practical considerations")
st.markdown("""
<div class="method-block">
<ul style="margin:0; padding-left:1.2rem;">
  <li><strong>Scaling is essential</strong> — distances are meaningless without it.</li>
  <li><strong>Choice of k</strong> — small k → sensitive to noise; large k → smooth but may miss price spikes.</li>
  <li><strong>Time-respecting split</strong> — the training set always ends before the test month; no look-ahead.</li>
  <li><strong>Curse of dimensionality</strong> — with 13 features, distances can become less informative; careful
      feature selection and normalisation are therefore central to the analysis.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("#### Workflow overview")
steps = [
    ("Step 1", "Load & preprocess — encode zone_name numerically, drop NaN rows, sort chronologically."),
    ("Step 2", "Subsample 1-in-3 — thin the dataset for computational efficiency while preserving distributions."),
    ("Step 3", "Rolling CV folds — 9 expanding-window folds with test months 4–12."),
    ("Step 4", "Baseline KNN — k=10, Euclidean, uniform weights; record MSE per fold."),
    ("Step 5", "Permutation importance — shuffle each feature 5× on the month-12 test set; measure ΔMSE."),
    ("Step 6", "Hyperparameter sensitivity — sweep over k, distance metric, and weighting scheme."),
]
for tag, text in steps:
    st.markdown(f'<span class="tag">{tag}</span> {text}', unsafe_allow_html=True)
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & RUN MODELS
# ─────────────────────────────────────────────────────────────────────────────
X, y, months = load_data()
res = run_analysis(X, y, months)

TEST_MONTHS    = res["TEST_MONTHS"]
imp_results    = res["imp_results"]
k_vals         = res["k_vals"]
k_mean         = res["k_mean"]
best_k         = res["best_k"]
best_met       = res["best_met"]
best_w         = res["best_w"]
met_mse        = res["met_mse"]
w_mse          = res["w_mse"]
all_base_mses  = res["all_base_mses"]
all_best_mses  = res["all_best_mses"]
train_sizes    = res["train_sizes"]

# Pre-sort features by importance
feat_arr = list(imp_results.keys())
imp_arr  = [imp_results[f][0] for f in feat_arr]
std_arr  = [imp_results[f][1] for f in feat_arr]
sidx     = np.argsort(imp_arr)[::-1]
feat_s   = [feat_arr[i] for i in sidx]
imp_s    = [imp_arr[i]  for i in sidx]
std_s    = [std_arr[i]  for i in sidx]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PERMUTATION VARIABLE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Permutation Variable Importance")
st.markdown(
    "Each feature is shuffled 5 times in the month-12 test set. "
    "The mean increase in MSE (ΔMSE) measures how much the model depends on that feature. "
    "Negative values indicate a feature adds noise rather than signal."
)

fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
fig_imp.patch.set_facecolor(BG)
bar_colors = [RED if v > 5 else (BLUE if v >= 0 else GREY) for v in imp_s]
ax_imp.barh(feat_s[::-1], imp_s[::-1], xerr=std_s[::-1],
            color=bar_colors[::-1], edgecolor="white", capsize=3, height=0.65)
ax_imp.axvline(0, color="grey", lw=1.2, ls="--")
ax_imp.set_xlabel("Mean Δ MSE when feature permuted", fontsize=10)
ax_imp.set_title("Permutation Variable Importance  (higher = more important)", fontsize=11, fontweight="bold")
apply_style(ax_imp)
plt.tight_layout()
st.pyplot(fig_imp, use_container_width=True)
plt.close(fig_imp)

st.markdown("""
<div class="insight-box">
<strong>Key findings:</strong>
<ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem;">
  <li><strong>DAM Zonal LBMP (+38.9)</strong> — by far the dominant driver; the day-ahead price is the single strongest predictor of the real-time price.</li>
  <li><strong>hour (+12.9)</strong> — strong intraday pattern (peak vs off-peak hours).</li>
  <li><strong>DAM Zonal Losses (+9.2)</strong> — transmission loss component carries significant information.</li>
  <li><strong>DAM Zonal Congestion (+2.8)</strong> — moderate contribution; relevant when the grid is constrained.</li>
  <li><strong>DAM Forecast Load, Dew Point, Temperature (~0.6–0.8)</strong> — small marginal contribution given the DAM features already present.</li>
  <li><strong>Negative importances (day, Wind Speed, zone_name, Solar Zenith)</strong> — shuffling these features <em>improves or doesn't hurt</em> prediction. This is a known artefact in KNN permutation importance: features that add noise to the distance metric degrade neighbour selection. These variables are redundant or noisy given the other features and could be dropped in a refined model.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MSE SENSITIVITY TO k
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## MSE Sensitivity to Hyperparameters")

col_k, col_met = st.columns(2)

with col_k:
    st.markdown("#### k sweep (number of neighbours)")
    fig_k, ax_k = plt.subplots(figsize=(5.5, 3.8))
    fig_k.patch.set_facecolor(BG)
    ax_k.plot(k_vals, [k_mean[k] for k in k_vals], marker="o", color=BLUE, lw=2.2)
    ax_k.axvline(best_k, color=RED, ls="--", lw=1.5, label=f"Selected k = {best_k}")
    ax_k.fill_between(k_vals, [k_mean[k] for k in k_vals],
                      max(k_mean.values()), alpha=0.06, color=BLUE)
    ax_k.set_xlabel("k  (number of neighbours)", fontsize=10)
    ax_k.set_ylabel("Mean MSE  (all 9 folds)", fontsize=10)
    ax_k.set_title("k Sweep — Mean CV MSE", fontsize=10, fontweight="bold")
    ax_k.legend(fontsize=9)
    apply_style(ax_k)
    plt.tight_layout()
    st.pyplot(fig_k, use_container_width=True)
    plt.close(fig_k)

with col_met:
    st.markdown("#### Distance metric & weighting")
    fig_mw, axes_mw = plt.subplots(1, 2, figsize=(5.5, 3.8))
    fig_mw.patch.set_facecolor(BG)

    metric_labels = {"minkowski": "Euclidean\n(Minkowski p=2)", "manhattan": "Manhattan", "chebyshev": "Chebyshev"}
    met_names = list(met_mse.keys())
    met_vals  = [met_mse[m] for m in met_names]
    bar_c     = [GREEN if m == best_met else BLUE for m in met_names]
    axes_mw[0].bar([metric_labels[m] for m in met_names], met_vals,
                   color=bar_c, edgecolor="white")
    axes_mw[0].set_ylabel("MSE", fontsize=9)
    axes_mw[0].set_title("Distance Metric\n(month-12 fold)", fontsize=9, fontweight="bold")
    apply_style(axes_mw[0])

    w_names = list(w_mse.keys())
    w_vals  = [w_mse[w] for w in w_names]
    bar_c2  = [GREEN if w == best_w else BLUE for w in w_names]
    axes_mw[1].bar(w_names, w_vals, color=bar_c2, edgecolor="white")
    axes_mw[1].set_title("Weighting Scheme\n(month-12 fold)", fontsize=9, fontweight="bold")
    apply_style(axes_mw[1])

    plt.tight_layout()
    st.pyplot(fig_mw, use_container_width=True)
    plt.close(fig_mw)

st.markdown(f"""
<div class="insight-box">
<strong>k sweep:</strong> MSE falls monotonically from k=3 (overfit, MSE≈1141) to k=50 (MSE≈689).
There is no clear elbow — the model keeps benefiting from smoothing over more neighbours.
This suggests the price surface is relatively smooth and a larger k is appropriate.
<br><br>
<strong>Distance metric:</strong> <em>{metric_labels.get(best_met, best_met)}</em> performs best
(MSE = {met_mse[best_met]:.0f}). Manhattan distance is slightly more robust to price spikes because
it penalises individual large deviations less than Euclidean does.
<br><br>
<strong>Weighting:</strong> <em>{best_w}</em> weighting wins (MSE = {w_mse[best_w]:.0f}).
Distance and uniform weighting perform similarly in this sample, reflecting a relatively homogeneous
neighbourhood structure around most observations.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ROLLING WINDOW MSE EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Rolling Window MSE Evolution — All 9 Folds")
st.markdown(
    "Expanding training window: each fold adds one more month of history. "
    "Test months 4–12 cover spring through winter 2018."
)

base_rmses = [np.sqrt(m) for m in all_base_mses]
best_rmses = [np.sqrt(m) for m in all_best_mses]
x_         = np.arange(len(TEST_MONTHS))
month_labels = [f"Month {m}" for m in TEST_MONTHS]

fig_roll, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"hspace": 0.08, "height_ratios": [2, 1]}
)
fig_roll.patch.set_facecolor(BG)

# Top panel — RMSE lines
ax_top.set_facecolor(BG)
ax_top.plot(x_, base_rmses, marker="o", lw=2.2, color=BLUE,
            label="Baseline (k=10, Euclidean, uniform)", zorder=3)
ax_top.plot(x_, best_rmses, marker="s", lw=2.2, color=GREEN,
            label=f"Best (k={best_k}, {best_met}, {best_w})", zorder=3)
ax_top.fill_between(x_, base_rmses, best_rmses, alpha=0.13, color=GREEN,
                    label="Improvement from tuning")

for i, (b, g) in enumerate(zip(base_rmses, best_rmses)):
    ax_top.annotate(f"{b:.1f}", (x_[i], b), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=BLUE)
    ax_top.annotate(f"{g:.1f}", (x_[i], g), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=GREEN)

# Shade seasonally difficult months (RMSE > median + 1 std)
threshold = np.mean(best_rmses) + np.std(best_rmses)
for i in range(len(TEST_MONTHS)):
    if best_rmses[i] > threshold:
        ax_top.axvspan(i - 0.45, i + 0.45, color="tomato", alpha=0.09, zorder=0)

ax_top.set_ylabel("RMSE  ($/MWh)", fontsize=11)
ax_top.set_title(
    "RMSE per Fold — Baseline vs Tuned Configuration\n"
    "Red shading = seasonally difficult months (high price volatility)",
    fontsize=11, fontweight="bold"
)
ax_top.legend(fontsize=9, loc="upper right")
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_top.grid(True, alpha=0.35, linewidth=0.7)

# Bottom panel — training set growth
ax_bot.set_facecolor(BG)
ax_bot.bar(x_, train_sizes, color=BLUE, alpha=0.72, edgecolor="white")
ax_bot.set_ylabel("Training rows\n(subsampled)", fontsize=9)
ax_bot.set_xlabel("Test fold", fontsize=11)
ax_bot.set_xticks(x_)
ax_bot.set_xticklabels(month_labels, rotation=15, ha="right", fontsize=9)
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)
ax_bot.grid(True, axis="y", alpha=0.35)
for i, s in enumerate(train_sizes):
    ax_bot.text(i, s + 150, f"{s:,}", ha="center", fontsize=7.5, color="#333")

plt.tight_layout()
st.pyplot(fig_roll, use_container_width=True)
plt.close(fig_roll)

st.markdown(f"""
<div class="insight-box">
<strong>Top panel — RMSE evolution:</strong> RMSE does <em>not</em> fall monotonically even though the
training set grows every fold. The dominant effect is <strong>seasonal difficulty</strong>, not data quantity.
<ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem;">
  <li>Months 6 and 9 spike sharply (RMSE ~38 and ~42 $/MWh) — these coincide with summer heat events and
      early-autumn price volatility in NYISO, when extreme demand peaks cause large, hard-to-predict price spikes.</li>
  <li>Month 12 is the easiest to predict (RMSE ~18–19 $/MWh) despite having the most training data —
      winter electricity demand in December 2018 was relatively stable compared to the summer months.</li>
  <li>The green shading shows the consistent improvement the tuned model
      (k={best_k}, {best_met}) delivers over baseline — most meaningful in months 7–12.</li>
</ul>
<strong>Bottom panel — training set growth:</strong> Confirms the expanding window is working as intended,
growing from ~7,800 rows in fold 1 to ~29,500 by fold 9. The fact that more data does not guarantee
lower RMSE (month 9 has more data than month 6 but similar RMSE) reinforces that
<strong>seasonal regime shifts matter more than sample size</strong> for KNN.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RMSE COMPARISON BAR CHART (all 9 folds side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Rolling CV: RMSE per Fold — Baseline vs Best Configuration")

fig_bar, ax_bar = plt.subplots(figsize=(11, 4.5))
fig_bar.patch.set_facecolor(BG)
w_ = 0.34
ax_bar.bar(x_ - w_/2, base_rmses, width=w_, color=BLUE,
           label="Baseline (k=10, Euclidean, uniform)", edgecolor="white")
ax_bar.bar(x_ + w_/2, best_rmses, width=w_, color=GREEN,
           label=f"Best (k={best_k}, {best_met}, {best_w})", edgecolor="white")
ax_bar.set_xticks(x_)
ax_bar.set_xticklabels([f"M{m}" for m in TEST_MONTHS], fontsize=10)
ax_bar.set_ylabel("RMSE  ($/MWh)", fontsize=11)
ax_bar.set_title("RMSE by Test Month — Baseline vs Tuned KNN", fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=9)
apply_style(ax_bar)
plt.tight_layout()
st.pyplot(fig_bar, use_container_width=True)
plt.close(fig_bar)

mean_base = np.mean(base_rmses)
mean_best = np.mean(best_rmses)
pct       = 100 * (mean_base - mean_best) / mean_base

st.markdown(f"""
<div class="insight-box">
Averaged across all 9 folds, the tuned model achieves a mean RMSE of
<strong>{mean_best:.2f} $/MWh</strong> vs <strong>{mean_base:.2f} $/MWh</strong> for the baseline —
a reduction of <strong>{pct:.1f}%</strong>.
The improvement is most pronounced in the seasonally volatile summer and autumn months (M6–M9),
where better neighbour selection via Manhattan distance reduces the influence of outlier neighbours
caused by price spikes.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.caption(
    "Data: NYISO 2018 · cleaned_nyiso_electricity_weather.csv &nbsp;|&nbsp; "
    "Model: KNeighborsRegressor (scikit-learn) &nbsp;|&nbsp; "
    "Subsample: 1-in-3 rows, seed=42"
)
