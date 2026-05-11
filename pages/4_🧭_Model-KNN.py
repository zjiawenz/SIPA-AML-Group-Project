#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_🧭_Model-KNN.py
Streamlit page for KNN spread prediction analysis.
Target: spread = int_LBMP − DAM Zonal LBMP
Place this file at: /Streamlit/pages/
Data expected at:   /Streamlit/data/cleaned_nyiso_electricity_weather.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Nearest Neighbors",
    page_icon="🧭",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
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
# Data & model pipeline — spread target, OHE encoding
# ─────────────────────────────────────────────────────────────────────────────
DATASET_YEAR = 2018
DATA_PATH    = "data/cleaned_nyiso_electricity_weather.csv"

CONTINUOUS_COLS = [
    "DAM Zonal Losses", "DAM Zonal Congestion", "DAM Forecast Load",
    "Dew Point", "Temperature", "Solar Zenith Angle",
    "Relative Humidity", "Wind Speed",
]
OHE_COLS          = ["month", "hour", "day_of_week", "zone_name"]
ORIGINAL_FEATURES = CONTINUOUS_COLS + OHE_COLS   # 12 original predictors

K_BASELINE  = 10
K_BEST      = 30
METRIC_BEST = "manhattan"
WEIGHT_BEST = "uniform"
TEST_MONTHS = list(range(4, 13))
K_SWEEP     = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
N_PERM      = 5


@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    df = pd.read_csv(DATA_PATH)

    RAW_COLS = [
        "DAM Zonal LBMP", "DAM Zonal Losses", "DAM Zonal Congestion",
        "DAM Forecast Load", "month", "day", "hour",
        "Dew Point", "Temperature", "Solar Zenith Angle",
        "Relative Humidity", "Wind Speed", "zone_name", "int_LBMP",
    ]
    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[RAW_COLS].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)

    # Spread target
    df["spread"] = df["int_LBMP"] - df["DAM Zonal LBMP"]

    # day_of_week from 2018 calendar; drop raw day column
    df["date"]        = pd.to_datetime({"year": DATASET_YEAR,
                                        "month": df["month"], "day": df["day"]})
    df["day_of_week"] = df["date"].dt.dayofweek
    df.drop(columns=["date", "day"], inplace=True)

    # 1-in-3 subsample
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(df), size=len(df) // 3, replace=False))
    df_sub = df.iloc[idx].reset_index(drop=True)

    # OHE structure (all categories present in subsample)
    ohe_structure = {col: sorted(df_sub[col].unique().tolist()) for col in OHE_COLS}

    # Build encoded feature matrix: continuous first, then OHE dummies
    df_ohe = df_sub[CONTINUOUS_COLS].copy()
    for col in OHE_COLS:
        dummies = pd.get_dummies(df_sub[col], prefix=col, drop_first=False).astype(int)
        df_ohe  = pd.concat([df_ohe, dummies], axis=1)

    encoded_cols    = list(df_ohe.columns)
    X_enc_raw       = df_ohe.values.astype(float)
    y_sub           = df_sub["spread"].values
    months_sub      = df_sub["month"].values
    n_cont          = len(CONTINUOUS_COLS)

    # Map original feature → column indices in the encoded matrix
    feature_col_indices = {}
    for feat in ORIGINAL_FEATURES:
        if feat in CONTINUOUS_COLS:
            feature_col_indices[feat] = [CONTINUOUS_COLS.index(feat)]
        else:
            prefix  = feat + "_"
            feature_col_indices[feat] = [
                i for i, c in enumerate(encoded_cols) if c.startswith(prefix)
            ]

    return (X_enc_raw, y_sub, months_sub, n_cont,
            encoded_cols, feature_col_indices, ohe_structure)


@st.cache_data(show_spinner="Running cross-validation & hyperparameter search…")
def run_analysis(_X_enc_raw, _y_sub, _months_sub, n_cont, feature_col_indices):
    X_enc_raw  = _X_enc_raw
    y_sub      = _y_sub
    months_sub = _months_sub

    folds = [
        (np.where(months_sub < tm)[0], np.where(months_sub == tm)[0])
        for tm in TEST_MONTHS
    ]

    def fit_fold(tr_idx, te_idx, k=10, metric="minkowski", weights="uniform"):
        Xtr = X_enc_raw[tr_idx].copy()
        Xte = X_enc_raw[te_idx].copy()
        sc  = StandardScaler()
        Xtr[:, :n_cont] = sc.fit_transform(Xtr[:, :n_cont])
        Xte[:, :n_cont] = sc.transform(Xte[:, :n_cont])
        knn = KNeighborsRegressor(n_neighbors=k, metric=metric,
                                   weights=weights, n_jobs=-1)
        knn.fit(Xtr, y_sub[tr_idx])
        return knn, sc, Xte, mean_squared_error(y_sub[te_idx], knn.predict(Xte))

    # ── Baseline rolling CV ──
    base_mses = [fit_fold(tr, te, k=K_BASELINE)[3] for tr, te in folds]

    # ── Best-config rolling CV ──
    best_mses, train_sizes = [], []
    for tr, te in folds:
        _, _, _, mse = fit_fold(tr, te, k=K_BEST, metric=METRIC_BEST, weights=WEIGHT_BEST)
        best_mses.append(mse)
        train_sizes.append(len(tr))

    # ── Permutation importance — month-12 fold, original features ──
    tr12, te12 = folds[-1]
    knn_vi, _, X_te_scaled, base_mse_vi = fit_fold(
        tr12, te12, k=K_BEST, metric=METRIC_BEST, weights=WEIGHT_BEST
    )
    y_te      = y_sub[te12]
    rng_perm  = np.random.default_rng(42)
    imp_results = {}
    for feat in ORIGINAL_FEATURES:
        col_indices = feature_col_indices[feat]
        deltas = []
        for _ in range(N_PERM):
            Xp = X_te_scaled.copy()
            if feat in CONTINUOUS_COLS:
                Xp[:, col_indices[0]] = rng_perm.permutation(Xp[:, col_indices[0]])
            else:
                perm_idx = rng_perm.permutation(len(Xp))
                for ci in col_indices:
                    Xp[:, ci] = Xp[perm_idx, ci]
            deltas.append(mean_squared_error(y_te, knn_vi.predict(Xp)) - base_mse_vi)
        imp_results[feat] = (np.mean(deltas), np.std(deltas))

    # ── k sweep — month-12 fold ──
    k_mses = {}
    for k in K_SWEEP:
        _, _, _, mse = fit_fold(tr12, te12, k=k, metric=METRIC_BEST, weights=WEIGHT_BEST)
        k_mses[k] = mse

    return dict(
        base_mses=base_mses,
        best_mses=best_mses,
        train_sizes=train_sizes,
        imp_results=imp_results,
        k_mses=k_mses,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧭  K-Nearest Neighbors Model Analysis")
st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
K-nearest neighbors (K-NN) is a <strong>nonparametric</strong> prediction method that estimates a target
variable by averaging the most similar historical observations in feature space.
The target here is the <strong>spread</strong> — the hourly deviation of the real-time integrated LBMP
from the day-ahead market forecast for that same hour.
The intuition is simple: <em>hours with similar market, weather, and time conditions
should have similar deviations from the day-ahead price.</em>
</div>
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
    st.markdown("#### Feature encoding (58 total)")
    st.markdown("""
<div class="formula-block">
Continuous (8) → StandardScaler<br>
DAM Zonal Losses<br>
DAM Zonal Congestion<br>
DAM Forecast Load<br>
Dew Point, Temperature<br>
Solar Zenith Angle<br>
Relative Humidity, Wind Speed<br>
month       → 12 dummies <br> 
hour        → 24 dummies <br>
day_of_week →  7 dummies <br>
zone_name   → 11 dummies <br>
─────────────────────────────────<br>
Total: 8 + 54 = 62 features
</div>
""", unsafe_allow_html=True)


with col2:
    st.markdown("#### Distance & prediction")
    st.markdown("""
<div class="formula-block">
d(x₀, xᵢ) = Σⱼ |x₀ⱼ − xᵢⱼ|  &nbsp;(Manhattan distance metric — tuned best)<br><br>
ŷ₀ = (1/k) Σᵢ∈Nₖ(x₀) spreadᵢ  &nbsp;(uniform weights)<br><br>
int_LBMP reconstructed = DAM_LBMP + ŷ₀
</div>
""", unsafe_allow_html=True)
    st.markdown("""
**Why Manhattan distance for OHE?** Euclidean is the square root of the sum of squared distances, while Manhattan is the sum of absolute values of distance.
Euclidean allows diagonal movement; Manhattan requires horizontal/vertical movement.
With binary columns, Manhattan distance counts the number of category mismatches directly (each mismatch contributes 2).
Euclidean would give partial credit for partial overlap, which is less meaningful
for nominal categories like zone or hour. Manhattan distance is often preferred in high-dimensional spaces.
""")


st.markdown("#### Workflow overview")
steps = [
    ("Step 1", "Load & preprocess — encode zone_name numerically, construct spread target, derive day_of_week, drop day-of-month. Build encoded matrix — one-hot encode month/hour/day_of_week/zone_name; scale continuous variables using StandardScaler."),
    ("Step 2", "Subsample 1-in-3 — thin the dataset for computational efficiency while preserving distributions."),
    ("Step 3", "Rolling Cross Validation folds — 9 expanding-window folds with test months 4–12."),
    ("Step 4", "Baseline KNN — k=10, Euclidean, uniform weights; record MSE per fold."),
    ("Step 5", "Best-configuration KNN — k=30, Manhattan, uniform; record MSE per fold."),
    ("Step 6", "Permutation importance — permute each original feature (OHE columns jointly) 5× on month-12 test set; measure ΔMSE."),
    ("Step 7", "k sweep — vary k=1–40 on month-12 fold to show sensitivity."),
]
for tag, text in steps:
    st.markdown(f'<span class="tag">{tag}</span> {text}', unsafe_allow_html=True)
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & RUN MODELS
# ─────────────────────────────────────────────────────────────────────────────
(X_enc_raw, y_sub, months_sub, n_cont,
 encoded_cols, feature_col_indices, ohe_structure) = load_data()

res = run_analysis(X_enc_raw, y_sub, months_sub, n_cont, feature_col_indices)

base_mses   = res["base_mses"]
best_mses   = res["best_mses"]
train_sizes = res["train_sizes"]
imp_results = res["imp_results"]
k_mses      = res["k_mses"]

# Pre-sort features by importance (descending)
sort_order = sorted(imp_results, key=lambda f: -imp_results[f][0])
feat_s = sort_order
imp_s  = [imp_results[f][0] for f in feat_s]
std_s  = [imp_results[f][1] for f in feat_s]

base_rmses   = [np.sqrt(m) for m in base_mses]
best_rmses   = [np.sqrt(m) for m in best_mses]
x_           = np.arange(len(TEST_MONTHS))
month_labels = [f"Month {m}" for m in TEST_MONTHS]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ROLLING WINDOW RMSE EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Rolling CV Model Fit")

st.markdown(
    "Expanding training window: each fold adds one more month of history. Baseline k-NN with k=10 and Best Tuned k-NN with k=50 are compared for each fold/window. "
)

fig_bar, ax_bar = plt.subplots(figsize=(11, 4.5))
fig_bar.patch.set_facecolor(BG)
w_ = 0.34
ax_bar.bar(x_ - w_/2, base_rmses, width=w_, color=BLUE,
           label=f"Baseline (k={K_BASELINE}, Euclidean, uniform)", edgecolor="white")
ax_bar.bar(x_ + w_/2, best_rmses, width=w_, color=GREEN,
           label=f"Best (k={K_BEST}, {METRIC_BEST}, {WEIGHT_BEST})", edgecolor="white")
ax_bar.set_xticks(x_)
ax_bar.set_xticklabels([f"M{m}" for m in TEST_MONTHS], fontsize=10)
ax_bar.set_ylabel("RMSE  ($/MWh)", fontsize=11)
ax_bar.set_title("Spread RMSE by Test Month — Baseline vs Tuned KNN",
                 fontsize=11, fontweight="bold")
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
RMSE does <em>not</em> fall monotonically with growing training data — seasonal difficulty
dominates data quantity, as in the raw LBMP analysis.
<ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem;">
  <li>High-RMSE months M9 and M10 coincide with summer and early-autumn NYISO volatility — periods
      where real-time prices deviate most unpredictably from day-ahead forecasts.</li>
  <li>Easier months (lower RMSE) reflect more stable spread behaviour, where the day-ahead market
      accurately captured real-time conditions.</li>
  <li>The green shading shows consistent improvement from the tuned configuration
      (k={K_BEST}, {METRIC_BEST}) over the baseline across most folds.</li>
</ul>
Averaged across all 9 folds, the tuned model achieves a mean RMSE of
<strong>{mean_best:.2f} $/MWh</strong> vs <strong>{mean_base:.2f} $/MWh</strong> for the baseline —
a reduction of <strong>{pct:.1f}%</strong>.
The improvement is most pronounced in the seasonally volatile summer and autumn months (M6–M9),
where better neighbour selection via Manhattan distance reduces the influence of outlier neighbours
caused by spread spikes driven by congestion and demand surges.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PERMUTATION VARIABLE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Permutation Variable Importance")
st.markdown(
    "Each of the 12 original features is permuted 5 times in the month-12 test set. "
    "For OHE features (month, hour, day_of_week, zone_name), **all dummy columns are permuted "
    "simultaneously** to preserve the one-hot constraint. "
    "ΔMSE > 1 (red) = dominant; ΔMSE > 0 (orange) = positive; negative (grey) = noisy/redundant."
)

fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
fig_imp.patch.set_facecolor(BG)
bar_colors = [RED if v > 1 else (ORANGE if v >= 0 else GREY) for v in imp_s]
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

top3 = feat_s[:3]
neg_feats = [f for f in feat_s if imp_results[f][0] < 0]
neg_str   = ", ".join(neg_feats) if neg_feats else "none"

top3_html = "".join(
    f"<li><strong>{f}</strong> (ΔMSE = {imp_results[f][0]:.3f} ± {imp_results[f][1]:.3f})</li>"
    for f in top3
)

st.markdown(f"""
<div class="insight-box">
<strong>Key findings — </strong>
<strong>Top features for spread prediction:</strong>
<ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem;">
  {top3_html}
    <li><strong>Positive importances (DAM Forecast Load, Zone Name, Temperature, DAM Congestion):</strong> these features are important to the spread signal. Permuting or dropping them hurts a model. </li>
  <li><strong>Negative importances ({neg_str}):</strong> these features add noise to the encoded distance
      metric, degrading neighbour selection. Permuting them improves or does not hurt prediction — a signal
      they could be dropped in a refined model.</li>
</ul>
<strong>Methodological note:</strong> Permuting all OHE dummy columns for a feature simultaneously
preserves the one-hot constraint (exactly one dummy active per row) and gives importance for the
<em>original variable</em>, not individual dummy columns — making results directly comparable across
continuous and categorical features.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — k SENSITIVITY SWEEP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## MSE Sensitivity to k")
st.markdown(
    "Varying the number of neighbours from k=1 to k=40 on the month-12 fold "
    f"(Manhattan distance, {WEIGHT_BEST} weights). "
    "This reveals whether the spread surface is smooth (favours large k) or local (favours small k)."
)

best_k_sweep = min(k_mses, key=k_mses.get)

fig_k, ax_k = plt.subplots(figsize=(8, 4))
fig_k.patch.set_facecolor(BG)
ax_k.plot(list(k_mses.keys()), list(k_mses.values()),
          marker="o", color=BLUE, lw=2.2)
ax_k.axvline(K_BEST, color=RED, ls="--", lw=1.5, label=f"Selected k = {K_BEST}")
ax_k.axvline(best_k_sweep, color=GREEN, ls=":", lw=1.5,
             label=f"Best on month-12 k = {best_k_sweep}")
ax_k.fill_between(list(k_mses.keys()), list(k_mses.values()),
                  max(k_mses.values()), alpha=0.06, color=BLUE)
ax_k.set_xlabel("k  (number of neighbours)", fontsize=10)
ax_k.set_ylabel("MSE  (month-12 test fold)", fontsize=10)
ax_k.set_title(
    f"k Sweep — MSE on Month-12 Fold\n"
    f"({METRIC_BEST.capitalize()} distance, {WEIGHT_BEST} weights, spread target)",
    fontsize=11, fontweight="bold"
)
ax_k.legend(fontsize=9)
apply_style(ax_k)
plt.tight_layout()
st.pyplot(fig_k, use_container_width=True)
plt.close(fig_k)

gain_1_to_best = 100 * (k_mses[1] - k_mses[best_k_sweep]) / k_mses[1]
st.markdown(f"""
<div class="insight-box">
MSE drops sharply from k=1 ({k_mses[1]:.1f}) to k={best_k_sweep} ({k_mses[best_k_sweep]:.1f}),
a reduction of <strong>{gain_1_to_best:.1f}%</strong>.
The best k on the month-12 fold is <strong>k={best_k_sweep}</strong>.
<strong>Selected k = {K_BEST}</strong> balances local accuracy with variance reduction and computational complexity. 
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

