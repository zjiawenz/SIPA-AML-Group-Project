#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3___Model-XGBoost-Spread.py
=============================================================================
XGBoost — Spread Analysis  (NYISO 2018)
Target: spread = int_LBMP - DAM Zonal LBMP

Streamlit page. Drop into the pages/ folder of your multi-page app.

Three analysis sections:
  1. Rolling CV — RMSE per fold (baseline 50 rounds vs best 100 rounds)
  2. Permutation variable importance (raw features, direct permutation)
  3. MSE vs n_estimators sensitivity sweep (10 to 100)

Key methodological points
--------------------------
  Target     : spread = int_LBMP - DAM Zonal LBMP ($/MWh)
  Features   : 12 raw features (DAM Zonal LBMP excluded — part of target)
               day_of_week added; day of month dropped
  Encoding   : NONE — XGBoost is tree-based and scale-invariant
  Scaling    : NONE — trees split on thresholds, not distances
  Dataset    : Full dataset (no subsampling)
  Folds      : 9 expanding-window, test months 4-12
  Baseline   : 50 boosting rounds, learning_rate=0.1, max_depth=6
  Best       : 100 boosting rounds, learning_rate=0.1, max_depth=6
  Sensitivity: n_estimators sweep 10-100, month-12 fold only

Key difference from Random Forest
----------------------------------
  RF uses bagging — trees are built independently on bootstrap samples
  and averaged. XGBoost uses boosting — trees are built sequentially,
  each correcting the residuals of the current ensemble. This makes
  XGBoost more powerful on structured tabular data but also more
  susceptible to overfitting the training-period spread patterns,
  particularly when those patterns shift seasonally.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)
from xgboost import XGBRegressor

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="XGBoost",
    page_icon="🚀",
    layout="wide",
)

# =============================================================================
# Custom CSS  (consistent with LR, KNN, and RF pages — orange theme)
# =============================================================================
st.markdown("""
<style>
  .method-block {
    background: #fffbeb;
    border-left: 4px solid #d97706;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
  }
  .insight-box {
    background: #fffbeb;
    border-left: 4px solid #d97706;
    border-radius: 4px;
    padding: 0.9rem 1.3rem;
    margin-top: 0.8rem;
    margin-bottom: 1.6rem;
    font-size: 0.92rem;
    line-height: 1.65;
  }
  .insight-box strong { color: #78350f; }
  .formula-block {
    background: #fef3c7;
    border: 1px solid #fcd34d;
    border-radius: 6px;
    padding: 0.7rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #78350f;
    margin: 0.6rem 0 1rem 0;
  }
  .tag {
    display: inline-block;
    background: #d97706;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 1px 8px;
    border-radius: 20px;
    margin-right: 5px;
    vertical-align: middle;
  }
  hr { border: none; border-top: 1px solid #fde68a; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Colour palette
# =============================================================================
ORANGE  = "#d97706"
DARK    = "#78350f"
LIGHT   = "#fef3c7"
BG      = "#fffbeb"
GREEN   = "#16a34a"
BLUE    = "#1a56db"
RED     = "#c62828"
GREY    = "#aaaaaa"


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
LEARNING_RATE  = 0.1
MAX_DEPTH      = 6
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

    Steps:
      1. Load CSV, label-encode zone_name to integer 0-10.
      2. Construct spread target: int_LBMP - DAM Zonal LBMP.
      3. Derive day_of_week from year/month/day; drop day of month.
      4. Return raw feature matrix (12 cols), spread target, month index.

    No subsampling — XGBoost prediction is O(n_trees * max_depth) per
    sample, independent of training set size, so the full dataset is used.
    No encoding or scaling — XGBoost is tree-based and scale-invariant.
    """
    df = pd.read_csv(DATA_PATH)

    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[RAW_COLS].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)

    # Spread target: real-time price minus day-ahead forecast for that hour
    df["spread"] = df["int_LBMP"] - df["DAM Zonal LBMP"]

    # Derive day_of_week (0=Monday ... 6=Sunday); drop day of month
    df["date"]        = pd.to_datetime({"year": DATASET_YEAR,
                                        "month": df["month"],
                                        "day":   df["day"]})
    df["day_of_week"] = df["date"].dt.dayofweek
    df.drop(columns=["date", "day"], inplace=True)

    X      = df[FEATURES].values
    y      = df["spread"].values
    months = df["month"].values

    return X, y, months


# =============================================================================
# Model pipeline (cached)
# =============================================================================
@st.cache_data(show_spinner="Training XGBoost models across 9 folds…")
def run_analysis(_X, _y, _months):
    """
    Rolling CV, permutation importance, and n_estimators sensitivity sweep.

    No scaling or encoding applied — XGBoost receives raw feature values.
    Identical data pipeline to the RF spread analysis; difference is in
    the model: sequential boosting vs independent bagged trees.
    """
    X, y, months = _X, _y, _months

    TEST_MONTHS = list(range(4, 13))
    folds = [(np.where(months < tm)[0], np.where(months == tm)[0])
             for tm in TEST_MONTHS]

    def fit_fold(tr, te, n_estimators=N_EST_BEST):
        """
        Fit XGBoost on one rolling-window fold.

        No scaling or encoding applied. XGBoost builds trees sequentially
        (boosting) — each tree corrects the residuals of the current
        ensemble. learning_rate shrinks each tree's contribution to
        prevent any single tree from dominating (regularisation).

        Returns fitted model and test MSE.
        """
        xgb = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=LEARNING_RATE,  # shrinkage — reduces overfitting
            max_depth=MAX_DEPTH,           # depth per tree (shallower than RF)
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,                   # suppress XGBoost console output
        )
        xgb.fit(X[tr], y[tr])
        mse = mean_squared_error(y[te], xgb.predict(X[te]))
        return xgb, mse

    # ── Baseline rolling CV (50 boosting rounds) ──────────────────────────────
    # 50 rounds with learning_rate=0.1 — a moderate ensemble. Used as the
    # weaker baseline to contrast with 100 rounds in the best config.
    base_mses = []
    for tr, te in folds:
        _, mse = fit_fold(tr, te, n_estimators=N_EST_BASELINE)
        base_mses.append(mse)

    # ── Best-config rolling CV (100 boosting rounds) ──────────────────────────
    # 100 rounds matches the settings used in train_spread_models.py —
    # the model actually deployed in the Streamlit prediction tool.
    best_mses, train_sizes = [], []
    for tr, te in folds:
        _, mse = fit_fold(tr, te, n_estimators=N_EST_BEST)
        best_mses.append(mse)
        train_sizes.append(len(tr))

    # ── Permutation importance (month-12 fold, raw feature permutation) ───────
    # XGBoost receives raw features — no OHE grouping needed (contrast with
    # KNN/LR Option A). Each of the 12 feature columns is permuted directly.
    # 5 repeats per feature for a stable importance estimate.
    #
    # Note: XGBoost also has built-in feature importance (gain, weight,
    # cover). Permutation importance is used here for direct comparability
    # with the KNN, RF, and LR spread analyses — all use the same approach.
    tr12, te12 = folds[-1]
    xgb_vi, base_mse_vi = fit_fold(tr12, te12, n_estimators=N_EST_BEST)
    yte = y[te12]
    rng = np.random.default_rng(RANDOM_SEED)
    imp_results = {}
    for j, feat in enumerate(FEATURES):
        deltas = []
        for _ in range(N_PERM_REPEATS):
            Xp       = X[te12].copy().astype(float)
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(
                mean_squared_error(yte, xgb_vi.predict(Xp)) - base_mse_vi
            )
        imp_results[feat] = (np.mean(deltas), np.std(deltas))

    # ── n_estimators sensitivity sweep (month-12 fold) ────────────────────────
    # Sweeps boosting rounds from 10 to 100, holding learning_rate and
    # max_depth fixed. Directly comparable to the RF n_estimators sweep —
    # more rounds = stronger ensemble, but with boosting the rate of
    # improvement differs from bagging due to sequential error correction.
    n_est_mses = {}
    for n in N_EST_SWEEP:
        _, mse        = fit_fold(tr12, te12, n_estimators=n)
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
st.title("🚀 XGBoost Model Analysis")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 1 — METHODOLOGY
# =============================================================================
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
<strong>XGBoost (Extreme Gradient Boosting)</strong> builds trees sequentially rather than
independently. Each new tree is fitted to the <em>residuals</em> of the current ensemble —
the errors that the previous trees have not yet explained. This sequential error-correction
is fundamentally different from Random Forest's bagging approach, where trees are built
independently on bootstrap samples and averaged.
<br><br>
For spread prediction, this sequential structure is particularly motivated: early boosting
rounds capture the "easy" spread patterns — hour-of-day effects, day-of-week demand cycles,
and seasonal load patterns. Later rounds then correct the residuals, focusing on the harder
congestion-driven and weather-driven deviations that early trees missed. This staged
refinement mirrors how a skilled analyst might approach the spread prediction problem.
<br><br>
<strong>No one-hot encoding or scaling is applied</strong> — XGBoost is tree-based and
scale-invariant, identical reasoning to Random Forest. Raw integers and floats are passed
directly. This contrasts with the KNN and Linear Regression spread analyses, which require
62 encoded features to handle categorical variables correctly.
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
    st.markdown("#### Feature set (12 raw features)")
    st.markdown("""
<div class="formula-block">
DAM Zonal Losses<br>
DAM Zonal Congestion<br>
DAM Forecast Load<br>
month, hour, day_of_week<br>
Dew Point, Temperature<br>
Solar Zenith Angle<br>
Relative Humidity, Wind Speed<br>
zone_name  (integer 0-10)
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("#### XGBoost settings")
    st.markdown("""
<div class="formula-block">
n_estimators  = 100  (boosting rounds)<br>
learning_rate = 0.1  (shrinkage)<br>
max_depth     = 6    (shallower than RF)<br>
random_state  = 42<br>
n_jobs        = -1   (all CPU cores)<br><br>
No StandardScaler applied.<br>
No one-hot encoding applied.<br>
Full dataset — no subsampling.
</div>
""", unsafe_allow_html=True)
    st.markdown("""
**learning_rate=0.1** (also called shrinkage) multiplies each tree's
contribution by 0.1 before adding it to the ensemble. This slows the
learning process, requiring more rounds to converge, but significantly
reduces overfitting — each individual tree has less influence over the
final prediction. This is XGBoost's primary regularisation mechanism
and is analogous to how max_depth limits RF tree complexity.

**max_depth=6** is XGBoost's default and is shallower than the RF
setting of 10. In boosting, shallow trees ("stumps" to medium-depth
trees) are preferred — the ensemble gains depth iteratively across
rounds rather than within a single tree.
""")

st.markdown("#### Boosting vs Bagging — key difference from Random Forest")
st.markdown("""
<div class="method-block">
<ul style="margin:0; padding-left:1.2rem;">
  <li><strong>Random Forest (bagging):</strong> Each tree is trained independently on a random
      bootstrap sample. Trees are decorrelated by random feature selection at each split.
      Final prediction = average of all trees. Variance is reduced by averaging; each tree
      has high variance but low bias.</li>
  <li><strong>XGBoost (boosting):</strong> Trees are built sequentially. Tree t+1 is fitted
      to the negative gradient of the loss (residuals for MSE) from trees 1..t.
      Final prediction = sum of all trees × learning_rate. Bias is reduced iteratively;
      each tree has low variance but targets a specific residual pattern.</li>
  <li><strong>Implication for spread prediction:</strong> XGBoost's sequential structure means
      it can capture finer-grained residual patterns than RF at the same number of trees.
      However, it is more sensitive to overfitting training-period spread dynamics that do not
      generalise across seasons — a known weakness on the chronological rolling CV we use.</li>
  <li><strong>Built-in vs permutation importance:</strong> XGBoost provides its own feature
      importance metrics (gain, weight, cover). We use permutation importance here for direct
      comparability across all four model pages — all use the same methodology.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("#### Workflow overview")
steps = [
    ("Step 1", "Load & preprocess — encode zone_name to integer, construct spread target, derive day_of_week, drop day of month."),
    ("Step 2", "Full dataset retained — no subsampling. XGBoost prediction is O(n_trees × depth), independent of training set size."),
    ("Step 3", "Rolling Cross Validation — 9 expanding-window folds, test months 4–12. Same fold structure as KNN, LR, and RF analyses."),
    ("Step 4", "Baseline XGBoost (50 rounds) vs best-config (100 rounds) — RMSE per fold."),
    ("Step 5", "Permutation importance — 12 raw features permuted directly. No OHE grouping needed."),
    ("Step 6", "n_estimators sensitivity sweep — MSE vs boosting rounds on month-12 fold (10 to 100)."),
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
    f"Baseline XGBoost ({N_EST_BASELINE} boosting rounds) vs best-config "
    f"({N_EST_BEST} rounds) across all 9 rolling folds. Both use "
    f"learning_rate={LEARNING_RATE} and max_depth={MAX_DEPTH}. "
    "The expanding training window means each fold adds one more month of "
    "history — the model trained on fold 9 (months 1–11) is the closest "
    "to what is deployed in the Streamlit prediction tool."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Mean RMSE ({N_EST_BEST} rounds)",      f"{np.sqrt(np.mean(best_mses)):.3f} $/MWh")
m2.metric(f"Mean RMSE ({N_EST_BASELINE} rounds)",   f"{np.sqrt(np.mean(base_mses)):.3f} $/MWh")
m3.metric("Best fold RMSE",                          f"{np.sqrt(min(best_mses)):.3f} $/MWh")
m4.metric("Worst fold RMSE",                         f"{np.sqrt(max(best_mses)):.3f} $/MWh")

fig_fit, ax_fit = plt.subplots(figsize=(11, 4.6))
fig_fit.patch.set_facecolor(BG)
w_ = 0.34
ax_fit.bar(x_ - w_/2, [np.sqrt(m) for m in base_mses], width=w_,
           color=GREY,
           label=f"Baseline: {N_EST_BASELINE} boosting rounds",
           edgecolor="white")
ax_fit.bar(x_ + w_/2, [np.sqrt(m) for m in best_mses], width=w_,
           color=ORANGE,
           label=f"Best: {N_EST_BEST} rounds, lr={LEARNING_RATE}, depth={MAX_DEPTH}",
           edgecolor="white")
ax_fit.set_xticks(x_)
ax_fit.set_xticklabels([f"M{m}" for m in TEST_MONTHS], fontsize=10)
ax_fit.set_ylabel("RMSE  ($/MWh)", fontsize=11)
ax_fit.set_title(
    f"RMSE by Test Month — Baseline ({N_EST_BASELINE} rounds) vs "
    f"Best Config ({N_EST_BEST} rounds)\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
ax_fit.legend(fontsize=9)
apply_style(ax_fit)
plt.tight_layout()
st.pyplot(fig_fit, use_container_width=True)
plt.close(fig_fit)

pct = (100 * (np.sqrt(np.mean(base_mses)) - np.sqrt(np.mean(best_mses)))
       / np.sqrt(np.mean(base_mses)))

st.markdown(f"""
<div class="insight-box">
<strong>Model fit:</strong> The best-config XGBoost ({N_EST_BEST} rounds) achieves a mean RMSE of
<strong>{np.sqrt(np.mean(best_mses)):.3f} $/MWh</strong> vs
<strong>{np.sqrt(np.mean(base_mses)):.3f} $/MWh</strong> for the {N_EST_BASELINE}-round baseline
(<strong>{abs(pct):.1f}% {'improvement' if pct > 0 else 'increase'}</strong>).
<br><br>
<strong>Seasonal pattern:</strong> The RMSE profile across test months reflects the difficulty
of each month's spread rather than the raw price level. Months with large congestion events,
unexpected demand surges, or weather extremes produce high spread RMSE — because the day-ahead
market's forecast was systematically wrong in ways that the model, trained on previous months,
could not anticipate.
<br><br>
<strong>Boosting and overfitting risk:</strong> XGBoost's sequential residual-correction
mechanism is powerful within the training period but carries a higher overfitting risk than RF
on chronological splits. When training on spring/summer months and testing on autumn/winter,
the residual patterns learned from summer congestion events may not generalise to winter
demand-driven spread behaviour. This can make XGBoost more sensitive to regime shifts than
RF — a potential explanation if XGBoost's RMSE profile is less stable across folds than RF's.
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 3 — PLOT 2: Permutation importance
# =============================================================================
st.markdown("## Permutation Variable Importance")
st.markdown(
    "Each of the 12 raw features is shuffled in the month-12 test set and "
    "the resulting increase in MSE (ΔMSE) is recorded. Because XGBoost — "
    "uses raw features without one-hot encoding, each "
    "feature column is permuted **directly** with no grouping required. "
    "ΔMSE > 1 (red) = dominant; ΔMSE > 0 (orange) = positive; negative (grey) = noisy/redundant."
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
    "Permutation Variable Importance  (raw features, direct permutation)",
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
  <li><strong>DAM Zonal Congestion</strong> is expected to be particularly important here.
      Congestion is the primary driver of zone-level real-time price deviations from the
      day-ahead system energy price in NYISO — it is the mechanism by which zonal real-time
      prices diverge from what the day-ahead market forecast. XGBoost's sequential boosting
      may be especially effective at capturing the nonlinear threshold behaviour of congestion:
      below a certain load or temperature threshold, congestion may be near-zero; above it,
      it can spike dramatically.</li>
  <li><strong>Built-in XGBoost importance vs permutation importance:</strong> XGBoost provides
      its own gain-based and weight-based importance metrics. These are not used here — permutation
      importance is applied to all four models for methodological consistency.</li>
  <li>Features with negative ΔMSE indicate that shuffling them does not hurt — or even slightly
      helps — the model. Unlike Linear Regression (where uninformative features distort
      coefficients), XGBoost can partially suppress uninformative features by not splitting on
      them. Negative importances in XGBoost are therefore less concerning than in LR, but still
      identify candidates for removal in a leaner model.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 4 — PLOT 3: n_estimators sensitivity
# =============================================================================
st.markdown("## Sensitivity: MSE vs n_estimators (Boosting Rounds)")
st.markdown(
    f"MSE on the month-12 test fold across n_estimators from "
    f"{min(N_EST_SWEEP)} to {max(N_EST_SWEEP)} boosting rounds, with "
    f"learning_rate={LEARNING_RATE} and max_depth={MAX_DEPTH} held fixed. "
    "Directly comparable to the RF n_estimators sweep — both show how "
    "ensemble size affects spread accuracy, but the mechanism differs: "
    "boosting reduces bias iteratively while bagging reduces variance by averaging."
)

fig_n, ax_n = plt.subplots(figsize=(8, 4))
fig_n.patch.set_facecolor(BG)
ax_n.plot(list(n_est_mses.keys()), list(n_est_mses.values()),
          marker="o", color=ORANGE, lw=2.2, zorder=3)
ax_n.axvline(N_EST_BEST, color=RED, ls="--", lw=1.8,
             label=f"Selected n={N_EST_BEST}")
best_n = min(n_est_mses, key=n_est_mses.get)
ax_n.scatter([best_n], [n_est_mses[best_n]], color=RED, s=90, zorder=5,
             label=f"Best n={best_n} (MSE={n_est_mses[best_n]:.3f})")
ax_n.set_xlabel(f"n_estimators  (boosting rounds, lr={LEARNING_RATE})", fontsize=10)
ax_n.set_ylabel("MSE  (month-12 test fold)", fontsize=10)
ax_n.set_title(
    f"MSE vs Boosting Rounds  (lr={LEARNING_RATE}, max_depth={MAX_DEPTH}, month-12 fold)\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
ax_n.legend(fontsize=9)
apply_style(ax_n)
plt.tight_layout()
st.pyplot(fig_n, use_container_width=True)
plt.close(fig_n)

small_n_mse = n_est_mses[min(N_EST_SWEEP)]
best_mse    = n_est_mses[best_n]
gain        = 100 * (small_n_mse - best_mse) / small_n_mse if small_n_mse != 0 else 0.0

st.markdown(f"""
<div class="insight-box">
<strong>Boosting rounds sensitivity:</strong> MSE falls from
{small_n_mse:.3f} at n={min(N_EST_SWEEP)} rounds to {best_mse:.3f} at n={best_n} rounds
({gain:.1f}% improvement).
<br><br>
<strong>How boosting converges differently from bagging:</strong> In Random Forest, adding
more trees always reduces variance (the average of more independent estimates is more stable),
so the MSE curve falls monotonically. In XGBoost, additional rounds reduce bias by correcting
smaller and smaller residuals — but with a risk: if the model overfits the training residuals
in later rounds, test MSE can actually rise after a minimum. This is called <em>over-boosting</em>
and is the reason early stopping (not used here for simplicity) is common in production XGBoost
deployments.
<br><br>
<strong>Interaction with learning_rate:</strong> The shape of this curve is specific to
learning_rate={LEARNING_RATE}. A lower learning_rate (e.g. 0.01) would require more rounds to
converge but typically achieves a lower final MSE. A higher learning_rate (e.g. 0.3) converges
faster but may overfit earlier. The selected learning_rate={LEARNING_RATE} with n={N_EST_BEST}
rounds is a standard, well-tested combination that balances convergence speed and generalisation.
<br><br>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)