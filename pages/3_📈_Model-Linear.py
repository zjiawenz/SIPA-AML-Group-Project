#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3___Model-Linear-Spread.py
=============================================================================
Linear Regression — Spread Analysis  (NYISO 2018)
Target: spread = int_LBMP - DAM Zonal LBMP

Streamlit page. Drop into the pages/ folder of your multi-page app.

Three analysis sections:
  1. Rolling CV — RMSE per fold (baseline vs full model)
  2. Permutation variable importance (Option A — original features)
  3. Ridge alpha sensitivity (LR equivalent of k-sweep / n_estimators sweep)

Key methodological points
--------------------------
  Target  : spread = int_LBMP - DAM Zonal LBMP ($/MWh)
  Features: 12 original (DAM Zonal LBMP excluded — part of target)
            day_of_week added; day of month dropped
  Encoding: month, hour, day_of_week, zone_name → one-hot, drop_first=True
            drop_first=True removes one reference category per group,
            preventing perfect multicollinearity with the intercept column.
            Without it, OLS is numerically unstable (MSE blows up at low
            alpha and collapses to near-zero at high alpha).
            Continuous features (8) → StandardScaler (fit on train only)
            Binary OHE columns → NOT scaled
            Total: 8 continuous + 50 OHE = 58 encoded features
  Baseline: DAM market variables only (Losses, Congestion, Forecast Load)
  Folds   : 9 expanding-window, test months 4-12
  Importance: Option A — for each original categorical feature, permute
              its raw integer values in the test set and re-encode to
              dummies on the fly. This correctly handles drop_first=True
              without needing to track shifted column indices.

Why drop_first=True for LR but not KNN
---------------------------------------
  KNN uses drop_first=False (all categories kept) because dropping one
  category creates an asymmetric distance structure — the dropped category
  maps to all-zeros, which is equidistant from all other categories, not
  properly represented. For KNN, keeping all dummies is correct.

  Linear Regression uses drop_first=True because OLS estimates coefficients
  by inverting X'X. When all category dummies sum to 1 (= the intercept),
  X'X is singular — no unique solution exists. Dropping one reference
  category per group breaks the collinearity and makes OLS well-defined.
  With 62 features and drop_first=False, OLS produces wildly unstable
  coefficients, causing MSE to blow up at low alpha and collapse to
  near-zero (predicting the intercept only) at high alpha in the Ridge sweep.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Linear Regression",
    page_icon="📈",
    layout="wide",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
  .method-block {
    background: #fff5f5;
    border-left: 4px solid #c62828;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
  }
  .insight-box {
    background: #fff5f5;
    border-left: 4px solid #c62828;
    border-radius: 4px;
    padding: 0.9rem 1.3rem;
    margin-top: 0.8rem;
    margin-bottom: 1.6rem;
    font-size: 0.92rem;
    line-height: 1.65;
  }
  .insight-box strong { color: #8e0000; }
  .formula-block {
    background: #fff0f0;
    border: 1px solid #ef9a9a;
    border-radius: 6px;
    padding: 0.7rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #8e0000;
    margin: 0.6rem 0 1rem 0;
  }
  .tag {
    display: inline-block;
    background: #c62828;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 1px 8px;
    border-radius: 20px;
    margin-right: 5px;
    vertical-align: middle;
  }
  hr { border: none; border-top: 1px solid #f0caca; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Colour palette
# =============================================================================
RED      = "#c62828"
DARK_RED = "#8e0000"
ORANGE   = "#f4a234"
GREY     = "#aaaaaa"
GREEN    = "#34a853"
BG       = "#fff5f5"
BLUE     = RED
PURPLE   = DARK_RED


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

RAW_COLS = [
    "DAM Zonal LBMP",       # for target construction only — not a feature
    "DAM Zonal Losses",
    "DAM Zonal Congestion",
    "DAM Forecast Load",
    "month",
    "day",                  # used to derive day_of_week, then dropped
    "hour",
    "Dew Point",
    "Temperature",
    "Solar Zenith Angle",
    "Relative Humidity",
    "Wind Speed",
    "zone_name",
    "int_LBMP",
]

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

# Categorical features — one-hot encoded with drop_first=True for LR
OHE_COLS = ["month", "hour", "day_of_week", "zone_name"]

# All 12 original features in display order
ORIGINAL_FEATURES = CONTINUOUS_COLS + OHE_COLS

# Baseline: DAM market variables only (all continuous — no OHE needed)
BASELINE_FEATURES = ["DAM Zonal Losses", "DAM Zonal Congestion", "DAM Forecast Load"]

# Ridge alpha sweep values
ALPHA_VALS = [0.0, 0.1, 1, 10, 100, 1000]

DATA_PATH = "data/cleaned_nyiso_electricity_weather.csv"

# =============================================================================
# Helper: build encoded matrix with drop_first=True
# =============================================================================
def build_encoded_matrix(df_input, ohe_structure):
    """
    Build the encoded feature matrix for Linear Regression.

    drop_first=True: removes one reference category per OHE group.
    This prevents perfect multicollinearity with the intercept — without
    it, OLS has no unique solution and produces numerically unstable
    coefficients.

    Column order: [continuous cols | OHE cols (drop_first)]
    Total: 8 continuous + 50 OHE (12-1 + 24-1 + 7-1 + 11-1) = 58 features.

    Parameters
    ----------
    df_input     : DataFrame with continuous and OHE columns present
    ohe_structure: dict {col: sorted list of all unique values from training}
                   Used to ensure consistent column order across folds.

    Returns
    -------
    X_enc        : ndarray, shape (n, 58)
    encoded_cols : list of column names in the encoded matrix
    """
    df_enc = df_input[CONTINUOUS_COLS].copy()
    for col in OHE_COLS:
        # Use pd.Categorical with fixed categories from training data
        # to ensure consistent column set and order across all folds.
        cat = pd.Categorical(df_input[col], categories=ohe_structure[col])
        dummies = pd.get_dummies(cat, prefix=col, drop_first=True).astype(int)
        df_enc  = pd.concat([df_enc, dummies], axis=1)
    return df_enc.values.astype(float), list(df_enc.columns)


# =============================================================================
# Data loading (cached)
# =============================================================================
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data():
    """
    Load and preprocess the NYISO 2018 dataset.

    Steps:
      1. Load CSV, label-encode zone_name to integer 0-10.
      2. Construct spread target: int_LBMP - DAM Zonal LBMP.
      3. Derive day_of_week from year/month/day; drop day of month.
      4. Record ohe_structure: sorted unique values per OHE column,
         fixed from the full dataset so all folds use consistent categories.
      5. Build encoded matrix using drop_first=True.
      6. Return raw dataframe slice (for per-fold re-encoding in importance),
         encoded matrix, target, month index, and OHE structure.
    """
    df = pd.read_csv(DATA_PATH)

    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"])
    df = df[RAW_COLS].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)

    # Spread target
    df["spread"] = df["int_LBMP"] - df["DAM Zonal LBMP"]

    # Derive day_of_week; drop day of month
    df["date"]        = pd.to_datetime({"year": DATASET_YEAR,
                                        "month": df["month"],
                                        "day":   df["day"]})
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday ... 6=Sunday
    df.drop(columns=["date", "day"], inplace=True)

    # OHE structure: sorted unique values per categorical column.
    # Fixed from the full dataset — consistent across all folds.
    ohe_structure = {col: sorted(df[col].unique().tolist()) for col in OHE_COLS}

    # Build encoded matrix (drop_first=True, 58 features)
    X_enc, encoded_cols = build_encoded_matrix(df, ohe_structure)

    y      = df["spread"].values
    months = df["month"].values
    n_cont = len(CONTINUOUS_COLS)

    # Also return the dataframe slice needed for per-fold importance re-encoding
    df_features = df[CONTINUOUS_COLS + OHE_COLS].copy()

    return X_enc, y, months, n_cont, encoded_cols, ohe_structure, df_features


# =============================================================================
# Model pipeline (cached)
# =============================================================================
@st.cache_data(show_spinner="Training Linear Regression models across 9 folds…")
def run_analysis(_X_enc, _y, _months, _n_cont, _ohe_structure, _df_features):
    """
    Rolling CV, permutation importance (Option A), and Ridge sensitivity.

    Permutation importance uses raw-value re-encoding:
      For each original categorical feature, shuffle its raw integer values
      in the test rows, then re-encode to dummies using build_encoded_matrix.
      This correctly handles drop_first=True without needing to track
      shifted column indices after dropping reference categories.
      For continuous features, permute the encoded column directly (no OHE
      is involved so there is no column-index ambiguity).
    """
    X_enc      = _X_enc
    y          = _y
    months     = _months
    n_cont     = _n_cont
    ohe_structure = _ohe_structure
    df_features   = _df_features

    TEST_MONTHS = list(range(4, 13))
    folds = [(np.where(months < tm)[0], np.where(months == tm)[0])
             for tm in TEST_MONTHS]

    def compute_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {"MSE": mse, "RMSE": np.sqrt(mse),
                "MAE": mean_absolute_error(y_true, y_pred),
                "EVS": explained_variance_score(y_true, y_pred),
                "R2":  r2_score(y_true, y_pred)}

    # ── Helper: fit LR/Ridge on one fold ─────────────────────────────────────
    def fit_fold_encoded(tr, te, model_type="ols", alpha=1.0):
        """
        Fit on encoded matrix. Scale continuous cols only (train fit).
        OHE binary cols are not scaled — already on [0,1].
        """
        X_tr = X_enc[tr].copy()
        X_te = X_enc[te].copy()
        sc   = StandardScaler()
        X_tr[:, :n_cont] = sc.fit_transform(X_tr[:, :n_cont])
        X_te[:, :n_cont] = sc.transform(X_te[:, :n_cont])
        model = Ridge(alpha=alpha, random_state=42) if model_type == "ridge" \
                else LinearRegression()
        model.fit(X_tr, y[tr])
        pred = model.predict(X_te)
        return model, sc, compute_metrics(y[te], pred), pred, X_te

    # ── Helper: fit baseline (continuous DAM vars only, no OHE) ──────────────
    def fit_fold_baseline(tr, te):
        """
        Fit LR on 3 continuous DAM features only — no encoding needed.
        All three are continuous so StandardScaler applied to all columns.
        """
        X_base = df_features[list(BASELINE_FEATURES)].values
        sc     = StandardScaler()
        Xtr    = sc.fit_transform(X_base[tr])
        Xte    = sc.transform(X_base[te])
        model  = LinearRegression()
        model.fit(Xtr, y[tr])
        return compute_metrics(y[te], model.predict(Xte))

    # ── Baseline rolling CV ───────────────────────────────────────────────────
    base_metrics = [fit_fold_baseline(tr, te) for tr, te in folds]

    # ── Full OLS rolling CV ───────────────────────────────────────────────────
    full_metrics, train_sizes = [], []
    for tr, te in folds:
        _, _, m, _, _ = fit_fold_encoded(tr, te)
        full_metrics.append(m)
        train_sizes.append(len(tr))

    # ── Permutation importance — Option A with raw-value re-encoding ──────────
    # Month-12 fold: largest training set → most stable model.
    #
    # For CATEGORICAL features (OHE):
    #   1. Copy the test rows of df_features (the raw integer values).
    #   2. Permute the raw integer column for the feature being assessed.
    #   3. Re-encode the entire test row block using build_encoded_matrix
    #      with drop_first=True — this produces correct dummies with
    #      consistent column order, regardless of which categories appear
    #      after permutation.
    #   4. Scale continuous cols using the training scaler, predict, record MSE.
    #
    # For CONTINUOUS features:
    #   Permute the corresponding encoded column directly (no re-encoding
    #   needed — there are no dummies to reconstruct).
    #
    # This approach is correct for drop_first=True because it never assumes
    # the reference category is absent — it re-builds the full dummy set
    # each time from the raw values, letting pd.get_dummies + pd.Categorical
    # handle the reference drop consistently.
    tr12, te12 = folds[-1]
    lr_vi, sc_vi, base_m_vi, _, Xte_s = fit_fold_encoded(tr12, te12)
    base_mse_vi = base_m_vi["MSE"]
    yte         = y[te12]

    # Test-set raw feature values (needed for categorical re-encoding)
    df_te = df_features.iloc[te12].copy().reset_index(drop=True)

    rng = np.random.default_rng(42)
    imp_results = {}

    for feat in ORIGINAL_FEATURES:
        deltas = []
        for _ in range(5):

            if feat in CONTINUOUS_COLS:
                # Continuous: permute the encoded column directly
                col_idx = CONTINUOUS_COLS.index(feat)
                Xp      = Xte_s.copy()
                Xp[:, col_idx] = rng.permutation(Xp[:, col_idx])
                perm_mse = mean_squared_error(yte, lr_vi.predict(Xp))

            else:
                # Categorical: permute raw values, re-encode, re-scale
                df_perm = df_te.copy()
                df_perm[feat] = rng.permutation(df_perm[feat].values)

                # Re-encode with drop_first=True using fixed OHE structure
                Xp_raw, _ = build_encoded_matrix(df_perm, ohe_structure)

                # Apply training scaler to continuous columns only
                Xp_raw[:, :n_cont] = sc_vi.transform(Xp_raw[:, :n_cont])
                perm_mse = mean_squared_error(yte, lr_vi.predict(Xp_raw))

            deltas.append(perm_mse - base_mse_vi)
        imp_results[feat] = (np.mean(deltas), np.std(deltas))

    # ── Ridge sensitivity sweep (month-12 fold) ───────────────────────────────
    ridge_mses = {}
    for a in ALPHA_VALS:
        mtype = "ols" if a == 0.0 else "ridge"
        _, _, m, _, _ = fit_fold_encoded(tr12, te12, model_type=mtype, alpha=a)
        ridge_mses[a] = m["MSE"]

    return dict(
        TEST_MONTHS  = TEST_MONTHS,
        base_metrics = base_metrics,
        full_metrics = full_metrics,
        train_sizes  = train_sizes,
        imp_results  = imp_results,
        ridge_mses   = ridge_mses,
        last_metrics = base_m_vi,
    )


# =============================================================================
# HEADER
# =============================================================================
st.title("📈 Linear Regression Model Analysis")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 1 — METHODOLOGY
# =============================================================================
st.markdown("## Methodology")

st.markdown("""
<div class="method-block">
<strong>Linear Regression (OLS)</strong> estimates a global linear relationship between the
predictors and the <strong>spread</strong> (real-time LBMP minus day-ahead LBMP, $/MWh).
The spread represents the deviation of the real-time market from the day-ahead forecast —
the quantity that convergence/virtual bidders trade in NYISO.
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
month       → 11 dummies (12-1)<br>
hour        → 23 dummies (24-1)<br>
day_of_week →  6 dummies  (7-1)<br>
zone_name   → 10 dummies (11-1)<br>
─────────────────────────────────<br>
Total: 8 + 50 = 58 features
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("#### How do we address multicollinearity?")
    st.markdown("""
<div class="formula-block">
Without drop_first:<br>
  Σ(month dummies) = 1 = intercept<br>
  → X'X is singular → OLS breaks<br><br>
With drop_first=True:<br>
  One reference category dropped<br>
  per group (e.g. month_1, hour_0,<br>
  day_of_week_0, zone_0 dropped)<br>
  → X'X invertible → OLS stable<br><br>
KNN uses drop_first=False (all<br>
categories kept for symmetric<br>
distance computation).
</div>
""", unsafe_allow_html=True)
    st.markdown("""
Without `drop_first=True`, the sum of all month dummy columns equals 1 for
every row — identical to the intercept column. This creates perfect
multicollinearity: OLS has no unique solution, producing wildly unstable
coefficients. In practice this causes MSE to blow up at low Ridge alpha
(OLS overfits with extreme coefficients) and collapse to near-zero at high
alpha (heavy shrinkage forces all coefficients to zero, predicting only the
intercept). `drop_first=True` removes this instability by dropping one
reference category per group, making X'X invertible.
""")

st.markdown("#### Workflow overview")
steps = [
    ("Step 1", "Load & preprocess — encode zone_name numerically, construct spread target, derive day_of_week, drop day-of-month. Build encoded matrix — one-hot encode month/hour/day_of_week/zone_name; scale continuous variables using StandardScaler."),
    ("Step 2", "Rolling Cross Validation — 9 expanding-window folds, test months 4–12."),
    ("Step 3", "Baseline Linear Regression — DAM market variables only (Losses, Congestion, Load). No OHE needed — all continuous."),
    ("Step 4", "Full OLS — all 12 original features (58 encoded) — record RMSE per fold."),
    ("Step 5", "Permutation importance — permute each original feature (OHE columns jointly) 5× on month-12 test set; measure ΔMSE."),
    ("Step 6", "Ridge sensitivity — sweep alpha from OLS to heavy shrinkage on month-12 fold."),
]
for tag, text in steps:
    st.markdown(f'<span class="tag">{tag}</span> {text}', unsafe_allow_html=True)
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA & RUN
# =============================================================================
(X_enc, y, months, n_cont,
 encoded_cols, ohe_structure, df_features) = load_data()

res = run_analysis(X_enc, y, months, n_cont, ohe_structure, df_features)

TEST_MONTHS  = res["TEST_MONTHS"]
base_metrics = res["base_metrics"]
full_metrics = res["full_metrics"]
train_sizes  = res["train_sizes"]
imp_results  = res["imp_results"]
ridge_mses   = res["ridge_mses"]
last_metrics = res["last_metrics"]

base_rmses = [m["RMSE"] for m in base_metrics]
full_rmses = [m["RMSE"] for m in full_metrics]
full_r2s   = [m["R2"]   for m in full_metrics]
x_         = np.arange(len(TEST_MONTHS))

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
    "The full linear model (OLS, 58 encoded features) is compared against "
    "a DAM-market-only baseline (Losses, Congestion, Forecast Load — 3 "
    "continuous features, no OHE) across all 9 rolling folds."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean RMSE (full)",     f"{np.mean(full_rmses):.3f} $/MWh")
m2.metric("Mean RMSE (baseline)", f"{np.mean(base_rmses):.3f} $/MWh")
m3.metric("Mean R² (full)",       f"{np.mean(full_r2s):.4f}")
m4.metric("Last-fold RMSE",       f"{last_metrics['RMSE']:.3f} $/MWh")

fig_fit, ax_fit = plt.subplots(figsize=(11, 4.6))
fig_fit.patch.set_facecolor(BG)
w_ = 0.34
ax_fit.bar(x_ - w_/2, base_rmses, width=w_, color=GREY,
           label="Baseline: DAM market only (Losses, Congestion, Load)",
           edgecolor="white")
ax_fit.bar(x_ + w_/2, full_rmses, width=w_, color=BLUE,
           label="Full LR: 12 features (58 encoded, drop_first=True)",
           edgecolor="white")
ax_fit.set_xticks(x_)
ax_fit.set_xticklabels([f"M{m}" for m in TEST_MONTHS], fontsize=10)
ax_fit.set_ylabel("RMSE  ($/MWh)", fontsize=11)
ax_fit.set_title(
    "RMSE by Test Month — DAM-market Baseline vs Full Linear Regression\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
ax_fit.legend(fontsize=9)
apply_style(ax_fit)
plt.tight_layout()
st.pyplot(fig_fit, use_container_width=True)
plt.close(fig_fit)

mean_base = np.mean(base_rmses)
mean_full = np.mean(full_rmses)
pct       = 100 * (mean_base - mean_full) / mean_base if mean_base != 0 else 0.0

st.markdown(f"""
<div class="insight-box">
<strong>Model fit:</strong> The full linear model achieves a mean RMSE of
<strong>{mean_full:.3f} $/MWh</strong> across the 9 rolling folds, compared with
<strong>{mean_base:.3f} $/MWh</strong> for the DAM-market-only baseline
(<strong>{abs(pct):.1f}% RMSE {'reduction' if pct > 0 else 'increase'}</strong>).
<br><br>
The spread target is inherently harder to predict than the raw real-time LBMP. This is because 
the dominant day-ahead price signal has been removed from the target by construction, 
leaving the model to explain <em>why</em> real-time deviated from the day-ahead forecast. 
The remaining predictors must collectively explain these deviations, which are driven by congestion
events, demand surprises, and weather-driven forecast errors.
<br><br>
The seasonal pattern in RMSE across folds reflects which months had the most unpredictable
deviations from day-ahead prices. Months with large congestion events or unexpected demand
surges produce high spread RMSE regardless of the absolute price level.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 3 — PLOT 2: Permutation importance (Option A, re-encoding)
# =============================================================================
st.markdown("## Permutation Variable Importance")
st.markdown(
    "Each of the 12 original features is permuted 5 times in the month-12 test set. "
    "For OHE features (month, hour, day_of_week, zone_name), **all dummy columns are permuted "
    "simultaneously** to preserve the one-hot constraint. "
    "ΔMSE > 1 (red) = dominant; ΔMSE > 0 (orange) = positive; negative (grey) = noisy/redundant."
)

fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
fig_imp.patch.set_facecolor(BG)
bar_colors = [
    RED    if v > 1 else
    ORANGE if v > 0 else
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
  <li>The importance ranking should be compared with the KNN, RF, and XGBoost
      spread pages. Features that rank highly across all four methods carry
      the strongest evidence of genuine predictive value for the spread.</li>
  <li>For Linear Regression, features with negative ΔMSE are more problematic
      than for tree-based models. RF and XGBoost can avoid splitting on
      uninformative features; LR always estimates a coefficient for every
      feature, and multicollinearity from uninformative OHE columns can
      distort adjacent coefficients.</li>
  <li>The raw-value re-encoding approach for permutation importance correctly
      handles <code>drop_first=True</code>: after permuting raw integers, the
      full dummy matrix is rebuilt, and the reference category is dropped
      consistently — the same way it was during training.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# SECTION 4 — PLOT 3: Ridge alpha sensitivity
# =============================================================================
st.markdown("## Regularisation Sensitivity: OLS vs Ridge")
st.markdown(
    "Ridge regression adds an L2 penalty (λ × Σβ²) that shrinks all "
    "coefficients toward zero. This sweep — from OLS (alpha=0) to heavy "
    "shrinkage (alpha=1000) — is the Linear Regression equivalent of the "
    "k-sweep in KNN and the n_estimators sweep in RF and XGBoost. "
    "With `drop_first=True` resolving the multicollinearity, the OLS "
    "solution is now stable and the Ridge curve shows the genuine "
    "regularisation trade-off rather than a numerical artefact."
)

fig_ridge, ax_ridge = plt.subplots(figsize=(8, 4))
fig_ridge.patch.set_facecolor(BG)

labels     = ["OLS" if a == 0 else str(a) for a in ALPHA_VALS]
vals       = [ridge_mses[a] for a in ALPHA_VALS]
best_alpha = min(ridge_mses, key=ridge_mses.get)
best_label = "OLS" if best_alpha == 0 else str(best_alpha)
best_idx   = ALPHA_VALS.index(best_alpha)

ax_ridge.plot(labels, vals, marker="o", color=PURPLE, lw=2.2, zorder=3)
ax_ridge.scatter([labels[best_idx]], [vals[best_idx]], color=RED,
                 s=90, zorder=5, label=f"Best alpha = {best_label}")
ax_ridge.set_xlabel("Ridge alpha  (0 = OLS, higher = more shrinkage)", fontsize=10)
ax_ridge.set_ylabel("MSE  (month-12 test fold)", fontsize=10)
ax_ridge.set_title(
    "MSE vs Ridge Penalty  (month-12 fold, drop_first=True)\n"
    "Target: spread = int_LBMP − DAM Zonal LBMP",
    fontsize=11, fontweight="bold"
)
ax_ridge.legend(fontsize=9)
apply_style(ax_ridge)
plt.tight_layout()
st.pyplot(fig_ridge, use_container_width=True)
plt.close(fig_ridge)

ols_mse    = ridge_mses[0.0]
best_mse   = ridge_mses[best_alpha]
ridge_gain = 100 * (ols_mse - best_mse) / ols_mse if ols_mse != 0 else 0.0

st.markdown(f"""
<div class="insight-box">
<strong>Regularisation check:</strong> The best alpha on the month-12 fold is
<strong>{best_label}</strong> (MSE = {best_mse:.4f}).
OLS achieves MSE = {ols_mse:.4f}
({'Ridge reduces MSE by ' + f'{ridge_gain:.1f}%' if best_alpha != 0.0
  else 'OLS is already optimal on this fold — regularisation does not improve performance'}).
<br><br>
With <code>drop_first=True</code> resolving the perfect multicollinearity, the Ridge curve
now shows the genuine regularisation trade-off: OLS has a well-defined, stable solution,
and the curve shows whether additional shrinkage helps generalisation. A flat curve means
the linear functional form (not overfitting) is the binding constraint — no amount of
regularisation can make a linear model capture threshold-driven spread behaviour. A falling
curve means OLS was still fitting training-period noise, and Ridge improves out-of-sample
generalisation by shrinking those noisy coefficients.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
