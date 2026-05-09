import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# Page config must be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NYISOPrice | KNN Model",
    page_icon="🧭",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Paths
# Recommended structure:
#   Home.py
#   pages/3_🧭_Model-KNN.py
#   cleaned_nyiso_electricity_weather.csv
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

# Streamlit/data/data.csv
DATA_PATH = PROJECT_DIR / "data" / "cleaned_nyiso_electricity_weather.csv"

FEATURES = [
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
    "zone_name",
]
TARGET = "int_LBMP"
TEST_MONTHS = list(range(4, 13))

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1.05rem; color: #5f6368; margin-bottom: 1.5rem;}
    .section-card {
        padding: 1.2rem 1.4rem; border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: rgba(250, 250, 250, 0.70); margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem; border-radius: 16px;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: rgba(248, 249, 250, 0.85); text-align: center; min-height: 105px;
    }
    .metric-number {font-size: 1.4rem; font-weight: 800; margin-bottom: 0.2rem;}
    .metric-label {font-size: 0.9rem; color: #5f6368;}
    .small-note {font-size: 0.92rem; color: #5f6368;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached data and model calculations
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_prepare_data(data_path: str, sample_fraction: float = 1 / 3, random_seed: int = 42):
    df = pd.read_csv(data_path)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    le = LabelEncoder()
    df["zone_name"] = le.fit_transform(df["zone_name"].astype(str))
    df = df[FEATURES + [TARGET]].dropna()
    df = df.sort_values(["month", "day", "hour", "zone_name"]).reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    sample_size = int(len(df) * sample_fraction)
    idx = np.sort(rng.choice(len(df), size=sample_size, replace=False))

    X = df[FEATURES].values[idx]
    y = df[TARGET].values[idx]
    months = df["month"].values[idx]
    return X, y, months, len(df), len(idx)


def fit_eval(X, y, tr, te, k=10, metric="minkowski", weights="uniform"):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    knn = KNeighborsRegressor(
        n_neighbors=k,
        metric=metric,
        weights=weights,
        n_jobs=-1,
    )
    knn.fit(Xtr, y[tr])
    pred = knn.predict(Xte)
    return knn, scaler, mean_squared_error(y[te], pred)


@st.cache_data(show_spinner=True)
def run_knn_analysis(data_path: str, sample_fraction: float, random_seed: int, repeats: int):
    X, y, months, full_n, sample_n = load_and_prepare_data(data_path, sample_fraction, random_seed)

    folds = [
        (np.where(months < tm)[0], np.where(months == tm)[0])
        for tm in TEST_MONTHS
    ]
    folds = [(tr, te) for tr, te in folds if len(tr) > 0 and len(te) > 0]
    test_months = [tm for tm in TEST_MONTHS if np.sum(months == tm) > 0 and np.sum(months < tm) > 0]

    base_mses = [fit_eval(X, y, tr, te, k=10, metric="minkowski", weights="uniform")[2] for tr, te in folds]

    def cv_mean_mse(k, metric="minkowski", weights="uniform"):
        return np.mean([fit_eval(X, y, tr, te, k, metric, weights)[2] for tr, te in folds])

    k_vals = [3, 5, 7, 10, 15, 20, 30, 40, 50]
    k_mean = {k: cv_mean_mse(k) for k in k_vals}
    best_k = min(k_mean, key=k_mean.get)

    tr_last, te_last = folds[-1]
    metrics = ["minkowski", "manhattan", "chebyshev"]
    metric_mse = {m: fit_eval(X, y, tr_last, te_last, k=best_k, metric=m)[2] for m in metrics}
    best_metric = min(metric_mse, key=metric_mse.get)

    weight_mse = {
        w: fit_eval(X, y, tr_last, te_last, k=best_k, metric=best_metric, weights=w)[2]
        for w in ["uniform", "distance"]
    }
    best_weight = min(weight_mse, key=weight_mse.get)

    best_mses = [
        fit_eval(X, y, tr, te, k=best_k, metric=best_metric, weights=best_weight)[2]
        for tr, te in folds
    ]
    train_sizes = [len(tr) for tr, _ in folds]

    # Permutation importance on the last fold using tuned model.
    knn_vi, scaler_vi, base_mse_vi = fit_eval(
        X, y, tr_last, te_last, k=best_k, metric=best_metric, weights=best_weight
    )
    Xte_s = scaler_vi.transform(X[te_last])
    yte = y[te_last]
    rng = np.random.default_rng(random_seed)
    imp_results = {}
    for j, feat in enumerate(FEATURES):
        deltas = []
        for _ in range(repeats):
            Xp = Xte_s.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(mean_squared_error(yte, knn_vi.predict(Xp)) - base_mse_vi)
        imp_results[feat] = (float(np.mean(deltas)), float(np.std(deltas)))

    return {
        "full_n": full_n,
        "sample_n": sample_n,
        "test_months": test_months,
        "base_mses": base_mses,
        "best_mses": best_mses,
        "train_sizes": train_sizes,
        "k_vals": k_vals,
        "k_mean": k_mean,
        "best_k": best_k,
        "metric_mse": metric_mse,
        "best_metric": best_metric,
        "weight_mse": weight_mse,
        "best_weight": best_weight,
        "imp_results": imp_results,
    }


def plot_feature_importance(imp_results):
    feat_arr = list(imp_results.keys())
    imp_arr = np.array([imp_results[f][0] for f in feat_arr])
    std_arr = np.array([imp_results[f][1] for f in feat_arr])
    order = np.argsort(imp_arr)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [feat_arr[i] for i in order],
        imp_arr[order],
        xerr=std_arr[order],
        capsize=3,
        edgecolor="white",
    )
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Mean ΔMSE when feature is permuted")
    ax.set_title("Permutation Feature Importance\nHigher values mean stronger predictive contribution")
    fig.tight_layout()
    return fig


def plot_k_sensitivity(k_vals, k_mean, best_k):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_vals, [k_mean[k] for k in k_vals], marker="o", linewidth=2)
    ax.axvline(best_k, linestyle="--", label=f"Selected k = {best_k}")
    ax.set_xlabel("k: number of neighbours")
    ax.set_ylabel("Mean MSE")
    ax.set_title("MSE Sensitivity to k")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rmse_by_fold(test_months, base_mses, best_mses, best_k, best_metric, best_weight):
    base_rmse = [np.sqrt(m) for m in base_mses]
    best_rmse = [np.sqrt(m) for m in best_mses]
    x = np.arange(len(test_months))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, base_rmse, width, label="Baseline: k=10, Euclidean, uniform", edgecolor="white")
    ax.bar(x + width / 2, best_rmse, width, label=f"Tuned: k={best_k}, {best_metric}, {best_weight}", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{m}" for m in test_months])
    ax.set_ylabel("RMSE ($/MWh)")
    ax.set_title("Rolling Cross-Validation: RMSE by Test Month")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rolling_evolution(test_months, base_mses, best_mses, train_sizes, best_k, best_metric, best_weight):
    x = np.arange(len(test_months))
    base_rmse = [np.sqrt(m) for m in base_mses]
    best_rmse = [np.sqrt(m) for m in best_mses]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"hspace": 0.08})

    ax_top = axes[0]
    ax_top.plot(x, base_rmse, marker="o", linewidth=2, label="Baseline: k=10, Euclidean, uniform")
    ax_top.plot(x, best_rmse, marker="s", linewidth=2, label=f"Tuned: k={best_k}, {best_metric}, {best_weight}")
    ax_top.fill_between(x, base_rmse, best_rmse, alpha=0.12, label="Improvement from tuning")
    for i, (b, t) in enumerate(zip(base_rmse, best_rmse)):
        ax_top.annotate(f"{b:.1f}", (x[i], b), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax_top.annotate(f"{t:.1f}", (x[i], t), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax_top.set_ylabel("RMSE ($/MWh)")
    ax_top.set_title("Rolling Window MSE Evolution\nExpanding training window across test months")
    ax_top.legend(fontsize=9, loc="upper right")
    ax_top.grid(True, alpha=0.3)

    ax_bottom = axes[1]
    ax_bottom.bar(x, train_sizes, edgecolor="white")
    for i, s in enumerate(train_sizes):
        ax_bottom.text(i, s + max(train_sizes) * 0.01, f"{s:,}", ha="center", fontsize=8)
    ax_bottom.set_ylabel("Training set size\nsubsampled rows")
    ax_bottom.set_xlabel("Test fold")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([f"Month {m}" for m in test_months], rotation=15, ha="right")
    ax_bottom.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🧭 K-Nearest Neighbors Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">NYISO real-time electricity price prediction using similar historical market, weather, time, and zone conditions.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-number">Target</div><div class="metric-label">Integrated LBMP</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-number">13</div><div class="metric-label">Predictor features</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-number">Rolling CV</div><div class="metric-label">Expanding-window validation</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-number">Built-in</div><div class="metric-label">Figures auto-rendered in page</div></div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Methodology
# -----------------------------------------------------------------------------
st.header("1. Methodology of KNN")
left, right = st.columns([1.2, 1])

with left:
    st.markdown(
        """
        K-nearest neighbors, or **KNN**, is a non-parametric prediction method. Instead of estimating a fixed global equation, it predicts the real-time price by finding historical hours that look most similar to the current hour.

        In this project, the target variable is **integrated LBMP (`int_LBMP`)**, and the predictors include day-ahead market variables, time indicators, weather variables, and zone information. The central assumption is simple: **hours with similar market, weather, time, and zone conditions should have similar real-time electricity prices.**
        """
    )
    st.markdown("**Distance between a new hour and a historical hour**")
    st.latex(r"d(x_0, x_i) = \sqrt{\sum_{j=1}^{p}(x_{0j} - x_{ij})^2}")
    st.markdown("**Standardization before distance calculation**")
    st.latex(r"z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}")
    st.markdown("**KNN prediction as local averaging**")
    st.latex(r"\hat{y}_0 = \frac{1}{k}\sum_{i \in N_k(x_0)} y_i")

with right:
    st.markdown(
        """
        <div class="section-card">
        <b>Why KNN is suitable here</b><br><br>
        Electricity prices are shaped by repeating patterns: demand cycles, weather, congestion, and time of day. KNN directly uses these historical similarities to generate predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Model inputs**")
    st.markdown(
        """
        - DAM Zonal LBMP, losses, congestion, and forecast load
        - Month, day, and hour
        - Dew point, temperature, solar zenith, relative humidity, and wind speed
        - Zone name / zone indicator
        """
    )
    st.info("Because KNN depends entirely on distance, feature scaling is essential.")

st.markdown("---")

# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------
st.header("2. Model Workflow")
cols = st.columns(3)
with cols[0]:
    st.subheader("Step 1–2")
    st.markdown("""**Load & preprocess**: keep selected features, encode `zone_name`, drop missing rows, and sort chronologically.  
**Subsample 1-in-3**: reduce runtime while preserving the broad data structure.""")
with cols[1]:
    st.subheader("Step 3–4")
    st.markdown("""**Rolling CV folds**: use expanding-window validation so each test month is predicted using only earlier months.  
**Baseline KNN**: start with k = 10, Euclidean distance, and uniform weights.""")
with cols[2]:
    st.subheader("Step 5–6")
    st.markdown("""**Permutation importance**: shuffle one feature at a time and measure the increase in MSE.  
**Sensitivity checks**: compare k values, distance metrics, and weighting schemes.""")

st.markdown("---")

# -----------------------------------------------------------------------------
# Built-in analysis run
# -----------------------------------------------------------------------------
st.header("3. KNN Model Results")

# These settings are fixed in the page code, so users do not need to generate
# figures manually. The app computes the figures automatically when the page loads.
SAMPLE_FRACTION = 1 / 3
PERMUTATION_REPEATS = 5
RANDOM_SEED = 42

X, y, months, full_n, sample_n = load_and_prepare_data(str(DATA_PATH))

best_k = results["best_k"]
best_metric = results["best_metric"]
best_weight = results["best_weight"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows after cleaning", f"{results['full_n']:,}")
m2.metric("Rows used by KNN", f"{results['sample_n']:,}")
m3.metric("Selected k", best_k)
m4.metric("Best metric", best_metric)

st.markdown("---")

# -----------------------------------------------------------------------------
# Graph 1: Feature importance
# -----------------------------------------------------------------------------
st.subheader("3.1 Permutation Feature Importance")
st.pyplot(plot_feature_importance(results["imp_results"]), use_container_width=True)

imp_table = (
    pd.DataFrame(
        [(k, v[0], v[1]) for k, v in results["imp_results"].items()],
        columns=["Feature", "Mean ΔMSE", "Std. dev."],
    )
    .sort_values("Mean ΔMSE", ascending=False)
    .reset_index(drop=True)
)
st.dataframe(imp_table, use_container_width=True, hide_index=True)

top_feature = imp_table.iloc[0]["Feature"]
top_delta = imp_table.iloc[0]["Mean ΔMSE"]
negative_features = imp_table.loc[imp_table["Mean ΔMSE"] < 0, "Feature"].tolist()

st.markdown("### Interpretation")
st.markdown(
    f"""
    The permutation importance results show that **{top_feature}** is the dominant predictor. Its ΔMSE is about **{top_delta:.1f}**, meaning that once this variable is shuffled, predictive performance deteriorates sharply. This is expected because day-ahead market information contains strong signals about real-time prices.

    Features with negative importance{': **' + ', '.join(negative_features) + '**' if negative_features else ''} appear redundant or noisy in the KNN distance calculation. In KNN, a feature can hurt prediction if it adds noise to the distance metric and makes the algorithm select less relevant neighbors. Therefore, negative ΔMSE is a useful signal that these variables could be removed in a refined version of the model.
    """
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Graph 2: Hyperparameter sensitivity
# -----------------------------------------------------------------------------
st.subheader("3.2 MSE Sensitivity to k")
st.pyplot(plot_k_sensitivity(results["k_vals"], results["k_mean"], best_k), use_container_width=True)

k_table = pd.DataFrame(
    {"k": results["k_vals"], "Mean MSE": [results["k_mean"][k] for k in results["k_vals"]]}
)
st.dataframe(k_table, use_container_width=True, hide_index=True)

first_k = results["k_vals"][0]
last_k = results["k_vals"][-1]
st.markdown("### Interpretation")
st.markdown(
    f"""
    The k-sensitivity plot shows how prediction error changes as KNN uses more neighbors. A smaller k, such as **k = {first_k}**, produces a more local but noisier prediction. A larger k, such as **k = {last_k}**, smooths over more neighboring hours.

    In this run, the selected value is **k = {best_k}**. If the curve continues falling near the largest tested k, it would be reasonable to extend the search to even larger values in a production version.
    """
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Graph 3: Baseline vs tuned RMSE
# -----------------------------------------------------------------------------
st.subheader("3.3 Rolling CV: Baseline vs Tuned Model")
st.pyplot(
    plot_rmse_by_fold(
        results["test_months"], results["base_mses"], results["best_mses"], best_k, best_metric, best_weight
    ),
    use_container_width=True,
)

rmse_table = pd.DataFrame(
    {
        "Test month": results["test_months"],
        "Baseline RMSE": [np.sqrt(m) for m in results["base_mses"]],
        "Tuned RMSE": [np.sqrt(m) for m in results["best_mses"]],
        "Training rows": results["train_sizes"],
    }
)
st.dataframe(rmse_table, use_container_width=True, hide_index=True)

st.markdown("### Interpretation")
st.markdown(
    f"""
    The tuned model uses **k = {best_k}**, **{best_metric} distance**, and **{best_weight} weights**. The bar chart compares this tuned version with the baseline model across rolling test months.

    The key point is that the tuned model usually improves on the baseline, but the improvement varies by month. This means model performance is shaped not only by hyperparameters, but also by the difficulty of the test period itself.
    """
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Graph 4: Rolling evolution
# -----------------------------------------------------------------------------
st.subheader("3.4 Rolling Window MSE Evolution")
st.pyplot(
    plot_rolling_evolution(
        results["test_months"], results["base_mses"], results["best_mses"], results["train_sizes"], best_k, best_metric, best_weight
    ),
    use_container_width=True,
)

st.markdown("### Top panel — RMSE evolution")
st.markdown(
    """
    The most important pattern is whether RMSE falls smoothly as the training set grows. If RMSE does **not** fall monotonically, then prediction error is driven more by **seasonal difficulty and market volatility** than by the amount of training data alone.

    Price spikes in high-demand or high-volatility months can remain difficult for KNN, even when more historical rows are available. This is because KNN is a local averaging method and can smooth away rare extreme events.
    """
)

st.markdown("### Bottom panel — training set growth")
st.markdown(
    """
    The bottom panel confirms that the expanding-window validation is working as intended: each fold adds more historical data to the training set.

    However, more training data does not automatically guarantee lower RMSE. If a later month has a different seasonal regime or more price volatility, the model can still perform worse than in earlier months.
    """
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Final model summary
# -----------------------------------------------------------------------------
st.header("4. Key Takeaways")
summary_left, summary_right = st.columns(2)
with summary_left:
    st.success("Day-ahead market variables, especially DAM Zonal LBMP, are central to real-time price prediction.")
    st.success(f"The tuned KNN model selected k = {best_k}, {best_metric} distance, and {best_weight} weights.")
with summary_right:
    st.warning("Features with negative permutation importance may add noise to the KNN distance metric.")
    st.warning("Seasonal regime shifts can matter more than sample size for KNN performance.")

st.markdown(
    """
    <p class="small-note">
    Suggested refinement: rerun the KNN model after dropping features with negative permutation importance, then compare the new rolling-window RMSE with the current tuned model.
    </p>
    """,
    unsafe_allow_html=True,
)
