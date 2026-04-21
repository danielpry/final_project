from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "modeling_weather_V2.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """
# Launch Modeling Weather V2: Base Model Vs Weather-Enhanced Subset

This notebook tests a different strategy for weather features:

- keep a base launch-history model for the full sample
- define a stricter **good-weather-coverage subset**
- compare a base model and a weather-enhanced model on that same subset

The idea is to stop forcing sparse weather variables into every row and instead ask whether weather helps when the matching quality is actually good.
"""
    ),
    code(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


pd.set_option("display.max_columns", 200)
sns.set_theme(style="whitegrid")

DATA_DIR = Path("data/derived")
RANDOM_STATE = 42
"""
    ),
    code(
        """
df = pd.read_csv(DATA_DIR / "us_launch_weather_merged.csv", low_memory=False)
df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
df = df.sort_values(["launch_date", "Launch Id"]).reset_index(drop=True)

df["launch_month"] = df["launch_date"].dt.month
df["launch_year_centered"] = df["launch_year"] - df["launch_year"].median()
df["mission_mass_log"] = np.log1p(df["mission_mass"].clip(lower=0))
df["rocket_payload_leo_log"] = np.log1p(df["rocket_payload_leo"].clip(lower=0))
df["config_liftoff_thrust_log"] = np.log1p(df["config_liftoff_thrust"].clip(lower=0))
df["dewpoint_depression"] = df["HourlyDryBulbTemperature"] - df["HourlyDewPointTemperature"]
df["visibility_log"] = np.log1p(df["HourlyVisibility"].clip(lower=0))
df["wind_x_visibility"] = df["HourlyWindSpeed"] * df["HourlyVisibility"]

print("Rows:", len(df))
display(
    df["launch_failure_binary"]
    .value_counts()
    .rename(index={0: "Success", 1: "Not Success"})
    .to_frame("count")
)
"""
    ),
    md(
        """
#### What this shows

The notebook starts from the same full merged table as the other modeling workflows. The purpose here is narrower: to test whether weather becomes more useful when it is only applied to launches with higher-quality weather coverage, rather than being forced into the full sample.
"""
    ),
    md(
        """
## 1. Define The Weather-Quality Subset

The subset is intentionally stricter than the original full sample. A launch is included if:

- weather was matched
- the weather observation is close enough to launch time
- the key weather variables needed for the weather-enhanced model are present
"""
    ),
    code(
        """
weather_subset = df[
    (df["weather_matched"] == True)
    & (df["weather_time_diff_minutes"] <= 30)
    & df["HourlyWindSpeed"].notna()
    & df["HourlyVisibility"].notna()
    & df["HourlyAltimeterSetting"].notna()
    & df["HourlyWetBulbTemperature"].notna()
].copy()

subset_summary = pd.DataFrame(
    [
        {
            "sample": "full sample",
            "rows": len(df),
            "failure_rate": df["launch_failure_binary"].mean(),
            "start_date": df["launch_date"].min().date(),
            "end_date": df["launch_date"].max().date(),
        },
        {
            "sample": "good weather subset",
            "rows": len(weather_subset),
            "failure_rate": weather_subset["launch_failure_binary"].mean(),
            "start_date": weather_subset["launch_date"].min().date(),
            "end_date": weather_subset["launch_date"].max().date(),
        },
    ]
)
subset_summary
"""
    ),
    md(
        """
#### What this section is doing

This step defines a stricter “good weather” subset. A launch is only kept if the weather match is close in time and the key weather variables are present.

#### How to interpret the output

The key takeaway is the tradeoff:

- the subset is much cleaner from a weather-data standpoint
- but it is also much smaller and lower-risk than the full dataset

That means any later performance gain has to be interpreted as a result on a narrower, easier sample rather than as a universal improvement.
"""
    ),
    code(
        """
coverage_by_facility = (
    df.assign(in_weather_subset=df.index.isin(weather_subset.index))
    .groupby("facility_group", dropna=False)
    .agg(
        launches=("Launch Id", "count"),
        subset_launches=("in_weather_subset", "sum"),
        subset_share=("in_weather_subset", "mean"),
    )
    .reset_index()
    .sort_values("launches", ascending=False)
)

coverage_by_facility
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

full_year = df.groupby("launch_year").size().reset_index(name="launches")
subset_year = weather_subset.groupby("launch_year").size().reset_index(name="launches")

sns.lineplot(data=full_year, x="launch_year", y="launches", marker="o", ax=axes[0], label="Full sample")
sns.lineplot(data=subset_year, x="launch_year", y="launches", marker="o", ax=axes[0], label="Weather subset")
axes[0].set_title("Launch coverage by year")
axes[0].set_xlabel("Launch year")
axes[0].set_ylabel("Launch count")

sns.barplot(data=coverage_by_facility.head(8), y="facility_group", x="subset_share", ax=axes[1], color="#4e79a7")
axes[1].set_title("Weather subset share by facility")
axes[1].set_xlabel("Share of launches retained")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### What these coverage checks mean

These outputs show whether the weather subset still looks representative. The answer is only partly yes: some facilities retain a decent share of launches, but others contribute little or nothing. That makes this notebook better suited to testing the *feasibility* of a weather-focused strategy than to establishing a final production workflow.
"""
    ),
    md(
        """
These checks matter because the weather-subset strategy only makes sense if the retained sample is still large enough and not dominated by a single site or era.
"""
    ),
    code(
        """
def chronological_split(frame):
    temp = frame.dropna(subset=["launch_date"]).sort_values(["launch_date", "Launch Id"]).reset_index(drop=True)
    n_rows = len(temp)
    train_end = int(n_rows * 0.60)
    val_end = int(n_rows * 0.80)
    return temp.iloc[:train_end].copy(), temp.iloc[train_end:val_end].copy(), temp.iloc[val_end:].copy()


full_train, full_val, full_test = chronological_split(df)
sub_train, sub_val, sub_test = chronological_split(weather_subset)

split_summary = pd.DataFrame(
    [
        {
            "sample": "full",
            "split": "train",
            "rows": len(full_train),
            "failure_rate": full_train["launch_failure_binary"].mean(),
        },
        {
            "sample": "full",
            "split": "validation",
            "rows": len(full_val),
            "failure_rate": full_val["launch_failure_binary"].mean(),
        },
        {
            "sample": "full",
            "split": "test",
            "rows": len(full_test),
            "failure_rate": full_test["launch_failure_binary"].mean(),
        },
        {
            "sample": "weather subset",
            "split": "train",
            "rows": len(sub_train),
            "failure_rate": sub_train["launch_failure_binary"].mean(),
        },
        {
            "sample": "weather subset",
            "split": "validation",
            "rows": len(sub_val),
            "failure_rate": sub_val["launch_failure_binary"].mean(),
        },
        {
            "sample": "weather subset",
            "split": "test",
            "rows": len(sub_test),
            "failure_rate": sub_test["launch_failure_binary"].mean(),
        },
    ]
)
split_summary
"""
    ),
    md(
        """
#### Why this split table matters

This is one of the most important cautionary outputs in the notebook. The weather subset is so small that the validation split has very weak positive-class support in your current run. That makes the validation-stage threshold selection and metric comparisons much less stable than in the main full-sample notebook.
"""
    ),
    md(
        """
## 2. Modeling Utilities
"""
    ),
    code(
        """
def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def choose_threshold(y_true, proba, thresholds=np.linspace(0.05, 0.95, 181)):
    rows = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                "failure_precision": precision_score(y_true, pred, zero_division=0),
                "failure_recall": recall_score(y_true, pred, zero_division=0),
                "failure_f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    scan = pd.DataFrame(rows)
    best = scan.sort_values(["balanced_accuracy", "failure_f1", "failure_recall"], ascending=False).iloc[0]
    return float(best["threshold"]), scan


def metric_frame(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "failure_precision": precision_score(y_true, pred, zero_division=0),
        "failure_recall": recall_score(y_true, pred, zero_division=0),
        "failure_f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
    }


def fit_eval(frame_train, frame_val, frame_test, model_name, estimator, numeric_features, categorical_features):
    feature_columns = numeric_features + categorical_features
    pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )
    pipe.fit(frame_train[feature_columns], frame_train["launch_failure_binary"])
    val_proba = pipe.predict_proba(frame_val[feature_columns])[:, 1]
    threshold, scan = choose_threshold(frame_val["launch_failure_binary"], val_proba)
    test_proba = pipe.predict_proba(frame_test[feature_columns])[:, 1]
    return {
        "pipe": pipe,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "scan": scan,
        "val_proba": val_proba,
        "test_proba": test_proba,
        "rows": [
            {"model": model_name, "split": "validation", **metric_frame(frame_val["launch_failure_binary"], val_proba, threshold)},
            {"model": model_name, "split": "test", **metric_frame(frame_test["launch_failure_binary"], test_proba, threshold)},
        ],
    }
"""
    ),
    md(
        """
## 3. Full-Sample Base Model
"""
    ),
    code(
        """
base_numeric = [
    "family_success_rate_pct",
    "mission_mass",
    "rocket_payload_leo",
    "config_liftoff_thrust",
    "config_stages",
    "config_strap_ons",
    "config_rocket_height",
    "launch_year_centered",
]

base_categorical = [
    "facility_group",
    "rocket_org_grouped",
    "rocket_family",
    "payload_bin",
]

full_base_result = fit_eval(
    full_train,
    full_val,
    full_test,
    "Full-sample base logistic",
    LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        C=0.5,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),
    base_numeric,
    base_categorical,
)

pd.DataFrame(full_base_result["rows"])
"""
    ),
    md(
        """
#### What this section is doing

This is the reference model for the notebook. It uses only the core non-weather features on the full chronological sample so the later subset results have a realistic benchmark.

#### What to notice

The full-sample base model performs credibly, which is important because it reminds us that a weather-subset strategy has to justify not just its score, but also its reduced coverage.
"""
    ),
    md(
        """
This is the reference point. The rest of the notebook asks whether restricting to a better weather subset lets weather variables add value beyond this baseline.
"""
    ),
    md(
        """
## 4. Within-Subset Comparison: Base Vs Weather-Enhanced
"""
    ),
    code(
        """
weather_numeric = base_numeric + [
    "HourlyWindSpeed",
    "HourlyVisibility",
    "HourlyAltimeterSetting",
    "HourlyWetBulbTemperature",
    "HourlyRelativeHumidity",
    "HourlyDryBulbTemperature",
    "HourlyDewPointTemperature",
    "dewpoint_depression",
    "visibility_log",
    "wind_x_visibility",
    "weather_time_diff_minutes",
]

weather_categorical = base_categorical + [
    "present_weather_rain_flag",
    "present_weather_fog_flag",
    "cloud_cover_broken_or_overcast_flag",
]

subset_base_result = fit_eval(
    sub_train,
    sub_val,
    sub_test,
    "Weather-subset base logistic",
    LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        C=0.5,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),
    base_numeric,
    base_categorical,
)

subset_weather_result = fit_eval(
    sub_train,
    sub_val,
    sub_test,
    "Weather-subset weather-enhanced logistic",
    LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        C=0.35,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),
    weather_numeric,
    weather_categorical,
)

subset_benchmark = pd.DataFrame(subset_base_result["rows"] + subset_weather_result["rows"]).sort_values(
    ["split", "balanced_accuracy", "pr_auc"],
    ascending=[True, False, False],
)
subset_benchmark
"""
    ),
    md(
        """
#### What this shows

This is the central within-subset comparison: a base subset model versus a weather-enhanced subset model on the same retained launches.

The current outputs look very strong, especially for the weather-subset base model, but they should be interpreted carefully. Because the subset is small and the validation split is weak, these scores are best treated as exploratory evidence that the subset may be easier to model, not proof that weather alone is driving a robust gain.
"""
    ),
    code(
        """
test_subset_benchmark = subset_benchmark[subset_benchmark["split"] == "test"].copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.barplot(data=test_subset_benchmark, x="model", y="balanced_accuracy", ax=axes[0], color="#4e79a7")
axes[0].set_title("Subset Test Balanced Accuracy")
axes[0].tick_params(axis="x", rotation=18)

sns.barplot(data=test_subset_benchmark, x="model", y="failure_recall", ax=axes[1], color="#f28e2b")
axes[1].set_title("Subset Test Failure Recall")
axes[1].tick_params(axis="x", rotation=18)

sns.barplot(data=test_subset_benchmark, x="model", y="pr_auc", ax=axes[2], color="#59a14f")
axes[2].set_title("Subset Test Failure PR AUC")
axes[2].tick_params(axis="x", rotation=18)

plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
#### How to read this chart

The chart compares how much the added weather block changes the precision-recall tradeoff inside the subset. In your current run, the base subset model is actually stronger on several thresholded test metrics, which suggests that simply restricting to better-measured rows may matter more than expanding the weather feature block.
"""
    ),
    md(
        """
The key comparison in this notebook is not against the original full-sample model. It is whether weather variables help **within the subset where they are measured well enough to be trusted**.
"""
    ),
    code(
        """
def plot_calibration_curves(prob_map, y_true, n_bins=8):
    plt.figure(figsize=(6.5, 5))
    for label, proba in prob_map.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted failure probability")
    plt.ylabel("Observed failure share")
    plt.title("Calibration on weather-subset test split")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_calibration_curves(
    {
        "Subset base": subset_base_result["test_proba"],
        "Subset weather-enhanced": subset_weather_result["test_proba"],
    },
    sub_test["launch_failure_binary"],
)
"""
    ),
    md(
        """
#### What the calibration plot means here

Calibration is especially important in a small subset experiment. Strong-looking classification metrics can happen by chance when only a few failures are present, so this plot acts as a qualitative check on whether the probability outputs still behave sensibly.
"""
    ),
    code(
        """
best_subset_output = (
    subset_base_result
    if subset_benchmark[subset_benchmark["split"] == "validation"].sort_values(
        ["balanced_accuracy", "pr_auc", "failure_f1"], ascending=False
    ).iloc[0]["model"] == "Weather-subset base logistic"
    else subset_weather_result
)

subset_test_df = sub_test.copy()
subset_test_df["predicted_failure_probability"] = best_subset_output["test_proba"]
subset_test_df["risk_decile"] = pd.qcut(
    subset_test_df["predicted_failure_probability"],
    q=10,
    duplicates="drop",
)

risk_bucket_summary = (
    subset_test_df.groupby("risk_decile", observed=False)
    .agg(
        launches=("Launch Id", "count"),
        observed_failure_rate=("launch_failure_binary", "mean"),
        mean_predicted_risk=("predicted_failure_probability", "mean"),
    )
    .reset_index()
)

plt.figure(figsize=(11, 4.8))
sns.barplot(data=risk_bucket_summary, x="risk_decile", y="observed_failure_rate", color="#f28e2b")
sns.pointplot(
    data=risk_bucket_summary,
    x="risk_decile",
    y="mean_predicted_risk",
    color="#4e79a7",
    linestyles="-",
    markers="o",
)
plt.title("Best subset model: observed failure rate by predicted-risk decile")
plt.xlabel("Predicted risk decile")
plt.ylabel("Failure probability")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

risk_bucket_summary
"""
    ),
    md(
        """
#### Why the risk-decile output matters

This is one of the most interpretable outputs in the notebook. It shows whether the best subset model is concentrating failures into the highest predicted-risk buckets. In your current run, the top decile carries most of the observed failures, which is promising.

At the same time, each decile contains very few launches, so the result should be read as “interesting evidence” rather than as a stable estimate of operational performance.
"""
    ),
    md(
        """
The risk-decile output is especially useful here. Even if the weather-enhanced model does not dominate every threshold metric, it may still be better at concentrating failures into the highest-risk buckets, which is often the more practical use case.
"""
    ),
    code(
        """
full_vs_subset = pd.DataFrame(full_base_result["rows"] + subset_benchmark.to_dict("records")).sort_values(
    ["split", "balanced_accuracy", "pr_auc"],
    ascending=[True, False, False],
)
full_vs_subset
"""
    ),
    md(
        """
#### How to interpret the full-vs-subset table

This is not a strict apples-to-apples leaderboard because the subset models are evaluated on a different population from the full-sample model. The right reading is:

- the full-sample base model is the robust general benchmark
- the subset models test whether a cleaner weather slice supports stronger local performance
- the subset results are promising but not yet strong enough to displace the full-sample approach
"""
    ),
    code(
        """
full_vs_subset.to_csv(DATA_DIR / "model_weather_subset_benchmark_results_v2.csv", index=False)
print("Saved benchmark results to", DATA_DIR / "model_weather_subset_benchmark_results_v2.csv")
"""
    ),
    md(
        """
#### Overall interpretation

This notebook does not yet prove that a weather-subset workflow is superior, but it does show that the idea is worth exploring. The main lesson is procedural: weather may be more useful when treated as a high-quality subset problem rather than a universal feature block. The next step would be to strengthen the subset definition and validation design before drawing firm conclusions.
"""
    ),
]

nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH.name}")
