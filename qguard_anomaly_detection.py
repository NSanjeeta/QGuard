"""
QGUARD - AI ANOMALY DETECTION
=============================

Run from the QGuard project root:
    python qguard_anomaly_detection.py

Input:
    data/processed/qguard_features.csv

Outputs:
    data/processed/qguard_anomaly_results.csv
    data/processed/qguard_anomaly_summary.csv
    models/qguard_isolation_forest_models.joblib

Model:
    Isolation Forest (unsupervised anomaly detection)

Design decisions:
- A separate Isolation Forest model is trained for each sensor so that
  one sensor's normal operating range does not dominate another's.
- Only numeric operational/communication/battery features are used.
- IDs, cycle numbers, timestamps, presence fields, target-distance fields,
  packet_rate_ppm and reading_value are excluded from model training.
- Presence/no-presence is not a sensor-health label.
- An anomaly is an unusual observation, NOT proof of hardware failure.
- The contamination setting is a configurable modelling assumption,
  not a measured real-world failure rate.
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURATION AND PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qguard_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_FILE = OUTPUT_DIR / "qguard_anomaly_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "qguard_anomaly_summary.csv"
MODEL_FILE = MODEL_DIR / "qguard_isolation_forest_models.joblib"

RANDOM_STATE = 42

# Approximate fraction of records the model is configured to flag.
# This is a modelling choice, not an expected real-world fault rate.
CONTAMINATION = 0.05

MIN_RECORDS_PER_SENSOR = 30

# Do not train on these fields. Presence and distance describe target
# activity, not sensor health; some distance records are already known
# to be inconsistent in the current dataset.
EXCLUDED_FEATURES = {
    "cycle",
    "presence_state",
    "target_distance_m",
    "packet_rate_ppm",
    "reading_value",
    "presence_transition",
    "presence_event",
    "presence_detection",
    "presence_distance_inconsistency",
    "distance_available",
    "target_distance_change_m",
    "absolute_distance_change_m",
    "cumulative_presence_detections",
    "presence_count_rolling_5",
    "zone_id",
}

IDENTIFIER_COLUMNS = {
    "sensor_id",
    "sensor_name",
    "sensor_type",
    "timestamp",
}


# ============================================================
# 2. LOAD FEATURE DATASET
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nFeature dataset not found.\n"
            f"Expected:\n{INPUT_FILE}\n\n"
            "Run qguard_feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"sensor_id", "cycle"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing from feature dataset: {sorted(missing)}"
        )

    df = df.copy()
    df["sensor_id"] = df["sensor_id"].astype("string").str.strip()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")

    if df["sensor_id"].isna().any():
        raise ValueError("Missing sensor_id values found.")

    if df["cycle"].isna().any():
        raise ValueError("Missing or invalid cycle values found.")

    if df.duplicated(["sensor_id", "cycle"]).any():
        raise ValueError(
            "Duplicate sensor-cycle records found. "
            "Resolve them before anomaly detection."
        )

    print("\nQGUARD AI ANOMALY DETECTION")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Records loaded: {len(df)}")
    print(f"Sensors found: {sorted(df['sensor_id'].unique().tolist())}")

    return df


# ============================================================
# 3. SELECT OPERATIONAL FEATURES
# ============================================================

def select_features(sensor_df):
    excluded = EXCLUDED_FEATURES | IDENTIFIER_COLUMNS

    numeric_columns = sensor_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    selected = [
        column for column in numeric_columns
        if column not in excluded
    ]

    # Remove columns that have no usable values or no variation.
    usable = []

    for column in selected:
        values = pd.to_numeric(
            sensor_df[column], errors="coerce"
        )

        if values.notna().sum() == 0:
            continue

        if values.nunique(dropna=True) <= 1:
            continue

        usable.append(column)

    return usable


# ============================================================
# 4. TRAIN AND SCORE ONE SENSOR
# ============================================================

def train_sensor_model(sensor_id, sensor_df):
    sensor_df = sensor_df.sort_values(
        "cycle", kind="stable"
    ).copy()

    feature_columns = select_features(sensor_df)

    if len(sensor_df) < MIN_RECORDS_PER_SENSOR:
        raise ValueError(
            f"{sensor_id} has only {len(sensor_df)} records. "
            f"At least {MIN_RECORDS_PER_SENSOR} are required."
        )

    if not feature_columns:
        raise ValueError(
            f"No suitable numeric operational features found for {sensor_id}."
        )

    X = sensor_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Drop features that are entirely missing for this sensor.
    feature_columns = [
        col for col in feature_columns
        if X[col].notna().any()
    ]
    X = X[feature_columns]

    if X.empty or not feature_columns:
        raise ValueError(
            f"No usable feature values available for {sensor_id}."
        )

    model = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=200,
                contamination=CONTAMINATION,
                random_state=RANDOM_STATE,
                max_samples="auto",
                n_jobs=-1,
            )
        ),
    ])

    model.fit(X)

    # sklearn: +1 means inlier; -1 means outlier.
    predictions = model.predict(X)

    # Higher decision_function values are more normal.
    # Negating it makes higher anomaly_score mean more unusual.
    anomaly_scores = -model.decision_function(X)

    sensor_results = sensor_df[
        ["cycle", "sensor_id"]
    ].copy()

    if "timestamp" in sensor_df.columns:
        sensor_results["timestamp"] = sensor_df["timestamp"].values

    sensor_results["anomaly_prediction"] = predictions
    sensor_results["anomaly_score"] = anomaly_scores
    sensor_results["anomaly_status"] = np.where(
        predictions == -1,
        "ANOMALY",
        "NORMAL"
    )
    sensor_results["features_used"] = ", ".join(feature_columns)

    model_info = {
        "model": model,
        "feature_columns": feature_columns,
        "training_records": int(len(sensor_df)),
        "contamination": CONTAMINATION,
    }

    return sensor_results, model_info


# ============================================================
# 5. PROCESS EACH SENSOR INDEPENDENTLY
# ============================================================

def run_anomaly_detection(df):
    result_frames = []
    models = {}
    errors = []

    for sensor_id, sensor_df in df.groupby(
        "sensor_id",
        sort=True
    ):
        sensor_id = str(sensor_id)

        try:
            sensor_results, model_info = train_sensor_model(
                sensor_id,
                sensor_df
            )

            result_frames.append(sensor_results)
            models[sensor_id] = model_info

            print(
                f"{sensor_id}: model trained using "
                f"{model_info['training_records']} records and "
                f"{len(model_info['feature_columns'])} features."
            )

        except Exception as error:
            errors.append({
                "sensor_id": sensor_id,
                "error": str(error)
            })
            print(f"WARNING - {sensor_id}: {error}")

    if not result_frames:
        raise RuntimeError(
            "No sensor models could be trained. "
            "Check the feature dataset and required columns."
        )

    results = pd.concat(
        result_frames,
        ignore_index=True
    )

    results = results.sort_values(
        ["cycle", "sensor_id"],
        kind="stable"
    ).reset_index(drop=True)

    return results, models, errors


# ============================================================
# 6. CREATE SENSOR-WISE SUMMARY
# ============================================================

def create_summary(results):
    rows = []

    for sensor_id, sensor_df in results.groupby(
        "sensor_id",
        sort=True
    ):
        total = len(sensor_df)
        anomaly_count = int(
            (sensor_df["anomaly_status"] == "ANOMALY").sum()
        )

        rows.append({
            "sensor_id": sensor_id,
            "records_analysed": total,
            "normal_records": total - anomaly_count,
            "anomaly_records": anomaly_count,
            "anomaly_percent": (
                anomaly_count / total * 100 if total else 0
            ),
            "mean_anomaly_score": sensor_df["anomaly_score"].mean(),
            "maximum_anomaly_score": sensor_df["anomaly_score"].max(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. SAVE RESULTS AND MODELS
# ============================================================

def save_outputs(results, summary, models, errors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    joblib.dump(
        {
            "models": models,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "description": (
                "Per-sensor Isolation Forest models for QGuard "
                "operational anomaly detection."
            ),
        },
        MODEL_FILE
    )

    print("\nANOMALY DETECTION COMPLETED")
    print("=" * 60)

    print("\nSensor-wise anomaly summary:")
    print(summary.to_string(index=False))

    total_anomalies = int(
        (results["anomaly_status"] == "ANOMALY").sum()
    )

    print(f"\nTotal records analysed: {len(results)}")
    print(f"Total anomalies flagged: {total_anomalies}")
    print(
        "Overall flagged percentage: "
        f"{total_anomalies / len(results) * 100:.2f}%"
    )

    print("\nOutput files:")
    print(f"- {RESULTS_FILE}")
    print(f"- {SUMMARY_FILE}")
    print(f"- {MODEL_FILE}")

    if errors:
        print("\nSensors not processed:")
        for item in errors:
            print(f"- {item['sensor_id']}: {item['error']}")

    print(
        "\nIMPORTANT: An anomaly is an unusual data pattern, "
        "not proof of sensor failure. Fault diagnosis is a separate phase."
    )
    print(
        "The model is trained and scored on the supplied dataset. "
        "Its anomaly flags must be reviewed and validated before "
        "being used for real-time decisions."
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    try:
        df = load_data()
        results, models, errors = run_anomaly_detection(df)
        summary = create_summary(results)
        save_outputs(results, summary, models, errors)

    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
