"""
QGUARD - FEATURE ENGINEERING

Run:
    python qguard_feature_engineering.py

Input:
    data/processed/quantumguard_cleaned.csv

Outputs:
    data/processed/qguard_features.csv
    data/processed/qguard_feature_summary.csv

Design:
- Features calculated independently for each sensor.
- Five-record rolling window.
- No future records used.
- Missing target distances preserved.
- No original readings modified.
- No fault labels or diagnoses generated.
"""

from pathlib import Path
import sys
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quantumguard_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "qguard_features.csv"

SUMMARY_FILE = OUTPUT_DIR / "qguard_feature_summary.csv"

ROLLING_WINDOW = 5

REQUIRED_COLUMNS = [
    "cycle",
    "timestamp",
    "sensor_id",
    "reading_value",
    "presence_state",
    "target_distance_m",
    "packet_rate_ppm",
    "response_time_ms",
    "battery_percent",
    "rssi_dbm",
    "missed_messages"
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Run qguard_data_preprocessing.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\n" + "=" * 60)
    print("QGUARD FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Input file: {INPUT_FILE}")
    print(f"Input rows: {len(df)}")

    return df


# ============================================================
# 3. PREPARE DATA TYPES AND SORTING
# ============================================================

def prepare_data(df):

    df = df.copy()

    numeric_columns = [
        "cycle",
        "reading_value",
        "presence_state",
        "target_distance_m",
        "packet_rate_ppm",
        "response_time_ms",
        "battery_percent",
        "rssi_dbm",
        "missed_messages"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Clean sensor identifiers

    df["sensor_id"] = (
        df["sensor_id"]
        .astype("string")
        .str.strip()
    )

    # The cleaned CSV stores timestamps as:
    # YYYY-MM-DD HH:MM:SS
    #
    # Original input format was:
    # DD-MM-YYYY HH:MM

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    if df["cycle"].isna().any():

        raise ValueError(
            "Invalid cycle values found."
        )

    if df["sensor_id"].isna().any():

        raise ValueError(
            "Missing sensor identifiers found."
        )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamps found in cleaned dataset."
        )

    if df.duplicated(
        subset=["sensor_id", "cycle"]
    ).any():

        raise ValueError(
            "Duplicate sensor-cycle records found."
        )

    # Sort independently by sensor and cycle

    df = df.sort_values(
        ["sensor_id", "cycle", "timestamp"],
        kind="stable"
    ).reset_index(drop=True)

    return df


# ============================================================
# 4. FEATURE ENGINEERING FOR EACH SENSOR
# ============================================================

def engineer_sensor_features(sensor_df):

    sensor_df = sensor_df.copy()

    # Previous record values

    previous_cycle = sensor_df["cycle"].shift(1)

    previous_presence = (
        sensor_df["presence_state"].shift(1)
    )

    previous_distance = (
        sensor_df["target_distance_m"].shift(1)
    )

    # Consecutive cycle validation

    sensor_df["cycle_gap"] = (
        sensor_df["cycle"].diff()
    )

    consecutive_cycle = (
        sensor_df["cycle_gap"] == 1
    )

    # ========================================================
    # A. TIME FEATURES
    # ========================================================

    sensor_df["elapsed_time_seconds"] = (
        sensor_df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    sensor_df["nonpositive_time_gap"] = (
        sensor_df["elapsed_time_seconds"].notna()
        & (sensor_df["elapsed_time_seconds"] <= 0)
    ).astype(int)

    # ========================================================
    # B. COMMUNICATION FEATURES
    # ========================================================

    communication_columns = [
        ("rssi_dbm", "rssi_change_db"),
        ("response_time_ms", "response_time_change_ms"),
        ("packet_rate_ppm", "packet_rate_change"),
        ("missed_messages", "missed_messages_change")
    ]

    for original, feature in communication_columns:

        change = sensor_df[original].diff()

        # Do not interpret skipped cycles as consecutive readings

        sensor_df[feature] = change.where(
            consecutive_cycle
        )

    # Rolling communication statistics

    rolling_columns = [
        ("rssi_dbm", "rssi"),
        ("response_time_ms", "response_time"),
        ("packet_rate_ppm", "packet_rate"),
        ("missed_messages", "missed_messages")
    ]

    for original, prefix in rolling_columns:

        rolling = sensor_df[original].rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )

        sensor_df[
            f"{prefix}_rolling_mean"
        ] = rolling.mean()

        sensor_df[
            f"{prefix}_rolling_std"
        ] = rolling.std(ddof=0)

    # Recent missed-message total

    sensor_df["missed_messages_rolling_sum"] = (
        sensor_df["missed_messages"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .sum()
    )

    # ========================================================
    # C. BATTERY FEATURES
    # ========================================================

    battery_change = (
        sensor_df["battery_percent"].diff()
    )

    sensor_df["battery_change_percent"] = (
        battery_change.where(consecutive_cycle)
    )

    # Positive value means battery decreased

    sensor_df["battery_drop_percent"] = (
        -sensor_df["battery_change_percent"]
    ).clip(lower=0)

    sensor_df["battery_rolling_mean"] = (
        sensor_df["battery_percent"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    sensor_df["battery_rolling_std"] = (
        sensor_df["battery_percent"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .std(ddof=0)
    )

    # ========================================================
    # D. PRESENCE AND MOTION FEATURES
    # ========================================================

    sensor_df["presence_change"] = (
        sensor_df["presence_state"].diff()
    )

    # Presence transition only between consecutive cycles

    sensor_df["presence_transition"] = (
        sensor_df["presence_state"].notna()
        & previous_presence.notna()
        & sensor_df["presence_state"].ne(
            previous_presence
        )
        & consecutive_cycle
    ).astype(int)

    # Distance availability

    sensor_df["distance_available"] = (
        sensor_df["target_distance_m"].notna()
    ).astype(int)

    # ========================================================
    # E. PRESENCE-DISTANCE CONSISTENCY
    # ========================================================

    no_presence_with_distance = (
        (sensor_df["presence_state"] == 0)
        & sensor_df["target_distance_m"].notna()
    )

    presence_without_distance = (
        (sensor_df["presence_state"] == 1)
        & sensor_df["target_distance_m"].isna()
    )

    sensor_df[
        "presence_distance_inconsistency"
    ] = (
        no_presence_with_distance
        | presence_without_distance
    ).astype(int)

    # ========================================================
    # F. TARGET DISTANCE CHANGE
    # ========================================================

    distance_change = (
        sensor_df["target_distance_m"].diff()
    )

    valid_distance_change = (
        consecutive_cycle
        & (sensor_df["presence_state"] == 1)
        & (previous_presence == 1)
        & sensor_df["target_distance_m"].notna()
        & previous_distance.notna()
    )

    sensor_df["target_distance_change_m"] = (
        distance_change.where(
            valid_distance_change
        )
    )

    # ========================================================
    # G. READING VALUE
    # ========================================================

    # reading_value remains unchanged.
    # Its physical meaning and unit are not confirmed.
    # Therefore, it is not used to create derived features.

    return sensor_df


# ============================================================
# 5. PROCESS EACH SENSOR INDEPENDENTLY
# ============================================================

def create_features(df):

    feature_frames = []

    for sensor_id, sensor_df in df.groupby(
        "sensor_id",
        sort=False
    ):

        engineered = engineer_sensor_features(
            sensor_df
        )

        feature_frames.append(engineered)

    if not feature_frames:

        raise ValueError(
            "No sensor records available."
        )

    features = pd.concat(
        feature_frames,
        ignore_index=True
    )

    # Restore global order

    features = features.sort_values(
        ["cycle", "sensor_id", "timestamp"],
        kind="stable"
    ).reset_index(drop=True)

    return features


# ============================================================
# 6. FEATURE SUMMARY
# ============================================================

def create_feature_summary(features):

    summary_rows = []

    for sensor_id, sensor_df in features.groupby(
        "sensor_id"
    ):

        detected_distances = sensor_df.loc[
            sensor_df["presence_state"] == 1,
            "target_distance_m"
        ]

        row = {

            "sensor_id": sensor_id,

            "record_count": len(sensor_df),

            "presence_detections": int(
                (sensor_df["presence_state"] == 1).sum()
            ),

            "no_presence_records": int(
                (sensor_df["presence_state"] == 0).sum()
            ),

            "distance_available_records": int(
                sensor_df["distance_available"].sum()
            ),

            "presence_distance_inconsistencies": int(
                sensor_df[
                    "presence_distance_inconsistency"
                ].sum()
            ),

            "presence_transitions": int(
                sensor_df["presence_transition"].sum()
            ),

            "minimum_detected_distance_m": (
                detected_distances.min()
            ),

            "maximum_detected_distance_m": (
                detected_distances.max()
            ),

            "average_rssi_dbm": (
                sensor_df["rssi_dbm"].mean()
            ),

            "average_response_time_ms": (
                sensor_df["response_time_ms"].mean()
            ),

            "average_battery_percent": (
                sensor_df["battery_percent"].mean()
            ),

            "total_missed_messages": (
                sensor_df["missed_messages"].sum()
            )
        }

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ============================================================
# 7. VALIDATE FEATURES
# ============================================================

def validate_features(original, features):

    print("\nFEATURE VALIDATION")
    print("=" * 60)

    # Row count must remain unchanged

    assert len(original) == len(features), (
        "Row count changed during feature engineering."
    )

    # Original target distances must remain unchanged

    original_distances = (
        original
        .set_index(["sensor_id", "cycle"])["target_distance_m"]
        .sort_index()
    )

    feature_distances = (
        features
        .set_index(["sensor_id", "cycle"])["target_distance_m"]
        .sort_index()
    )

    pd.testing.assert_series_equal(
        original_distances,
        feature_distances,
        check_names=False
    )

    # Missing distances must remain missing

    assert (
        original["target_distance_m"].isna().sum()
        ==
        features["target_distance_m"].isna().sum()
    )

    # Inconsistencies must be preserved

    inconsistency_count = int(
        features[
            "presence_distance_inconsistency"
        ].sum()
    )

    print("Original rows preserved: YES")
    print("Original distances preserved: YES")
    print("Missing distances preserved: YES")

    print(
        f"Presence-distance inconsistencies: "
        f"{inconsistency_count}"
    )

    print(
        "Maximum target distance:",
        features["target_distance_m"].max(),
        "m"
    )

    # Detect accidental cross-sensor feature mixing

    sensor_counts = features["sensor_id"].value_counts()

    print("\nRecords per sensor:")

    for sensor_id, count in sensor_counts.items():

        print(f"{sensor_id}: {count}")


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

def save_outputs(features):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = create_feature_summary(
        features
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print("\nFEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print(f"Rows retained: {len(features)}")

    print(
        f"Columns in feature dataset: "
        f"{len(features.columns)}"
    )

    print("\nFiles created:")

    print(f"- {OUTPUT_FILE}")
    print(f"- {SUMMARY_FILE}")

    print(
        "\nOriginal and cleaned datasets "
        "have NOT been modified."
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    try:

        df = load_data()

        df = prepare_data(df)

        features = create_features(df)

        validate_features(df, features)

        save_outputs(features)

        print("\nNext: inspect the feature summary.")

    except Exception as error:

        print(f"\nERROR: {error}")

        sys.exit(1)


if __name__ == "__main__":

    main()