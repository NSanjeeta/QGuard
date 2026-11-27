"""
QGUARD - BATTERY MONITORING

Run from project root:
    python qguard_battery_monitor.py

Input:
    data/processed/quantumguard_cleaned.csv

Configuration:
    config/qguard_thresholds.yaml

Outputs:
    data/processed/qguard_battery_status.csv
    data/processed/qguard_battery_summary.csv
    data/processed/qguard_battery_events.csv

Features:
    - Low battery detection
    - Critical battery detection
    - Rapid battery drop detection
    - Rolling battery statistics
    - Sensor-wise battery monitoring
    - Warning and critical severity classification

All thresholds are provisional until validated against
actual battery measurement hardware.
"""

from pathlib import Path
import sys

import pandas as pd
import yaml


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quantumguard_cleaned.csv"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "qguard_thresholds.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

STATUS_FILE = OUTPUT_DIR / "qguard_battery_status.csv"
SUMMARY_FILE = OUTPUT_DIR / "qguard_battery_summary.csv"
EVENTS_FILE = OUTPUT_DIR / "qguard_battery_events.csv"


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "cycle",
    "timestamp",
    "sensor_id",
    "battery_percent",
]

EXPECTED_SENSOR_IDS = {"S1", "S2", "S3", "S4", "S5"}


# ============================================================
# 3. LOAD CONFIGURATION
# ============================================================

def load_thresholds():

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    battery_config = config.get("battery", {})
    statistical_config = config.get("statistical", {})

    required_keys = [
        "low_percent",
        "critical_percent",
        "rapid_drop_warning",
        "rapid_drop_critical",
    ]

    missing = [
        key for key in required_keys
        if key not in battery_config
        or battery_config[key] is None
    ]

    if missing:
        raise ValueError(
            f"Missing battery thresholds: {missing}"
        )

    thresholds = {
        key: float(battery_config[key])
        for key in required_keys
    }

    thresholds["rolling_window"] = int(
        statistical_config.get("rolling_window", 5)
    )

    thresholds["statistical_multiplier"] = float(
        statistical_config.get("multiplier", 3)
    )

    # Validate threshold relationships.

    if not (
        0 <= thresholds["critical_percent"]
        < thresholds["low_percent"]
        <= 100
    ):
        raise ValueError(
            "Battery thresholds must satisfy: "
            "0 <= critical < low <= 100."
        )

    if not (
        0 < thresholds["rapid_drop_warning"]
        < thresholds["rapid_drop_critical"]
    ):
        raise ValueError(
            "Rapid-drop critical threshold must exceed warning threshold."
        )

    if thresholds["rolling_window"] < 1:
        raise ValueError("Rolling window must be at least 1.")

    if thresholds["statistical_multiplier"] <= 0:
        raise ValueError("Statistical multiplier must be positive.")

    print("\nBATTERY THRESHOLDS LOADED")
    print("=" * 60)

    for key, value in thresholds.items():
        print(f"{key}: {value}")

    return thresholds


# ============================================================
# 4. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Run qguard_data_preprocessing.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df["sensor_id"] = (
        df["sensor_id"].astype("string").str.strip()
    )

    df["cycle"] = pd.to_numeric(
        df["cycle"], errors="coerce"
    )

    df["battery_percent"] = pd.to_numeric(
        df["battery_percent"], errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    if df["cycle"].isna().any():
        raise ValueError(
            "Missing or invalid cycle values found."
        )

    if df["sensor_id"].isna().any():
        raise ValueError("Missing sensor IDs found.")

    if df.duplicated(["sensor_id", "cycle"]).any():
        raise ValueError(
            "Duplicate sensor-cycle records found."
        )

    invalid_battery = (
        (df["battery_percent"] < 0)
        | (df["battery_percent"] > 100)
    ).fillna(False)

    if invalid_battery.any():
        raise ValueError(
            "Battery readings outside the valid 0-100% range found."
        )

    if df["battery_percent"].isna().any():
        print(
            "\nWARNING: Missing battery readings detected. "
            "They will remain missing and will not be treated as zero."
        )

    print("\nQGUARD BATTERY MONITORING")
    print("=" * 60)

    print(f"Input: {INPUT_FILE}")
    print(f"Records loaded: {len(df)}")

    print(
        "Sensors found:",
        sorted(df["sensor_id"].unique().tolist())
    )

    return df


# ============================================================
# 5. CALCULATE BATTERY FEATURES
# ============================================================

def calculate_sensor_battery_features(
    sensor_df,
    thresholds
):

    sensor_df = sensor_df.sort_values(
        ["cycle", "timestamp"],
        kind="stable",
        na_position="last"
    ).copy()

    window = thresholds["rolling_window"]

    # --------------------------------------------------------
    # A. Battery change
    # --------------------------------------------------------

    sensor_df["battery_change_percentage_points"] = (
        sensor_df["battery_percent"].diff()
    )

    # Positive value means battery decreased.

    sensor_df["battery_drop_percentage_points"] = (
        -sensor_df["battery_change_percentage_points"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # B. Rolling statistics
    # --------------------------------------------------------

    battery_series = sensor_df["battery_percent"]

    sensor_df["battery_rolling_mean"] = (
        battery_series
        .rolling(window=window, min_periods=1)
        .mean()
    )

    sensor_df["battery_rolling_min"] = (
        battery_series
        .rolling(window=window, min_periods=1)
        .min()
    )

    sensor_df["battery_rolling_std"] = (
        battery_series
        .rolling(window=window, min_periods=2)
        .std()
    )

    sensor_df["battery_rolling_drop_sum"] = (
        sensor_df["battery_drop_percentage_points"]
        .rolling(window=window, min_periods=1)
        .sum()
    )

    # --------------------------------------------------------
    # C. Threshold evaluation
    # --------------------------------------------------------

    statuses = []
    severities = []
    reasons = []
    threshold_flags = []

    for _, row in sensor_df.iterrows():

        battery = row["battery_percent"]

        drop = row["battery_drop_percentage_points"]

        flags = []

        if pd.isna(battery):

            status = "UNKNOWN"
            severity = "UNKNOWN"

            reason = "Battery reading is missing"

        else:

            # Battery level evaluation.

            if battery <= thresholds["critical_percent"]:

                flags.append("CRITICAL_BATTERY")

            elif battery <= thresholds["low_percent"]:

                flags.append("LOW_BATTERY")

            # Rapid battery drop evaluation.

            if pd.notna(drop):

                if drop >= thresholds["rapid_drop_critical"]:

                    flags.append("CRITICAL_RAPID_DROP")

                elif drop >= thresholds["rapid_drop_warning"]:

                    flags.append("RAPID_BATTERY_DROP")

            # ------------------------------------------------
            # Determine primary status.
            # ------------------------------------------------

            if (
                "CRITICAL_BATTERY" in flags
                or "CRITICAL_RAPID_DROP" in flags
            ):

                status = "CRITICAL"
                severity = "CRITICAL"

            elif (
                "LOW_BATTERY" in flags
                or "RAPID_BATTERY_DROP" in flags
            ):

                status = "WARNING"
                severity = "WARNING"

            else:

                status = "NORMAL"
                severity = "NORMAL"

            # ------------------------------------------------
            # Explanation.
            # ------------------------------------------------

            explanations = []

            if "CRITICAL_BATTERY" in flags:

                explanations.append(
                    f"Critical battery level: {battery:.2f}%"
                )

            elif "LOW_BATTERY" in flags:

                explanations.append(
                    f"Low battery level: {battery:.2f}%"
                )

            if "CRITICAL_RAPID_DROP" in flags:

                explanations.append(
                    f"Critical battery decrease: {drop:.2f} percentage points"
                )

            elif "RAPID_BATTERY_DROP" in flags:

                explanations.append(
                    f"Rapid battery decrease: {drop:.2f} percentage points"
                )

            if not explanations:

                explanations.append(
                    "No configured battery threshold exceeded"
                )

            reason = "; ".join(explanations)

        statuses.append(status)
        severities.append(severity)
        reasons.append(reason)

        threshold_flags.append(
            "; ".join(flags) if flags else "NONE"
        )

    sensor_df["battery_status"] = statuses

    sensor_df["battery_severity"] = severities

    sensor_df["battery_status_reason"] = reasons

    sensor_df["battery_threshold_flags"] = threshold_flags

    # Preserve the previous feature name for compatibility.

    sensor_df["rapid_battery_drop"] = (
        sensor_df["battery_drop_percentage_points"]
        >= thresholds["rapid_drop_warning"]
    ).fillna(False).astype(int)

    # Preserve the previous rolling feature names.

    sensor_df["battery_rolling_mean_5"] = (
        sensor_df["battery_rolling_mean"]
    )

    sensor_df["battery_rolling_min_5"] = (
        sensor_df["battery_rolling_min"]
    )

    sensor_df["battery_rolling_drop_sum_5"] = (
        sensor_df["battery_rolling_drop_sum"]
    )

    return sensor_df


# ============================================================
# 6. PROCESS ALL SENSORS
# ============================================================

def monitor_all_sensors(df, thresholds):

    frames = []

    sensor_ids = sorted(
        df["sensor_id"].astype(str).unique().tolist()
    )

    missing_expected = EXPECTED_SENSOR_IDS - set(sensor_ids)

    if missing_expected:

        print(
            "\nWARNING: Expected sensors missing:",
            sorted(missing_expected)
        )

    for sensor_id, sensor_df in df.groupby(
        "sensor_id",
        sort=True
    ):

        processed = calculate_sensor_battery_features(
            sensor_df,
            thresholds
        )

        frames.append(processed)

    if not frames:
        raise ValueError(
            "No sensor records available."
        )

    result = pd.concat(
        frames,
        ignore_index=True
    )

    result = result.sort_values(
        ["cycle", "sensor_id"],
        kind="stable"
    ).reset_index(drop=True)

    return result


# ============================================================
# 7. CREATE SENSOR-WISE SUMMARY
# ============================================================

def create_summary(status_df):

    summary_rows = []

    for sensor_id, sensor_df in status_df.groupby(
        "sensor_id",
        sort=True
    ):

        valid_battery = (
            sensor_df["battery_percent"].dropna()
        )

        if len(valid_battery) > 0:

            first_battery = valid_battery.iloc[0]
            last_battery = valid_battery.iloc[-1]

            net_change = (
                last_battery - first_battery
            )

            minimum_battery = valid_battery.min()
            maximum_battery = valid_battery.max()
            average_battery = valid_battery.mean()

        else:

            first_battery = float("nan")
            last_battery = float("nan")
            net_change = float("nan")
            minimum_battery = float("nan")
            maximum_battery = float("nan")
            average_battery = float("nan")

        status_counts = (
            sensor_df["battery_status"].value_counts()
        )

        severity_counts = (
            sensor_df["battery_severity"].value_counts()
        )

        summary_rows.append({

            "sensor_id": sensor_id,

            "records_checked": len(sensor_df),

            "average_battery_percent": average_battery,

            "minimum_battery_percent": minimum_battery,

            "maximum_battery_percent": maximum_battery,

            "first_record_battery_percent": first_battery,

            "last_record_battery_percent": last_battery,

            "net_battery_change_percentage_points": net_change,

            "normal_records": status_counts.get("NORMAL", 0),

            "warning_records": status_counts.get("WARNING", 0),

            "critical_records": status_counts.get("CRITICAL", 0),

            "unknown_records": status_counts.get("UNKNOWN", 0),

            "normal_severity_records": severity_counts.get("NORMAL", 0),

            "warning_severity_records": severity_counts.get("WARNING", 0),

            "critical_severity_records": severity_counts.get("CRITICAL", 0),

            "rapid_drop_events": int(
                sensor_df["rapid_battery_drop"].sum()
            ),

            "maximum_single_cycle_drop_percentage_points": (
                sensor_df["battery_drop_percentage_points"].max()
            ),

        })

    return pd.DataFrame(summary_rows)


# ============================================================
# 8. EXTRACT BATTERY EVENTS
# ============================================================

def create_events(status_df):

    event_mask = status_df["battery_status"].isin(
        ["WARNING", "CRITICAL", "UNKNOWN"]
    )

    event_columns = [

        "cycle",
        "timestamp",
        "sensor_id",

        "battery_percent",

        "battery_change_percentage_points",

        "battery_drop_percentage_points",

        "battery_rolling_mean",

        "battery_rolling_min",

        "battery_rolling_std",

        "battery_status",

        "battery_severity",

        "battery_threshold_flags",

        "battery_status_reason",

    ]

    return (
        status_df.loc[event_mask, event_columns]
        .sort_values(["cycle", "sensor_id"])
        .reset_index(drop=True)
    )


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(
    status_df,
    summary_df,
    events_df,
    thresholds
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    status_df.to_csv(
        STATUS_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    events_df.to_csv(
        EVENTS_FILE,
        index=False
    )

    print("\nBATTERY MONITORING COMPLETED")
    print("=" * 60)

    print("\nSensor-wise battery summary:")

    print(
        summary_df.to_string(index=False)
    )

    print("\nBattery status:")

    print(
        status_df["battery_status"]
        .value_counts()
        .to_string()
    )

    print("\nBattery severity:")

    print(
        status_df["battery_severity"]
        .value_counts()
        .to_string()
    )

    print("\nThresholds used:")

    print(
        f"LOW: <= {thresholds['low_percent']}%"
    )

    print(
        f"CRITICAL: <= {thresholds['critical_percent']}%"
    )

    print(
        "Rapid-drop warning: >= "
        f"{thresholds['rapid_drop_warning']} percentage points"
    )

    print(
        "Rapid-drop critical: >= "
        f"{thresholds['rapid_drop_critical']} percentage points"
    )

    print("\nGenerated files:")

    print(STATUS_FILE)
    print(SUMMARY_FILE)
    print(EVENTS_FILE)

    print(
        "\nNOTE: Rapid battery drops are calculated between "
        "consecutive recorded cycles, not per hour or per day."
    )

    print(
        "\nNOTE: Battery thresholds are provisional and require "
        "validation against actual battery measurement hardware."
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    try:

        thresholds = load_thresholds()

        df = load_data()

        status_df = monitor_all_sensors(
            df,
            thresholds
        )

        summary_df = create_summary(
            status_df
        )

        events_df = create_events(
            status_df
        )

        save_outputs(
            status_df,
            summary_df,
            events_df,
            thresholds
        )

    except Exception as error:

        print(f"\nERROR: {error}")

        sys.exit(1)


if __name__ == "__main__":

    main()