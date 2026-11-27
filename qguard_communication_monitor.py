"""
QGUARD - COMMUNICATION MONITORING

Run from project root:
    python qguard_communication_monitor.py

Input:
    data/processed/qguard_features.csv

Configuration:
    config/qguard_thresholds.yaml

Outputs:
    data/processed/qguard_communication_status.csv
    data/processed/qguard_communication_summary.csv
    data/processed/qguard_communication_events.csv
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
    PROJECT_ROOT / "data" / "processed" / "qguard_features.csv"
)

CONFIG_FILE = PROJECT_ROOT / "config" / "qguard_thresholds.yaml"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

STATUS_FILE = OUTPUT_DIR / "qguard_communication_status.csv"
SUMMARY_FILE = OUTPUT_DIR / "qguard_communication_summary.csv"
EVENTS_FILE = OUTPUT_DIR / "qguard_communication_events.csv"


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "cycle",
    "sensor_id",
    "rssi_dbm",
    "response_time_ms",
    "missed_messages",
]


# ============================================================
# 3. LOAD THRESHOLD CONFIGURATION
# ============================================================

def load_thresholds():

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Threshold configuration not found: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    communication = config.get("communication")

    if not isinstance(communication, dict):
        raise ValueError("Missing communication configuration.")

    required_keys = [
        "rssi_warning_dbm",
        "rssi_critical_dbm",
        "response_warning_ms",
        "response_critical_ms",
        "missed_messages_warning",
        "missed_messages_critical",
        "recent_gap_window_cycles",
    ]

    missing = [
        key for key in required_keys
        if key not in communication
        or communication[key] is None
    ]

    if missing:
        raise ValueError(
            f"Missing communication thresholds: {missing}"
        )

    thresholds = {
        key: float(communication[key])
        for key in required_keys
        if key != "recent_gap_window_cycles"
    }

    thresholds["recent_gap_window_cycles"] = int(
        communication["recent_gap_window_cycles"]
    )

    if thresholds["rssi_critical_dbm"] > thresholds["rssi_warning_dbm"]:
        raise ValueError(
            "Critical RSSI threshold must be lower than warning RSSI."
        )

    if thresholds["response_critical_ms"] < thresholds["response_warning_ms"]:
        raise ValueError(
            "Critical response threshold must be >= warning threshold."
        )

    if thresholds["missed_messages_critical"] < thresholds["missed_messages_warning"]:
        raise ValueError(
            "Critical missed-message threshold must be >= warning threshold."
        )

    if thresholds["recent_gap_window_cycles"] < 1:
        raise ValueError("Recent gap window must be at least 1.")

    print("\nCOMMUNICATION THRESHOLDS LOADED")

    for key, value in thresholds.items():
        print(f"{key}: {value}")

    return thresholds


# ============================================================
# 4. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}\n"
            "Run qguard_feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    for column in [
        "cycle",
        "rssi_dbm",
        "response_time_ms",
        "missed_messages",
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["sensor_id"] = (
        df["sensor_id"].astype("string").str.strip()
    )

    if df["cycle"].isna().any():
        raise ValueError("Invalid or missing cycle values.")

    if df["sensor_id"].isna().any():
        raise ValueError("Missing sensor IDs found.")

    if df.duplicated(["sensor_id", "cycle"]).any():
        raise ValueError(
            "Duplicate sensor_id/cycle records found."
        )

    print("\nQGUARD COMMUNICATION MONITORING")
    print("=" * 60)
    print(f"Records loaded: {len(df)}")
    print(
        "Sensors found:",
        sorted(df["sensor_id"].unique().tolist())
    )

    return df


# ============================================================
# 5. ALIGN SENSORS BY CYCLE
# ============================================================

def create_cycle_grid(df):

    sensor_ids = sorted(
        df["sensor_id"].astype(str).unique().tolist()
    )

    cycles = sorted(df["cycle"].unique().tolist())

    if not sensor_ids or not cycles:
        raise ValueError("No valid sensors or cycles found.")

    grid = pd.MultiIndex.from_product(
        [sensor_ids, cycles],
        names=["sensor_id", "cycle"]
    ).to_frame(index=False)

    aligned = grid.merge(
        df,
        on=["sensor_id", "cycle"],
        how="left",
        indicator=True,
        validate="one_to_one"
    )

    aligned["record_received"] = (
        aligned["_merge"] == "both"
    ).astype(int)

    aligned.drop(columns=["_merge"], inplace=True)

    return aligned, sensor_ids


# ============================================================
# 6. FEATURE-LEVEL THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(row, thresholds):

    warnings = []
    criticals = []

    rssi = row["rssi_dbm"]
    response = row["response_time_ms"]
    missed = row["missed_messages"]

    # RSSI

    if pd.notna(rssi):

        if rssi <= thresholds["rssi_critical_dbm"]:
            criticals.append(
                f"Critical RSSI ({rssi:g} dBm)"
            )

        elif rssi <= thresholds["rssi_warning_dbm"]:
            warnings.append(
                f"Weak RSSI ({rssi:g} dBm)"
            )

    # Response time

    if pd.notna(response):

        if response >= thresholds["response_critical_ms"]:
            criticals.append(
                f"Critical response time ({response:g} ms)"
            )

        elif response >= thresholds["response_warning_ms"]:
            warnings.append(
                f"High response time ({response:g} ms)"
            )

    # Missed messages

    if pd.notna(missed):

        if missed >= thresholds["missed_messages_critical"]:
            criticals.append(
                f"Critical missed messages ({missed:g})"
            )

        elif missed >= thresholds["missed_messages_warning"]:
            warnings.append(
                f"Elevated missed messages ({missed:g})"
            )

    return warnings, criticals


# ============================================================
# 7. CLASSIFY COMMUNICATION STATUS
# ============================================================

def classify_communication(aligned, sensor_ids, thresholds):

    results = []

    gap_window = thresholds["recent_gap_window_cycles"]

    for sensor_id in sensor_ids:

        sensor_df = (
            aligned[aligned["sensor_id"] == sensor_id]
            .sort_values("cycle")
            .copy()
            .reset_index(drop=True)
        )

        statuses = []
        severities = []
        reasons_list = []
        warning_counts = []
        critical_counts = []
        threshold_flags = []

        for position, row in sensor_df.iterrows():

            received = int(row["record_received"]) == 1

            if not received:

                statuses.append("ISOLATED")
                severities.append("CRITICAL")

                reasons_list.append(
                    "No record for this sensor in an observed network cycle"
                )

                warning_counts.append(0)
                critical_counts.append(0)
                threshold_flags.append("MISSING_RECORD")

                continue

            start = max(0, position - gap_window)

            previous_window = sensor_df.iloc[start:position]

            recent_missing_count = int(
                (previous_window["record_received"] == 0).sum()
            )

            warnings, criticals = evaluate_thresholds(
                row, thresholds
            )

            warning_counts.append(len(warnings))
            critical_counts.append(len(criticals))

            if recent_missing_count > 0:

                statuses.append("INTERMITTENT")
                severities.append("CRITICAL")

                reason = (
                    "Communication returned after "
                    f"{recent_missing_count} missing cycle(s)"
                )

            elif criticals:

                statuses.append("DEGRADED")
                severities.append("CRITICAL")

                reason = "; ".join(criticals + warnings)

            elif warnings:

                statuses.append("DEGRADED")
                severities.append("WARNING")

                reason = "; ".join(warnings)

            else:

                statuses.append("HEALTHY")
                severities.append("NORMAL")

                reason = (
                    "Record received; no configured link threshold exceeded"
                )

            reasons_list.append(reason)

            all_flags = criticals + warnings

            threshold_flags.append(
                "; ".join(all_flags) if all_flags else "NONE"
            )

        sensor_df["communication_status"] = statuses

        sensor_df["communication_severity"] = severities

        sensor_df["warning_indicator_count"] = warning_counts

        sensor_df["critical_indicator_count"] = critical_counts

        sensor_df["threshold_flags"] = threshold_flags

        sensor_df["communication_reason"] = reasons_list

        sensor_df["missing_cycle_record"] = (
            sensor_df["record_received"] == 0
        ).astype(int)

        results.append(sensor_df)

    status_df = pd.concat(results, ignore_index=True)

    status_df = status_df.sort_values(
        ["cycle", "sensor_id"]
    ).reset_index(drop=True)

    return status_df


# ============================================================
# 8. SENSOR-WISE SUMMARY
# ============================================================

def create_summary(status_df, sensor_ids):

    rows = []

    for sensor_id in sensor_ids:

        sensor_df = status_df[
            status_df["sensor_id"] == sensor_id
        ]

        counts = sensor_df[
            "communication_status"
        ].value_counts().to_dict()

        severity_counts = sensor_df[
            "communication_severity"
        ].value_counts().to_dict()

        rows.append({

            "sensor_id": sensor_id,

            "cycles_checked": len(sensor_df),

            "healthy_cycles": counts.get("HEALTHY", 0),

            "degraded_cycles": counts.get("DEGRADED", 0),

            "intermittent_cycles": counts.get("INTERMITTENT", 0),

            "isolated_cycles": counts.get("ISOLATED", 0),

            "warning_cycles": severity_counts.get("WARNING", 0),

            "critical_cycles": severity_counts.get("CRITICAL", 0),

            "average_rssi_dbm": sensor_df["rssi_dbm"].mean(),

            "average_response_time_ms": (
                sensor_df["response_time_ms"].mean()
            ),

            "total_recorded_missed_messages": (
                sensor_df["missed_messages"].sum()
            ),

            "minimum_rssi_dbm": sensor_df["rssi_dbm"].min(),

            "maximum_response_time_ms": (
                sensor_df["response_time_ms"].max()
            )

        })

    return pd.DataFrame(rows)


# ============================================================
# 9. EXTRACT NON-HEALTHY EVENTS
# ============================================================

def create_events(status_df):

    events = status_df[
        status_df["communication_status"] != "HEALTHY"
    ].copy()

    event_columns = [

        "cycle",
        "sensor_id",

        "communication_status",
        "communication_severity",

        "communication_reason",
        "threshold_flags",

        "warning_indicator_count",
        "critical_indicator_count",

        "record_received",
        "missing_cycle_record",

        "rssi_dbm",
        "response_time_ms",
        "missed_messages"

    ]

    return events[event_columns].reset_index(drop=True)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(status_df, summary_df, events_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    status_df.to_csv(STATUS_FILE, index=False)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    events_df.to_csv(EVENTS_FILE, index=False)

    print("\nCOMMUNICATION MONITORING COMPLETED")
    print("=" * 60)

    print("\nSensor-wise summary:")
    print(summary_df.to_string(index=False))

    print("\nCommunication status:")
    print(
        status_df["communication_status"]
        .value_counts()
        .to_string()
    )

    print("\nSeverity summary:")
    print(
        status_df["communication_severity"]
        .value_counts()
        .to_string()
    )

    print("\nGenerated files:")

    print(STATUS_FILE)
    print(SUMMARY_FILE)
    print(EVENTS_FILE)

    print(
        "\nIMPORTANT: ISOLATED means a sensor-cycle record "
        "is missing from the observed cycle grid. "
        "It does not prove physical sensor failure."
    )

    print(
        "\nThresholds are provisional and require "
        "hardware calibration."
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    try:

        thresholds = load_thresholds()

        df = load_data()

        aligned, sensor_ids = create_cycle_grid(df)

        status_df = classify_communication(
            aligned,
            sensor_ids,
            thresholds
        )

        summary_df = create_summary(
            status_df,
            sensor_ids
        )

        events_df = create_events(status_df)

        save_results(
            status_df,
            summary_df,
            events_df
        )

    except Exception as error:

        print(f"\nERROR: {error}")

        sys.exit(1)


if __name__ == "__main__":
    main()