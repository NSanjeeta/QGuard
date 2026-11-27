"""
QGUARD - PRESENCE AND TARGET-ACTIVITY MONITORING
================================================

Run:
    python qguard_motion_monitor.py

Input:
    data/processed/qguard_features.csv

Outputs:
    data/processed/qguard_motion_status.csv
    data/processed/qguard_motion_summary.csv
    data/processed/qguard_motion_events.csv

Important:
- Monitors binary presence_state (0/1).
- Does not classify moving vs stationary targets.
- No presence does not mean sensor failure.
- Presence-distance inconsistencies are flagged.
- Each sensor is processed independently.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qguard_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

STATUS_FILE = OUTPUT_DIR / "qguard_motion_status.csv"
SUMMARY_FILE = OUTPUT_DIR / "qguard_motion_summary.csv"
EVENTS_FILE = OUTPUT_DIR / "qguard_motion_events.csv"

EXPECTED_SENSOR_IDS = {"S1", "S2", "S3", "S4", "S5"}

REQUIRED_COLUMNS = [
    "cycle",
    "timestamp",
    "sensor_id",
    "presence_state",
    "target_distance_m",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nFeature dataset not found:\n{INPUT_FILE}\n"
            "Run qguard_feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    df["sensor_id"] = (
        df["sensor_id"]
        .astype("string")
        .str.strip()
    )

    df["cycle"] = pd.to_numeric(
        df["cycle"],
        errors="coerce"
    )

    df["presence_state"] = pd.to_numeric(
        df["presence_state"],
        errors="coerce"
    )

    df["target_distance_m"] = pd.to_numeric(
        df["target_distance_m"],
        errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["cycle"].isna().any():
        raise ValueError(
            "Missing or invalid cycle values found."
        )

    if df["sensor_id"].isna().any():
        raise ValueError(
            "Missing sensor IDs found."
        )

    if df.duplicated(
        ["sensor_id", "cycle"]
    ).any():

        raise ValueError(
            "Duplicate sensor-cycle records found."
        )

    invalid_presence = (
        ~df["presence_state"].isin([0, 1])
        & df["presence_state"].notna()
    )

    if invalid_presence.any():
        raise ValueError(
            f"{int(invalid_presence.sum())} invalid "
            "presence_state values found. "
            "Expected 0 or 1."
        )

    if (
        df["target_distance_m"] < 0
    ).fillna(False).any():

        raise ValueError(
            "Negative target distances found."
        )

    print("\nQGUARD PRESENCE MONITORING")
    print("=" * 60)

    print(f"Input file: {INPUT_FILE}")
    print(f"Records loaded: {len(df)}")

    print(
        "Sensors found:",
        sorted(df["sensor_id"].unique())
    )

    return df


# ============================================================
# 3. PROCESS ONE SENSOR
# ============================================================

def process_one_sensor(sensor_df):

    sensor_df = sensor_df.sort_values(
        ["cycle", "timestamp"],
        kind="stable",
        na_position="last"
    ).copy()

    # --------------------------------------------------------
    # Previous recorded states
    # --------------------------------------------------------

    previous_presence = (
        sensor_df["presence_state"].shift(1)
    )

    previous_distance = (
        sensor_df["target_distance_m"].shift(1)
    )

    previous_cycle = (
        sensor_df["cycle"].shift(1)
    )

    # --------------------------------------------------------
    # Presence transitions
    # --------------------------------------------------------

    sensor_df["presence_transition"] = (
        sensor_df["presence_state"].notna()
        & previous_presence.notna()
        & sensor_df["presence_state"].ne(
            previous_presence
        )
    ).astype(int)

    sensor_df["presence_event"] = "UNKNOWN"

    sensor_df.loc[
        sensor_df["presence_state"] == 0,
        "presence_event"
    ] = "NO_PRESENCE"

    sensor_df.loc[
        sensor_df["presence_state"] == 1,
        "presence_event"
    ] = "PRESENCE_DETECTED"

    # --------------------------------------------------------
    # Transition descriptions
    # --------------------------------------------------------

    sensor_df["presence_transition_type"] = (
        "NO_CHANGE_OR_FIRST_RECORD"
    )

    sensor_df.loc[
        (previous_presence == 0)
        & (sensor_df["presence_state"] == 1),
        "presence_transition_type"
    ] = "PRESENCE_STARTED"

    sensor_df.loc[
        (previous_presence == 1)
        & (sensor_df["presence_state"] == 0),
        "presence_transition_type"
    ] = "PRESENCE_ENDED"

    sensor_df.loc[
        sensor_df["presence_state"].isna(),
        "presence_transition_type"
    ] = "UNKNOWN_STATE"

    # --------------------------------------------------------
    # Distance availability
    # --------------------------------------------------------

    sensor_df["distance_available"] = (
        sensor_df["target_distance_m"]
        .notna()
        .astype(int)
    )

    # --------------------------------------------------------
    # Presence-distance consistency
    # --------------------------------------------------------

    sensor_df["presence_distance_inconsistency"] = 0

    no_presence_but_distance = (
        (sensor_df["presence_state"] == 0)
        & sensor_df["target_distance_m"].notna()
    )

    presence_but_no_distance = (
        (sensor_df["presence_state"] == 1)
        & sensor_df["target_distance_m"].isna()
    )

    inconsistent_records = (
        no_presence_but_distance
        | presence_but_no_distance
    )

    sensor_df.loc[
        inconsistent_records,
        "presence_distance_inconsistency"
    ] = 1

    # --------------------------------------------------------
    # Check whether consecutive cycles are adjacent
    # --------------------------------------------------------

    consecutive_cycle = (
        sensor_df["cycle"] - previous_cycle == 1
    )

    # --------------------------------------------------------
    # Target distance change
    # --------------------------------------------------------

    sensor_df["target_distance_change_m"] = (
        sensor_df["target_distance_m"]
        - previous_distance
    )

    valid_distance_pair = (
        (sensor_df["presence_state"] == 1)
        & (previous_presence == 1)
        & sensor_df["target_distance_m"].notna()
        & previous_distance.notna()
        & consecutive_cycle
    )

    sensor_df.loc[
        ~valid_distance_pair,
        "target_distance_change_m"
    ] = pd.NA

    sensor_df["absolute_distance_change_m"] = (
        sensor_df["target_distance_change_m"].abs()
    )

    # --------------------------------------------------------
    # Cumulative presence detections
    # --------------------------------------------------------

    sensor_df["cumulative_presence_detections"] = (
        (sensor_df["presence_state"] == 1)
        .astype(int)
        .cumsum()
    )

    # --------------------------------------------------------
    # Rolling presence count
    # Current record + previous 4 records
    # --------------------------------------------------------

    sensor_df["presence_count_rolling_5"] = (
        (sensor_df["presence_state"] == 1)
        .astype(int)
        .rolling(
            window=5,
            min_periods=1
        )
        .sum()
    )

    return sensor_df


# ============================================================
# 4. PROCESS ALL SENSORS
# ============================================================

def monitor_all_sensors(df):

    sensor_ids = sorted(
        df["sensor_id"].unique().tolist()
    )

    missing_expected = (
        EXPECTED_SENSOR_IDS - set(sensor_ids)
    )

    if missing_expected:
        print(
            "\nWARNING: Expected sensors missing:",
            sorted(missing_expected)
        )

    frames = []

    for sensor_id, sensor_df in df.groupby(
        "sensor_id",
        sort=True
    ):

        processed = process_one_sensor(
            sensor_df
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
# 5. SENSOR-WISE SUMMARY
# ============================================================

def create_summary(status_df):

    rows = []

    for sensor_id, sensor_df in status_df.groupby(
        "sensor_id",
        sort=True
    ):

        valid_presence = (
            sensor_df["presence_state"].dropna()
        )

        valid_distance = sensor_df.loc[
            sensor_df["presence_state"] == 1,
            "target_distance_m"
        ].dropna()

        presence_count = int(
            (sensor_df["presence_state"] == 1).sum()
        )

        no_presence_count = int(
            (sensor_df["presence_state"] == 0).sum()
        )

        unknown_count = int(
            sensor_df["presence_state"].isna().sum()
        )

        transition_count = int(
            sensor_df["presence_transition"].sum()
        )

        rows.append({

            "sensor_id": sensor_id,

            "records_checked": len(sensor_df),

            "presence_detections": presence_count,

            "no_presence_records": no_presence_count,

            "unknown_presence_records": unknown_count,

            "presence_detection_percent": (
                presence_count / len(valid_presence) * 100
                if len(valid_presence) > 0
                else float("nan")
            ),

            "presence_started_events": int(
                (
                    sensor_df["presence_transition_type"]
                    == "PRESENCE_STARTED"
                ).sum()
            ),

            "presence_ended_events": int(
                (
                    sensor_df["presence_transition_type"]
                    == "PRESENCE_ENDED"
                ).sum()
            ),

            "total_presence_transitions": transition_count,

            "distance_available_records": int(
                sensor_df["distance_available"].sum()
            ),

            "mean_detected_target_distance_m": (
                valid_distance.mean()
                if len(valid_distance) > 0
                else float("nan")
            ),

            "minimum_detected_target_distance_m": (
                valid_distance.min()
                if len(valid_distance) > 0
                else float("nan")
            ),

            "maximum_detected_target_distance_m": (
                valid_distance.max()
                if len(valid_distance) > 0
                else float("nan")
            ),

            "presence_distance_inconsistencies": int(
                sensor_df[
                    "presence_distance_inconsistency"
                ].sum()
            ),

        })

    return pd.DataFrame(rows)


# ============================================================
# 6. EXTRACT EVENTS
# ============================================================

def create_events(status_df):

    event_mask = (
        status_df["presence_transition"].eq(1)
        | status_df[
            "presence_distance_inconsistency"
        ].eq(1)
    )

    columns = [

        "cycle",
        "timestamp",
        "sensor_id",

        "presence_state",
        "presence_event",
        "presence_transition_type",

        "target_distance_m",
        "target_distance_change_m",

        "presence_distance_inconsistency",

    ]

    events_df = status_df.loc[
        event_mask,
        columns
    ].copy()

    events_df = events_df.sort_values(
        ["cycle", "sensor_id"]
    ).reset_index(drop=True)

    return events_df


# ============================================================
# 7. SAVE OUTPUTS AND DISPLAY RESULTS
# ============================================================

def save_outputs(
    status_df,
    summary_df,
    events_df
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

    print("\nPRESENCE MONITORING COMPLETED")
    print("=" * 60)

    print("\nSENSOR-WISE SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nPRESENCE EVENT COUNTS")

    print(
        status_df["presence_event"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTOTAL PRESENCE TRANSITIONS")

    print(
        int(status_df["presence_transition"].sum())
    )

    print("\nPRESENCE-DISTANCE INCONSISTENCIES")

    print(
        int(
            status_df[
                "presence_distance_inconsistency"
            ].sum()
        )
    )

    print("\nOUTPUT FILES")

    print(f"- {STATUS_FILE}")
    print(f"- {SUMMARY_FILE}")
    print(f"- {EVENTS_FILE}")

    print("\nIMPORTANT INTERPRETATION")

    print(
        "This dataset supports binary presence monitoring only."
    )

    print(
        "Moving-versus-stationary classification requires "
        "a separate target-state field."
    )

    print(
        "No-presence records are NOT classified as sensor faults."
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    try:

        df = load_data()

        status_df = monitor_all_sensors(df)

        summary_df = create_summary(
            status_df
        )

        events_df = create_events(
            status_df
        )

        save_outputs(
            status_df,
            summary_df,
            events_df
        )

    except Exception as error:

        print(f"\nERROR: {error}")

        sys.exit(1)


if __name__ == "__main__":
    main()