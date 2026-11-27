"""
QGUARD - INTEGRATED CYCLE-WISE FAULT DETECTION + TEMPORAL ANALYSIS
==================================================================

Run from the QuantumGuard project root:
    python qguard_fault_detection.py

Inputs:
    data/processed/qguard_features.csv
    data/processed/qguard_fault_detection_status.csv
        (used only as the existing communication-status source)
    data/processed/qguard_battery_status.csv
    data/processed/qguard_motion_status.csv

Outputs:
    qguard_fault_detection_status.csv
    qguard_fault_detection_events.csv
    qguard_fault_detection_summary.csv
    qguard_fault_detection_network.csv

Important:
- Every sensor-cycle is analysed independently.
- Communication, battery and data-quality faults remain separate flags.
- Consecutive degraded cycles are grouped into events.
- A degraded run shorter than PERSISTENCE_CYCLES is transient.
- A run at least PERSISTENCE_CYCLES long is persistent.
- The first healthy cycle after a degraded run is marked RECOVERED.
- Repeated separate transient events are reported as recurring/intermittent
  at sensor-summary level; they are not automatically called persistent.
- Missing expected communication records are marked ISOLATED only under the
  assumption that every sensor is expected to report once per cycle.
- This is rule-based analysis. Thresholds and labels need hardware validation.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_FILE = DATA_DIR / "qguard_features.csv"
COMMUNICATION_SOURCE_FILE = DATA_DIR / "qguard_fault_detection_status.csv"
BATTERY_FILE = DATA_DIR / "qguard_battery_status.csv"
MOTION_FILE = DATA_DIR / "qguard_motion_status.csv"

STATUS_FILE = DATA_DIR / "qguard_fault_detection_status.csv"
EVENTS_FILE = DATA_DIR / "qguard_fault_detection_events.csv"
SUMMARY_FILE = DATA_DIR / "qguard_fault_detection_summary.csv"
NETWORK_FILE = DATA_DIR / "qguard_fault_detection_network.csv"

EXPECTED_SENSOR_IDS = ["S1", "S2", "S3", "S4", "S5"]
PERSISTENCE_CYCLES = 3
RECURRENCE_EVENT_COUNT = 3

KEYS = ["sensor_id", "cycle"]


def read_csv(path, required):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{path}\n"
            "Run the relevant preprocessing/monitoring script first."
        )
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    df = df.copy()
    df["sensor_id"] = df["sensor_id"].astype("string").str.strip()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    if df[KEYS].isna().any().any():
        raise ValueError(f"Missing sensor_id/cycle key in {path.name}")
    if df.duplicated(KEYS).any():
        raise ValueError(f"Duplicate sensor-cycle keys in {path.name}")
    return df


def load_inputs():
    features = read_csv(
        FEATURE_FILE,
        ["sensor_id", "cycle", "timestamp", "presence_state",
         "target_distance_m", "battery_percent", "rssi_dbm",
         "response_time_ms", "missed_messages"]
    )

    # This file is the previously generated integrated output. We reuse only
    # its communication monitor result, never its old battery/data/overall flags.
    comm = read_csv(
        COMMUNICATION_SOURCE_FILE,
        ["sensor_id", "cycle", "communication_status"]
    )
    comm_cols = ["sensor_id", "cycle", "communication_status"]
    if "communication_reason" in comm.columns:
        comm_cols.append("communication_reason")
    if "record_received" in comm.columns:
        comm_cols.append("record_received")
    comm = comm[comm_cols].copy()

    battery = read_csv(
        BATTERY_FILE,
        ["sensor_id", "cycle", "battery_status", "battery_percent"]
    )
    battery_cols = ["sensor_id", "cycle", "battery_status", "battery_percent"]
    if "battery_status_reason" in battery.columns:
        battery_cols.append("battery_status_reason")
    battery = battery[battery_cols].copy().rename(
        columns={"battery_percent": "battery_percent_from_monitor"}
    )

    motion = read_csv(
        MOTION_FILE,
        ["sensor_id", "cycle", "presence_distance_inconsistency"]
    )
    motion_cols = ["sensor_id", "cycle", "presence_distance_inconsistency"]
    motion = motion[motion_cols].copy().rename(
        columns={
            "presence_distance_inconsistency":
                "motion_presence_distance_inconsistency"
        }
    )

    return features, comm, battery, motion


def merge_sources(features, comm, battery, motion):
    # Use the feature dataset as the row backbone, then left-join monitor outputs.
    df = features.merge(comm, on=KEYS, how="left", validate="one_to_one")
    df = df.merge(battery, on=KEYS, how="left", validate="one_to_one")
    df = df.merge(motion, on=KEYS, how="left", validate="one_to_one")

    # The feature dataset may already contain a column with the same
    # meaning. Use the motion-monitor value explicitly for the diagnostic
    # flag and keep the feature-engineering column unchanged for traceability.
    df["communication_record_available"] = df["communication_status"].notna().astype(int)
    df["battery_record_available"] = df["battery_status"].notna().astype(int)
    df["motion_record_available"] = df["presence_distance_inconsistency"].notna().astype(int)

    # Explicitly create expected sensor-cycle grid to expose missing feature rows.
    cycles = sorted(features["cycle"].dropna().unique().tolist())
    grid = pd.MultiIndex.from_product(
        [EXPECTED_SENSOR_IDS, cycles], names=KEYS
    ).to_frame(index=False)

    grid = grid.merge(
        df, on=KEYS, how="left", indicator="feature_merge",
        validate="one_to_one"
    )
    grid["feature_record_available"] = (
        grid["feature_merge"] == "both"
    ).astype(int)
    grid = grid.drop(columns=["feature_merge"])

    # Missing monitor rows are unknown, not silently normal.
    grid["communication_record_available"] = (
        grid["communication_record_available"].fillna(0).astype(int)
    )
    grid["battery_record_available"] = (
        grid["battery_record_available"].fillna(0).astype(int)
    )
    grid["motion_record_available"] = (
        grid["motion_record_available"].fillna(0).astype(int)
    )

    return grid.sort_values(KEYS).reset_index(drop=True)


def normalise_and_flag(df):
    df = df.copy()

    comm = df["communication_status"].astype("string").str.upper().str.strip()
    comm_missing = df["communication_record_available"].eq(0)
    df["communication_status"] = comm.fillna("UNKNOWN")

    # Existing communication monitor uses HEALTHY / DEGRADED / ISOLATED.
    df["communication_degraded_flag"] = (
        df["communication_status"].isin(["DEGRADED", "WARNING", "CRITICAL"])
    ).astype(int)
    df["communication_isolated_flag"] = (
        comm_missing | df["communication_status"].eq("ISOLATED")
    ).astype(int)

    battery = df["battery_status"].astype("string").str.upper().str.strip()
    df["battery_status"] = battery.fillna("UNKNOWN")
    df["battery_fault_flag"] = (
        df["battery_status"].isin(
            ["WARNING", "LOW", "CRITICAL", "RAPID_DROP"]
        )
    ).astype(int)
    df["battery_unknown_flag"] = (
        df["battery_record_available"].eq(0)
        | df["battery_status"].eq("UNKNOWN")
    ).astype(int)

    inconsistency = pd.to_numeric(
        df["motion_presence_distance_inconsistency"], errors="coerce"
    )
    df["data_inconsistency_flag"] = inconsistency.fillna(0).gt(0).astype(int)
    df["data_unknown_flag"] = df["motion_record_available"].eq(0).astype(int)

    df["communication_temporal_status"] = "HEALTHY"
    df["communication_event_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["consecutive_degraded_cycles"] = 0
    df["communication_recovery_flag"] = 0

    # Analyse contiguous runs separately for each sensor.
    for sensor_id, idx in df.groupby("sensor_id", sort=False).groups.items():
        ordered_idx = df.loc[idx].sort_values("cycle").index.tolist()
        run_positions = []
        event_number = 0

        def finish_run(positions):
            nonlocal event_number
            if not positions:
                return
            event_number += 1
            run_len = len(positions)
            event_id = f"{sensor_id}_COMM_{event_number:02d}"
            state = (
                "DEGRADED_PERSISTENT"
                if run_len >= PERSISTENCE_CYCLES
                else "DEGRADED_TRANSIENT"
            )
            for n, row_idx in enumerate(positions, start=1):
                df.at[row_idx, "communication_temporal_status"] = state
                df.at[row_idx, "communication_event_id"] = event_id
                df.at[row_idx, "consecutive_degraded_cycles"] = n

        for row_idx in ordered_idx:
            is_degraded = (
                df.at[row_idx, "communication_degraded_flag"] == 1
                or df.at[row_idx, "communication_isolated_flag"] == 1
            )
            if is_degraded:
                run_positions.append(row_idx)
            else:
                if run_positions:
                    finish_run(run_positions)
                    # First healthy cycle after an event is a recovery transition.
                    df.at[row_idx, "communication_recovery_flag"] = 1
                    df.at[row_idx, "communication_temporal_status"] = "RECOVERED"
                    run_positions = []
        if run_positions:
            finish_run(run_positions)

    # A missing monitor output remains explicitly unknown.
    df.loc[
        df["communication_record_available"].eq(0)
        & df["feature_record_available"].eq(1),
        "communication_temporal_status"
    ] = "COMMUNICATION_STATUS_UNKNOWN"

    # Independent diagnostic labels. A flag on one dimension never overwrites another.
    df["communication_fault"] = "HEALTHY"
    df.loc[df["communication_degraded_flag"].eq(1), "communication_fault"] = "DEGRADED"
    df.loc[df["communication_isolated_flag"].eq(1), "communication_fault"] = "ISOLATED"
    df.loc[df["communication_record_available"].eq(0), "communication_fault"] = "UNKNOWN"

    df["battery_fault"] = "NORMAL"
    df.loc[df["battery_fault_flag"].eq(1), "battery_fault"] = "WARNING"
    df.loc[df["battery_status"].eq("CRITICAL"), "battery_fault"] = "CRITICAL"
    df.loc[df["battery_unknown_flag"].eq(1), "battery_fault"] = "UNKNOWN"

    df["data_fault"] = "NORMAL"
    df.loc[df["data_inconsistency_flag"].eq(1), "data_fault"] = "INCONSISTENCY"
    df.loc[df["data_unknown_flag"].eq(1), "data_fault"] = "UNKNOWN"

    df["any_fault_flag"] = (
        df["communication_degraded_flag"].eq(1)
        | df["communication_isolated_flag"].eq(1)
        | df["battery_fault_flag"].eq(1)
        | df["data_inconsistency_flag"].eq(1)
    ).astype(int)

    def overall(row):
        active = []
        if row["communication_isolated_flag"]:
            active.append("COMMUNICATION_ISOLATED")
        elif row["communication_degraded_flag"]:
            active.append("COMMUNICATION_DEGRADED")
        if row["battery_fault_flag"]:
            active.append("BATTERY_PROBLEM")
        if row["data_inconsistency_flag"]:
            active.append("DATA_INCONSISTENCY")
        if active:
            return " + ".join(active)
        if (
            row["communication_record_available"] == 0
            or row["battery_unknown_flag"] == 1
            or row["data_unknown_flag"] == 1
        ):
            return "INCOMPLETE_DIAGNOSTIC_DATA"
        return "HEALTHY"

    df["overall_sensor_status"] = df.apply(overall, axis=1)
    return df


def make_event_table(df):
    rows = []
    for sensor_id, group in df.groupby("sensor_id", sort=True):
        group = group.sort_values("cycle")
        degraded = (
            group["communication_degraded_flag"].eq(1)
            | group["communication_isolated_flag"].eq(1)
        )
        active = False
        run = []
        for idx, is_bad in zip(group.index, degraded.tolist()):
            if is_bad:
                run.append(idx)
                active = True
            elif active:
                part = df.loc[run]
                rows.append({
                    "sensor_id": sensor_id,
                    "event_id": part["communication_event_id"].iloc[0],
                    "event_type": "COMMUNICATION",
                    "start_cycle": int(part["cycle"].min()),
                    "end_cycle": int(part["cycle"].max()),
                    "duration_cycles": int(len(part)),
                    "temporal_classification": part["communication_temporal_status"].iloc[0],
                    "recovery_cycle": int(df.loc[idx, "cycle"]),
                    "recovered": 1,
                })
                run = []
                active = False
        if run:
            part = df.loc[run]
            rows.append({
                "sensor_id": sensor_id,
                "event_id": part["communication_event_id"].iloc[0],
                "event_type": "COMMUNICATION",
                "start_cycle": int(part["cycle"].min()),
                "end_cycle": int(part["cycle"].max()),
                "duration_cycles": int(len(part)),
                "temporal_classification": part["communication_temporal_status"].iloc[0],
                "recovery_cycle": pd.NA,
                "recovered": 0,
            })

    # Add battery and data issue runs/records, preserving their own categories.
    for sensor_id, group in df.groupby("sensor_id", sort=True):
        for category, flag_col, event_type in [
            ("BATTERY", "battery_fault_flag", "BATTERY"),
            ("DATA", "data_inconsistency_flag", "DATA_INCONSISTENCY"),
        ]:
            bad = group.loc[group[flag_col].eq(1)].sort_values("cycle")
            if bad.empty:
                continue
            # Group consecutive cycles into events.
            current = []
            previous = None
            event_num = 0
            for _, row in bad.iterrows():
                cycle = int(row["cycle"])
                if previous is not None and cycle != previous + 1:
                    event_num += 1
                    rows.append({
                        "sensor_id": sensor_id,
                        "event_id": f"{sensor_id}_{category}_{event_num:02d}",
                        "event_type": event_type,
                        "start_cycle": int(current[0]),
                        "end_cycle": int(current[-1]),
                        "duration_cycles": len(current),
                        "temporal_classification": "ACTIVE",
                        "recovery_cycle": pd.NA,
                        "recovered": 0,
                    })
                    current = []
                current.append(cycle)
                previous = cycle
            if current:
                event_num += 1
                rows.append({
                    "sensor_id": sensor_id,
                    "event_id": f"{sensor_id}_{category}_{event_num:02d}",
                    "event_type": event_type,
                    "start_cycle": int(current[0]),
                    "end_cycle": int(current[-1]),
                    "duration_cycles": len(current),
                    "temporal_classification": "ACTIVE",
                    "recovery_cycle": pd.NA,
                    "recovered": 0,
                })

    return pd.DataFrame(rows, columns=[
        "sensor_id", "event_id", "event_type", "start_cycle", "end_cycle",
        "duration_cycles", "temporal_classification", "recovery_cycle", "recovered"
    ]).sort_values(["sensor_id", "start_cycle", "event_type"]).reset_index(drop=True)


def make_summary(df, events):
    rows = []
    for sensor_id, group in df.groupby("sensor_id", sort=True):
        comm_events = events[
            (events["sensor_id"] == sensor_id)
            & (events["event_type"] == "COMMUNICATION")
        ] if not events.empty else pd.DataFrame()

        persistent = int(
            (comm_events["temporal_classification"] == "DEGRADED_PERSISTENT").sum()
        ) if not comm_events.empty else 0
        transient = int(
            (comm_events["temporal_classification"] == "DEGRADED_TRANSIENT").sum()
        ) if not comm_events.empty else 0
        comm_event_count = len(comm_events)

        if int(group["communication_isolated_flag"].sum()) > 0:
            comm_diagnosis = "COMMUNICATION_ISOLATION_DETECTED"
        elif persistent > 0:
            comm_diagnosis = "PERSISTENT_COMMUNICATION_DEGRADATION"
        elif comm_event_count >= RECURRENCE_EVENT_COUNT:
            comm_diagnosis = "RECURRENT_TRANSIENT_COMMUNICATION_ANOMALIES"
        elif transient > 0:
            comm_diagnosis = "TRANSIENT_COMMUNICATION_ANOMALIES"
        else:
            comm_diagnosis = "COMMUNICATION_HEALTHY"

        battery_cycles = group.loc[group["battery_fault_flag"].eq(1), "cycle"].astype(int).tolist()
        data_cycles = group.loc[group["data_inconsistency_flag"].eq(1), "cycle"].astype(int).tolist()
        comm_cycles = group.loc[
            group["communication_degraded_flag"].eq(1)
            | group["communication_isolated_flag"].eq(1),
            "cycle"
        ].astype(int).tolist()

        battery_diagnosis = (
            "BATTERY_PROBLEM"
            if battery_cycles else
            ("BATTERY_DATA_UNKNOWN" if group["battery_unknown_flag"].any() else "BATTERY_NORMAL")
        )
        data_diagnosis = (
            "DATA_INCONSISTENCY_DETECTED"
            if data_cycles else
            ("DATA_STATUS_UNKNOWN" if group["data_unknown_flag"].any() else "DATA_NORMAL")
        )

        diagnoses = []
        if comm_diagnosis != "COMMUNICATION_HEALTHY":
            diagnoses.append(comm_diagnosis)
        if battery_diagnosis == "BATTERY_PROBLEM":
            diagnoses.append(battery_diagnosis)
        if data_diagnosis == "DATA_INCONSISTENCY_DETECTED":
            diagnoses.append(data_diagnosis)

        if diagnoses:
            overall = "ABNORMAL: " + " | ".join(diagnoses)
        elif "UNKNOWN" in battery_diagnosis or "UNKNOWN" in data_diagnosis:
            overall = "REVIEW: INCOMPLETE MONITOR DATA"
        else:
            overall = "HEALTHY"

        rows.append({
            "sensor_id": sensor_id,
            "records_analysed": int(len(group)),
            "communication_degraded_cycles": int(group["communication_degraded_flag"].sum()),
            "communication_isolated_cycles": int(group["communication_isolated_flag"].sum()),
            "communication_event_count": int(comm_event_count),
            "transient_communication_events": int(transient),
            "persistent_communication_events": int(persistent),
            "longest_consecutive_degradation": int(group["consecutive_degraded_cycles"].max()),
            "communication_recovery_count": int(group["communication_recovery_flag"].sum()),
            "communication_diagnosis": comm_diagnosis,
            "battery_warning_cycles": int(group["battery_fault_flag"].sum()),
            "battery_affected_cycles": ", ".join(map(str, battery_cycles)),
            "battery_diagnosis": battery_diagnosis,
            "data_inconsistency_cycles": int(group["data_inconsistency_flag"].sum()),
            "data_affected_cycles": ", ".join(map(str, data_cycles)),
            "data_diagnosis": data_diagnosis,
            "communication_affected_cycles": ", ".join(map(str, comm_cycles)),
            "overall_sensor_diagnosis": overall,
        })
    return pd.DataFrame(rows)


def make_network_summary(df, summary):
    rows = []
    for cycle, group in df.groupby("cycle", sort=True):
        rows.append({
            "cycle": int(cycle),
            "sensors_expected": len(EXPECTED_SENSOR_IDS),
            "sensor_records_available": int(group["feature_record_available"].sum()),
            "sensors_with_communication_degradation": int(group["communication_degraded_flag"].sum()),
            "sensors_isolated": int(group["communication_isolated_flag"].sum()),
            "sensors_with_battery_problem": int(group["battery_fault_flag"].sum()),
            "sensors_with_data_inconsistency": int(group["data_inconsistency_flag"].sum()),
            "sensors_with_any_fault": int(group["any_fault_flag"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    try:
        print("\nQGUARD INTEGRATED FAULT DETECTION + TEMPORAL ANALYSIS")
        print("=" * 68)
        features, comm, battery, motion = load_inputs()
        print(f"Feature records: {len(features)}")
        print(f"Communication-source records: {len(comm)}")
        print(f"Battery records: {len(battery)}")
        print(f"Motion records: {len(motion)}")

        df = merge_sources(features, comm, battery, motion)
        df = normalise_and_flag(df)
        events = make_event_table(df)
        summary = make_summary(df, events)
        network = make_network_summary(df, summary)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(STATUS_FILE, index=False)
        events.to_csv(EVENTS_FILE, index=False)
        summary.to_csv(SUMMARY_FILE, index=False)
        network.to_csv(NETWORK_FILE, index=False)

        print("\nTEMPORAL SETTINGS")
        print(f"Persistent degradation threshold: {PERSISTENCE_CYCLES} consecutive cycles")
        print(f"Recurring transient threshold: {RECURRENCE_EVENT_COUNT} separate events")

        print("\nSENSOR-WISE FINAL DIAGNOSIS")
        display_cols = [
            "sensor_id", "communication_diagnosis", "communication_event_count",
            "transient_communication_events", "persistent_communication_events",
            "battery_diagnosis", "battery_warning_cycles",
            "data_diagnosis", "data_inconsistency_cycles",
            "overall_sensor_diagnosis"
        ]
        print(summary[display_cols].to_string(index=False))

        print("\nCYCLE-WISE CHECKS")
        print("Communication degraded:", int(df["communication_degraded_flag"].sum()))
        print("Battery warning/fault:", int(df["battery_fault_flag"].sum()))
        print("Data inconsistencies:", int(df["data_inconsistency_flag"].sum()))
        print("Communication recoveries:", int(df["communication_recovery_flag"].sum()))
        print("Missing feature records:", int((df["feature_record_available"] == 0).sum()))
        print("Missing communication monitor records:", int((df["communication_record_available"] == 0).sum()))
        print("Missing battery monitor records:", int((df["battery_record_available"] == 0).sum()))
        print("Missing motion monitor records:", int((df["motion_record_available"] == 0).sum()))

        print("\nOUTPUT FILES")
        print(STATUS_FILE)
        print(EVENTS_FILE)
        print(SUMMARY_FILE)
        print(NETWORK_FILE)

        print("\nNote: persistent/transient labels are based on configured cycle rules.")
        print("They are not proof of physical sensor failure; validate thresholds on hardware.")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
