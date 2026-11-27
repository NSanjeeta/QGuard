"""
QUANTUMGUARD - DATA PREPROCESSING AND VALIDATION

Run:
    python qguard_data_preprocessing.py

Input:
    data/QGuard_dataset.csv

Outputs:
    data/processed/quantumguard_cleaned.csv
    data/processed/sensor_wise_summary.csv
    data/processed/flagged_records.csv
    data/processed/validation_summary.txt

IMPORTANT:
- Original dataset is never overwritten.
- Missing target distances remain missing.
- Suspicious records are flagged, not deleted.
- Timestamp format: DD-MM-YYYY HH:MM.
- No feature engineering or ML training in this phase.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "data" / "QGuard_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_SENSOR_IDS = {"S1", "S2", "S3", "S4", "S5"}

EXPECTED_COLUMNS = [
    "cycle",
    "timestamp",
    "sensor_id",
    "sensor_name",
    "sensor_type",
    "zone_id",
    "reading_value",
    "presence_state",
    "target_distance_m",
    "packet_rate_ppm",
    "response_time_ms",
    "battery_percent",
    "rssi_dbm",
    "missed_messages"
]

NUMERIC_COLUMNS = [
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


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found!\n"
            f"Expected location:\n{INPUT_FILE}\n"
            "Check that QGuard_dataset.csv is inside data/."
        )

    df = pd.read_csv(INPUT_FILE)

    print("\n" + "=" * 60)
    print("QUANTUMGUARD DATASET LOADED")
    print("=" * 60)

    print(f"Input file : {INPUT_FILE}")
    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")

    return df


# ============================================================
# 3. VALIDATE DATASET STRUCTURE
# ============================================================

def validate_structure(df):

    issues = []

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        issues.append(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        issues.append(
            f"Unexpected columns: {unexpected_columns}"
        )

    return issues, missing_columns


# ============================================================
# 4. CONVERT DATA TYPES
# ============================================================

def clean_data_types(df):

    df = df.copy()

    # Clean text columns
    text_columns = [
        "sensor_id",
        "sensor_name",
        "sensor_type",
        "zone_id"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # Convert numerical columns
    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Explicit timestamp format:
    # 02-03-2026 00:00 = 2 March 2026

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # Convert discrete values to nullable integers
    for column in [
        "cycle",
        "presence_state",
        "missed_messages"
    ]:

        df[column] = df[column].astype("Int64")

    return df


# ============================================================
# 5. DATA QUALITY CHECKS
# ============================================================

def build_quality_report(df):

    checks = {}

    checks["row_count"] = int(len(df))

    checks["column_count"] = int(len(df.columns))

    checks["duplicate_full_rows"] = int(
        df.duplicated().sum()
    )

    checks["duplicate_sensor_cycle_pairs"] = int(
        df.duplicated(
            subset=["sensor_id", "cycle"]
        ).sum()
    )

    # Sensor validation

    sensor_ids = sorted(
        df["sensor_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    checks["sensor_ids_found"] = sensor_ids

    checks["unexpected_sensor_ids"] = sorted(
        set(sensor_ids) - EXPECTED_SENSOR_IDS
    )

    checks["records_per_sensor"] = (
        df["sensor_id"]
        .value_counts(dropna=False)
        .to_dict()
    )

    # Cycle validation

    checks["minimum_cycle"] = (
        int(df["cycle"].min())
        if df["cycle"].notna().any()
        else None
    )

    checks["maximum_cycle"] = (
        int(df["cycle"].max())
        if df["cycle"].notna().any()
        else None
    )

    checks["missing_cycle_values"] = int(
        df["cycle"].isna().sum()
    )

    # Timestamp validation

    checks["missing_or_invalid_timestamps"] = int(
        df["timestamp"].isna().sum()
    )

    checks["unique_timestamps"] = int(
        df["timestamp"].nunique()
    )

    # Missing values

    checks["missing_values_by_column"] = {
        column: int(count)
        for column, count in df.isna().sum().items()
    }

    # Presence validation

    checks["invalid_presence_states"] = int(
        (
            ~df["presence_state"].isin([0, 1])
            & df["presence_state"].notna()
        ).sum()
    )

    checks["presence_state_counts"] = (
        df["presence_state"]
        .value_counts(dropna=False)
        .to_dict()
    )

    # Battery validation

    checks["battery_out_of_range"] = int(
        (
            (df["battery_percent"] < 0)
            | (df["battery_percent"] > 100)
        ).fillna(False).sum()
    )

    # Communication validation

    checks["negative_response_times"] = int(
        (df["response_time_ms"] < 0)
        .fillna(False).sum()
    )

    checks["negative_packet_rate_values"] = int(
        (df["packet_rate_ppm"] < 0)
        .fillna(False).sum()
    )

    checks["negative_missed_message_counts"] = int(
        (df["missed_messages"] < 0)
        .fillna(False).sum()
    )

    # Distance validation

    checks["negative_target_distances"] = int(
        (df["target_distance_m"] < 0)
        .fillna(False).sum()
    )

    # Presence-distance consistency

    checks["no_presence_and_missing_distance"] = int(
        (
            (df["presence_state"] == 0)
            & df["target_distance_m"].isna()
        ).sum()
    )

    checks["presence_but_missing_distance"] = int(
        (
            (df["presence_state"] == 1)
            & df["target_distance_m"].isna()
        ).sum()
    )

    checks["no_presence_but_distance_recorded"] = int(
        (
            (df["presence_state"] == 0)
            & df["target_distance_m"].notna()
        ).sum()
    )

    checks["presence_with_distance_recorded"] = int(
        (
            (df["presence_state"] == 1)
            & df["target_distance_m"].notna()
        ).sum()
    )

    checks["maximum_target_distance_m"] = (
        float(df["target_distance_m"].max())
        if df["target_distance_m"].notna().any()
        else None
    )

    return checks


# ============================================================
# 6. FLAG SUSPICIOUS RECORDS
# ============================================================

def flag_records(df):

    # Preserve original values in flagged output
    flagged = df.copy()

    # Each row gets its own list of reasons
    reasons = pd.Series(
        "",
        index=df.index,
        dtype="string"
    )

    def add_flag(mask, message):

        mask = pd.Series(
            mask,
            index=df.index
        ).fillna(False)

        existing = reasons.loc[mask]

        reasons.loc[mask] = existing.apply(
            lambda old: (
                f"{old}; {message}"
                if old else message
            )
        )

    # Battery checks

    add_flag(
        (df["battery_percent"] < 0)
        | (df["battery_percent"] > 100),
        "Battery percentage outside 0-100 range"
    )

    # Communication checks

    add_flag(
        df["response_time_ms"] < 0,
        "Negative response time"
    )

    add_flag(
        df["packet_rate_ppm"] < 0,
        "Negative packet-rate value"
    )

    add_flag(
        df["missed_messages"] < 0,
        "Negative missed-message count"
    )

    # Distance checks

    add_flag(
        df["target_distance_m"] < 0,
        "Negative target distance"
    )

    # Presence-distance consistency checks

    # Case 1:
    # Presence detected, but distance is missing

    presence_missing_distance = (
        (df["presence_state"] == 1)
        & df["target_distance_m"].isna()
    )

    add_flag(
        presence_missing_distance,
        "Presence reported but target distance is missing"
    )

    # Case 2:
    # No presence detected, but distance is recorded

    no_presence_with_distance = (
        (df["presence_state"] == 0)
        & df["target_distance_m"].notna()
    )

    add_flag(
        no_presence_with_distance,
        "No presence reported but target distance is recorded"
    )

    # Invalid presence state

    add_flag(
        ~df["presence_state"].isin([0, 1])
        & df["presence_state"].notna(),
        "Invalid presence state"
    )

    # Attach flag reasons

    flagged["flag_reasons"] = reasons

    # Keep only suspicious records
    flagged = flagged[
        flagged["flag_reasons"] != ""
    ].copy()

    return flagged


# ============================================================
# 7. SENSOR-WISE SUMMARY
# ============================================================

def create_sensor_summary(df):

    numeric_columns = [
        "reading_value",
        "presence_state",
        "target_distance_m",
        "packet_rate_ppm",
        "response_time_ms",
        "battery_percent",
        "rssi_dbm",
        "missed_messages"
    ]

    summary = (
        df.groupby("sensor_id", dropna=False)[numeric_columns]
        .agg(["count", "mean", "min", "max", "std"])
    )

    # Flatten column names

    summary.columns = [
        f"{column}_{statistic}"
        for column, statistic in summary.columns
    ]

    summary = summary.reset_index()

    # Presence detection count

    detections = (
        df.assign(
            presence_detected=(
                df["presence_state"] == 1
            ).astype(int)
        )
        .groupby("sensor_id", dropna=False)[
            "presence_detected"
        ]
        .sum()
        .reset_index(
            name="presence_detection_records"
        )
    )

    summary = summary.merge(
        detections,
        on="sensor_id",
        how="left"
    )

    return summary


# ============================================================
# 8. WRITE VALIDATION REPORT
# ============================================================

def write_validation_report(
    df,
    checks,
    structural_issues,
    flagged
):

    report_path = (
        OUTPUT_DIR / "validation_summary.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "QUANTUMGUARD - DATA VALIDATION REPORT\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(f"Input file: {INPUT_FILE}\n")
        file.write(f"Rows: {len(df)}\n")
        file.write(f"Columns: {len(df.columns)}\n\n")

        # Structural validation

        file.write("STRUCTURAL VALIDATION\n")
        file.write("-" * 60 + "\n")

        if structural_issues:

            for issue in structural_issues:
                file.write(f"ISSUE: {issue}\n")

        else:

            file.write(
                "All expected columns are present.\n"
            )

        # Data quality checks

        file.write("\nDATA QUALITY CHECKS\n")
        file.write("-" * 60 + "\n")

        for key, value in checks.items():

            file.write(
                f"{key}: {value}\n"
            )

        # Flagged records

        file.write("\nFLAGGED RECORDS\n")
        file.write("-" * 60 + "\n")

        file.write(
            f"Flagged record count: {len(flagged)}\n"
        )

        file.write(
            "Flagged records are retained for review. "
            "They are not automatically deleted.\n"
        )

        # Decisions

        file.write("\nPREPROCESSING DECISIONS\n")
        file.write("-" * 60 + "\n")

        file.write(
            "1. Original dataset is not overwritten.\n"
        )

        file.write(
            "2. Missing target distances are preserved.\n"
        )

        file.write(
            "3. reading_value is retained without assuming its unit.\n"
        )

        file.write(
            "4. Suspicious records are flagged, not deleted.\n"
        )

        file.write(
            "5. No feature scaling, imputation or ML training "
            "is performed in this phase.\n"
        )

        file.write(
            "6. Timestamp format used: DD-MM-YYYY HH:MM.\n"
        )

    return report_path


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load original dataset

    df = load_dataset()

    # Validate structure

    structural_issues, missing_columns = (
        validate_structure(df)
    )

    if structural_issues:

        print("\nSTRUCTURAL VALIDATION MESSAGES:")

        for issue in structural_issues:
            print(f"- {issue}")

    if missing_columns:

        raise ValueError(
            "Required columns are missing. "
            "Preprocessing stopped."
        )

    # Convert data types

    df = clean_data_types(df)

    # Stop if critical parsing failures occur.
    # Do not silently continue with invalid cycle/timestamp data.

    if df["cycle"].isna().any():

        raise ValueError(
            "Invalid or missing cycle values found. "
            "Please inspect the original dataset."
        )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Timestamp parsing failed. "
            "Expected DD-MM-YYYY HH:MM."
        )

    # Sort records

    df = df.sort_values(
        by=["timestamp", "cycle", "sensor_id"],
        kind="stable",
        na_position="last"
    ).reset_index(drop=True)

    # Generate validation outputs

    checks = build_quality_report(df)

    flagged = flag_records(df)

    sensor_summary = create_sensor_summary(df)

    # Output paths

    cleaned_path = (
        OUTPUT_DIR / "quantumguard_cleaned.csv"
    )

    summary_path = (
        OUTPUT_DIR / "sensor_wise_summary.csv"
    )

    flagged_path = (
        OUTPUT_DIR / "flagged_records.csv"
    )

    # Save outputs

    df.to_csv(
        cleaned_path,
        index=False
    )

    sensor_summary.to_csv(
        summary_path,
        index=False
    )

    flagged.to_csv(
        flagged_path,
        index=False
    )

    # Write report

    report_path = write_validation_report(
        df=df,
        checks=checks,
        structural_issues=structural_issues,
        flagged=flagged
    )

    # Final terminal output

    print("\n" + "=" * 60)

    print(
        "PREPROCESSING AND VALIDATION COMPLETED"
    )

    print("=" * 60)

    print(
        f"Rows retained                 : {len(df)}"
    )

    print(
        f"Flagged records for review    : {len(flagged)}"
    )

    print(
        f"Missing target distances      : "
        f"{df['target_distance_m'].isna().sum()}"
    )

    print(
        f"Maximum target distance       : "
        f"{df['target_distance_m'].max()} m"
    )

    print(
        f"Presence-distance mismatches  : "
        f"{checks['no_presence_but_distance_recorded']}"
    )

    print("\nOUTPUT FILES")
    print("-" * 60)

    print(f"Cleaned dataset : {cleaned_path}")
    print(f"Sensor summary  : {summary_path}")
    print(f"Flagged records : {flagged_path}")
    print(f"Validation      : {report_path}")

    print("\nOriginal dataset has NOT been modified.")

    print(
        "\nNext: review flagged_records.csv "
        "before feature engineering."
    )


# ============================================================
# 10. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()