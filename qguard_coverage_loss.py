"""
QUANTUMGUARD - ACCESSIBLE COVERAGE LOSS CALCULATION
Run from project root: python qguard_coverage_loss.py

Input:
  data/processed/qguard_fault_detection_status.csv

Outputs:
  data/processed/coverage_loss_by_cycle.csv
  data/processed/sensor_coverage_loss_summary.csv
  data/processed/coverage_loss_summary.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
INPUT = DATA / "qguard_fault_detection_status.csv"

AREA_W, AREA_H = 22.0, 22.0
RADIUS = 3.5
GRID = 0.10

# Final user-approved coordinates: do not round or change.
SENSORS = {
    "S1": (15.449747468305834, 15.449747468305834),
    "S2": (15.449747468305834, 5.550252531694167),
    "S3": (10.5, 10.5),
    "S4": (5.550252531694168, 15.449747468305834),
    "S5": (5.550252531694168, 5.550252531694167),
}
IDS = list(SENSORS)

def load_status():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run qguard_fault_detection.py first."
        )
    df = pd.read_csv(INPUT)
    required = [
        "sensor_id", "cycle", "communication_isolated_flag",
        "backend_isolation_flag", "uart_interruption_flag",
        "battery_fault_flag", "data_inconsistency_flag"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in fault status file: {missing}")

    df = df.copy()
    df["sensor_id"] = df["sensor_id"].astype(str).str.strip()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    if df[["sensor_id", "cycle"]].isna().any().any():
        raise ValueError("sensor_id/cycle contains missing values.")
    df["cycle"] = df["cycle"].astype(int)
    if df.duplicated(["sensor_id", "cycle"]).any():
        raise ValueError("Duplicate sensor_id + cycle rows found.")
    unknown = sorted(set(df.sensor_id) - set(IDS))
    if unknown:
        raise ValueError(f"Unexpected sensor IDs: {unknown}")

    # Complete the expected sensor-cycle grid. Missing diagnostic rows remain
    # UNKNOWN; they are never silently assumed to be healthy/available.
    cycles = sorted(df.cycle.unique())
    expected = pd.MultiIndex.from_product(
        [IDS, cycles], names=["sensor_id", "cycle"]
    ).to_frame(index=False)
    df = expected.merge(df, on=["sensor_id", "cycle"], how="left")
    flags = [
        "communication_isolated_flag", "backend_isolation_flag",
        "uart_interruption_flag", "battery_fault_flag",
        "data_inconsistency_flag"
    ]
    df["fault_record_missing"] = df[flags].isna().all(axis=1).astype(int)
    for col in flags:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df

def make_geometry():
    xs = np.arange(GRID / 2, AREA_W, GRID)
    ys = np.arange(GRID / 2, AREA_H, GRID)
    xx, yy = np.meshgrid(xs, ys)
    px, py = xx.ravel(), yy.ravel()
    masks = []
    for sid in IDS:
        sx, sy = SENSORS[sid]
        masks.append((px - sx) ** 2 + (py - sy) ** 2 <= RADIUS ** 2)
    return np.column_stack(masks), GRID ** 2

def main():
    print("\\nQUANTUMGUARD - ACCESSIBLE COVERAGE LOSS CALCULATION")
    print("=" * 60)
    df = load_status()
    masks, cell_area = make_geometry()

    # Baseline coverage with all five sensors healthy.
    baseline_count = masks.sum(axis=1)
    baseline_covered = baseline_count > 0
    baseline_area = float(baseline_covered.sum() * cell_area)
    total_area = AREA_W * AREA_H

    # Unique area contributed by each sensor when all other sensors are healthy.
    individual_loss = {}
    for i, sid in enumerate(IDS):
        remaining = baseline_count - masks[:, i].astype(int)
        individual_loss[sid] = float(
            (baseline_covered & (remaining == 0)).sum() * cell_area
        )

    cycle_rows = []
    unavailable_cycles = {sid: 0 for sid in IDS}
    reason_counts = {sid: {} for sid in IDS}

    for cycle, group in df.groupby("cycle", sort=True):
        group = group.set_index("sensor_id").reindex(IDS)
        data_unavailable, diagnostic_unknown, reasons = {}, {}, {}

        for sid in IDS:
            row = group.loc[sid]
            if int(row["fault_record_missing"]) == 1:
                diagnostic_unknown[sid] = True
                data_unavailable[sid] = False
                reasons[sid] = "DIAGNOSTIC_STATUS_UNKNOWN"
                continue

            diagnostic_unknown[sid] = False
            why = []
            if row["communication_isolated_flag"] == 1:
                why.append("COMMUNICATION_ISOLATED")
            if row["backend_isolation_flag"] == 1:
                why.append("BACKEND_ISOLATED")
            if row["uart_interruption_flag"] == 1:
                why.append("UART_INTERRUPTED")

            data_unavailable[sid] = bool(why)
            reasons[sid] = " + ".join(why) if why else "DATA_AVAILABLE"

            if why:
                unavailable_cycles[sid] += 1
                reason_counts[sid][reasons[sid]] = (
                    reason_counts[sid].get(reasons[sid], 0) + 1
                )

        # Only sensors with known-available data contribute to accessible
        # backend coverage. Unknown records are reported separately.
        available = [
            sid for sid in IDS
            if not data_unavailable[sid] and not diagnostic_unknown[sid]
        ]
        unknown_sensors = [sid for sid in IDS if diagnostic_unknown[sid]]
        unavailable_sensors = [sid for sid in IDS if data_unavailable[sid]]
        healthy_idx = [IDS.index(sid) for sid in available]
        if healthy_idx:
            after_count = masks[:, healthy_idx].sum(axis=1)
        else:
            after_count = np.zeros(len(baseline_count), dtype=int)

        after_covered = after_count > 0
        lost_cells = baseline_covered & ~after_covered
        lost_area = float(lost_cells.sum() * cell_area)
        after_area = float(after_covered.sum() * cell_area)
        uncovered_area = float((~after_covered).sum() * cell_area)

        if unknown_sensors:
            status = "DIAGNOSTIC_STATUS_INCOMPLETE"
        elif not available:
            status = "NO_SENSOR_DATA_AVAILABLE"
        elif lost_area > 0:
            status = "ACCESSIBLE_COVERAGE_LOSS"
        else:
            status = "BASELINE_ACCESSIBLE_COVERAGE_RETAINED"

        cycle_rows.append({
            "cycle": int(cycle),
            "data_available_sensor_count": len(available),
            "data_unavailable_sensor_count": len(unavailable_sensors),
            "diagnostic_unknown_sensor_count": len(unknown_sensors),
            "data_unavailable_sensors": ", ".join(unavailable_sensors),
            "diagnostic_unknown_sensors": ", ".join(unknown_sensors),
            "data_unavailability_reasons": " | ".join(
                f"{sid}: {reasons[sid]}" for sid in unavailable_sensors
            ),
            "diagnostic_unknown_reasons": " | ".join(
                f"{sid}: {reasons[sid]}" for sid in unknown_sensors
            ),
            "data_available_sensors": ", ".join(available),
            "baseline_covered_area_m2": round(baseline_area, 4),
            "covered_area_after_fault_m2": round(after_area, 4),
            "coverage_lost_area_m2": round(lost_area, 4),
            "uncovered_area_after_fault_m2": round(uncovered_area, 4),
            "coverage_retained_percent": round(
                after_area / baseline_area * 100 if baseline_area else 0, 2
            ),
            "coverage_status": status
        })

    sensor_rows = []
    for sid in IDS:
        x, y = SENSORS[sid]
        sensor_rows.append({
            "sensor_id": sid, "x_m": x, "y_m": y,
            "normal_radius_m": RADIUS,
            "nominal_circular_footprint_m2": round(np.pi * RADIUS ** 2, 4),
            "individual_unique_accessible_coverage_lost_if_data_unavailable_m2": round(
                individual_loss[sid], 4
            ),
            "data_unavailable_cycles": unavailable_cycles[sid],
            "data_unavailability_reason_counts": "; ".join(
                f"{k}: {v}" for k, v in reason_counts[sid].items()
            )
        })

    cycle_df = pd.DataFrame(cycle_rows)
    sensor_df = pd.DataFrame(sensor_rows)
    summary_df = pd.DataFrame([{
        "area_width_m": AREA_W, "area_height_m": AREA_H,
        "total_area_m2": round(total_area, 4),
        "normal_sensor_radius_m": RADIUS,
        "grid_resolution_m": GRID,
        "cycles_analysed": len(cycle_df),
        "baseline_covered_area_m2": round(baseline_area, 4),
        "baseline_coverage_percent": round(baseline_area / total_area * 100, 2),
        "cycles_with_accessible_coverage_loss": int(
            cycle_df.coverage_lost_area_m2.gt(0).sum()
        ),
        "cycles_with_incomplete_diagnostic_status": int(
            cycle_df.diagnostic_unknown_sensor_count.gt(0).sum()
        ),
        "average_coverage_lost_area_m2": round(
            cycle_df.coverage_lost_area_m2.mean(), 4
        ),
        "maximum_coverage_lost_area_m2": round(
            cycle_df.coverage_lost_area_m2.max(), 4
        ),
        "coverage_loss_method": "GRID_BASED_CIRCLE_UNION"
    }])

    DATA.mkdir(parents=True, exist_ok=True)
    cycle_df.to_csv(DATA / "coverage_loss_by_cycle.csv", index=False)
    sensor_df.to_csv(DATA / "sensor_coverage_loss_summary.csv", index=False)
    summary_df.to_csv(DATA / "coverage_loss_summary.csv", index=False)

    print(f"Area: {AREA_W} m x {AREA_H} m")
    print(f"Radius: {RADIUS} m | Grid: {GRID} m")
    print(f"Baseline covered area: {baseline_area:.4f} m^2")
    print(f"Baseline coverage: {baseline_area / total_area * 100:.2f}%")
    print("\\nSensor-wise results:")
    print(sensor_df.to_string(index=False))
    print("\\nCycle-wise coverage loss:")
    print(cycle_df[[
        "cycle", "data_unavailable_sensors", "diagnostic_unknown_sensors",
        "coverage_lost_area_m2", "uncovered_area_after_fault_m2",
        "coverage_retained_percent", "coverage_status"
    ]].to_string(index=False))
    print("\\nCreated:")
    print(DATA / "coverage_loss_by_cycle.csv")
    print(DATA / "sensor_coverage_loss_summary.csv")
    print(DATA / "coverage_loss_summary.csv")
    print("\\nInterpretation:")
    print("- Coverage loss means surveillance data is unavailable to the backend.")
    print("- Communication isolation, backend isolation, and UART interruption")
    print("  make that sensor's data unavailable for the affected cycles.")
    print("- Battery warnings and data inconsistencies alone do not remove coverage.")
    print("- Missing diagnostic records are marked UNKNOWN, not assumed healthy.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\\nERROR: {exc}")
        sys.exit(1)
