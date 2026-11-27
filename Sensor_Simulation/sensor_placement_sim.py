"""
QuantumGuard - sensor_placement_sim.py
NORMAL-MODE ONLY (3.5 m radius).

Purpose:
- Preserve the central S3 + four surrounding sensors architecture.
- Place five equal circular footprints without overlap.
- Search the rotation angle of the four surrounding sensors to minimise
  the worst distance from any point in the monitoring area to its nearest
  sensor centre (a uniformity / worst-gap objective).
- Do NOT calculate 6 m recovery coverage in this version.

This is an ideal circular geometry model, not measured LD2410C performance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# -------------------- Configuration --------------------
AREA_WIDTH = 21.0
AREA_HEIGHT = 21.0
NORMAL_RADIUS_M = 3.5
GRID_RESOLUTION_M = 0.10
ANGLE_STEP_DEGREES = 0.25

SENSOR_NAMES = ["S1", "S2", "S3", "S4", "S5"]
SENSOR_COLORS = ["red", "orange", "blue", "green", "purple"]

# Keep S3 at the centre of the area.
CENTRE = (AREA_WIDTH / 2.0, AREA_HEIGHT / 2.0)

OUTPUT_DIR = Path("quantumguard_sensor_placement")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid cell centres for area calculations and layout scoring.
x_values = np.arange(GRID_RESOLUTION_M / 2, AREA_WIDTH, GRID_RESOLUTION_M)
y_values = np.arange(GRID_RESOLUTION_M / 2, AREA_HEIGHT, GRID_RESOLUTION_M)
X, Y = np.meshgrid(x_values, y_values)
CELL_AREA_M2 = GRID_RESOLUTION_M ** 2
TOTAL_AREA_M2 = AREA_WIDTH * AREA_HEIGHT


def positions_for_angle(angle_degrees):
    """S3 centre; four outer sensors at 90-degree intervals, 7 m away."""
    angle = np.deg2rad(angle_degrees)
    cx, cy = CENTRE
    d = 2.0 * NORMAL_RADIUS_M  # tangent circles: no normal-mode overlap

    # S1, S2, S4, S5 map to four successive compass directions.
    offsets = [
        (d * np.cos(angle + np.deg2rad(45)), d * np.sin(angle + np.deg2rad(45))),
        (d * np.cos(angle - np.deg2rad(45)), d * np.sin(angle - np.deg2rad(45))),
        (d * np.cos(angle + np.deg2rad(135)), d * np.sin(angle + np.deg2rad(135))),
        (d * np.cos(angle - np.deg2rad(135)), d * np.sin(angle - np.deg2rad(135))),
    ]
    # These four offsets are intentionally arranged as two diagonally
    # opposite pairs; angle controls the overall rotation.
    return {
        "S1": (cx + offsets[0][0], cy + offsets[0][1]),
        "S2": (cx + offsets[1][0], cy + offsets[1][1]),
        "S3": (cx, cy),
        "S4": (cx + offsets[2][0], cy + offsets[2][1]),
        "S5": (cx + offsets[3][0], cy + offsets[3][1]),
    }


def footprint(sensor_position):
    sx, sy = sensor_position
    return (X - sx) ** 2 + (Y - sy) ** 2 <= NORMAL_RADIUS_M ** 2


def evaluate_layout(positions):
    """Return normal coverage, overlap, and worst nearest-sensor distance."""
    masks = {name: footprint(positions[name]) for name in SENSOR_NAMES}
    coverage_count = np.sum(list(masks.values()), axis=0)

    # Every grid cell's distance to its nearest sensor centre.
    nearest_distance_sq = np.full(X.shape, np.inf)
    for name in SENSOR_NAMES:
        sx, sy = positions[name]
        nearest_distance_sq = np.minimum(
            nearest_distance_sq, (X - sx) ** 2 + (Y - sy) ** 2
        )
    nearest_distance = np.sqrt(nearest_distance_sq)

    covered_cells = int(np.count_nonzero(coverage_count >= 1))
    overlap_cells = int(np.count_nonzero(coverage_count >= 2))
    uncovered_cells = int(np.count_nonzero(coverage_count == 0))

    # Worst gap distance: max distance from a grid cell to nearest sensor.
    # Also report worst distance beyond the 3.5 m sensing radius.
    worst_nearest_distance = float(nearest_distance.max())
    worst_uncovered_gap_depth = max(0.0, worst_nearest_distance - NORMAL_RADIUS_M)

    return {
        "masks": masks,
        "coverage_count": coverage_count,
        "covered_area_m2": covered_cells * CELL_AREA_M2,
        "overlap_area_m2": overlap_cells * CELL_AREA_M2,
        "uncovered_area_m2": uncovered_cells * CELL_AREA_M2,
        "coverage_percentage": 100.0 * covered_cells * CELL_AREA_M2 / TOTAL_AREA_M2,
        "worst_nearest_sensor_distance_m": worst_nearest_distance,
        "worst_uncovered_gap_depth_m": worst_uncovered_gap_depth,
    }


def pairwise_separation_check(positions):
    violations = []
    for i, first in enumerate(SENSOR_NAMES):
        for second in SENSOR_NAMES[i + 1:]:
            p1 = np.array(positions[first])
            p2 = np.array(positions[second])
            distance = float(np.linalg.norm(p1 - p2))
            if distance < 2 * NORMAL_RADIUS_M - 1e-8:
                violations.append((first, second, distance))
    return violations


def optimise_rotation():
    """
    Search 0..89.75 degrees. Because the four outer sensors are arranged
    symmetrically, the objective repeats every 90 degrees.
    Select the angle with the smallest worst nearest-sensor distance.
    """
    best = None
    angles = np.arange(0.0, 90.0, ANGLE_STEP_DEGREES)
    for angle in angles:
        positions = positions_for_angle(angle)
        # Require all 3.5 m circles to stay inside the monitoring rectangle.
        inside = all(
            NORMAL_RADIUS_M <= px <= AREA_WIDTH - NORMAL_RADIUS_M
            and NORMAL_RADIUS_M <= py <= AREA_HEIGHT - NORMAL_RADIUS_M
            for px, py in positions.values()
        )
        if not inside or pairwise_separation_check(positions):
            continue

        metrics = evaluate_layout(positions)
        score = metrics["worst_nearest_sensor_distance_m"]
        if best is None or score < best["score"]:
            best = {
                "angle_degrees": float(angle),
                "positions": positions,
                "metrics": metrics,
                "score": score,
            }

    if best is None:
        raise RuntimeError("No feasible normal-mode sensor layout found.")
    return best


def create_coverage_map(best):
    positions = best["positions"]
    count = best["metrics"]["coverage_count"]

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.pcolormesh(
        X, Y, count, shading="auto", cmap="viridis", vmin=0, vmax=5
    )

    for name, color in zip(SENSOR_NAMES, SENSOR_COLORS):
        sx, sy = positions[name]
        ax.add_patch(Circle(
            (sx, sy), NORMAL_RADIUS_M,
            fill=False, edgecolor=color, linewidth=2.5
        ))
        ax.scatter(sx, sy, color=color, edgecolor="black", s=95, zorder=5)
        ax.text(sx, sy + 0.22, name, ha="center", weight="bold", zorder=6)

    ax.add_patch(Rectangle(
        (0, 0), AREA_WIDTH, AREA_HEIGHT,
        fill=False, edgecolor="black", linewidth=2
    ))
    ax.set(
        xlim=(0, AREA_WIDTH), ylim=(0, AREA_HEIGHT),
        xlabel="X position (m)", ylabel="Y position (m)",
        title=f"QuantumGuard — Optimised Normal Mode ({NORMAL_RADIUS_M:.1f} m)"
    )
    ax.set_aspect("equal")
    fig.colorbar(
        image, ax=ax, ticks=range(6),
        label="Number of sensors covering each grid cell"
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output = OUTPUT_DIR / "final_sensor_placement.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_results(best):
    positions = best["positions"]
    metrics = best["metrics"]

    coordinate_rows = []
    for name in SENSOR_NAMES:
        px, py = positions[name]
        coordinate_rows.append({
            "sensor_id": name,
            "x_m": px,
            "y_m": py,
            "role": "centre" if name == "S3" else "surrounding",
            "normal_radius_m": NORMAL_RADIUS_M,
        })
    pd.DataFrame(coordinate_rows).to_csv(
        OUTPUT_DIR / "sensor_coordinates.csv", index=False
    )

    # Individual unique/shared footprint area per sensor.
    masks = metrics["masks"]
    rows = []
    for name in SENSOR_NAMES:
        others = np.sum(
            [masks[other] for other in SENSOR_NAMES if other != name],
            axis=0
        )
        own = masks[name]
        total = int(np.count_nonzero(own)) * CELL_AREA_M2
        unique = int(np.count_nonzero(own & (others == 0))) * CELL_AREA_M2
        shared = int(np.count_nonzero(own & (others >= 1))) * CELL_AREA_M2
        rows.append({
            "sensor_id": name,
            "total_footprint_m2": total,
            "unique_coverage_m2": unique,
            "shared_coverage_m2": shared,
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "sensor_redundancy_metrics.csv", index=False
    )

    summary = {
        "monitoring_area_m2": TOTAL_AREA_M2,
        "normal_radius_m": NORMAL_RADIUS_M,
        "grid_resolution_m": GRID_RESOLUTION_M,
        "optimised_rotation_degrees": best["angle_degrees"],
        "covered_area_m2": metrics["covered_area_m2"],
        "uncovered_area_m2": metrics["uncovered_area_m2"],
        "overlap_area_m2": metrics["overlap_area_m2"],
        "coverage_percentage": metrics["coverage_percentage"],
        "worst_nearest_sensor_distance_m": metrics["worst_nearest_sensor_distance_m"],
        "worst_uncovered_gap_depth_m": metrics["worst_uncovered_gap_depth_m"],
        "normal_overlap_constraint_satisfied": (
            len(pairwise_separation_check(positions)) == 0
        ),
        "scope": "Normal mode only; no 6 m calculations performed.",
    }
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    ).to_csv(OUTPUT_DIR / "placement_summary.csv", index=False)


def main():
    best = optimise_rotation()
    metrics = best["metrics"]

    print("\\n" + "=" * 68)
    print("QUANTUMGUARD — NORMAL-MODE PLACEMENT OPTIMISATION")
    print("=" * 68)
    print(f"Monitoring area: {AREA_WIDTH:.1f} m x {AREA_HEIGHT:.1f} m")
    print(f"Normal sensing radius: {NORMAL_RADIUS_M:.2f} m")
    print(f"Optimised rotation: {best['angle_degrees']:.2f} degrees")
    print("\\nOPTIMISED SENSOR COORDINATES")
    for name in SENSOR_NAMES:
        px, py = best["positions"][name]
        print(f"{name}: x={px:.3f} m, y={py:.3f} m")

    print("\\nNORMAL-MODE COVERAGE")
    print(f"Covered area: {metrics['covered_area_m2']:.2f} m²")
    print(f"Uncovered area: {metrics['uncovered_area_m2']:.2f} m²")
    print(f"Overlap area: {metrics['overlap_area_m2']:.2f} m²")
    print(f"Coverage percentage: {metrics['coverage_percentage']:.2f}%")
    print(
        "Worst distance from any grid cell to nearest sensor centre: "
        f"{metrics['worst_nearest_sensor_distance_m']:.2f} m"
    )
    print(
        "Worst uncovered-gap depth beyond 3.5 m range: "
        f"{metrics['worst_uncovered_gap_depth_m']:.2f} m"
    )

    violations = pairwise_separation_check(best["positions"])
    if violations:
        print("\\nFAIL: normal-mode circles overlap.")
        for first, second, distance in violations:
            print(f"{first}-{second}: {distance:.3f} m")
    else:
        print("\\nPASS: all normal-mode circles are non-overlapping.")

    print("\\nNo 6 m extended-range or isolation-recovery calculations were run.")
    map_path = create_coverage_map(best)
    save_results(best)
    print(f"\\nCoverage map: {map_path}")
    print(f"CSV outputs: {OUTPUT_DIR.resolve()}")
    print("\\nSimulation completed.")


if __name__ == "__main__":
    main()
