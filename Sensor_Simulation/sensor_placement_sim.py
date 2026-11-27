"""
QUANTUMGUARD - CENTRAL + FOUR-SURROUNDING SENSOR PLACEMENT
---------------------------------------------------------

This script implements the refined architecture requested:

                 S1                 S2

                         S3

                 S5                 S4

Coordinate plane: 15 m x 12 m
S3 is fixed at the centre.
The other four sensors are placed around S3.

This version uses the refined positions shown in the latest layout:
    S1 = (5.0, 7.6)  upper-left
    S2 = (10.5, 7.6) upper-right
    S3 = (7.5, 6.0)  centre
    S4 = (10.5, 4.2) lower-right
    S5 = (5.0, 4.2)  lower-left

The sensing radius is set to 4.3 m to extend the outer coverage slightly.

IMPORTANT:
- These are simulation coordinates, not validated installation coordinates.
- Circular sensing areas are an idealised model.
- The actual LD2410C detection footprint must be measured in the demo room.
- The script reports the coverage of each sensor's footprint by the other
  four sensors if that sensor is isolated.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# ============================================================
# 1. CONFIGURATION
# ============================================================

AREA_WIDTH = 15.0
AREA_HEIGHT = 12.0

SENSING_RADIUS_M = 4.3
GRID_RESOLUTION_M = 0.05

SENSOR_NAMES = ["S1", "S2", "S3", "S4", "S5"]
SENSOR_COLORS = ["red", "orange", "blue", "green", "purple"]

# Refined central-plus-four-surrounding architecture.
# Order: S1 upper-left, S2 upper-right, S3 centre,
#        S4 lower-right, S5 lower-left.
SENSOR_POSITIONS = np.array([
    [5.0, 7.6],     # S1
    [10.5, 7.6],    # S2
    [7.5, 6.0],     # S3 - centre
    [10.5, 4.2],    # S4
    [5.0, 4.2],     # S5
], dtype=float)

OUTPUT_DIR = Path("quantumguard_sensor_placement")
OUTPUT_DIR.mkdir(exist_ok=True)

# Grid cell centres
x_values = np.arange(
    GRID_RESOLUTION_M / 2,
    AREA_WIDTH,
    GRID_RESOLUTION_M
)
y_values = np.arange(
    GRID_RESOLUTION_M / 2,
    AREA_HEIGHT,
    GRID_RESOLUTION_M
)
X, Y = np.meshgrid(x_values, y_values)

CELL_AREA_M2 = GRID_RESOLUTION_M ** 2
TOTAL_AREA_M2 = AREA_WIDTH * AREA_HEIGHT


# ============================================================
# 2. COVERAGE CALCULATION
# ============================================================

def get_sensor_footprint(sensor_position):
    """Return a Boolean grid showing the ideal circular footprint."""
    sx, sy = sensor_position
    distance_squared = (X - sx) ** 2 + (Y - sy) ** 2

    return distance_squared <= SENSING_RADIUS_M ** 2


def get_all_footprints():
    """Return one Boolean coverage grid per sensor."""
    return np.stack(
        [get_sensor_footprint(position) for position in SENSOR_POSITIONS],
        axis=0
    )


def get_coverage_count():
    """Number of sensors covering each grid cell."""
    footprints = get_all_footprints()
    return np.sum(footprints, axis=0)


# ============================================================
# 3. OVERLAP AND ISOLATION METRICS
# ============================================================

def calculate_metrics():
    footprints = get_all_footprints()
    coverage_count = np.sum(footprints, axis=0)

    sensor_results = []
    rescue_percentages = []

    for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
        own_footprint = footprints[sensor_index]

        # Coverage supplied by all sensors except the isolated sensor.
        other_sensor_count = np.sum(
            np.delete(footprints, sensor_index, axis=0),
            axis=0
        )

        own_cells = int(np.count_nonzero(own_footprint))
        rescued_cells = int(np.count_nonzero(
            own_footprint & (other_sensor_count >= 1)
        ))
        unrescued_cells = own_cells - rescued_cells

        own_area = own_cells * CELL_AREA_M2
        rescued_area = rescued_cells * CELL_AREA_M2
        unrescued_area = unrescued_cells * CELL_AREA_M2

        rescue_percentage = (
            100.0 * rescued_cells / own_cells if own_cells else 0.0
        )

        rescue_percentages.append(rescue_percentage)

        sensor_results.append({
            "sensor_id": sensor_name,
            "own_footprint_area_m2": own_area,
            "covered_by_other_four_m2": rescued_area,
            "uncovered_if_isolated_m2": unrescued_area,
            "redundancy_percentage": rescue_percentage,
        })

    overlap_results = {}

    # Grid area covered by at least 1, 2, 3, 4 or 5 sensors.
    for number in range(1, 6):
        covered_cells = int(np.count_nonzero(
            coverage_count >= number
        ))
        covered_area = covered_cells * CELL_AREA_M2

        overlap_results[f"at_least_{number}_sensors_m2"] = covered_area
        overlap_results[f"at_least_{number}_sensors_pct"] = (
            100.0 * covered_area / TOTAL_AREA_M2
        )

    uncovered_cells = int(np.count_nonzero(coverage_count == 0))
    overlap_results["uncovered_area_m2"] = (
        uncovered_cells * CELL_AREA_M2
    )
    overlap_results["uncovered_area_pct"] = (
        100.0 * uncovered_cells * CELL_AREA_M2 / TOTAL_AREA_M2
    )

    summary = {
        "worst_case_redundancy_pct": min(rescue_percentages),
        "average_redundancy_pct": float(np.mean(rescue_percentages)),
        **overlap_results,
    }

    return sensor_results, summary


# ============================================================
# 4. MAIN ARCHITECTURE VISUALISATION
# ============================================================

def create_main_layout(sensor_results, summary):
    coverage_count = get_coverage_count()

    fig, ax = plt.subplots(figsize=(12, 9))

    image = ax.pcolormesh(
        X,
        Y,
        coverage_count,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=5
    )

    # Draw sensor circles and labels.
    for index, (name, position) in enumerate(
        zip(SENSOR_NAMES, SENSOR_POSITIONS)
    ):
        sx, sy = position

        circle = Circle(
            (sx, sy),
            SENSING_RADIUS_M,
            fill=False,
            edgecolor=SENSOR_COLORS[index],
            linewidth=2.8
        )
        ax.add_patch(circle)

        ax.scatter(
            sx,
            sy,
            color=SENSOR_COLORS[index],
            edgecolor="black",
            linewidth=1.0,
            s=125,
            zorder=5
        )

        label_y = sy + 0.30
        ax.text(
            sx,
            label_y,
            name,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            zorder=6
        )

    # Monitoring-area boundary.
    ax.add_patch(Rectangle(
        (0, 0),
        AREA_WIDTH,
        AREA_HEIGHT,
        fill=False,
        edgecolor="black",
        linewidth=2.0
    ))

    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X position (metres)", fontsize=12)
    ax.set_ylabel("Y position (metres)", fontsize=12)

    ax.set_title(
        "QuantumGuard - Central + Four-Surrounding Sensor Placement\n"
        f"Worst-case redundancy: "
        f"{summary['worst_case_redundancy_pct']:.1f}% | "
        f"Average redundancy: "
        f"{summary['average_redundancy_pct']:.1f}%",
        fontsize=14,
        fontweight="bold"
    )

    colorbar = fig.colorbar(image, ax=ax, ticks=range(6))
    colorbar.set_label(
        "Number of sensors covering each grid cell",
        fontsize=11
    )

    ax.grid(alpha=0.15)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "final_sensor_placement.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return output_path


# ============================================================
# 5. INDIVIDUAL SENSOR ISOLATION MAPS
# ============================================================

def create_isolation_maps(sensor_results):
    footprints = get_all_footprints()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for failed_index, sensor_name in enumerate(SENSOR_NAMES):
        ax = axes[failed_index]

        own_footprint = footprints[failed_index]
        remaining_coverage = np.sum(
            np.delete(footprints, failed_index, axis=0),
            axis=0
        )

        # Map interpretation:
        # 0 = outside the isolated sensor's original footprint
        # 1 = inside its footprint but not covered by another sensor
        # 2 = inside its footprint and rescued by another sensor
        isolation_map = np.zeros(own_footprint.shape, dtype=int)

        isolation_map[
            own_footprint & (remaining_coverage == 0)
        ] = 1

        isolation_map[
            own_footprint & (remaining_coverage >= 1)
        ] = 2

        ax.pcolormesh(
            X,
            Y,
            isolation_map,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=2
        )

        # Dashed circle identifies the isolated sensor's original footprint.
        sx, sy = SENSOR_POSITIONS[failed_index]

        ax.add_patch(Circle(
            (sx, sy),
            SENSING_RADIUS_M,
            fill=False,
            edgecolor=SENSOR_COLORS[failed_index],
            linestyle="--",
            linewidth=2.3
        ))

        # Show all five installation points.
        for index, (name, position) in enumerate(
            zip(SENSOR_NAMES, SENSOR_POSITIONS)
        ):
            px, py = position

            ax.scatter(
                px,
                py,
                color=SENSOR_COLORS[index],
                edgecolor="black",
                s=48,
                zorder=5
            )

            ax.text(
                px,
                py + 0.15,
                name,
                ha="center",
                fontsize=8,
                fontweight="bold"
            )

        rescue_pct = sensor_results[failed_index][
            "redundancy_percentage"
        ]

        ax.set_title(
            f"{sensor_name} isolated: {rescue_pct:.1f}% rescued"
        )
        ax.set_xlim(0, AREA_WIDTH)
        ax.set_ylim(0, AREA_HEIGHT)
        ax.set_aspect("equal")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.grid(alpha=0.12)

    axes[5].axis("off")

    fig.suptitle(
        "QuantumGuard - Individual Sensor Isolation Analysis\n"
        "Within dashed circle: remaining coverage vs. uncovered footprint",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "sensor_isolation_maps.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return output_path


# ============================================================
# 6. SAVE CSV FILES
# ============================================================

def save_csv_files(sensor_results, summary):
    coordinates_path = OUTPUT_DIR / "sensor_coordinates.csv"

    with coordinates_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sensor_id", "x_m", "y_m", "placement_role"])

        roles = [
            "upper_left",
            "upper_right",
            "centre",
            "lower_right",
            "lower_left"
        ]

        for index, (name, position) in enumerate(
            zip(SENSOR_NAMES, SENSOR_POSITIONS)
        ):
            writer.writerow([
                name,
                round(float(position[0]), 3),
                round(float(position[1]), 3),
                roles[index]
            ])

    redundancy_path = OUTPUT_DIR / "sensor_redundancy_metrics.csv"

    with redundancy_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sensor_id",
            "own_footprint_area_m2",
            "covered_by_other_four_m2",
            "uncovered_if_isolated_m2",
            "redundancy_percentage"
        ])

        for row in sensor_results:
            writer.writerow([
                row["sensor_id"],
                round(row["own_footprint_area_m2"], 3),
                round(row["covered_by_other_four_m2"], 3),
                round(row["uncovered_if_isolated_m2"], 3),
                round(row["redundancy_percentage"], 3)
            ])

    summary_path = OUTPUT_DIR / "placement_summary.csv"

    with summary_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for key, value in summary.items():
            writer.writerow([key, round(float(value), 3)])

        writer.writerow(["area_width_m", AREA_WIDTH])
        writer.writerow(["area_height_m", AREA_HEIGHT])
        writer.writerow(["sensing_radius_m", SENSING_RADIUS_M])
        writer.writerow(["grid_resolution_m", GRID_RESOLUTION_M])

    return coordinates_path, redundancy_path, summary_path


# ============================================================
# 7. PRINT RESULTS
# ============================================================

def print_results(sensor_results, summary):
    print("\n" + "=" * 68)
    print("QUANTUMGUARD - CENTRAL + FOUR-SURROUNDING SENSOR PLACEMENT")
    print("=" * 68)

    print(f"Monitoring coordinate plane: {AREA_WIDTH} m x {AREA_HEIGHT} m")
    print(f"Assumed sensing radius: {SENSING_RADIUS_M:.2f} m")
    print(f"Grid resolution: {GRID_RESOLUTION_M:.2f} m")

    print("\nSENSOR COORDINATES")
    for name, position in zip(SENSOR_NAMES, SENSOR_POSITIONS):
        print(
            f"{name}: x={position[0]:.2f} m, "
            f"y={position[1]:.2f} m"
        )

    print("\nISOLATION / REDUNDANCY ANALYSIS")
    for row in sensor_results:
        print(
            f"{row['sensor_id']}: "
            f"{row['redundancy_percentage']:.2f}% of its own footprint "
            f"is covered by the other four sensors; "
            f"{row['uncovered_if_isolated_m2']:.2f} m2 remains uncovered."
        )

    print("\nKEY RESULTS")
    print(
        f"Worst-case redundancy: "
        f"{summary['worst_case_redundancy_pct']:.2f}%"
    )
    print(
        f"Average redundancy: "
        f"{summary['average_redundancy_pct']:.2f}%"
    )

    print("\nGRID OVERLAP (CONTEXT METRICS)")
    for number in range(1, 6):
        print(
            f"At least {number} sensor(s): "
            f"{summary[f'at_least_{number}_sensors_m2']:.2f} m2 "
            f"({summary[f'at_least_{number}_sensors_pct']:.2f}%)"
        )

    print(
        f"Uncovered grid area: {summary['uncovered_area_m2']:.2f} m2 "
        f"({summary['uncovered_area_pct']:.2f}%)"
    )

    print("\nNOTE:")
    print(
        "The 4.3 m circular footprint is an idealised simulation assumption. "
        "It is not a verified LD2410C detection range."
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("Calculating coverage for the refined architecture...")

    sensor_results, summary = calculate_metrics()

    print_results(sensor_results, summary)

    layout_file = create_main_layout(sensor_results, summary)
    isolation_file = create_isolation_maps(sensor_results)
    csv_files = save_csv_files(sensor_results, summary)

    print("\nOUTPUT FILES")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(layout_file)
    print(isolation_file)

    for path in csv_files:
        print(path)

    print("\nSimulation completed.")


if __name__ == "__main__":
    main()
