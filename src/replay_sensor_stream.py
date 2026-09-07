from pathlib import Path
import time

import pandas as pd


# Find the main QuantumGuard project folder.
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

# Location of the sorted sensor dataset.
DATA_FILE = PROJECT_FOLDER / "data" / "quantumguard_sensor_data_sorted.csv"

# Demo speed:
# 1 second in the demo represents 5 minutes of sensor time.
DELAY_SECONDS = 1

# Start with only 5 time cycles for testing.
# Change this to None after the test works.
MAX_CYCLES = 5


def load_sensor_data():
    """Load the CSV and arrange it in timestamp order."""
    data = pd.read_csv(DATA_FILE)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data.sort_values(["timestamp", "sensor_id"])


def display_sensor_reading(reading):
    """Show one sensor message received by the virtual gateway."""
    print(
        f"  {reading['sensor_id']} | "
        f"{reading['sensor_type']} | "
        f"Zone: {reading['zone_id']} | "
        f"Value: {reading['reading_value']} | "
        f"Packet rate: {reading['packet_rate']} | "
        f"Response: {reading['response_time_ms']} ms | "
        f"Battery: {reading['battery_percent']}% | "
        f"Condition: {reading['sensor_condition']}"
    )


def replay_sensor_stream():
    """Replay all readings as a live sensor stream."""
    sensor_data = load_sensor_data()

    print("\nQUANTUMGUARD VIRTUAL GATEWAY STARTED\n")

    grouped_readings = sensor_data.groupby("timestamp", sort=True)

    for cycle_number, (timestamp, readings) in enumerate(grouped_readings, start=1):
        if MAX_CYCLES is not None and cycle_number > MAX_CYCLES:
            break

        print(f"Time: {timestamp}")
        print("Gateway received sensor readings:")

        for _, reading in readings.iterrows():
            display_sensor_reading(reading)

        print("-" * 120)
        time.sleep(DELAY_SECONDS)

    print("\nSensor replay test completed.")


if __name__ == "__main__":
    replay_sensor_stream()