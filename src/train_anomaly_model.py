from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Find the main project folder.
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

# Input dataset and output locations.
DATA_FILE = PROJECT_FOLDER / "data" / "quantumguard_sensor_data_sorted.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
MODEL_FILE = OUTPUT_FOLDER / "quantumguard_anomaly_model.joblib"
PREDICTIONS_FILE = OUTPUT_FOLDER / "test_predictions.csv"


# These are the sensor details the AI is allowed to study.
NUMERIC_FEATURES = [
    "reading_value",
    "packet_rate",
    "response_time_ms",
    "battery_percent",
    "signal_strength_dbm",
    "consistency_score",
    "missed_messages",
]

CATEGORICAL_FEATURES = [
    "sensor_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# This is the correct answer column:
# 0 = normal sensor behaviour
# 1 = suspicious sensor behaviour
TARGET = "anomaly_label"


def load_data():
    """Load the QuantumGuard dataset."""
    return pd.read_csv(DATA_FILE)


def train_model(data):
    """Train and test the Random Forest anomaly detector."""
    x = data[FEATURES]
    y = data[TARGET]

    # 80% data for learning, 20% data for testing.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Convert sensor type words into numbers the AI can understand.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", "passthrough", NUMERIC_FEATURES),
            ("sensor_type", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    # Random Forest learns patterns of normal and suspicious sensor behaviour.
    model = Pipeline(
        steps=[
            ("prepare_data", preprocessor),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    predicted_labels = model.predict(x_test)

    # Probability that a reading is suspicious becomes the anomaly score.
    anomaly_scores = model.predict_proba(x_test)[:, 1]

    return model, x_test, y_test, predicted_labels, anomaly_scores


def save_results(model, x_test, y_test, predicted_labels, anomaly_scores):
    """Save the trained model and test predictions."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    results = x_test.copy()
    results["actual_anomaly_label"] = y_test.values
    results["predicted_anomaly_label"] = predicted_labels
    results["anomaly_score"] = anomaly_scores

    results.to_csv(PREDICTIONS_FILE, index=False)


def show_evaluation(y_test, predicted_labels):
    """Print AI performance results."""
    accuracy = accuracy_score(y_test, predicted_labels)
    matrix = confusion_matrix(y_test, predicted_labels)

    print("\nPHASE 3: AI ANOMALY DETECTION RESULTS")
    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nConfusion Matrix")
    print(matrix)

    print("\nClassification Report")
    print(classification_report(y_test, predicted_labels, zero_division=0))


def main():
    data = load_data()

    print(f"Dataset loaded: {len(data)} sensor readings")

    model, x_test, y_test, predicted_labels, anomaly_scores = train_model(data)

    show_evaluation(y_test, predicted_labels)

    save_results(model, x_test, y_test, predicted_labels, anomaly_scores)

    print(f"\nTrained model saved to: {MODEL_FILE}")
    print(f"Test predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()