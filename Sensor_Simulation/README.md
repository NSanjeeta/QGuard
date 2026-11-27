# QuantumGuard Sensor Placement Simulation

## 1. Requirements
- Python 3.10 or later
- VS Code
- Python extension in VS Code

## 2. Open the project
Extract this ZIP, then open the `QuantumGuard_Sensor_Simulation` folder in VS Code.

## 3. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## 4. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run the simulation

```bash
python sensor_placement_sim.py
```

The script optimises five sensor centre coordinates and creates:
- `outputs/optimised_layout.png`
- `outputs/optimised_metrics.csv`
- `outputs/sensor_coordinates.csv`

## 6. Important modelling limitation

The starter version uses an ideal circular detection footprint with a configurable 4 m radius.
This is ONLY a baseline geometry for learning and code validation. It is not an accurate,
manufacturer-validated ceiling-mounted LD2410C detection map.

Before claiming real-world performance:
1. Measure the sensor's detection boundary at the chosen ceiling height.
2. Record detection probability at multiple floor positions.
3. Repeat separately for moving and stationary targets.
4. Convert those measurements into a 2D probability/coverage map.
5. Replace the circle calculation in `coverage_count()` with that measured footprint.

The objective currently prioritises 2+ sensor coverage, then total coverage, then 3+ and 4+ coverage.
The optimisation is a classical baseline, not QAOA.
