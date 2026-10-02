# 🦾 6-DOF Robotic Arm Simulator (EMG & Hardware-Ready)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()
[![Physics](https://img.shields.io/badge/Physics-PyBullet%20%7C%20Kinematics-orange.svg)](https://pybullet.org/)
[![Tests](https://img.shields.io/badge/Tests-70%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A modular, real-time Python simulation and control environment for a **6-DOF desktop metal robotic arm** (Hiwonder / LewanSoul style). 

Designed with a clean, decoupled architecture:
* **EMG gesture signals** (e.g., from a BioWave armband, ML classifier, or keyboard mock) drive the arm in 3D Cartesian space.
* The **simulation backend** (PyBullet physics or pure-NumPy kinematics) can be seamlessly swapped for a **physical robotic arm controller** without altering the control or EMG pipelines.

---

## 📑 Table of Contents

* [✨ Features & Highlights](#-features--highlights)
* [💻 System Requirements](#-system-requirements)
* [🚀 Quick Start & Installation](#-quick-start--installation)
  * [🍏 macOS Setup (Apple Silicon & Intel)](#-macos-setup-apple-silicon-m1m2m3m4--intel)
  * [🪟 Windows Setup (PowerShell / Command Prompt)](#-windows-setup-powershell--cmd)
  * [🐧 Linux / WSL2 Setup](#-linux--wsl2-setup)
* [🎮 How to Run the App](#-how-to-run-the-app)
* [🕹️ Interactive Controls Guide](#️-interactive-controls-guide)
  * [Manual Mode Keyboard Controls](#manual-mode-keyboard-controls)
  * [EMG Gesture Mode Keys](#emg-gesture-mode-mock-keys)
  * [3D GUI Sliders & Buttons](#3d-gui-sliders--interactive-panel)
* [🏗️ System Architecture](#️-system-architecture)
* [🧠 Connecting Custom EMG / AI Classifiers](#-connecting-custom-emg--ai-classifiers)
* [⚙️ Configuration & Calibration (Zero-Code)](#️-configuration--calibration-zero-code)
* [📐 Coordinate System, Kinematics & Joints](#-coordinate-system-kinematics--joints)
* [🛡️ Safety & Fault Protection](#️-safety--fault-protection)
* [🦾 Connecting Physical Hardware (LewanSoul Arm)](#-connecting-physical-hardware-lewansoul-arm)
* [🧪 Running Automated Tests](#-running-automated-tests)
* [📁 Project File Structure](#-project-file-structure)
* [❓ Troubleshooting & FAQ](#-troubleshooting--faq)

---

## ✨ Features & Highlights

* **🎯 EMG Gesture-Driven Control**: Seamlessly translates gesture predictions (`left`, `right`, `up`, `down`, `fist_close`, `rest`) into smooth Cartesian jogging and gripper actuations.
* **🛡️ Smart Gesture Filtering**: Confidence gating ($P \ge 0.75$) and sliding-window majority voting prevent spurious trigger motions.
* **📐 Numerical Inverse Kinematics (Damped Least Squares)**: Fast (sub-millisecond) limit-aware IK with posture constraints and automatic singularity recovery.
* **⚡ Decoupled Multiprocess Architecture**: The 240 Hz real-time physics and motion controller runs independently of the PyBullet 3D rendering process, preventing UI lag.
* **🛡️ Active Safety Layers**: Capsule-plane self-collision avoidance, workspace envelope clamping, torque overload monitoring, and latching emergency stop (`ESC`).
* **📦 Portable & Hardware-Ready**: Standard SI units throughout, instant URDF export for ROS/ROS 2, and clean hardware controller abstractions.

---

## 💻 System Requirements

| Component | Minimum Requirement | Recommended |
|---|---|---|
| **Operating System** | macOS 11+, Windows 10/11 (64-bit), Ubuntu 20.04+ / WSL2 | macOS (Apple Silicon M1–M4) or Windows 11 |
| **Python** | Python 3.10 | **Python 3.11 or 3.12** |
| **Memory (RAM)** | 4 GB | 8 GB+ |
| **Graphics** | OpenGL 2.1+ compatible GPU | Dedicated GPU or Apple Silicon Integrated GPU |
| **Input** | Standard Keyboard | Keyboard + (Optional) BioWave EMG Armband |

---

## 🚀 Quick Start & Installation

Choose your operating system below for detailed, step-by-step setup instructions:

### 🍏 macOS Setup (Apple Silicon M1/M2/M3/M4 & Intel)

<details open>
<summary><b>Click to expand macOS Instructions</b></summary>

#### 1. Open Terminal and Navigate to the Project
```bash
cd /path/to/RoboticArm
```

#### 2. Create and Activate a Virtual Environment
```bash
# Recommended: Python 3.11 or 3.12
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> **macOS Build Note (Python 3.13 / newer Xcode SDKs):**
> If PyBullet compilation reports an issue with `fdopen` in `zlib`, install it with compiler compatibility flags:
> ```bash
> CFLAGS="-Dfdopen=fdopen -Wno-error" CXXFLAGS="-Dfdopen=fdopen -Wno-error" pip install pybullet
> ```

#### 4. Launch the 3D Simulator
```bash
python main.py
```
</details>

---

### 🪟 Windows Setup (PowerShell / CMD)

<details open>
<summary><b>Click to expand Windows Instructions</b></summary>

#### 1. Open PowerShell or Command Prompt
Ensure [Python 3.10–3.12](https://www.python.org/downloads/) is installed with the **"Add Python to PATH"** checkbox selected during setup.

#### 2. Navigate to the Project Directory
```powershell
cd C:\path\to\RoboticArm
```

#### 3. Enable Script Execution & Create Virtual Environment (PowerShell)
```powershell
# Allow local scripts in PowerShell (run once if needed):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If you are using classic **Command Prompt (cmd.exe)**, activate with: `.\.venv\Scripts\activate.bat`)*

#### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Launch the 3D Simulator
```powershell
python main.py
```
</details>

---

### 🐧 Linux / WSL2 Setup

<details>
<summary><b>Click to expand Linux / WSL2 Instructions</b></summary>

```bash
# 1. Install system OpenGL dependencies (Ubuntu / Debian)
sudo apt update && sudo apt install -y python3-venv python3-pip libgl1-mesa-glx libglib2.0-0

# 2. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python packages and run
pip install -r requirements.txt
python main.py
```
*(Note for WSL2: Ensure you have an active X-Server like VcXsrv or WSLg enabled to view the 3D GUI window).*
</details>

---

## 🎮 How to Run the App

The main entry point is [`main.py`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/RoboticArm/main.py). You can customize execution modes via CLI arguments:

```bash
# ==============================================================================
# 1. 3D GUI - MANUAL MODE (Default: sliders, buttons, keyboard jogging)
# ==============================================================================
python main.py

# ==============================================================================
# 2. 3D GUI - EMG GESTURE MODE (Mock EMG driven by keys 1-5 and 0)
# ==============================================================================
python main.py --mode emg

# ==============================================================================
# 3. 3D GUI - SCRIPTED EMG DEMO (Plays automated sequence of gestures)
# ==============================================================================
python main.py --mode emg --emg scripted

# ==============================================================================
# 4. 3D GUI - NOISY EMG SIMULATION (Injected false predictions & low confidence)
# ==============================================================================
python main.py --mode emg --emg noisy

# ==============================================================================
# 5. HEADLESS BENCHMARK (Console telemetry only, no 3D window, high speed)
# ==============================================================================
python main.py --headless --fast --mode emg --emg noisy --duration 12

# ==============================================================================
# 6. AUTONOMOUS PICK-AND-PLACE DEMO (3D Physics Grasping)
# ==============================================================================
python examples/example_pick_and_place.py --gui

# ==============================================================================
# 7. EXPORT URDF MODEL (For ROS, MoveIt, or external robotics tools)
# ==============================================================================
python main.py --export-urdf arm.urdf

# ==============================================================================
# 8. STANDALONE ALL-IN-ONE RUNNER
# ==============================================================================
python robotic_arm.py
```

### 📋 CLI Options Reference

| Flag | Values | Default | Description |
|---|---|---|---|
| `--mode` | `manual`, `emg` | `manual` | Primary control mode (manual keyboard/sliders vs EMG gesture feed) |
| `--emg` | `keyboard`, `scripted`, `noisy` | `keyboard` | Mock EMG input source provider |
| `--config` | Path to JSON | None | Load custom link/joint calibration overrides |
| `--headless` | *flag* | `False` | Run without opening the 3D PyBullet GUI window |
| `--fast` | *flag* | `False` | Execute simulation as fast as CPU permits (headless only) |
| `--duration` | Float (seconds) | Unlimited | Automatically exit after specified duration |
| `--no-physics` | *flag* | `False` | Run pure NumPy kinematic backend (disables PyBullet physics) |
| `--control-hz` | Float | `240.0` | Motion controller & physics loop execution frequency (Hz) |
| `--threshold` | Float (0.0–1.0) | `0.75` | Minimum confidence required to accept an EMG gesture |
| `--window` | Integer | `5` | Size of sliding window for majority-vote smoothing |
| `--no-log` | *flag* | `False` | Disable writing telemetry to `logs/session_NNN.csv` |

---

## 🕹️ Interactive Controls Guide

Click inside the **3D PyBullet window** to focus keyboard input:

### Manual Mode Keyboard Controls

```
                 [ ↑ ] Tool Up (+Z)
  [ ← ] Left (-X)     [ → ] Right (+X)
                 [ ↓ ] Tool Down (-Z)
          [ I ] Forward (+Y)   [ K ] Backward (-Y)
```

| Key Group | Key | Action | Details |
|---|---|---|---|
| **Joint Jogging** | `Q` / `A` | **J1 Base** | Rotate base CCW (+Z) / CW (-Z) |
| | `W` / `S` | **J2 Shoulder** | Pitch forward (+Y) / backward (-Y) |
| | `E` / `D` | **J3 Elbow** | Pitch forward / backward |
| | `R` / `F` | **J4 Wrist Pitch** | Pitch tool down / up |
| | `T` / `G` | **J5 Wrist Roll** | Rotate tool CCW / CW |
| | `Y` / `H` | **Gripper** | Open fingers (`Y`) / Close fingers (`H`) |
| **Cartesian Jogging** | `←` / `→` | **X-Axis** | Move Tool Left (−X) / Right (+X) |
| | `↑` / `↓` | **Z-Axis** | Move Tool Up (+Z) / Down (−Z) |
| | `I` / `K` | **Y-Axis** | Move Tool Forward (+Y) / Backward (−Y) |
| **Safety & Reset** | `ESC` | **EMERGENCY STOP** | Instant stop (latched fault state) |
| | `X` | **Reset** | Re-arm arm after stop or fault |
| | `Z` | **Home** | Return arm smoothly to home position |

---

### EMG Gesture Mode Mock Keys

When running with `--mode emg`, use the number keys to simulate real-time BioWave gesture classifications:

| Key | Gesture Simulated | Resulting Robot Action |
|:---:|:---|:---|
| `1` | **LEFT** | Continuous Cartesian move Left (−X direction) at 40 mm/s |
| `2` | **RIGHT** | Continuous Cartesian move Right (+X direction) at 40 mm/s |
| `3` | **UP** | Continuous Cartesian move Up (+Z direction) at 40 mm/s |
| `4` | **DOWN** | Continuous Cartesian move Down (−Z direction) at 40 mm/s |
| `5` | **FIST_CLOSE** | Close parallel gripper fingers |
| `0` | **REST** *(or release key)* | Hold current pose (motion halt) |

---

### 3D GUI Sliders & Interactive Panel

On the right side of the PyBullet window, an interactive GUI sidebar provides direct controls:

* **Joint Sliders (J1–J6)**: Drag to command explicit joint angles (degrees).
* **Cartesian Sliders (X, Y, Z, Roll, Pitch, Yaw)**: Dial in precise coordinates in millimeters and angles in degrees.
* **Action Buttons**:
  * `GO to XYZ (tool down)`: Computes IK with vertical downward tool orientation.
  * `GO to full pose`: Computes full 6-DOF target pose.
  * `HOME` / `RESET` / `STOP` / `E-STOP` / `OPEN GRIPPER` / `CLOSE GRIPPER`.

---

## 🏗️ System Architecture

The simulator employs a multi-tiered, thread-safe pipeline designed for high responsiveness:

```
 ┌────────────────────────────────────────────────────────┐
 │   BioWave Armband (8 EMG + 3 IMU) / Mock Keyboard Keys │
 └──────────────────────────┬─────────────────────────────┘
                            │ Raw Gesture + Confidence
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  EMG Controller (Confidence Gate ≥ 0.75 + 5-Sample MV)  │
 └──────────────────────────┬─────────────────────────────┘
                            │ MotionCommand (Thread-safe Queue)
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Motion Planner (Cartesian / Joint Jog, Path Trajectory)│
 └──────────────────────────┬─────────────────────────────┘
                            │ Target Pose
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Inverse Kinematics (Numerical DLS with Joint Limits)  │
 └──────────────────────────┬─────────────────────────────┘
                            │ Servo Target Angles
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Safety Monitor (Collision Veto, Limits, Overload)     │
 └──────────────────────────┬─────────────────────────────┘
                            │ Verified Angles
              ┌─────────────┴─────────────┐
              ▼                           ▼
 ┌─────────────────────────┐ ┌─────────────────────────────┐
 │ PyBullet Physics Engine │ │ Physical Arm (LewanSoul)    │
 │ (240 Hz Real-Time Loop) │ │ (Serial Bus Controller)     │
 └────────────┬────────────┘ └─────────────────────────────┘
              ▼
 ┌─────────────────────────┐
 │ PyBullet 3D GUI Process │
 │ (~60 Hz GUI / Render)   │
 └─────────────────────────┘
```

### Multithreading & Process Breakdown

| Unit | Rate | Thread / Process | Responsibility |
|---|---|---|---|
| **EMG Poller** | 50 Hz | Worker Thread | Polls EMG classifier, applies majority voting, creates `MotionCommand` |
| **Control Loop** | 240 Hz | Main Worker Thread | Trajectory planning, IK solver, simulated servo dynamics, safety checks, CSV logging |
| **Viewer Link** | 30 Hz | IPC Bridge Thread | Transmits robot state to GUI process, fetches slider/button commands |
| **3D Viewer** | ~60 Hz | **Separate Process** | Handles OpenGL window rendering and PyBullet user events without blocking physics |

---

## 📡 Wireless EMG Armband + Control Dashboard (`robotic_arm.py`)

`python robotic_arm.py` opens a BioWave-style dashboard (same dark theme and workflow as the BioWave mouse controller) next to the 3D view.

```bash
pip install numpy mujoco PyQt5 pyqtgraph pyserial joblib scikit-learn
python robotic_arm.py                       # one window: dashboard + MuJoCo 3D view (no mjpython needed)
python robotic_arm.py --model path/to/rf_realtime_model.joblib   # pre-load a model
python robotic_arm.py --no-dashboard        # console only (no window)
python robotic_arm.py --no-shadows --gui-fps 20   # cooler on a fanless MacBook Air
```

1. **Connect Device** - Wireless: *Discover* (ESP32-S3 on the same Wi-Fi), enter the access key, *Connect Wireless*. Wired: pick a serial port. New board? *Provision New Device (USB)* sends Wi-Fi credentials.
2. **Load Pretrained Model** - the `.joblib` trained in BioWave (8 EMG + 3 IMU = 11 channels, 500 Hz). Feature extraction is the same code as the trainer.
3. **Calibrate** - REST then FLEX. Calibration is rejected if a channel is dead, noisy, saturated or weak.
4. **Mapping** - each model class is mapped to an arm action (defaults guessed from the class name; Rest = hold).
5. **ENABLE ARM CONTROL** - gestures now move the arm. Control switches itself off (and the arm stops) on signal-quality failure, packet loss, a stalled stream, a lost link, or **EMERGENCY STOP**.

No armband? Use *Test Without Device* (hold a gesture button) - it goes through the same pipeline.

## 🧠 Connecting Custom EMG / AI Classifiers

Integrating a custom machine learning model (e.g., Random Forest, SVM, or Neural Network trained on BioWave armband data) requires subclassing [`EMGSource`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/RoboticArm/emg/emg_controller.py#L13-L25):

```python
from emg.emg_controller import EMGSource, EMGController
from control.controller import SimulatedRobotController

class MyTrainedModelEMGSource(EMGSource):
    def __init__(self, real_time_stream):
        self.stream = real_time_stream

    def connect(self):
        self.stream.start()

    def disconnect(self):
        self.stream.stop()

    def read(self) -> dict | None:
        """Returns the latest classification dictionary."""
        sample = self.stream.get_latest()
        if sample is None:
            return None
        
        return {
            "gesture": sample.predicted_label,  # "left", "right", "up", "down", "fist_close", "rest"
            "confidence": float(sample.prob),   # 0.0 to 1.0
            "timestamp": sample.timestamp       # epoch seconds
        }

# Usage:
controller = SimulatedRobotController()
controller.connect()

emg_source = MyTrainedModelEMGSource(my_stream)
emg_pipeline = EMGController(
    source=emg_source,
    sink=controller.submit,
    status_callback=controller.set_emg_status,
    latency=controller.latency
)
emg_pipeline.connect()
emg_pipeline.start()
```

---

## ⚙️ Configuration & Calibration (Zero-Code)

All physical parameters (link lengths, masses, limits, velocity curves) are stored in [`config/robot_config.py`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/RoboticArm/config/robot_config.py).

You can override any parameter using a **JSON calibration file** without changing code:

```json
{
  "_comment": "Human units: degrees, mm, kg, kg*cm",
  "joints": {
    "joint_1": { "zero_offset": 0, "direction": 1, "min_angle": -90, "max_angle": 90, "home_position": 0 },
    "joint_2": { "direction": -1, "zero_offset": 0, "torque_limit_kgcm": 17 },
    "joint_3": { "max_velocity_deg_s": 90 }
  },
  "links": {
    "link2": { "length_mm": 110, "mass_kg": 0.15 },
    "link3": { "length_mm": 105, "mass_kg": 0.12 }
  },
  "tcp_offset_mm": 90
}
```

Run with your custom configuration:
```bash
python main.py --config config/calibration_example.json
```

---

## 📐 Coordinate System, Kinematics & Joints

The robot uses a **Right-Handed Base Frame** fixed to the tabletop:
* **+X**: Points to the operator's right
* **+Y**: Points forward (arm reach direction)
* **+Z**: Points vertically up
* **Zero Pose**: All joint angles $= 0^\circ$ (arm straight up, TCP at $Z = 450\text{ mm}$)

```
               [+Z] Up
                 │
                 │   [+Y] Forward
                 │  /
                 │ /
  [Origin]───────┴────────► [+X] Right
```

### Joint Specifications

| Joint | Role | Axis | Positive Direction | Working Limit |
|---|---|---|---|---|
| **J1** | Base Turntable | $+Z$ | Counter-clockwise seen from above | $\pm 120^\circ$ |
| **J2** | Shoulder Pitch | $-X$ | Lean forward (toward $+Y$) | $\pm 90^\circ$ |
| **J3** | Elbow Pitch | $-X$ | Pitch forearm forward | $\pm 120^\circ$ |
| **J4** | Wrist Pitch | $-X$ | Pitch tool tip forward | $\pm 120^\circ$ |
| **J5** | Wrist Roll | $+Z$ (axial) | Counter-clockwise roll | $\pm 150^\circ$ |
| **J6** | Parallel Gripper | Prismatic | Finger opening | $0^\circ\text{ (closed)} \dots 60^\circ\text{ (open)}$ |

---

## 🛡️ Safety & Fault Protection

The safety monitor ([`control/safety.py`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/RoboticArm/control/safety.py)) runs at 240 Hz to guard against damage:

1. **Joint Limits Clamp**: Exceeding angular limits gracefully clamps motion with warnings.
2. **Predictive Collision Veto**: Analytically tests swept capsules against the floor and adjacent links. If a trajectory would collide, the command is **vetoed before servos move**.
3. **Gravity Torque Overload Detection**: Monitors static load per joint; triggers a fault if torque exceeds servo capacity for $> 0.5\text{ s}$.
4. **Locked-Rotor Detection**: Flags a fault if physical joint angle lags commanded angle by $> 0.12\text{ rad}$ for $> 0.5\text{ s}$ (gripper exempt during grasping).
5. **Emergency Stop (`ESC`)**: Instantly freezes all servos without ramping, discards remaining path commands, and requires an explicit Reset (`X`) key to re-arm.

---

## 🦾 Connecting Physical Hardware (LewanSoul Arm)

[`control/hardware_controller.py`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/RoboticArm/control/hardware_controller.py) provides a drop-in hardware backend implementing `RobotController`:

1. Connect the LewanSoul 6-channel STM32 bus servo board via USB/Serial (`/dev/ttyUSB0` or `COM3`).
2. Calibrate joint directions and zero-offsets in JSON (`direction: 1` or `-1`, `zero_offset`).
3. Angle to servo tick conversion is performed automatically:
   $$\text{ticks} = \text{center\_ticks} + \text{ticks\_per\_rad} \times (\text{direction} \cdot q + \text{zero\_offset})$$
4. Replace `SimulatedRobotController` with `HardwareRobotController` in your runtime script.

---

## 🧪 Running Automated Tests

A comprehensive test suite of **70 unit and integration tests** verifies kinematics, IK convergence, safety vetoes, EMG smoothing, and physics:

```bash
# Run all tests cleanly
pytest -q

# Or run via Python module
python -m pytest
```

---

## 📁 Project File Structure

```
RoboticArm/
├── main.py                     # Primary executable entry point
├── robotic_arm.py              # Standalone single-file simulator
├── runtime.py                  # Multithreaded/multiprocess runtime coordinator
├── requirements.txt            # Python package dependencies
├── config/
│   ├── robot_config.py         # Robot geometry, joints, link dataclasses
│   ├── emg_config.py           # EMG thresholds, voting window, speed tunables
│   └── calibration_example.json# Zero-code human-readable calibration file
├── control/
│   ├── controller.py           # SimulatedRobotController & base interface
│   ├── hardware_controller.py  # LewanSoul bus servo serial driver stub
│   ├── motion_command.py       # Input-agnostic MotionCommand definitions
│   ├── motion_planner.py       # Cartesian & Joint trajectory generator
│   ├── pick_and_place.py       # Autonomous pick-and-place state machine
│   └── safety.py               # Active limit, collision, and stall monitor
├── emg/
│   ├── emg_controller.py       # EMG consumer pipeline & polling thread
│   ├── gesture_mapper.py       # Gesture smoothing & MotionCommand mapping
│   └── mock_emg.py             # Scripted, noisy, and keyboard mock sources
├── examples/
│   ├── example_emg_control.py  # End-to-end EMG latency analysis demo
│   └── example_pick_and_place.py # 3D physics pick-and-place demonstration
├── robot/
│   ├── kinematics.py           # Forward & Damped-Least-Squares Inverse Kinematics
│   ├── robot_arm.py            # Arm assembly and pose calculation
│   ├── joints.py / links.py    # Joint and link representations
│   ├── servo.py                # Velocity/acceleration-limited servo model
│   └── gripper.py              # Parallel finger gripper actuation
├── simulation/
│   ├── physics.py              # PyBullet & pure-NumPy physics backends
│   ├── collision.py            # Capsule-based analytical collision detection
│   ├── objects.py              # Dynamic world objects (table, target cubes)
│   └── urdf.py                 # Dynamic URDF generator
├── ui/
│   ├── gui.py                  # PyBullet 3D GUI process (sliders & buttons)
│   ├── controls.py             # Keyboard input mapping
│   └── viewer_link.py          # IPC state synchronization bridge
├── utils/
│   ├── logger.py               # CSV telemetry logger & LatencyTracker
│   └── math_utils.py           # SO(3)/SE(3) transforms, quaternions, RPY
└── tests/                      # 70 automated pytest test cases
```

---

## ❓ Troubleshooting & FAQ

<details>
<summary><b>Q1: PyBullet fails to install on macOS (zlib fdopen error)</b></summary>
Install PyBullet with the following C/C++ compiler definitions:

```bash
CFLAGS="-Dfdopen=fdopen -Wno-error" CXXFLAGS="-Dfdopen=fdopen -Wno-error" pip install pybullet
```
</details>

<details>
<summary><b>Q2: PowerShell says "running scripts is disabled on this system" (Windows)</b></summary>
Run PowerShell as your user and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then re-run `.\.venv\Scripts\Activate.ps1`.
</details>

<details>
<summary><b>Q3: The arm stops moving during an EMG gesture</b></summary>
Continuous EMG commands feature a **0.30 second safety keep-alive timeout** (`COMMAND_TIMEOUT_S`). If new gesture predictions drop below confidence threshold ($0.75$) or cease arriving, the planner automatically halts the robot for safety.
</details>

<details>
<summary><b>Q4: How do I recover after an Emergency Stop?</b></summary>
Press the `X` key on your keyboard or click the `RESET` button on the GUI panel to re-arm the servos.
</details>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
