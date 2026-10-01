# 6-DOF Robotic Arm Simulator (EMG-ready)

A modular Python simulator of a Hiwonder/LewanSoul-style 6-DOF metal desktop arm, built so that
EMG gestures from the BioWave armband can later replace keyboard input **without touching the robot core**,
and so a physical arm can later replace the simulator **without touching the EMG layer**.

> **Placeholder geometry.** Only the published figures (465 mm height, 1.24 kg, six servos, up to 17 kg·cm,
> controller specs) are used as given. Every other number (link lengths, masses, joint limits, servo speeds,
> per-joint torque, TCP offset) is a **clearly marked placeholder** (`# TODO: calibrate using physical Hiwonder arm`
> in `config/robot_config.py`) that sums to the published height/weight. Calibrate them from your physical arm
> via a JSON file (no code change) – see [Configuration](#configuration--calibration).

## Contents
1. [Install](#install) · 2. [Run](#run) · 3. [Architecture](#architecture) · 4. [Coordinate system & joints](#coordinate-system--joints)
5. [Kinematics](#kinematics) · 6. [Configuration](#configuration--calibration) · 7. [Controls](#controls)
8. [EMG interface](#emg-interface) · 9. [Replacing the mock EMG](#replacing-the-mock-emg-with-your-classifier)
10. [Real hardware](#connecting-the-physical-lewansoul-controller) · 11. [Safety](#safety) · 12. [Latency](#latency)
13. [Tests](#tests) · 14. [Phases & limitations](#development-phases--known-limitations)

## Install

Python 3.10+ (tested on 3.12). NumPy is enough for kinematics, planning, EMG layer and most tests;
PyBullet adds physics, grasping and the 3D window.

```bash
cd robotic_arm_sim
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy pytest pybullet
```

**macOS / Python 3.13+ note:** there is no PyBullet wheel for every platform, and building from source can fail on
a new SDK in `zlib` (`expected ')'` around `fdopen`). This works around it:

```bash
CFLAGS="-Dfdopen=fdopen -Wno-error" CXXFLAGS="-Dfdopen=fdopen -Wno-error" pip install pybullet
```

(It failed to build on Python 3.14 in the environment this was developed in; 3.12 built fine with the flags above.)

## Run

```bash
python main.py                                  # 3D window, manual: keyboard + sliders + buttons
python main.py --mode emg                       # 3D window, keys 1-5/0 act as mock EMG gestures
python main.py --mode emg --emg scripted        # scripted gesture demo in the 3D window
python main.py --mode emg --emg noisy           # same, with injected classifier noise
python main.py --headless --fast --mode emg --emg noisy --duration 14   # no window, deterministic, prints telemetry
python main.py --export-urdf arm.urdf           # write the generated URDF (for ROS/other tools)
python main.py --config config/calibration_example.json                 # use calibration overrides

python examples/example_emg_control.py          # EMG chain end-to-end, prints latency
python examples/example_pick_and_place.py [--gui]   # physics pick & place of the cube
python -m pytest -q                              # 70 tests
```

Each run writes `logs/session_NNN.csv` (disable with `--no-log`).

## Architecture

```
 EMG device (8 EMG + 3 IMU) ─► Gesture model (Random Forest)          ← YOUR code, later
                                    │  {"gesture","confidence","timestamp"}
                                    ▼
 emg/emg_controller.py   EMGSource.read() → confidence gate → sliding-window majority vote   (emg/gesture_mapper.py)
                                    ▼
 emg/gesture_mapper.py   Gesture Mapper → MotionCommand        ◄── keyboard / GUI / ROS produce the SAME MotionCommand
                                    ▼
 control/controller.py   RobotController.submit()  (thread-safe queue)
                                    ▼
 control/motion_planner.py   Motion Planner  (Cartesian jog / move-to / joint jog; keep-alive timeout)
                                    ▼
 robot/kinematics.py     Inverse kinematics (numerical DLS, limit-aware)
                                    ▼
 control/controller.py   limit clamp + collision veto  →  SimulatedServo (vel/accel-limited) targets
                                    ▼
 simulation/physics.py   PhysicsBackend: PyBullet (gravity, contacts, friction grasp)  |  KinematicBackend
                                    ▼
 ui/                     PyBullet GUI window (separate process) + terminal telemetry
```

Threads / processes (`runtime.py`):

| unit | rate | does |
|---|---|---|
| `emg-input` thread | 50 Hz (`EMG_POLL_HZ`) | poll source → smooth → MotionCommand → `submit()` |
| `control` thread | 240 Hz (`--control-hz`) | planner + IK (100 Hz), servos, physics step, safety, CSV log |
| `viewer-link` thread | 30 Hz | ships state to the viewer, receives its commands, prints the dashboard |
| `viewer` **process** | ~60 Hz | PyBullet GUI: sliders, buttons, keyboard, 3D scene |

The GUI is a separate process on purpose: any call into PyBullet's GUI client blocked 100+ ms in testing (and on
macOS it must own a main thread), which would starve a 240 Hz loop. Physics runs in a fast headless PyBullet client.

Package map: `config/` (robot + EMG parameters) · `robot/` (arm, joints, links, servo, gripper, kinematics/IK) ·
`simulation/` (physics backends, URDF generator, collision, objects, environment) · `control/` (MotionCommand,
planner, controller interface + simulated/hardware controllers, safety, pick-and-place) · `emg/` · `ui/` · `utils/` · `tests/`.

Rules the code follows: SI units internally (m, kg, s, rad; mm/deg only in UI/JSON/logs headers say `_m`/`_rad`);
EMG never maps to servo angles (always Gesture → MotionCommand → Planner → IK → Joint targets → Servo);
hardware control is separate from simulation; everything is deterministic given the same inputs
(IK restarts use a fixed seed; the planner has no clock of its own).

## Coordinate system & joints

Right-handed world frame fixed to the base (origin = centre of the base plate on the table):

* **+X** operator's right · **+Y** forward (arm reach direction when J1 = 0) · **+Z** up
* Zero pose (all joints 0): arm straight up, TCP at z = 450 mm.
* Gesture/direction convention: LEFT = −X, RIGHT = +X, UP = +Z, DOWN = −Z (FORWARD/BACKWARD = ±Y via keys I/K).

| joint | name | axis | positive direction | placeholder limit |
|---|---|---|---|---|
| J1 | base rotation | +Z | counter-clockwise from above (toward −X, "left") | ±120° |
| J2 | shoulder | −X (pitch) | lean forward (toward +Y) | ±90° |
| J3 | elbow | −X | forward | ±120° |
| J4 | wrist pitch | −X | forward | ±120° |
| J5 | wrist rotation | link axis (roll) | CCW seen from the tool tip's base | ±150° |
| J6 | gripper | fingers (prismatic pair) | opening | 0° (closed) … 60° (open) |

Home pose (placeholder): J = (0, 20°, 50°, 110°, 0, open) → tool pointing straight down, TCP ≈ (0, 136, 64) mm.

Link chain (lengths are configurable; they add to 465 mm): base plate 12 mm → link1 68 → link2 110 (upper arm) →
link3 105 (forearm) → link4 65 (wrist) → end-effector 105 (palm + fingers). TCP = grip point 90 mm along the tool axis.

## Kinematics

Homogeneous 4×4 transforms, built from the config (`robot/kinematics.py`):

```
T_tcp = Π_{i=1..5} [ Trans(0,0,len(link_{i-1})) · Rot(axis_i, q_i) ] · Trans(0,0,tcp_offset)
```

`RobotArm.forward_kinematics(q)` → `Pose(position xyz, rpy roll/pitch/yaw, matrix 4×4)`;
RPY convention `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. Note a tool pointing straight down reports roll = ±180°.

**The arm has 5 pose DOF** (J6 is the gripper), so a general 6-D pose is *not always reachable*. IK
(`robot/kinematics.py: NumericalIKSolver`, damped least squares with joint-limit clamping and deterministic restarts)
therefore has three modes:

| call | constrains | use |
|---|---|---|
| `inverse_kinematics(pos)` | position (3) | default |
| `inverse_kinematics(pos, tool_pitch=φ)` | position + tool tilt φ (0 = up, π = down) (4) | EMG jogging, pick & place |
| `inverse_kinematics(pos, orientation=(r,p,y))` | full pose (6) | succeeds only if that orientation is achievable |

It returns an `IKResult(success, joint_angles, position_error, orientation_error, message)`. Failures are explicit:
*"Target position unreachable: 500 mm from shoulder, max reach 370 mm"*, *"…within joint limits (best residual …)"*,
*"…reachable but requested orientation is not…"*. Joint limits are never silently violated. Results are cached;
warm local solves take ≈0.6 ms. `IKSolver` is an abstract base so an analytical solver can be added later.

## Configuration / calibration

All in `config/robot_config.py` (dataclasses) and overridable **without code changes** by a JSON file in human
units (deg, mm, kg, kg·cm) – see `config/calibration_example.json`:

```json
{ "joints": { "joint_1": {"zero_offset": 0, "direction": 1, "min_angle": -90, "max_angle": 90, "home_position": 0},
              "joint_2": {"torque_limit_kgcm": 17, "max_velocity_deg_s": 90} },
  "links":  { "link2": {"length_mm": 110, "mass_kg": 0.15} },
  "tcp_offset_mm": 90 }
```

Per joint: `direction`, `zero_offset`, `min/max`, `home`, `max_velocity`, `max_acceleration` (and deceleration),
`torque_limit`. Per link: `length`, `mass`, `com`, `size` (visual+collision box), optional `inertia`
(default: box approximation). `direction`/`zero_offset` map joint angle → hardware angle
(`hw = direction·q + zero_offset`) and are used by the hardware path; the simulator works in joint space.
EMG tunables are in `config/emg_config.py` (`EMG_CONFIDENCE_THRESHOLD = 0.75`, `SMOOTHING_WINDOW`, speed, timeout …).

## Controls

**Manual mode** (`python main.py`) – hold a key to jog (one motion key at a time):

| key | action |
|---|---|
| Q / A | J1 + / − |
| W / S | J2 |
| E / D | J3 |
| R / F | J4 |
| T / G | J5 |
| Y / H | gripper open / close |
| ← → ↑ ↓ | end effector LEFT (−X) / RIGHT (+X) / UP (+Z) / DOWN (−Z) |
| I / K | forward / back (+Y / −Y) |
| Z | home |
| **ESC** | **emergency stop** (latching) |
| X | reset: explicitly re-arm after e-stop / fault |

Window sidebar: J1–J6 sliders (degrees, drag = command), Cartesian X/Y/Z/Roll/Pitch/Yaw sliders with
**GO to XYZ (tool down)** and **GO to full pose** buttons, and **HOME / RESET / STOP / OPEN / CLOSE / E-STOP**.
(PyBullet sliders cannot be moved programmatically, so they show what *you* last set, not the live pose.)

**EMG mode** (`--mode emg`): hold **1** left · **2** right · **3** up · **4** down · **5** fist_close · **0** rest;
releasing a key = rest. ESC / X still work. Telemetry (FPS-independent): a live dashboard in the terminal
(sim time, control Hz, J1–J6, XYZ + RPY, gripper, gesture, confidence, command, collision, safety) and a one-line
status in the 3D window.

## EMG interface

```python
from emg.emg_controller import EMGController, EMGSource
emg = EMGController(source, sink=controller.submit, status_callback=controller.set_emg_status)
emg.connect(); emg.start()                  # input thread polls source at 50 Hz
emg.process_gesture("left", confidence=0.94)   # or call it yourself, one prediction at a time
emg.process_prediction({"gesture": "left", "confidence": 0.94, "timestamp": 1720000000.123})
```

Per prediction: (1) confidence < threshold (0.75) → counted as REST; (2) sliding-window majority vote
(window 5, winner needs ≥ 60 % = 3 votes, else REST) – so `L L L L R` → LEFT, but `L R L REST R` → REST, and a single
sample can never start a motion; (3) the smoothed gesture becomes a `MotionCommand`:

| gesture | MotionCommand |
|---|---|
| left / right / up / down | `CARTESIAN` LEFT(−X) / RIGHT / UP / DOWN, continuous at `EMG_SPEED_M_S` (40 mm/s) |
| fist_close | `GRIPPER` CLOSE |
| rest | `HOLD` (stop generating motion, keep position) |

Continuous commands are keep-alive: if no new command arrives for `COMMAND_TIMEOUT_S` (0.3 s) – EMG dropout,
crash – the planner stops by itself. `FIST_CLOSE_TOGGLES = True` makes a fist toggle close/open instead (default
follows the spec: close only; open with `Y` or the button). Unknown gesture labels are treated as REST.

`MotionCommand` (`control/motion_command.py`) is input-agnostic: `command_type` (CARTESIAN, MOVE_TO, JOINT,
JOINT_TARGET, GRIPPER, HOME, STOP, HOLD, ESTOP, RESET), `direction`, `magnitude` (None = continuous), `duration`,
`speed`, plus provenance (`source`, `gesture`, `confidence`). Magnitudes/speeds are SI; helpers such as
`MotionCommand.cartesian("LEFT", magnitude_mm=20, speed_mm_s=50)` take mm.

## Replacing the mock EMG with your classifier

Only the *source* changes. Implement `EMGSource.read()` returning the newest prediction (or `None`), e.g. wrapping
`code/realtime_pipeline.py` / your trained Random Forest:

```python
from emg.emg_controller import EMGSource, EMGController

class RandomForestEMGSource(EMGSource):
    def __init__(self, pipeline):                # your existing 8-EMG + 3-IMU feature pipeline + RF model
        self.pipeline = pipeline
    def connect(self):    self.pipeline.start()
    def disconnect(self): self.pipeline.stop()
    def read(self):
        r = self.pipeline.latest_prediction()    # -> label, probability, time of the window
        if r is None:
            return None
        return {"gesture": r.label,                 # "left" | "right" | "up" | "down" | "fist_close" | "rest"
                "confidence": float(r.proba.max()), # RF predict_proba max
                "timestamp": r.time}                # epoch seconds → enables `classifier_age_ms`

emg = EMGController(RandomForestEMGSource(pipe), sink=controller.submit,
                    status_callback=controller.set_emg_status, latency=controller.latency)
emg.connect(); emg.start()
```

For a quick callable instead of a class use `CallbackEMGSource(fn)`. Label names are normalised (`"Fist Close"`,
`"fist"` → `fist_close`; add aliases in `config/emg_config.py: GESTURE_ALIASES`). Set the poll rate
(`poll_hz=`) a bit above your classifier's output rate. `runtime.py` builds the mock source in one place
(`SimulationRuntime.__init__`); put your source there (or build your own entry script) – nothing in `robot/`,
`simulation/` or `control/` changes.

## Connecting the physical LewanSoul controller

`control/hardware_controller.py: HardwareRobotController` implements the **same** `RobotController` interface
(`move_joint(s)`, `move_cartesian`, `open_gripper`, `close_gripper`, `stop`, `emergency_stop`, `reset`, `home`,
`submit`, `snapshot`) but its transport is a stub. To finish it:

1. Open the serial/USB link to the 6-channel bus-servo controller (port/baud from its manual).
2. Encode "servo N → position P in T ms" frames per the controller's protocol document (**nothing about the
   frame format is assumed here**). `HardwareRobotController.joint_to_ticks(q)` already applies your calibrated
   `direction`/`zero_offset`; the tick scale (`ServoConfig.ticks_per_rad`, `center_ticks`) is a placeholder – verify it.
3. Reuse the simulator's `MotionPlanner`, IK, `SafetyMonitor` and `SimulatedServo` as the motion generator, and
   send each tick's servo angle to the board; read back position/voltage for telemetry and the low-voltage alarm.
4. Calibrate directions, offsets, limits and link lengths in the JSON file; verify with the arm unloaded first.

Then pass the hardware controller where the runtime uses `SimulatedRobotController` – the EMG layer does not care.
Real servos can stall and brown out: keep ESC/e-stop within reach for first runs. ROS 2 can be added later as another
producer of `MotionCommand`s (topic → `submit`) and a publisher of `snapshot()`; the core has no ROS dependency.

## Safety

`control/safety.py` + `simulation/collision.py` + the controller:

* **Joint limits** – targets are clamped, `"J2 joint limit reached"` is raised.
* **Collision** – every commanded pose is checked analytically (link capsules vs table and vs each other); a
  colliding target is **rejected** (old target kept, planner path dropped) with `"Collision detected: target rejected (…)"`.
  PyBullet contacts (robot vs table/objects; finger–object grasp contact is allowed) add `"Collision detected: …"` warnings.
* **Overload** – static gravity torque per joint vs the servo torque limit: warning at once,
  `"Potential servo overload"`, latched FAULT if sustained 0.5 s.
* **Locked-rotor-like** – physics joint angle deviating from the commanded servo angle > 0.12 rad for 0.5 s → FAULT
  (the gripper is exempt: holding an object is normal).
* **Unreachable IK / workspace** – `"Target position unreachable…"`, motion stops instead of continuing blindly.
* **Emergency stop** – `emergency_stop()` / ESC: all servos stop on the spot (no ramp), pending commands are
  dropped, commands are refused until an explicit `reset()` (X key / RESET button). `stop()` is a non-latching halt.

## Latency

Measured by `controller.latency` (printed at shutdown and by the examples), on an Apple-silicon laptop,
scripted noisy EMG, 240 Hz control, PyBullet physics:

| stage | mean | p95 | max |
|---|---|---|---|
| EMG prediction → MotionCommand | 0.01 ms | 0.03 ms | ≈2–3 ms |
| command → first joint target (planner + IK) | ≈3 ms | ≈3 ms | **≈55 ms** |

Both targets (< 20 ms) are met on average. The ~55 ms maximum comes from rare cases when the held tool pitch becomes
infeasible (e.g. wrist limit reached while jogging UP) and the planner falls back to a second IK solve; it is a known
worst case, not a typical one. Keep-alive timeout (0.3 s) and the 3-of-5 vote add deliberate *decision* latency
(≈ 3 samples ≈ 60 ms at 50 Hz) on top – tune with `window` / `min_agreement`.

## Tests

`python -m pytest -q` (70 tests, ≈25 s): forward kinematics (closed-form checks, Jacobian vs finite differences),
IK (round trips, limits, unreachable messages, determinism/cache), servo dynamics (no teleport, velocity/accel limits,
no overshoot), joint limits, gripper, collision (floor, self, veto), emergency stop + reset, EMG mapping, confidence
threshold, smoothing, dropout timeout, latency budget, calibration JSON, URDF validity, hardware-controller interface,
and – if PyBullet is installed – physics tracking and a real pick-and-place of the cube.

## Development phases & known limitations

Phases 1–9 are implemented (model, FK, joint control, IK, physics + collision, gripper, GUI, MotionCommand layer,
mock EMG). Phase 10 = plug in your classifier (one `EMGSource`, above). Phase 11 = finish `HardwareRobotController`.

Known limitations / honest notes:

* **All geometry/limits/masses/servo speeds are placeholders** until calibrated; physics tuning (motor gains) is only
  validated on the placeholder arm.
* 5 pose DOF: arbitrary 6-D end-effector poses are not reachable; EMG jogging holds the tool pitch while feasible and
  otherwise continues from the nearest feasible pitch.
* Collision checking is capsule/plane based (conservative approximation), not mesh-accurate; objects are only checked
  via physics contacts.
* The visual model is simple boxes (aluminum links, dark servo housings are only partly visible because PyBullet's
  viewer tints a whole link with one color). Functional correctness was prioritised over graphics.
* PyBullet sliders cannot be repositioned programmatically (they show your last input, not the live pose).
* Debug text in PyBullet's GUI is slow (~75 ms/update); full telemetry therefore lives in the terminal dashboard.
* Not verified: behaviour with a real EMG device, with the physical arm, or on Windows/Linux.
