#!/usr/bin/env python3
"""6-DOF robotic arm simulator driven by EMG gestures - everything in ONE file.

Just run it (no arguments needed):

    python robotic_arm.py

That starts every component together:

    * PyBullet physics            (control process, 120 Hz, DIRECT client)
    * 3D viewer window            (separate process, keyboard-driven)
    * control loop thread         (planner + IK + servos + safety + collision veto + CSV log in ./logs)
    * EMG input thread            (mock EMG source -> confidence gate -> smoothing -> MotionCommand)
    * terminal telemetry dashboard

Keys in the 3D window (click it first):
    Q/A W/S E/D R/F T/G Y/H   jog joints J1..J6        arrows + I/K   jog the tool (X/Z, Y)
    1 left  2 right  3 up  4 down  5 fist(close)  0 rest      <- mock EMG gestures (hold the key)
    O open gripper   C close gripper   SPACE stop   Z home   X reset after a stop   ESC emergency stop
    P  pick-and-place demo

Other ways to run it:

    python robotic_arm.py --emg scripted                       # scripted EMG gesture demo in the 3D view
    python robotic_arm.py --mode emg                           # only EMG drives the arm (no manual jogging)
    python robotic_arm.py --panel                              # add PyBullet's slider/button sidebar (slow on M1)
    python robotic_arm.py --headless --emg noisy --fast --duration 12   # console-only, deterministic
    python robotic_arm.py --control-hz 240                     # the original physics rate
    python robotic_arm.py --export-urdf arm.urdf

MacBook Air M1 notes (what was tuned, and why):
    * BLAS/OpenMP pinned to 1 thread (set before numpy loads): the maths is tiny, thread hand-offs only cost.
    * Control/physics rate 240 -> 120 Hz and resting objects may sleep: ~2x less CPU / heat on a fanless Air.
    * Kinematics: cached forward kinematics, closed-form rotations, analytic pitch Jacobian (IK ~2x faster).
    * Viewer: PyBullet's sidebar costs ~8 ms per widget per frame on the M1's OpenGL driver (21 widgets ->
      5 fps), so the default view has no sidebar (-> ~30 fps); shadows are off; window sized for a 13" screen.
    * caffeinate keeps App Nap from throttling the real-time timers; warns if Python runs under Rosetta.

Setup (Apple Silicon, NATIVE arm64 Python 3.10-3.13):   pip install numpy pybullet
Without PyBullet the arm still runs headless on the pure-NumPy kinematic backend.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------
# Apple-Silicon (M1) tuning. Must happen BEFORE numpy is imported.
# This model only does tiny (3x3 .. 6x5) matrix maths, 240+ times per second. Letting Accelerate/OpenBLAS
# fan each of those out over all cores costs far more in thread hand-offs than it saves, and keeps every
# core awake on a fanless MacBook Air. One BLAS thread is both faster and cooler here.
# ----------------------------------------------------------------------------------------------------------
import os as _os

for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import copy
import csv
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

IS_MACOS = sys.platform == "darwin"


def running_under_rosetta() -> bool:
    """True when an Intel (x86_64) Python is being translated on an Apple-Silicon Mac (2-3x slower)."""
    if not IS_MACOS:
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True,
                             timeout=2).stdout.strip()
        return out == "1"
    except Exception:
        return False


# ====================================================================================================
# MATH UTILITIES   (from utils/math_utils.py)
# ====================================================================================================
# Small SO(3)/SE(3) helpers. SI units (metres, radians) throughout.
#
# Euler convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (extrinsic x-y-z, a.k.a. ZYX).

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rot_axis(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about a (not necessarily unit) axis."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def make_transform(R: np.ndarray | None = None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def translation(x: float, y: float, z: float) -> np.ndarray:
    return make_transform(None, (x, y, z))


def rotation_transform(axis, angle: float) -> np.ndarray:
    return make_transform(rot_axis(axis, angle), None)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_to_matrix. pitch in [-pi/2, pi/2]; roll = 0 at gimbal lock."""
    sp = -float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    cos_a = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Near pi: axis from the symmetric part.
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] < 1e-12:
            return np.zeros(3)
        for i in range(3):
            if i != k:
                axis[i] = math.copysign(axis[i], B[k, i])
        axis /= np.linalg.norm(axis)
        return axis * angle
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (angle / (2.0 * math.sin(angle)))


def rotation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector that takes R_current onto R_target."""
    return so3_log(R_target @ R_current.T)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def segment_distance(p1, q1, p2, q2) -> float:
    """Minimum distance between segments p1-q1 and p2-q2 (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, clamp(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, clamp(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = clamp((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, clamp((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        sq = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / sq, (R[0, 2] - R[2, 0]) / sq, (R[1, 0] - R[0, 1]) / sq, 0.25 * sq)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    sq = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = [0.0] * 4
    q[i] = 0.25 * sq
    q[j] = (R[j, i] + R[i, j]) / sq
    q[k] = (R[k, i] + R[i, k]) / sq
    q[3] = (R[k, j] - R[j, k]) / sq
    return tuple(q)


# ====================================================================================================
# ROBOT CONFIGURATION & CALIBRATION   (from config/robot_config.py)
# ====================================================================================================
# Robot description + calibration parameters (SI units: m, kg, s, rad, N*m).
#
# Only the figures in ``PUBLISHED_SPEC`` come from the kit's published description.
# EVERYTHING ELSE below (link lengths, masses, joint limits, servo speeds, torque
# per joint ...) is a PLACEHOLDER chosen to be plausible and to add up to the
# published overall height/weight. Each is tagged ``# TODO: calibrate using
# physical Hiwonder arm``.  Override them without touching code via a JSON file
# (see ``load_config`` and ``config/calibration_example.json``).
#
# Kinematic frame convention (world == base frame, right-handed):
#     +X : operator's right      +Y : forward (arm reach direction at J1 = 0)
#     +Z : up                    origin: centre of the base plate on the table
# Zero pose = every joint at 0 -> arm points straight up.
#     J1 : rotation about +Z  (positive = counter-clockwise seen from above, i.e. toward -X / "left")
#     J2-J4 : pitch about local -X (positive = lean FORWARD toward +Y)
#     J5 : roll about the link axis
#     J6 : gripper (not part of the pose chain)

GRAVITY = 9.81
KGCM_TO_NM = 0.0980665
deg = math.radians

# Figures published for the kit (informational; not all are used by the model).
PUBLISHED_SPEC = {
    "overall_size_m": (0.285, 0.120, 0.465),   # W x D x H
    "total_mass_kg": 1.24,
    "servo_count": 6,
    "max_torque_kgcm": 17.0,
    "controller": "LewanSoul / STM32 6-ch bus servo controller",
    "controller_size_mm": (70.3, 80.2),
    "controller_voltage_v": (6.4, 8.4),
    "controller_features": [
        "BUS servo communication", "PS2 controller", "Android/iOS control",
        "PC graphical control", "over-current protection", "low-voltage alarm",
        "offline operation", "8 MB flash",
    ],
}


@dataclass
class JointLimit:
    min_angle: float
    max_angle: float
    max_velocity: float
    max_acceleration: float
    max_deceleration: Optional[float] = None  # defaults to max_acceleration

    @property
    def decel(self) -> float:
        return self.max_acceleration if self.max_deceleration is None else self.max_deceleration


@dataclass
class ServoConfig:
    torque_limit: float                       # N*m
    direction: int = 1                        # +1 / -1: hardware angle = direction * joint + zero_offset
    zero_offset: float = 0.0                  # rad of hardware angle at joint angle 0
    model: str = "digital metal-gear bus servo"
    # Hardware count conversion (used only by HardwareRobotController).
    # TODO: verify against the actual servo datasheet / LewanSoul protocol.
    ticks_per_rad: float = 1000.0 / deg(240.0)
    center_ticks: float = 500.0


@dataclass
class JointConfig:
    joint_id: int                             # 1..6
    name: str
    role: str
    axis: tuple                               # in the joint's own (parent-aligned) frame
    limit: JointLimit
    servo: ServoConfig
    home: float = 0.0


@dataclass
class LinkConfig:
    name: str
    length: float                             # m, along local +Z to the next joint origin
    mass: float                               # kg
    com: tuple                                # m, in link frame
    size: tuple                               # (width, depth) of the box used for visual/collision
    inertia: Optional[tuple] = None           # (Ixx, Iyy, Izz); None -> box approximation

    def inertia_diag(self) -> tuple:
        if self.inertia is not None:
            return self.inertia
        w, d, L = self.size[0], self.size[1], max(self.length, 1e-3)
        m = self.mass
        return (m / 12 * (d * d + L * L), m / 12 * (w * w + L * L), m / 12 * (w * w + d * d))


@dataclass
class GripperConfig:
    finger_travel: float = 0.030              # m per finger (width 2x -> 60 mm)   # TODO: calibrate
    finger_thickness: float = 0.008
    finger_length: float = 0.060
    finger_depth: float = 0.014
    finger_mass: float = 0.020
    palm_length: float = 0.045                # fingers start this far along the EE link
    max_grip_force: float = 8.0               # N per finger (simulation motor cap)   # TODO


@dataclass
class WorkspaceLimits:
    x: tuple = (-0.30, 0.30)
    y: tuple = (-0.30, 0.32)
    z: tuple = (0.008, 0.42)                  # z_min keeps the tool above the table


@dataclass
class RobotConfig:
    name: str
    joints: list
    links: list
    gripper: GripperConfig
    tcp_offset: float                         # m from J5 origin along tool +Z to the grip point
    workspace: WorkspaceLimits
    base_size: tuple = (0.140, 0.120)         # base plate width, depth (placeholder)
    table_z: float = 0.0

    def joint(self, joint_id: int) -> JointConfig:
        return self.joints[joint_id - 1]

    def link(self, name: str) -> LinkConfig:
        for l in self.links:
            if l.name == name:
                return l
        raise KeyError(name)

    @property
    def total_mass(self) -> float:
        return sum(l.mass for l in self.links)

    @property
    def total_height(self) -> float:
        return sum(l.length for l in self.links)

    @property
    def home_angles(self) -> list:
        return [j.home for j in self.joints]


def default_config() -> RobotConfig:
    # TODO: calibrate using physical Hiwonder arm -- all numbers below are placeholders.
    t17 = 17.0 * KGCM_TO_NM     # servos stated as "up to 17 kg*cm"
    t10 = 10.0 * KGCM_TO_NM     # TODO: weaker servos assumed on J1/J4/J5/J6

    def lim(lo, hi, vmax=deg(115), amax=8.0):
        return JointLimit(deg(lo), deg(hi), vmax, amax)

    joints = [
        JointConfig(1, "J1_base_rotation", "Base rotation", (0, 0, 1), lim(-120, 120), ServoConfig(t10), 0.0),
        JointConfig(2, "J2_shoulder", "Shoulder pitch", (-1, 0, 0), lim(-90, 90), ServoConfig(t17), deg(20)),
        JointConfig(3, "J3_elbow", "Elbow pitch", (-1, 0, 0), lim(-120, 120), ServoConfig(t17), deg(50)),
        JointConfig(4, "J4_wrist_pitch", "Wrist pitch", (-1, 0, 0), lim(-120, 120), ServoConfig(t10), deg(110)),
        JointConfig(5, "J5_wrist_rotation", "Wrist roll", (0, 0, 1), lim(-150, 150), ServoConfig(t10), 0.0),
        # J6 = gripper actuation: 0 rad = closed, max = fully open.
        JointConfig(6, "J6_gripper", "Gripper", (0, 0, 1), lim(0, 60, vmax=deg(115)), ServoConfig(t10), deg(60)),
    ]
    links = [
        #            name     length  mass  com (x,y,z)          (w, d)
        LinkConfig("base",  0.012, 0.630, (0, 0, 0.006),  (0.140, 0.120)),   # large metal base plate
        LinkConfig("link1", 0.068, 0.120, (0, 0, 0.034),  (0.060, 0.060)),   # turntable -> shoulder axis
        LinkConfig("link2", 0.110, 0.150, (0, 0, 0.055),  (0.040, 0.030)),   # upper arm
        LinkConfig("link3", 0.105, 0.120, (0, 0, 0.052),  (0.036, 0.028)),   # forearm
        LinkConfig("link4", 0.065, 0.080, (0, 0, 0.030),  (0.036, 0.030)),   # wrist bracket
        LinkConfig("ee",    0.105, 0.140, (0, 0, 0.050),  (0.050, 0.030)),   # palm + gripper (fingers incl.)
    ]
    return RobotConfig(
        name="hiwonder_6dof_placeholder",
        joints=joints,
        links=links,
        gripper=GripperConfig(),
        tcp_offset=0.090,
        workspace=WorkspaceLimits(),
    )


# --------------------------------------------------------------------------- calibration file
_LINK_ALIASES = {"base": "base", "link1": "link1", "link2": "link2", "link3": "link3",
                 "link4": "link4", "ee": "ee", "end_effector": "ee"}


def _joint_key(k) -> int:
    return int(str(k).lower().replace("joint_", "").replace("j", ""))


def apply_overrides(cfg: RobotConfig, data: dict) -> RobotConfig:
    """Apply a calibration dict in HUMAN units (deg, mm, kg*cm, deg/s) to ``cfg``."""
    cfg = copy.deepcopy(cfg)
    for key, o in data.get("joints", {}).items():
        j = cfg.joint(_joint_key(key))
        for name, val in o.items():
            if name == "direction":
                j.servo.direction = int(val)
            elif name in ("zero_offset", "zero_offset_deg"):
                j.servo.zero_offset = deg(val)
            elif name in ("min_angle", "min_deg", "joint_min"):
                j.limit.min_angle = deg(val)
            elif name in ("max_angle", "max_deg", "joint_max"):
                j.limit.max_angle = deg(val)
            elif name in ("home", "home_deg", "home_position"):
                j.home = deg(val)
            elif name == "max_velocity_deg_s":
                j.limit.max_velocity = deg(val)
            elif name == "max_acceleration_deg_s2":
                j.limit.max_acceleration = deg(val)
            elif name == "max_deceleration_deg_s2":
                j.limit.max_deceleration = deg(val)
            elif name == "torque_limit_kgcm":
                j.servo.torque_limit = val * KGCM_TO_NM
            else:
                raise KeyError(f"unknown joint calibration key '{name}' for joint {j.joint_id}")
    for key, o in data.get("links", {}).items():
        link = cfg.link(_LINK_ALIASES.get(key, key))
        for name, val in o.items():
            if name in ("length_mm", "link_length"):
                link.length = val / 1000.0
            elif name in ("mass_kg", "link_mass"):
                link.mass = float(val)
            elif name == "com_mm":
                link.com = tuple(v / 1000.0 for v in val)
            elif name == "size_mm":
                link.size = tuple(v / 1000.0 for v in val)
            else:
                raise KeyError(f"unknown link calibration key '{name}' for {link.name}")
    if "tcp_offset_mm" in data:
        cfg.tcp_offset = data["tcp_offset_mm"] / 1000.0
    if "gripper" in data:
        for name, val in data["gripper"].items():
            if name.endswith("_mm"):
                setattr(cfg.gripper, name[:-3], val / 1000.0)
            else:
                setattr(cfg.gripper, name, val)
    if "workspace_mm" in data:
        for axis, (lo, hi) in data["workspace_mm"].items():
            setattr(cfg.workspace, axis, (lo / 1000.0, hi / 1000.0))
    return cfg


def load_config(path: str | Path | None = None) -> RobotConfig:
    """Default placeholder config, optionally overridden by a JSON calibration file."""
    cfg = default_config()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        return apply_overrides(cfg, json.load(fh))


# ====================================================================================================
# EMG CONFIGURATION   (from config/emg_config.py)
# ====================================================================================================
# EMG-layer tunables. Nothing here touches the robot core.

GESTURES = ("left", "right", "up", "down", "fist_close", "rest")

# Predictions below this confidence are replaced by REST before smoothing.
EMG_CONFIDENCE_THRESHOLD = 0.75

# Sliding-window majority vote.
SMOOTHING_WINDOW = 5            # number of recent predictions considered
SMOOTHING_MIN_AGREEMENT = 0.6   # fraction of the window the winner needs, else REST

# Cartesian jog performed while a direction gesture is held.
EMG_SPEED_M_S = 0.040           # 40 mm/s
# Continuous commands expire if not refreshed (EMG dropout -> robot stops itself).
COMMAND_TIMEOUT_S = 0.30
EMG_POLL_HZ = 50

# gesture -> (command_type, direction).  REST -> HOLD (stop moving, keep position).
GESTURE_TO_COMMAND = {
    "left": ("CARTESIAN", "LEFT"),       # -X
    "right": ("CARTESIAN", "RIGHT"),     # +X
    "up": ("CARTESIAN", "UP"),           # +Z
    "down": ("CARTESIAN", "DOWN"),       # -Z
    "fist_close": ("GRIPPER", "CLOSE"),
    "rest": ("HOLD", "NONE"),
}

# Accepted spellings coming from a classifier.
GESTURE_ALIASES = {
    "fist": "fist_close", "close": "fist_close", "fistclose": "fist_close", "fist close": "fist_close",
    "idle": "rest", "none": "rest", "neutral": "rest", "hold": "rest",
}

# If True, a fist while the gripper is closed re-opens it (toggle). Default follows the spec:
# FIST_CLOSE only closes; opening is done from keyboard / GUI.
FIST_CLOSE_TOGGLES = False


# ====================================================================================================
# LOGGING, CSV SESSION LOG, LATENCY STATS   (from utils/logger.py)
# ====================================================================================================
# Console logging, CSV session logging and latency statistics.

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
        _CONFIGURED = True
    return logging.getLogger(name)


CSV_FIELDS = (["wall_time", "sim_time", "gesture", "confidence", "command"]
              + [f"j{i}_rad" for i in range(1, 7)]
              + ["x_m", "y_m", "z_m", "collision", "safety"])


class SessionLogger:
    """Writes logs/session_NNN.csv (SI units: rad, m, s). Thread-safe, rate-limited."""

    def __init__(self, directory: str | Path = "logs", rate_hz: float = 50.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        nums = [int(m.group(1)) for p in self.dir.glob("session_*.csv")
                if (m := re.match(r"session_(\d+)\.csv", p.name))]
        self.path = self.dir / f"session_{(max(nums) + 1 if nums else 1):03d}.csv"
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(CSV_FIELDS)
        self._lock = threading.Lock()
        self._period = 1.0 / rate_hz
        self._last = -1e9

    def log(self, sim_time: float, gesture, confidence, command, joint_angles, tcp, collision, safety,
            force: bool = False) -> bool:
        if not force and sim_time - self._last < self._period:
            return False
        self._last = sim_time
        row = [f"{time.time():.4f}", f"{sim_time:.4f}", gesture or "", "" if confidence is None else f"{confidence:.3f}",
               command or ""] + [f"{a:.5f}" for a in joint_angles] + [f"{v:.5f}" for v in tcp] + [collision, safety]
        with self._lock:
            if not self._fh.closed:
                self._w.writerow(row)
        return True

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class LatencyTracker:
    """Rolling latency samples in milliseconds, keyed by stage name."""

    def __init__(self, maxlen: int = 2000):
        self._d: dict = {}
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def record(self, name: str, ms: float) -> None:
        with self._lock:
            self._d.setdefault(name, deque(maxlen=self._maxlen)).append(ms)

    def stats(self, name: str) -> dict:
        with self._lock:
            v = np.array(self._d.get(name, ()), dtype=float)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    def summary(self) -> str:
        lines = []
        for k in sorted(self._d):
            s = self.stats(k)
            lines.append(f"{k}: n={s['n']} mean={s['mean']:.2f} ms p95={s['p95']:.2f} ms max={s['max']:.2f} ms")
        return "\n".join(lines) or "no latency samples"


# ====================================================================================================
# SERVO MODEL   (from robot/servo.py)
# ====================================================================================================
# Simulated digital servo with velocity/acceleration-limited motion.
#
# The servo works in JOINT space (radians). ``direction`` / ``zero_offset`` only matter
# when converting to/from hardware angles or counts (``to_hardware`` / ``from_hardware``),
# which is what a future HardwareRobotController will use.

class SimulatedServo:
    def __init__(self, servo_id: int, limit: JointLimit, cfg: ServoConfig, home: float = 0.0):
        self.id = servo_id
        self.limit = limit
        self.cfg = cfg
        self.direction = cfg.direction
        self.offset = cfg.zero_offset
        self.torque_limit = cfg.torque_limit
        self.max_velocity = limit.max_velocity
        self.home_angle = home
        self.angle = home
        self.target_angle = home
        self.velocity = 0.0
        self.limit_hit = False          # set when the last set_target() had to clamp

    # ------------------------------------------------------------------ commands
    def set_target(self, angle: float) -> bool:
        """Set a target, clamped to the joint limits. Returns True if it had to clamp."""
        lo, hi = self.limit.min_angle, self.limit.max_angle
        clamped = angle < lo - 1e-9 or angle > hi + 1e-9
        self.target_angle = min(max(angle, lo), hi)
        self.limit_hit = clamped
        return clamped

    def stop(self) -> None:
        """Freeze immediately (no deceleration ramp): target = current angle."""
        self.target_angle = self.angle
        self.velocity = 0.0

    def reset(self, angle: float | None = None) -> None:
        a = self.home_angle if angle is None else angle
        self.angle = self.target_angle = min(max(a, self.limit.min_angle), self.limit.max_angle)
        self.velocity = 0.0
        self.limit_hit = False

    @property
    def at_target(self) -> bool:
        return abs(self.target_angle - self.angle) < 1e-4 and abs(self.velocity) < 1e-3

    # ------------------------------------------------------------------ dynamics
    def update(self, dt: float) -> float:
        """Advance one step: accelerate -> cruise at max_velocity -> brake to arrive with ~zero speed.

        Braking starts when the stopping distance (v^2 / 2*decel, plus one step of lookahead) reaches the
        remaining error, so the servo neither overshoots nor teleports.
        """
        err = self.target_angle - self.angle
        if abs(err) < 1e-9 and abs(self.velocity) < 1e-9:
            return self.angle
        sgn = 1.0 if err >= 0 else -1.0
        acc, dec = self.limit.max_acceleration, self.limit.decel
        vv = self.velocity * sgn                       # speed toward the target (negative = moving away)
        stop_dist = vv * vv / (2.0 * dec) if vv > 0 else 0.0
        if vv > 0 and stop_dist + vv * dt >= abs(err):
            vv -= dec * dt                             # brake
        elif vv < self.max_velocity:
            vv = min(self.max_velocity, vv + acc * dt) # accelerate (also reverses if moving away)
        elif vv > self.max_velocity:
            vv = max(self.max_velocity, vv - dec * dt) # e.g. max_velocity lowered while moving
        vv = max(vv, -self.max_velocity)
        self.velocity = vv * sgn
        step = self.velocity * dt
        crossed = (self.target_angle - (self.angle + step)) * sgn <= 0.0
        if crossed or (abs(err) < 1e-5 and abs(self.velocity) < 2.0 * dec * dt):
            self.angle = self.target_angle             # arrive exactly; speed here is already ~ dec*dt
            self.velocity = 0.0
        else:
            self.angle += step
        return self.angle

    # ------------------------------------------------------------------ hardware mapping
    def to_hardware(self, joint_angle: float | None = None) -> float:
        """Joint angle (rad) -> hardware servo angle (rad)."""
        a = self.angle if joint_angle is None else joint_angle
        return self.direction * a + self.offset

    def from_hardware(self, hw_angle: float) -> float:
        return (hw_angle - self.offset) / self.direction

    def to_ticks(self, joint_angle: float | None = None) -> int:
        return int(round(self.cfg.center_ticks + self.cfg.ticks_per_rad * self.to_hardware(joint_angle)))


# ====================================================================================================
# JOINTS   (from robot/joints.py)
# ====================================================================================================
# Joint = static description (JointConfig) + its simulated servo.

class Joint:
    def __init__(self, cfg: JointConfig):
        self.cfg = cfg
        self.id = cfg.joint_id
        self.name = cfg.name
        self.limit = cfg.limit
        self.servo = SimulatedServo(cfg.joint_id, cfg.limit, cfg.servo, cfg.home)

    @property
    def label(self) -> str:
        return f"J{self.id}"

    @property
    def angle(self) -> float:
        return self.servo.angle

    @property
    def target(self) -> float:
        return self.servo.target_angle

    def within_limits(self, angle: float, tol: float = 1e-9) -> bool:
        return self.limit.min_angle - tol <= angle <= self.limit.max_angle + tol

    def clamp(self, angle: float) -> float:
        return min(max(angle, self.limit.min_angle), self.limit.max_angle)


# ====================================================================================================
# GRIPPER   (from robot/gripper.py)
# ====================================================================================================
# Two-finger parallel gripper driven by servo J6.
#
# opening: 0.0 = completely closed, 1.0 = completely open (J6 angle limits map linearly).

class Gripper:
    def __init__(self, joint: Joint, cfg: GripperConfig):
        self.joint = joint
        self.cfg = cfg

    # ---- helpers
    def _to_angle(self, value: float) -> float:
        lim = self.joint.limit
        return lim.min_angle + min(max(value, 0.0), 1.0) * (lim.max_angle - lim.min_angle)

    def _to_value(self, angle: float) -> float:
        lim = self.joint.limit
        return (angle - lim.min_angle) / (lim.max_angle - lim.min_angle)

    # ---- state
    @property
    def opening(self) -> float:
        return self._to_value(self.joint.servo.angle)

    @property
    def target_opening(self) -> float:
        return self._to_value(self.joint.servo.target_angle)

    @property
    def finger_travel(self) -> float:
        """Displacement of EACH finger from the closed position (m)."""
        return self.opening * self.cfg.finger_travel

    @property
    def width(self) -> float:
        return 2.0 * self.finger_travel

    # ---- commands
    def set_position(self, value: float) -> None:
        self.joint.servo.set_target(self._to_angle(value))

    def open(self) -> None:
        self.set_position(1.0)

    def close(self) -> None:
        self.set_position(0.0)


# ====================================================================================================
# LINKS   (from robot/links.py)
# ====================================================================================================
# Rigid link: geometry + inertial properties (all from LinkConfig, all configurable).

class Link:
    def __init__(self, index: int, cfg: LinkConfig):
        self.index = index            # 0 = base, 1..4 = arm links, 5 = end-effector
        self.cfg = cfg
        self.name = cfg.name
        self.length = cfg.length
        self.mass = cfg.mass
        self.com = np.asarray(cfg.com, dtype=float)
        self.size = cfg.size          # (width, depth) box used for visual + collision
        self.inertia = cfg.inertia_diag()

    @property
    def collision_radius(self) -> float:
        """Capsule radius used by the analytic collision checker."""
        return 0.5 * max(self.size)


# ====================================================================================================
# KINEMATICS & INVERSE KINEMATICS   (from robot/kinematics.py)
# ====================================================================================================
# Forward kinematics (homogeneous transforms), Jacobian, statics and numerical IK.
#
# Chain (5 pose joints; J6 is the gripper and does not move the TCP):
#
#     T_tcp = prod_{i=1..5} [ Trans(0,0,len(link_{i-1})) * Rot(axis_i, q_i) ] * Trans(0,0,tcp_offset)
#
# The arm has 5 pose DOF, so a general 6-D pose is NOT always reachable. IK therefore has
# three modes:  "position" (3 constraints),  "pitch" (position + tool pitch, 4),
# "full" (position + full orientation, 6 - succeeds only for reachable orientations).

N_POSE_JOINTS = 5


def _cross3(a, b) -> np.ndarray:
    """3-vector cross product; np.cross costs ~20x more than this on such tiny inputs."""
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


@dataclass
class Pose:
    position: np.ndarray            # (3,) m
    rpy: np.ndarray                 # (3,) rad  roll, pitch, yaw
    matrix: np.ndarray              # 4x4

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose":
        return cls(T[:3, 3].copy(), np.array(matrix_to_rpy(T[:3, :3])), T.copy())


class Kinematics:
    def __init__(self, config: RobotConfig):
        self.config = config
        links = config.links
        self._offsets = [np.array([0.0, 0.0, links[i].length]) for i in range(N_POSE_JOINTS)]
        self._axes = []
        for i in range(N_POSE_JOINTS):
            a = np.asarray(config.joints[i].axis, dtype=float)
            self._axes.append(a / np.linalg.norm(a))
        self._tcp = translation(0, 0, config.tcp_offset)
        # Fast-FK precomputation: Trans(offset) @ Rot(axis, q) == [[R, offset], [0, 1]] with
        # R = I + sin(q) K + (1 - cos(q)) K^2 (Rodrigues; K = skew(axis)).
        self._skew, self._skew2, self._A = [], [], []
        for i in range(N_POSE_JOINTS):
            x, y, z = self._axes[i]
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            A = np.eye(4)
            A[:3, 3] = self._offsets[i]
            self._skew.append(K)
            self._skew2.append(K @ K)
            self._A.append(A)
        self._eye3 = np.eye(3)
        self._frames_cache: dict = {}
        self.lower = np.array([j.limit.min_angle for j in config.joints[:N_POSE_JOINTS]])
        self.upper = np.array([j.limit.max_angle for j in config.joints[:N_POSE_JOINTS]])

    # ------------------------------------------------------------------ FK
    def frames(self, q) -> list:
        """[T_base, T_1, ..., T_5]; T_i is the frame of link i (origin on joint i axis).

        The returned matrices are shared through a small cache (the control step asks for the same pose
        several times: safety torques, collision check, logging): treat them as READ-ONLY.
        """
        q = np.ascontiguousarray(q[:N_POSE_JOINTS], dtype=float)
        key = q.tobytes()
        cache = self._frames_cache
        hit = cache.get(key)
        if hit is not None:
            return hit
        T = np.eye(4)
        out = [T]
        for i in range(N_POSE_JOINTS):
            qi = float(q[i])
            A = self._A[i].copy()          # fresh array: never mutate shared state (thread safety)
            A[:3, :3] = self._eye3 + math.sin(qi) * self._skew[i] + (1.0 - math.cos(qi)) * self._skew2[i]
            T = T @ A
            out.append(T)
        if len(cache) >= 256:               # bounded; plain dict ops are atomic under the GIL (multi-thread safe)
            cache.clear()
        cache[key] = out
        return out

    def forward(self, q) -> np.ndarray:
        return self.frames(q)[-1] @ self._tcp

    def pose(self, q) -> Pose:
        return Pose.from_matrix(self.forward(q))

    def joint_positions(self, q) -> np.ndarray:
        """(6,3): origins of J1..J5 and the gripper (J6) reference point (palm base)."""
        fr = self.frames(q)
        pts = [fr[i][:3, 3] for i in range(1, N_POSE_JOINTS + 1)]
        palm = fr[5] @ np.array([0, 0, self.config.gripper.palm_length, 1.0])
        pts.append(palm[:3])
        return np.array(pts)

    def tool_pitch(self, T: np.ndarray, q1: float) -> float:
        """Tilt of the tool axis in the arm's vertical plane: 0 = up, +90 deg = horizontal forward, 180 = down."""
        z = T[:3, 2]
        radial = z[0] * -math.sin(q1) + z[1] * math.cos(q1)   # component along the arm's forward direction
        return math.atan2(radial, z[2])

    # ------------------------------------------------------------------ Jacobian
    def jacobian(self, q) -> np.ndarray:
        """6x5 geometric Jacobian of the TCP in the world frame: [v; w]."""
        fr = self.frames(q)
        p_tcp = (fr[-1] @ self._tcp)[:3, 3]
        J = np.zeros((6, N_POSE_JOINTS))
        for i in range(N_POSE_JOINTS):
            T = fr[i + 1]
            a = T[:3, :3] @ self._axes[i]
            J[:3, i] = _cross3(a, p_tcp - T[:3, 3])
            J[3:, i] = a
        return J

    # ------------------------------------------------------------------ statics
    def com_positions(self, q) -> list:
        fr = self.frames(q)
        links = self.config.links
        return [(fr[k] @ np.append(links[k].com, 1.0))[:3] for k in range(len(links))]

    def gravity_torques(self, q, payload_kg: float = 0.0) -> np.ndarray:
        """Static torque magnitude (N*m) each pose joint must hold against gravity."""
        fr = self.frames(q)
        coms = self.com_positions(q)
        links = self.config.links
        tcp = (fr[-1] @ self._tcp)[:3, 3]
        g = np.array([0.0, 0.0, -GRAVITY])
        tau = np.zeros(N_POSE_JOINTS)
        for i in range(N_POSE_JOINTS):
            T = fr[i + 1]
            a, p = T[:3, :3] @ self._axes[i], T[:3, 3]
            t = 0.0
            for k in range(i + 1, len(links)):          # links carried by joint i+1 (child links)
                t += a @ _cross3(coms[k] - p, links[k].mass * g)
            if payload_kg:
                t += a @ _cross3(tcp - p, payload_kg * g)
            tau[i] = abs(t)
        return tau

    @property
    def max_reach(self) -> float:
        """Max distance from the shoulder (J2) axis to the TCP."""
        L = self.config.links
        return L[2].length + L[3].length + L[4].length + self.config.tcp_offset

    @property
    def shoulder_origin(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.config.links[0].length + self.config.links[1].length])


# ====================================================================== IK
@dataclass
class IKResult:
    success: bool
    joint_angles: np.ndarray
    position_error: float            # m
    orientation_error: float         # rad (0 if orientation not constrained)
    iterations: int
    message: str
    mode: str = "position"


class IKSolver(ABC):
    """Interface so an analytical solver can replace/augment the numerical one later."""

    @abstractmethod
    def solve(self, target_position, target_orientation=None, seed=None, tool_pitch=None,
              restarts=None) -> IKResult: ...


class NumericalIKSolver(IKSolver):
    """Damped-least-squares IK with joint-limit clamping and deterministic random restarts."""

    def __init__(self, kin: Kinematics, max_iters: int = 120, damping: float = 0.03,
                 pos_tol: float = 1e-3, ori_tol: float = math.radians(1.0),
                 restarts: int = 8, seed: int = 0, ori_weight: float = 0.15):
        self.kin = kin
        self.max_iters = max_iters
        self.damping = damping
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.restarts = restarts
        self.seed = seed
        self.ori_weight = ori_weight            # metres-per-radian trade-off between rows
        self._cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------ public
    def solve(self, target_position, target_orientation=None, seed=None, tool_pitch=None,
              restarts=None) -> IKResult:
        """
        target_position:    (x, y, z) m.
        target_orientation: (roll, pitch, yaw) rad or 3x3 matrix -> "full" mode.
        tool_pitch:         rad; constrain only the tool tilt (position + pitch) -> "pitch" mode.
        seed:               starting joint angles (5 or 6 values); default = home pose.
        restarts:           random restarts after the seed attempt fails (default: solver setting;
                            0 = purely local solve, as used for incremental tracking).
        """
        kin = self.kin
        p_t = np.asarray(target_position, dtype=float).reshape(3)
        R_t = None
        if target_orientation is not None:
            o = np.asarray(target_orientation, dtype=float)
            R_t = o if o.shape == (3, 3) else rpy_to_matrix(*o.reshape(3))
            mode = "full"
        elif tool_pitch is not None:
            mode = "pitch"
        else:
            mode = "position"
        q0 = np.array(kin.config.home_angles[:N_POSE_JOINTS]) if seed is None else np.asarray(seed, float)[:N_POSE_JOINTS]
        q0 = np.clip(q0, kin.lower, kin.upper)

        key = (mode, tuple(np.round(p_t, 6)), None if R_t is None else tuple(np.round(R_t.ravel(), 5)),
               None if tool_pitch is None else round(float(tool_pitch), 6), tuple(np.round(q0, 3)),
               self.restarts if restarts is None else restarts)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        reach = float(np.linalg.norm(p_t - kin.shoulder_origin))
        if reach > kin.max_reach + self.pos_tol:
            res = IKResult(False, q0, reach - kin.max_reach, 0.0, 0,
                           f"Target position unreachable: {reach*1000:.0f} mm from shoulder, "
                           f"max reach {kin.max_reach*1000:.0f} mm", mode)
        else:
            res = self._solve_with_restarts(p_t, R_t, tool_pitch, q0, mode,
                                            self.restarts if restarts is None else restarts)
        self._cache[key] = res
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return res

    # ------------------------------------------------------------------ internals
    def _errors(self, q, p_t, R_t, pitch_t, mode):
        T = self.kin.forward(q)
        e_p = p_t - T[:3, 3]
        if mode == "full":
            e_o = rotation_error(R_t, T[:3, :3])
        elif mode == "pitch":
            e_o = np.array([wrap_angle(pitch_t - self.kin.tool_pitch(T, q[0]))])
        else:
            e_o = np.zeros(0)
        return e_p, e_o

    def _pitch_row(self, q, pitch_t):
        """d(tool_pitch)/dq, analytic. pitch = atan2(rad, z_z) with rad = -z_x sin(q1) + z_y cos(q1) and z the
        tool axis; rotating joint i turns z at rate a_i x z (a_i = world axis of joint i). Exact, and one FK
        pass instead of the six a finite-difference row needs."""
        kin = self.kin
        fr = kin.frames(q)
        z = fr[-1][:3, 2]
        s1, c1 = math.sin(q[0]), math.cos(q[0])
        rad, zz = -z[0] * s1 + z[1] * c1, z[2]
        den = rad * rad + zz * zz
        row = np.zeros(N_POSE_JOINTS)
        if den < 1e-18:
            return row
        for i in range(N_POSE_JOINTS):
            dz = _cross3(fr[i + 1][:3, :3] @ kin._axes[i], z)
            drad = -dz[0] * s1 + dz[1] * c1
            if i == 0:
                drad += -z[0] * c1 - z[1] * s1
            row[i] = (zz * drad - rad * dz[2]) / den
        return row

    def _run(self, q, p_t, R_t, pitch_t, mode):
        kin, w = self.kin, self.ori_weight
        q = q.copy()
        e_p, e_o = self._errors(q, p_t, R_t, pitch_t, mode)
        for it in range(self.max_iters):
            if np.linalg.norm(e_p) < self.pos_tol and (mode == "position" or np.linalg.norm(e_o) < self.ori_tol):
                return q, e_p, e_o, it, True
            J = kin.jacobian(q)
            if mode == "position":
                Jm, e = J[:3], e_p
            elif mode == "full":
                Jm, e = np.vstack([J[:3], w * J[3:]]), np.concatenate([e_p, w * e_o])
            else:
                Jm, e = np.vstack([J[:3], w * self._pitch_row(q, pitch_t)]), np.concatenate([e_p, w * e_o])
            JJt = Jm @ Jm.T + (self.damping ** 2) * np.eye(Jm.shape[0])
            dq = Jm.T @ np.linalg.solve(JJt, e)
            n = np.linalg.norm(dq)
            if n > 0.35:
                dq *= 0.35 / n
            q = np.clip(q + dq, kin.lower, kin.upper)
            e_p, e_o = self._errors(q, p_t, R_t, pitch_t, mode)
        ok = np.linalg.norm(e_p) < self.pos_tol and (mode == "position" or np.linalg.norm(e_o) < self.ori_tol)
        return q, e_p, e_o, self.max_iters, ok

    def _solve_with_restarts(self, p_t, R_t, pitch_t, q0, mode, n_restarts) -> IKResult:
        rng = np.random.default_rng(self.seed)
        starts = [q0]
        for _ in range(n_restarts):
            s = rng.uniform(self.kin.lower, self.kin.upper)
            if mode != "full":
                s[4] = q0[4]               # wrist roll is redundant for position/pitch goals: keep it
            starts.append(s)
        best, best_cost, total = None, np.inf, 0
        for s in starts:
            q, e_p, e_o, it, ok = self._run(s, p_t, R_t, pitch_t, mode)
            total += it
            cost = np.linalg.norm(e_p) + self.ori_weight * np.linalg.norm(e_o)
            if ok:
                return IKResult(True, q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o)), total, "ok", mode)
            if cost < best_cost:
                best, best_cost = (q, e_p, e_o), cost
        q, e_p, e_o = best
        pe, oe = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o))
        if pe >= self.pos_tol:
            msg = (f"Target position unreachable within joint limits "
                   f"(best residual {pe*1000:.1f} mm, mode={mode})")
        else:
            msg = (f"Target position reachable but requested orientation is not "
                   f"(residual {math.degrees(oe):.1f} deg; the arm has only 5 pose DOF)")
        return IKResult(False, q, pe, oe, total, msg, mode)


# ====================================================================================================
# ROBOT ARM MODEL   (from robot/robot_arm.py)
# ====================================================================================================
# RobotArm: the kinematic + servo model of the 6-DOF arm. Knows nothing about EMG or physics.

class RobotArm:
    def __init__(self, config: RobotConfig | None = None, ik_solver=None):
        self.config = config or default_config()
        self.joints = [Joint(c) for c in self.config.joints]          # J1..J6
        self.links = [Link(i, c) for i, c in enumerate(self.config.links)]
        self.kinematics = Kinematics(self.config)
        self.ik_solver = ik_solver or NumericalIKSolver(self.kinematics)
        self.gripper = Gripper(self.joints[5], self.config.gripper)
        self.servo_parameters = {j.id: j.servo.cfg for j in self.joints}
        self.joint_limits = {j.id: j.limit for j in self.joints}
        self._warnings: list[str] = []
        self.reset()

    # ------------------------------------------------------------------ state access
    @property
    def joint_angles(self) -> np.ndarray:
        """Current servo angles of J1..J6 (rad)."""
        return np.array([j.angle for j in self.joints])

    @property
    def joint_targets(self) -> np.ndarray:
        return np.array([j.target for j in self.joints])

    @property
    def end_effector(self) -> Pose:
        return self.get_end_effector_pose()

    def forward_kinematics(self, joint_angles) -> Pose:
        """Pose (position, roll/pitch/yaw, 4x4 matrix) of the TCP for J1..J5 (a 6th value is ignored)."""
        return self.kinematics.pose(np.asarray(joint_angles, dtype=float)[:N_POSE_JOINTS])

    def get_end_effector_pose(self) -> Pose:
        return self.forward_kinematics(self.joint_angles)

    def get_joint_positions(self) -> np.ndarray:
        """(6,3) world positions of J1..J6."""
        return self.kinematics.joint_positions(self.joint_angles[:N_POSE_JOINTS])

    def inverse_kinematics(self, target_position, target_orientation=None, seed=None, tool_pitch=None) -> IKResult:
        seed = self.joint_targets if seed is None else seed
        return self.ik_solver.solve(target_position, target_orientation, seed=seed, tool_pitch=tool_pitch)

    # ------------------------------------------------------------------ commands (set servo TARGETS)
    def set_joint_angle(self, joint_id: int, angle: float) -> bool:
        """Target a single joint (1-based id). Clamps at the limit; returns True if clamped."""
        j = self.joints[joint_id - 1]
        clamped = j.servo.set_target(angle)
        if clamped:
            self._warnings.append(f"J{joint_id} joint limit reached")
        return clamped

    def set_joint_angles(self, joint_angles, immediate: bool = False) -> list:
        """Target J1..J5 (or J1..J6). Returns the list of joint ids that were clamped."""
        clamped = []
        for i, a in enumerate(joint_angles[:6]):
            if self.set_joint_angle(i + 1, float(a)):
                clamped.append(i + 1)
        if immediate:
            for j in self.joints:
                j.servo.reset(j.servo.target_angle)
        return clamped

    def home(self) -> None:
        for j in self.joints:
            j.servo.set_target(j.cfg.home)

    def stop(self) -> None:
        for j in self.joints:
            j.servo.stop()

    def reset(self) -> None:
        """Snap every servo to its home angle and clear warnings (simulation reset)."""
        for j in self.joints:
            j.servo.reset()
        self._warnings.clear()

    # ------------------------------------------------------------------ simulation step
    def update(self, dt: float) -> None:
        for j in self.joints:
            j.servo.update(dt)

    def drain_warnings(self) -> list:
        w, self._warnings = self._warnings, []
        return w

    def gravity_torques(self, payload_kg: float = 0.0) -> np.ndarray:
        return self.kinematics.gravity_torques(self.joint_angles[:N_POSE_JOINTS], payload_kg)


# ====================================================================================================
# MOTION COMMAND   (from control/motion_command.py)
# ====================================================================================================
# MotionCommand: the single, input-agnostic command format.
#
# Keyboard, GUI sliders, EMG and (future) ROS all emit these; nothing downstream knows
# which one produced a command. SI units: magnitude/target in metres or radians, speed in m/s
# (or rad/s for joints, opening/s for the gripper).

_ids = itertools.count(1)

# command_type values
CARTESIAN = "CARTESIAN"          # direction (+ optional magnitude) -> jog the TCP in world axes
MOVE_TO = "MOVE_TO"              # absolute TCP target (straight line)
JOINT = "JOINT"                  # jog one joint: joint=1..6, direction "+"/"-"
JOINT_TARGET = "JOINT_TARGET"    # absolute joint angle(s)
GRIPPER = "GRIPPER"              # direction OPEN / CLOSE / SET (target=0..1)
HOME = "HOME"
STOP = "STOP"                    # stop NOW (targets = current angles)
HOLD = "HOLD"                    # stop generating motion, keep current target (EMG "rest")
ESTOP = "ESTOP"
RESET = "RESET"

# World-frame unit vectors. LEFT = -X, RIGHT = +X, UP = +Z, DOWN = -Z, FORWARD = +Y.
DIRECTIONS = {
    "LEFT": (-1.0, 0.0, 0.0), "RIGHT": (1.0, 0.0, 0.0),
    "UP": (0.0, 0.0, 1.0), "DOWN": (0.0, 0.0, -1.0),
    "FORWARD": (0.0, 1.0, 0.0), "BACKWARD": (0.0, -1.0, 0.0),
}


@dataclass
class MotionCommand:
    command_type: str
    direction: str = "NONE"
    magnitude: Optional[float] = None      # total distance/angle; None = continuous until refresh timeout
    duration: Optional[float] = None       # s; explicit lifetime for continuous commands
    speed: Optional[float] = None          # m/s | rad/s | opening/s ; None = planner default
    joint: Optional[int] = None
    target: Optional[Tuple[float, ...]] = None       # absolute xyz (m) / joint angles (rad) / gripper value
    orientation: Optional[Tuple[float, float, float]] = None   # roll, pitch, yaw (rad), MOVE_TO only
    tool_pitch: Optional[float] = None     # rad, MOVE_TO only (hold/set tool tilt)
    # provenance / telemetry
    source: str = "unknown"
    gesture: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[float] = None      # classifier timestamp (epoch s), if any
    created_at: float = field(default_factory=time.perf_counter)   # monotonic, for latency
    command_id: int = field(default_factory=lambda: next(_ids))

    @property
    def is_continuous(self) -> bool:
        return self.magnitude is None

    def describe(self) -> str:
        if self.command_type == CARTESIAN:
            axis = {"LEFT": "X -", "RIGHT": "X +", "UP": "Z +", "DOWN": "Z -",
                    "FORWARD": "Y +", "BACKWARD": "Y -"}.get(self.direction, self.direction)
            return f"MOVE {axis}"
        if self.command_type == JOINT:
            return f"JOG J{self.joint}{self.direction}"
        if self.command_type == GRIPPER:
            return f"GRIPPER {self.direction}"
        if self.command_type == MOVE_TO and self.target is not None:
            return "MOVE_TO " + ",".join(f"{v*1000:.0f}" for v in self.target) + " mm"
        return self.command_type

    # ---- convenience constructors (mm / deg at the call site, SI inside)
    @classmethod
    def cartesian(cls, direction: str, magnitude_mm: float | None = None,
                  speed_mm_s: float | None = None, **kw) -> "MotionCommand":
        return cls(CARTESIAN, direction.upper(),
                   None if magnitude_mm is None else magnitude_mm / 1000.0,
                   speed=None if speed_mm_s is None else speed_mm_s / 1000.0, **kw)

    @classmethod
    def gripper(cls, direction: str, value: float | None = None, **kw) -> "MotionCommand":
        return cls(GRIPPER, direction.upper(), target=None if value is None else (float(value),), **kw)

    @classmethod
    def hold(cls, **kw) -> "MotionCommand":
        return cls(HOLD, "NONE", **kw)

    @classmethod
    def stop(cls, **kw) -> "MotionCommand":
        return cls(STOP, "NONE", **kw)

    @classmethod
    def home(cls, **kw) -> "MotionCommand":
        return cls(HOME, "NONE", **kw)

    @classmethod
    def move_to(cls, position, speed: float | None = None, orientation=None,
                tool_pitch: float | None = None, **kw) -> "MotionCommand":
        return cls(MOVE_TO, "NONE", target=tuple(float(v) for v in position), speed=speed,
                   orientation=None if orientation is None else tuple(orientation),
                   tool_pitch=tool_pitch, **kw)

    @classmethod
    def joint_jog(cls, joint: int, sign: int, speed: float | None = None, **kw) -> "MotionCommand":
        return cls(JOINT, "+" if sign >= 0 else "-", joint=joint, speed=speed, **kw)

    @classmethod
    def joint_target(cls, angles, joint: int | None = None, **kw) -> "MotionCommand":
        a = (angles,) if joint is not None else tuple(angles)
        return cls(JOINT_TARGET, "NONE", joint=joint, target=tuple(float(v) for v in a), **kw)


# ====================================================================================================
# SAFETY MONITOR   (from control/safety.py)
# ====================================================================================================
# Safety monitor inspired by the real controller's protections (limits, over-current, stall).
#
# Events carry a severity. WARNING events expire if they stop being re-reported; FAULT events
# are latched and require ``SimulatedRobotController.reset()``.

class Severity(IntEnum):
    OK = 0
    WARNING = 1
    FAULT = 2


JOINT_LIMIT = "JOINT_LIMIT"
OVERLOAD = "OVERLOAD"
STALL = "STALL"
COLLISION = "COLLISION"
UNREACHABLE = "UNREACHABLE"
ESTOP = "ESTOP"
CONTACT = "CONTACT"


@dataclass
class SafetyStatus:
    level: Severity = Severity.OK
    messages: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.level == Severity.OK

    def __str__(self) -> str:
        return "OK" if self.ok else f"{self.level.name}: " + "; ".join(self.messages)


class SafetyMonitor:
    WARNING_HOLD_S = 1.0           # a warning stays visible this long after its last report

    def __init__(self, config: RobotConfig, overload_time: float = 0.5, stall_time: float = 0.5,
                 stall_error: float = 0.12, payload_kg: float = 0.0):
        self.config = config
        self.overload_time = overload_time
        self.stall_time = stall_time
        self.stall_error = stall_error
        self.payload_kg = payload_kg
        self._events: dict = {}             # (code, key) -> (severity, message, last_t)
        self._overload_t = np.zeros(6)
        self._stall_t = np.zeros(6)
        self.history: list = []             # (sim_time, code, message) of everything ever raised

    # ---------------------------------------------------------------- reporting
    def report(self, t: float, code: str, message: str, severity: Severity = Severity.WARNING, key: str = "") -> None:
        k = (code, key)
        if k not in self._events:
            self.history.append((t, code, message))
        elif self._events[k][0] == Severity.FAULT and severity < Severity.FAULT:
            return
        self._events[k] = (severity, message, t)

    def clear(self) -> None:
        self._events.clear()
        self._overload_t[:] = 0
        self._stall_t[:] = 0

    # ---------------------------------------------------------------- per-step evaluation
    def evaluate(self, t: float, dt: float, arm, actual_angles=None, torques=None) -> SafetyStatus:
        """Check overload + locked-rotor-like conditions; return the merged current status."""
        if torques is None:
            torques = arm.gravity_torques(self.payload_kg)
        for i in range(5):
            joint = arm.joints[i]
            limit = joint.servo.torque_limit
            if torques[i] > limit:
                self._overload_t[i] += dt
                sev = Severity.FAULT if self._overload_t[i] >= self.overload_time else Severity.WARNING
                self.report(t, OVERLOAD, f"Potential servo overload on J{i+1} "
                            f"({torques[i]:.2f} > {limit:.2f} N*m)", sev, key=f"J{i+1}")
            else:
                self._overload_t[i] = 0.0

        if actual_angles is not None:                      # physics reports where joints really are
            for i in range(5):                             # J6 (gripper) is allowed to stall on an object
                joint = arm.joints[i]
                err = abs(joint.servo.angle - actual_angles[i])
                if err > self.stall_error:
                    self._stall_t[i] += dt
                    if self._stall_t[i] >= self.stall_time:
                        self.report(t, STALL, f"Locked-rotor-like condition on J{i+1} "
                                    f"(tracking error {np.degrees(err):.0f} deg)", Severity.FAULT, key=f"J{i+1}")
                else:
                    self._stall_t[i] = 0.0
        return self.status(t)

    def status(self, t: float) -> SafetyStatus:
        level, msgs = Severity.OK, []
        for k in list(self._events):
            sev, msg, last = self._events[k]
            if sev == Severity.WARNING and t - last > self.WARNING_HOLD_S:
                del self._events[k]
                continue
            level = max(level, sev)
            msgs.append(msg)
        return SafetyStatus(level, msgs)

    @property
    def faulted(self) -> bool:
        return any(sev == Severity.FAULT for sev, _, _ in self._events.values())


# ====================================================================================================
# MOTION PLANNER   (from control/motion_planner.py)
# ====================================================================================================
# MotionPlanner: MotionCommand -> Cartesian/joint targets -> IK -> joint targets.
#
# Pure logic (no threads, no simulation, no clock): time only advances through ``update(dt)``,
# so behaviour is deterministic and unit-testable. Continuous commands (CARTESIAN / JOINT with
# no magnitude) are kept alive by being re-sent; if they stop arriving for ``command_timeout``
# seconds (e.g. EMG dropout) the planner stops generating motion by itself.

@dataclass
class PlanOutput:
    joint_targets: Optional[np.ndarray] = None     # J1..J5 absolute targets (rad)
    gripper: Optional[float] = None                # opening 0..1
    stop_now: bool = False
    events: list = field(default_factory=list)     # (code, message, severity)
    done: bool = False                             # a finite motion just completed

    def event(self, code, msg, sev=Severity.WARNING):
        self.events.append((code, msg, sev))


class MotionPlanner:
    def __init__(self, kin: Kinematics, ik: NumericalIKSolver, workspace, home_angles, gripper_range,
                 default_speed: float = 0.04, joint_jog_speed: float = math.radians(45),
                 gripper_rate: float = 0.6, command_timeout: float = 0.30, home_gripper: float = 1.0):
        self.kin, self.ik, self.ws = kin, ik, workspace
        self.home_angles = np.asarray(home_angles, dtype=float)[:5]
        self.gripper_range = gripper_range
        self.default_speed = default_speed
        self.joint_jog_speed = joint_jog_speed
        self.gripper_rate = gripper_rate
        self.command_timeout = command_timeout
        self.home_gripper = home_gripper
        self.t = 0.0
        self._mode: Optional[str] = None
        self._m: dict = {}

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self._mode is not None

    def cancel(self) -> None:
        self._mode, self._m = None, {}

    # ------------------------------------------------------------------ command intake
    def handle(self, cmd: MotionCommand, q_cmd, gripper_cmd: float) -> PlanOutput:
        out = PlanOutput()
        q_cmd = np.asarray(q_cmd, dtype=float)[:5]
        ct = cmd.command_type

        if ct == CARTESIAN:
            self._start_cartesian(cmd, q_cmd, out)
        elif ct == JOINT:
            self._start_jog(cmd)
        elif ct == MOVE_TO:
            self._start_move_to(cmd, q_cmd, out)
        elif ct == JOINT_TARGET:
            self.cancel()
            tgt = np.array(cmd.target, dtype=float)
            if cmd.joint is not None:
                if cmd.joint == 6:
                    out.gripper = self._angle_to_opening(tgt[0])
                else:
                    q = q_cmd.copy()
                    q[cmd.joint - 1] = tgt[0]
                    out.joint_targets = q
            else:
                out.joint_targets = tgt[:5]
                if len(tgt) > 5:
                    out.gripper = self._angle_to_opening(tgt[5])
        elif ct == GRIPPER:
            d = cmd.direction
            if d == "OPEN":
                out.gripper = 1.0
            elif d == "CLOSE":
                out.gripper = 0.0
            elif d == "TOGGLE":
                out.gripper = 0.0 if gripper_cmd > 0.5 else 1.0
            elif d in ("SET", "PARTIAL_OPEN") and cmd.target:
                out.gripper = float(np.clip(cmd.target[0], 0.0, 1.0))
        elif ct == HOME:
            self.cancel()
            out.joint_targets = self.home_angles.copy()
            out.gripper = self.home_gripper
        elif ct == HOLD:
            self.cancel()
        elif ct == STOP:
            self.cancel()
            out.stop_now = True
        return out

    def _angle_to_opening(self, angle: float) -> float:
        lo, hi = self.gripper_range
        return float(np.clip((angle - lo) / (hi - lo), 0.0, 1.0))

    def _lifetime(self, cmd) -> float:
        return self.t + (cmd.duration if cmd.duration is not None else self.command_timeout)

    def _start_cartesian(self, cmd, q_cmd, out: PlanOutput) -> None:
        if cmd.direction not in DIRECTIONS:
            out.event(UNREACHABLE, f"Unknown direction '{cmd.direction}'")
            return
        d = np.array(DIRECTIONS[cmd.direction])
        same = (self._mode == "velocity" and np.allclose(self._m["dir"], d) and cmd.is_continuous
                and self._m["remaining"] is None)
        if same:                                   # refresh keep-alive, keep integrating the same target
            self._m["deadline"] = self._lifetime(cmd)
            self._m["speed"] = cmd.speed or self.default_speed
            return
        T = self.kin.forward(q_cmd)
        self._mode = "velocity"
        self._m = {
            "dir": d, "speed": cmd.speed or self.default_speed,
            "remaining": cmd.magnitude,
            "deadline": None if cmd.magnitude is not None else self._lifetime(cmd),
            "target": T[:3, 3].copy(),
            "pitch": self.kin.tool_pitch(T, q_cmd[0]),
        }

    def _start_jog(self, cmd) -> None:
        j = cmd.joint
        sign = 1.0 if cmd.direction == "+" else -1.0
        if self._mode == "jog" and self._m["joint"] == j and self._m["sign"] == sign:
            self._m["deadline"] = self._lifetime(cmd)
            return
        default = self.gripper_rate if j == 6 else self.joint_jog_speed
        self._mode = "jog"
        self._m = {"joint": j, "sign": sign, "speed": cmd.speed or default, "deadline": self._lifetime(cmd)}

    def _start_move_to(self, cmd, q_cmd, out: PlanOutput) -> None:
        self.cancel()
        end = np.array(cmd.target, dtype=float)
        if not self._in_workspace(end):
            out.event(UNREACHABLE, "Target position unreachable: outside configured workspace")
            return
        T = self.kin.forward(q_cmd)
        start, pitch0 = T[:3, 3].copy(), self.kin.tool_pitch(T, q_cmd[0])
        if cmd.orientation is not None:               # full pose: joint-space move to the IK solution
            res = self.ik.solve(end, target_orientation=cmd.orientation, seed=q_cmd)
            if res.success:
                out.joint_targets = res.joint_angles
            else:
                out.event(UNREACHABLE, res.message)
            return
        pitch1 = pitch0 if cmd.tool_pitch is None else cmd.tool_pitch
        res = self.ik.solve(end, tool_pitch=pitch1, seed=q_cmd, restarts=3)
        if not res.success:
            res = self.ik.solve(end, seed=q_cmd, restarts=3)      # fall back: ignore tool pitch
            if not res.success:
                out.event(UNREACHABLE, res.message)
                return
        self._mode = "moveto"
        self._m = {"start": start, "end": end, "p0": pitch0, "p1": pitch1, "s": 0.0,
                   "length": float(np.linalg.norm(end - start)), "speed": cmd.speed or self.default_speed}

    # ------------------------------------------------------------------ per-tick
    def update(self, dt: float, q_cmd, gripper_cmd: float) -> PlanOutput:
        self.t += dt
        out = PlanOutput()
        if self._mode is None:
            return out
        q_cmd = np.asarray(q_cmd, dtype=float)[:5]
        m = self._m
        if m.get("deadline") is not None and self.t > m["deadline"]:
            self.cancel()                              # keep-alive lapsed -> stop moving
            return out

        if self._mode == "velocity":
            step = m["speed"] * dt
            if m["remaining"] is not None:
                step = min(step, m["remaining"])
                m["remaining"] -= step
            new = np.clip(m["target"] + m["dir"] * step,
                          [self.ws.x[0], self.ws.y[0], self.ws.z[0]], [self.ws.x[1], self.ws.y[1], self.ws.z[1]])
            if np.allclose(new, m["target"]):
                out.event(UNREACHABLE, "Target position unreachable: workspace limit")
            else:
                res = self._solve(new, m["pitch"], q_cmd)
                if res is None:
                    out.event(UNREACHABLE, "Target position unreachable")
                    self.cancel()
                    return out
                m["target"] = new
                out.joint_targets = res[0]
                m["pitch"] = res[1]          # if the held pitch was infeasible, continue from the achieved one
            if m["remaining"] is not None and m["remaining"] <= 1e-9:
                self.cancel()
                out.done = True

        elif self._mode == "jog":
            if m["joint"] == 6:
                out.gripper = float(np.clip(gripper_cmd + m["sign"] * m["speed"] * dt, 0.0, 1.0))
            else:
                q = q_cmd.copy()
                q[m["joint"] - 1] += m["sign"] * m["speed"] * dt
                out.joint_targets = q

        elif self._mode == "moveto":
            m["s"] = min(m["s"] + m["speed"] * dt, m["length"])
            f = 1.0 if m["length"] < 1e-9 else m["s"] / m["length"]
            pos = m["start"] + (m["end"] - m["start"]) * f
            pitch = m["p0"] + (m["p1"] - m["p0"]) * f
            res = self._solve(pos, pitch, q_cmd)
            if res is None:
                out.event(UNREACHABLE, "Target position unreachable along path")
                self.cancel()
                return out
            out.joint_targets = res[0]
            if f >= 1.0:
                self.cancel()
                out.done = True
        return out

    def _solve(self, pos, pitch, seed):
        """-> (joint angles, pitch actually achieved) or None. Prefers holding ``pitch``; falls back to position-only."""
        r = self.ik.solve(pos, tool_pitch=pitch, seed=seed, restarts=0)      # local solve: fast, continuous
        if r.success:
            return r.joint_angles, pitch
        r = self.ik.solve(pos, seed=seed, restarts=0)
        if not r.success:
            return None
        T = self.kin.forward(r.joint_angles)
        return r.joint_angles, self.kin.tool_pitch(T, r.joint_angles[0])

    def _in_workspace(self, p) -> bool:
        ws = self.ws
        return bool(ws.x[0] <= p[0] <= ws.x[1] and ws.y[0] <= p[1] <= ws.y[1] and ws.z[0] <= p[2] <= ws.z[1])


# ====================================================================================================
# SIMULATION OBJECTS   (from simulation/objects.py)
# ====================================================================================================
# Simple graspable objects (SI units).

@dataclass
class SimObject:
    name: str
    kind: str                      # "cube" | "sphere" | "cylinder"
    dims: tuple                    # cube: (edge,)  sphere: (radius,)  cylinder: (radius, height)
    mass: float
    position: tuple                # initial centre (x, y, z)
    color: tuple = (0.85, 0.25, 0.2, 1.0)
    body_id: Optional[int] = None  # filled by the physics backend

    @property
    def half_height(self) -> float:
        if self.kind == "cube":
            return self.dims[0] / 2
        return self.dims[0] if self.kind == "sphere" else self.dims[1] / 2

    @property
    def width(self) -> float:
        """Size across a gripper that grips it along X."""
        return self.dims[0] if self.kind == "cube" else 2 * self.dims[0]


def make_cube(name: str, x: float, y: float, edge: float = 0.030, mass: float = 0.025,
              color=(0.85, 0.25, 0.2, 1.0)) -> SimObject:
    return SimObject(name, "cube", (edge,), mass, (x, y, edge / 2), color)


def make_sphere(name: str, x: float, y: float, radius: float = 0.015, mass: float = 0.02,
                color=(0.2, 0.5, 0.85, 1.0)) -> SimObject:
    return SimObject(name, "sphere", (radius,), mass, (x, y, radius), color)


def make_cylinder(name: str, x: float, y: float, radius: float = 0.013, height: float = 0.045,
                  mass: float = 0.03, color=(0.25, 0.7, 0.35, 1.0)) -> SimObject:
    return SimObject(name, "cylinder", (radius, height), mass, (x, y, height / 2), color)


# ====================================================================================================
# SIMULATION ENVIRONMENT   (from simulation/environment.py)
# ====================================================================================================
# Scene description: a table (top surface at z = 0) and a set of objects.

@dataclass
class Environment:
    table_size: tuple = (0.80, 0.60)           # x, y extent of the table top (m)
    table_thickness: float = 0.02
    objects: list = field(default_factory=list)

    def add(self, obj: SimObject) -> SimObject:
        self.objects.append(obj)
        return obj

    def get(self, name: str) -> SimObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    @classmethod
    def default_scene(cls) -> "Environment":
        env = cls()
        env.add(make_cube("cube", 0.0, 0.17))
        env.add(make_sphere("sphere", 0.10, 0.19))
        env.add(make_cylinder("cylinder", -0.10, 0.19))
        return env


# ====================================================================================================
# URDF GENERATION   (from simulation/urdf.py)
# ====================================================================================================
# Generate a URDF from RobotConfig, so geometry/mass/limits live in ONE place (the config).
#
# Link frames follow robot/kinematics.py: link i's frame sits on joint i's axis and the link
# extends along +Z by ``length`` to the next joint origin.

ALU = "0.78 0.80 0.83 1"
DARK = "0.10 0.10 0.12 1"
FINGER = "0.62 0.65 0.70 1"


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


AX_COLORS = {"ax_x": "1 0 0 1", "ax_y": "0 0.8 0 1", "ax_z": "0 0 1 1"}
_MATERIALS = {"alu": ALU, "dark": DARK, "finger": FINGER}
_MATERIALS.update(AX_COLORS)


def _axis_links(parent: str, origin_z: float = 0.0, length: float = 0.03, r: float = 0.0008, tag: str = "") -> str:
    """Coordinate-frame arrows as three tiny fixed child links (the OpenGL viewer tints a whole link with ONE
    color, so each arrow needs its own link). Visual only: no collision."""
    out = ""
    for (name, size, xyz) in (("x", (length, 2 * r, 2 * r), (length / 2, 0, 0)),
                              ("y", (2 * r, length, 2 * r), (0, length / 2, 0)),
                              ("z", (2 * r, 2 * r, length), (0, 0, length / 2))):
        ln = f"ax_{parent}{tag}_{name}"
        out += (f'<link name="{ln}"><inertial><mass value="1e-4"/><inertia ixx="1e-9" iyy="1e-9" izz="1e-9" '
                f'ixy="0" ixz="0" iyz="0"/></inertial><visual><origin xyz="{_fmt(xyz)}"/><geometry>'
                f'<box size="{_fmt(size)}"/></geometry><material name="ax_{name}"/></visual></link>'
                f'<joint name="{ln}_j" type="fixed"><parent link="{parent}"/><child link="{ln}"/>'
                f'<origin xyz="0 0 {origin_z:.6g}"/></joint>')
    return out


def _box(size, xyz, color: str, name: str, collision: bool = True) -> str:
    mat = next(k for k, v in _MATERIALS.items() if v == color)
    vis = (f'<visual><origin xyz="{_fmt(xyz)}"/><geometry><box size="{_fmt(size)}"/></geometry>'
           f'<material name="{mat}"/></visual>')
    col = (f'<collision><origin xyz="{_fmt(xyz)}"/><geometry><box size="{_fmt(size)}"/></geometry></collision>'
           if collision else "")
    return vis + col


def _inertial(link) -> str:
    ixx, iyy, izz = link.inertia_diag()
    return (f'<inertial><origin xyz="{_fmt(link.com)}"/><mass value="{link.mass:.6g}"/>'
            f'<inertia ixx="{ixx:.6g}" iyy="{iyy:.6g}" izz="{izz:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>')


def build_urdf(cfg: RobotConfig, axes: bool = False) -> str:
    """``axes=True`` adds coordinate-frame arrows to every link + TCP (used by the GUI viewer only)."""
    L = cfg.links
    g = cfg.gripper
    parts = [f'<?xml version="1.0"?>\n<robot name="{cfg.name}">']
    parts += [f'<material name="{k}"><color rgba="{v}"/></material>' for k, v in _MATERIALS.items()]

    # ---- base: large metal plate
    w, d = cfg.base_size
    parts.append(f'<link name="base">{_inertial(L[0])}'
                 + _box((w, d, L[0].length), (0, 0, L[0].length / 2), ALU, "alu_base")
                 + "</link>")

    # ---- arm links 1..4 and end-effector body
    for i in range(1, 6):
        link = L[i]
        lw, ld = link.size
        # NOTE: the OpenGL viewer tints a whole link with its LAST visual's color, so the aluminum bracket
        # goes last and the dark servo housing first (it stays visible only in the collision/inertial model).
        body = ""
        if i == 5:
            body += _box((lw, ld, g.palm_length), (0, 0, g.palm_length / 2), ALU, f"alu_{i}")
        else:
            body += _box((lw * 1.15, ld * 1.5, 0.036), (0, 0, link.length - 0.018), DARK, f"servo_{i}", collision=False)
            body += _box((lw, ld, link.length), (0, 0, link.length / 2), ALU, f"alu_{i}")
        parts.append(f'<link name="{link.name}">{_inertial(link)}{body}</link>')

    # ---- revolute joints J1..J5
    for i in range(1, 6):
        j = cfg.joints[i - 1]
        a = j.axis
        parts.append(
            f'<joint name="j{i}" type="revolute"><parent link="{L[i-1].name}"/><child link="{L[i].name}"/>'
            f'<origin xyz="0 0 {L[i-1].length:.6g}"/><axis xyz="{_fmt(a)}"/>'
            f'<limit lower="{j.limit.min_angle:.6g}" upper="{j.limit.max_angle:.6g}" '
            f'effort="{j.servo.torque_limit:.6g}" velocity="{j.limit.max_velocity:.6g}"/>'
            f'<dynamics damping="0.02" friction="0.01"/></joint>')

    # ---- gripper fingers (prismatic, J6 drives both)
    fx = g.finger_thickness / 2
    for side, sgn in (("l", 1), ("r", -1)):
        name = f"finger_{side}"
        mass = g.finger_mass
        ix = mass / 12 * (g.finger_depth ** 2 + g.finger_length ** 2)
        iy = mass / 12 * (g.finger_thickness ** 2 + g.finger_length ** 2)
        iz = mass / 12 * (g.finger_thickness ** 2 + g.finger_depth ** 2)
        parts.append(
            f'<link name="{name}"><inertial><origin xyz="0 0 {g.finger_length/2:.6g}"/><mass value="{mass}"/>'
            f'<inertia ixx="{ix:.6g}" iyy="{iy:.6g}" izz="{iz:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>'
            + _box((g.finger_thickness, g.finger_depth, g.finger_length), (0, 0, g.finger_length / 2), FINGER,
                   f"finger_{side}") + "</link>")
        parts.append(
            f'<joint name="j6_{name}" type="prismatic"><parent link="ee"/><child link="{name}"/>'
            f'<origin xyz="{sgn*fx:.6g} 0 {g.palm_length:.6g}"/><axis xyz="{sgn} 0 0"/>'
            f'<limit lower="0" upper="{g.finger_travel:.6g}" effort="{g.max_grip_force}" velocity="0.2"/></joint>')

    # ---- TCP marker (fixed)
    parts.append('<link name="tcp"><inertial><mass value="0.001"/><inertia ixx="1e-9" iyy="1e-9" izz="1e-9" '
                 'ixy="0" ixz="0" iyz="0"/></inertial>' + '</link>')
    parts.append(f'<joint name="tcp_fixed" type="fixed"><parent link="ee"/><child link="tcp"/>'
                 f'<origin xyz="0 0 {cfg.tcp_offset:.6g}"/></joint>')
    if axes:                                   # appended LAST so the real joints keep their indices
        parts.append(_axis_links("base", L[0].length, 0.04, tag="_j1"))
        for i in range(1, 6):
            parts.append(_axis_links(L[i].name, 0.0, 0.03))
        parts.append(_axis_links("ee", g.palm_length, 0.02, tag="_j6"))
        parts.append(_axis_links("tcp", 0.0, 0.05, 0.001))
    parts.append("</robot>")
    return "\n".join(parts)


def export_urdf(cfg: RobotConfig, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(build_urdf(cfg), encoding="utf-8")
    return path


# ====================================================================================================
# ANALYTIC COLLISION CHECKER   (from simulation/collision.py)
# ====================================================================================================
# Analytic (NumPy-only) collision checks used to veto unsafe targets before they are executed.
#
# Link capsules are tested against the table plane and against each other (non-adjacent pairs).
# Contacts with simulated objects are reported separately by the physics backend.

@dataclass
class CollisionReport:
    colliding: bool = False
    pairs: list = field(default_factory=list)

    def __str__(self) -> str:
        return "COLLISION: " + ", ".join(self.pairs) if self.colliding else "none"


class CollisionChecker:
    def __init__(self, kin: Kinematics, config: RobotConfig, margin: float = 0.002):
        self.kin = kin
        self.config = config
        self.margin = margin
        self.table_z = config.table_z
        self.radii = {i: 0.5 * max(l.size) for i, l in enumerate(config.links)}
        # Slimmer self-collision radii: brackets overlap by design at joints.
        self.self_scale = 0.6

    def _capsules(self, q, opening: float):
        """Segment (a, b, radius, name) per link 1..5 plus the turntable column."""
        fr = self.kin.frames(q)
        links = self.config.links
        caps = {}
        top = fr[1][:3, 3] + np.array([0, 0, links[1].length])
        caps["base"] = (np.zeros(3), top, self.radii[1])
        for k in range(2, 5):
            caps[f"link{k}"] = (fr[k][:3, 3], fr[k + 1][:3, 3], self.radii[k])
        tip = (fr[5] @ np.array([0, 0, links[5].length, 1.0]))[:3]
        caps["ee"] = (fr[5][:3, 3], tip, self.radii[5])
        # Fingertip points (outer corners) for the floor test.
        g = self.config.gripper
        half = opening * g.finger_travel + g.finger_thickness
        tips = [(fr[5] @ np.array([s * half, 0, links[5].length, 1.0]))[:3] for s in (-1, 1)]
        return caps, tips

    def check(self, q, opening: float = 1.0) -> CollisionReport:
        q = np.asarray(q, dtype=float)[:5]
        caps, tips = self._capsules(q, opening)
        floor = self.table_z + self.margin
        pairs = []
        for name, (a, b, r) in caps.items():
            if name == "base":
                continue
            if min(a[2], b[2]) - r * 0.5 < floor and name != "ee":
                pairs.append(f"{name}-table")
        if any(t[2] < floor for t in tips) or min(caps["ee"][0][2], caps["ee"][1][2]) < floor:
            pairs.append("gripper-table")
        s = self.self_scale
        for n1, n2 in (("base", "link4"), ("base", "ee"), ("link2", "link4"), ("link2", "ee"), ("link3", "ee")):
            a1, b1, r1 = caps[n1]
            a2, b2, r2 = caps[n2]
            if segment_distance(a1, b1, a2, b2) < (r1 + r2) * s:
                pairs.append(f"{n1}-{n2}")
        return CollisionReport(bool(pairs), pairs)


# ====================================================================================================
# PHYSICS BACKENDS (KINEMATIC / PYBULLET)   (from simulation/physics.py)
# ====================================================================================================
# Physics backends. The controller talks only to ``PhysicsBackend``.
#
# * KinematicBackend - no dependencies; joints track their commands perfectly, no objects/contacts.
# * PyBulletBackend  - gravity, collisions, joint motors with torque caps, friction grasping, GUI.
#
# All PyBullet calls are serialised through ``backend.lock`` (control thread + UI thread share one client).

try:                                       # optional dependency
    import pybullet as _p
    import pybullet_data as _pd
except Exception:                          # pragma: no cover
    _p = _pd = None


SOLVER_ITERATIONS = 80          # PyBullet constraint-solver iterations per step (cost is ~linear in this)


def pybullet_available() -> bool:
    return _p is not None


class PhysicsBackend(ABC):
    supports_physics = False

    @abstractmethod
    def connect(self, env: Environment | None = None) -> None: ...

    @abstractmethod
    def set_targets(self, q5: np.ndarray, opening: float) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def read_state(self) -> tuple:
        """-> (q5 actual (rad), gripper opening actual 0..1)"""

    def get_contacts(self) -> list:
        return []

    def object_position(self, name: str):
        return None

    def disconnect(self) -> None:
        pass


class KinematicBackend(PhysicsBackend):
    """Perfect tracking, no physics. Useful for tests and machines without PyBullet."""

    def __init__(self):
        self._q = np.zeros(5)
        self._opening = 1.0
        self.env: Environment | None = None

    def connect(self, env=None):
        self.env = env

    def set_targets(self, q5, opening):
        self._q, self._opening = np.asarray(q5, float).copy(), float(opening)

    def step(self):
        pass

    def read_state(self):
        return self._q.copy(), self._opening

    def object_position(self, name):
        return None if self.env is None else self.env.get(name).position


class _World:
    """Handles of one PyBullet client."""

    def __init__(self, client: int):
        self.client = client
        self.robot = None
        self.table = None
        self.objects: dict[str, int] = {}
        self.joint_idx: dict[str, int] = {}
        self.link_names: dict[int, str] = {-1: "base"}

    @property
    def arm_idx(self) -> list:
        return [self.joint_idx[f"j{i}"] for i in range(1, 6)]

    @property
    def finger_idx(self) -> list:
        return [self.joint_idx["j6_finger_l"], self.joint_idx["j6_finger_r"]]


def _add_object(p, w: _World, obj: SimObject) -> int:
    c = w.client
    if obj.kind == "cube":
        h = obj.dims[0] / 2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h] * 3, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h] * 3, rgbaColor=list(obj.color), physicsClientId=c)
    elif obj.kind == "sphere":
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=obj.dims[0], physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=obj.dims[0], rgbaColor=list(obj.color), physicsClientId=c)
    else:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obj.dims[0], height=obj.dims[1], physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obj.dims[0], length=obj.dims[1],
                                  rgbaColor=list(obj.color), physicsClientId=c)
    bid = p.createMultiBody(obj.mass, col, vis, list(obj.position), physicsClientId=c)
    p.changeDynamics(bid, -1, lateralFriction=1.0, spinningFriction=0.005, rollingFriction=0.001, physicsClientId=c)
    # Let a resting object fall asleep (no solver work until the gripper touches it): ~20% less CPU per step.
    p.changeDynamics(bid, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING, physicsClientId=c)
    w.objects[obj.name] = bid
    return bid


def make_world(client: int, cfg: RobotConfig, env: Environment, dt: float, gui: bool = False,
               show_frames: bool = True) -> _World:
    """Scene (table + floor), robot from the generated URDF, and objects, inside one PyBullet client.
    Used for the DIRECT physics client AND for the GUI viewer process (identical geometry)."""
    p, c = _p, client
    w = _World(client)
    p.setAdditionalSearchPath(_pd.getDataPath(), physicsClientId=c)
    p.setGravity(0, 0, -9.81, physicsClientId=c)
    p.setTimeStep(dt, physicsClientId=c)
    p.setPhysicsEngineParameter(numSolverIterations=SOLVER_ITERATIONS, physicsClientId=c)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=c)   # keys are ours
        # Shadow-map pass is the single most expensive thing the OpenGL viewer does on the M1's integrated GPU.
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=c)
        for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            p.configureDebugVisualizer(flag, 0, physicsClientId=c)
        p.resetDebugVisualizerCamera(0.65, 35, -28, [0.0, 0.10, 0.10], physicsClientId=c)
    tx, ty = env.table_size
    th = env.table_thickness
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tx / 2, ty / 2, th / 2], physicsClientId=c)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[tx / 2, ty / 2, th / 2],
                              rgbaColor=[0.55, 0.42, 0.30, 1], physicsClientId=c)
    w.table = p.createMultiBody(0, col, vis, [0, ty / 2 - 0.2, -th / 2], physicsClientId=c)
    p.changeDynamics(w.table, -1, lateralFriction=0.9, physicsClientId=c)
    fcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 3, 0.05], physicsClientId=c)
    fvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 0.05], rgbaColor=[0.85, 0.85, 0.87, 1], physicsClientId=c)
    p.createMultiBody(0, fcol, fvis, [0, 0, -0.75], physicsClientId=c)
    tmp = Path(tempfile.mkdtemp(prefix="arm_urdf_")) / "arm.urdf"
    tmp.write_text(build_urdf(cfg, axes=gui and show_frames), encoding="utf-8")
    w.robot = p.loadURDF(str(tmp), [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE,
                         physicsClientId=c)
    for i in range(p.getNumJoints(w.robot, physicsClientId=c)):
        info = p.getJointInfo(w.robot, i, physicsClientId=c)
        w.joint_idx[info[1].decode()] = i
        w.link_names[i] = info[12].decode()
    for obj in env.objects:
        _add_object(p, w, obj)
    return w


class PyBulletBackend(PhysicsBackend):
    """Physics in a fast DIRECT client. Rendering is a separate concern: the GUI lives in its own process
    (ui/gui.py) and is fed ``view_state()``; any call into PyBullet's GUI client can block ~100+ ms."""

    supports_physics = True

    def __init__(self, config: RobotConfig, dt: float = 1.0 / 240.0, position_gain: float = 0.5,
                 velocity_gain: float = 1.0):
        if _p is None:
            raise RuntimeError("pybullet is not installed (pip install pybullet)")
        self.cfg = config
        self.dt = dt
        self.kp, self.kd = position_gain, velocity_gain
        self.p = _p
        self.lock = threading.RLock()
        self.phys: _World | None = None
        self.objects: dict[str, SimObject] = {}

    @property
    def client(self):
        return self.phys.client if self.phys else None

    def connect(self, env: Environment | None = None) -> None:
        p = self.p
        env = env or Environment()
        with self.lock:
            self.phys = make_world(p.connect(p.DIRECT), self.cfg, env, self.dt)
            for o in env.objects:
                self.objects[o.name] = o
                o.body_id = self.phys.objects[o.name]
            w, c = self.phys, self.phys.client
            self._torque = [self.cfg.joints[i].servo.torque_limit for i in range(5)]
            p.setCollisionFilterPair(w.robot, w.robot, *w.finger_idx, 0, physicsClientId=c)
            for fi in w.finger_idx:
                p.changeDynamics(w.robot, fi, lateralFriction=1.2, spinningFriction=0.01, physicsClientId=c)
            for ji in w.arm_idx:                         # pure position control: disable default velocity motor
                p.setJointMotorControl2(w.robot, ji, p.VELOCITY_CONTROL, force=0, physicsClientId=c)
            self.reset_state(np.array(self.cfg.home_angles[:5]), 1.0)

    # ------------------------------------------------------------------ objects
    def add_object(self, obj: SimObject) -> int:
        with self.lock:
            self.objects[obj.name] = obj
            obj.body_id = _add_object(self.p, self.phys, obj)
            return obj.body_id

    def object_position(self, name: str):
        with self.lock:
            pos, _ = self.p.getBasePositionAndOrientation(self.phys.objects[name], physicsClientId=self.client)
        return tuple(pos)

    def reset_object(self, name: str, position) -> None:
        with self.lock:
            bid = self.phys.objects[name]
            self.p.resetBasePositionAndOrientation(bid, list(position), [0, 0, 0, 1], physicsClientId=self.client)
            self.p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

    # ------------------------------------------------------------------ control / stepping
    def reset_state(self, q5, opening: float) -> None:
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            for ji, q in zip(w.arm_idx, q5):
                p.resetJointState(w.robot, ji, float(q), physicsClientId=c)
            s = opening * self.cfg.gripper.finger_travel
            for fi in w.finger_idx:
                p.resetJointState(w.robot, fi, s, physicsClientId=c)

    def set_targets(self, q5, opening: float) -> None:
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            p.setJointMotorControlArray(
                w.robot, w.arm_idx, p.POSITION_CONTROL, targetPositions=[float(v) for v in q5],
                forces=self._torque, positionGains=[self.kp] * 5, velocityGains=[self.kd] * 5, physicsClientId=c)
            s = float(np.clip(opening, 0, 1)) * self.cfg.gripper.finger_travel
            p.setJointMotorControlArray(
                w.robot, w.finger_idx, p.POSITION_CONTROL, targetPositions=[s, s],
                forces=[self.cfg.gripper.max_grip_force] * 2, physicsClientId=c)

    def step(self) -> None:
        with self.lock:
            self.p.stepSimulation(physicsClientId=self.client)

    def read_state(self):
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            st = p.getJointStates(w.robot, w.arm_idx, physicsClientId=c)
            fs = p.getJointStates(w.robot, w.finger_idx, physicsClientId=c)
        q = np.array([s[0] for s in st])
        travel = float(np.mean([s[0] for s in fs]))
        return q, travel / self.cfg.gripper.finger_travel

    def view_state(self) -> dict:
        """Everything a viewer needs to pose its copy of the scene (picklable)."""
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            idx = w.arm_idx + w.finger_idx
            joints = [s[0] for s in p.getJointStates(w.robot, idx, physicsClientId=c)]
            objs = {n: p.getBasePositionAndOrientation(b, physicsClientId=c) for n, b in w.objects.items()}
        return {"joints": joints, "objects": objs}

    def get_contacts(self) -> list:
        """Unwanted contacts: robot vs table/objects, excluding base-table and finger-object (grasp)."""
        p, c, w = self.p, self.client, self.phys
        found = []
        with self.lock:
            pts = p.getContactPoints(bodyA=w.robot, physicsClientId=c)
        finger_links = set(w.finger_idx)
        obj_ids = {bid: n for n, bid in w.objects.items()}
        for cp in pts:
            link, other = cp[3], cp[2]
            if other == w.robot:
                continue
            if link == -1 and other == w.table:
                continue
            if link in finger_links and other in obj_ids:
                continue
            tag = "table" if other == w.table else obj_ids.get(other, f"body{other}")
            found.append(f"{w.link_names.get(link, link)}-{tag}")
        return sorted(set(found))

    def disconnect(self) -> None:
        with self.lock:
            if self.phys is not None and self.p.isConnected(self.phys.client):
                self.p.disconnect(self.phys.client)
            self.phys = None


# ====================================================================================================
# ROBOT CONTROLLER   (from control/controller.py)
# ====================================================================================================
# RobotController interface + SimulatedRobotController.
#
# EMG / keyboard / GUI code only ever sees ``RobotController``; swapping the simulator for
# the real arm means swapping this one object (see control/hardware_controller.py).
#
# Pipeline inside ``SimulatedRobotController.step``:
#     queued MotionCommands -> MotionPlanner (+IK) -> safety/limit/collision vetting
#     -> servo targets -> SimulatedServo dynamics -> physics backend -> safety evaluation -> CSV log

class RobotController(ABC):
    """Everything an input layer (EMG, keyboard, GUI, ROS) may ask of a robot - simulated or real."""

    @abstractmethod
    def move_joint(self, joint_id: int, angle: float, wait: bool = False) -> None: ...

    @abstractmethod
    def move_joints(self, angles, wait: bool = False) -> None: ...

    @abstractmethod
    def move_cartesian(self, position, orientation=None, tool_pitch=None, speed=None, wait: bool = False) -> bool: ...

    @abstractmethod
    def open_gripper(self, wait: bool = False) -> None: ...

    @abstractmethod
    def close_gripper(self, wait: bool = False) -> None: ...

    @abstractmethod
    def set_gripper_position(self, value: float, wait: bool = False) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def home(self, wait: bool = False) -> None: ...

    @abstractmethod
    def submit(self, command: MotionCommand) -> bool: ...

    @abstractmethod
    def snapshot(self): ...


@dataclass
class Telemetry:
    sim_time: float = 0.0
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(6))      # rad (servo state)
    joint_targets: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tcp_position: np.ndarray = field(default_factory=lambda: np.zeros(3))      # m
    tcp_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))           # rad
    gripper_opening: float = 1.0
    gesture: str = "-"
    confidence: float | None = None
    command: str = "-"
    collision: str = "NONE"
    safety: SafetyStatus = field(default_factory=SafetyStatus)
    estopped: bool = False
    connected: str = "SIMULATION"
    control_hz: float = 0.0
    torques: np.ndarray = field(default_factory=lambda: np.zeros(5))


class SimulatedRobotController(RobotController):
    def __init__(self, config: RobotConfig | None = None, backend: PhysicsBackend | None = None,
                 control_hz: float = 240.0, planner_hz: float = 100.0, command_timeout: float = 0.30,
                 session_logger: SessionLogger | None = None, payload_kg: float = 0.0):
        self.config = config or default_config()
        self.arm = RobotArm(self.config)
        self.kin = self.arm.kinematics
        self.dt = 1.0 / control_hz
        self.planner_period = 1.0 / planner_hz
        self.planner = MotionPlanner(
            self.kin, self.arm.ik_solver, self.config.workspace, self.config.home_angles,
            (self.arm.joints[5].limit.min_angle, self.arm.joints[5].limit.max_angle),
            command_timeout=command_timeout)
        self.safety = SafetyMonitor(self.config, payload_kg=payload_kg)
        self.collision_checker = CollisionChecker(self.kin, self.config)
        self.backend = backend or KinematicBackend()
        self.logger = session_logger
        self.latency = LatencyTracker()

        self._lock = threading.RLock()
        self._pending: deque = deque()
        self._planner_accum = 0.0
        self._step_count = 0
        self.sim_time = 0.0
        self.estopped = False
        self.rejected_commands = 0
        self.current_command: MotionCommand | None = None
        self._awaiting_target: MotionCommand | None = None
        self._gesture, self._confidence = "-", None
        self._collision_text, self._collision_t = "NONE", -1e9
        self._actual_q = self.arm.joint_angles[:5].copy()
        self.control_hz_measured = 0.0
        self.external_stepper = False          # True when a runtime thread calls step() for us

    # ------------------------------------------------------------------ lifecycle
    def connect(self, env=None) -> None:
        self.backend.connect(env)
        self._sync_backend()

    def shutdown(self) -> None:
        self.backend.disconnect()
        if self.logger:
            self.logger.close()

    # ------------------------------------------------------------------ helpers
    def _q_cmd(self) -> np.ndarray:
        return self.arm.joint_targets[:5]

    def _grip_cmd(self) -> float:
        return self.arm.gripper.target_opening

    def _sync_backend(self) -> None:
        self.backend.set_targets(self.arm.joint_angles[:5], self.arm.gripper.opening)

    @property
    def faulted(self) -> bool:
        return self.safety.faulted

    @property
    def blocked(self) -> bool:
        return self.estopped or self.faulted

    def set_emg_status(self, gesture: str, confidence: float | None) -> None:
        self._gesture, self._confidence = gesture, confidence

    # ------------------------------------------------------------------ command intake (thread-safe)
    def submit(self, command: MotionCommand) -> bool:
        """Queue a command for the next control step. Returns False if rejected (e-stop/fault)."""
        if command.command_type in (ESTOP,):
            self.emergency_stop()                       # never wait for the next tick
            return True
        if self.blocked and command.command_type != RESET:
            self.rejected_commands += 1
            return False
        with self._lock:
            self._pending.append(command)
        return True

    def _process(self, cmd: MotionCommand) -> None:
        if cmd.command_type == RESET:
            self.reset()
            return
        if self.blocked:
            self.rejected_commands += 1
            return
        self.current_command = cmd
        self.latency.record("command_queue_ms", (time.perf_counter() - cmd.created_at) * 1000.0)
        self._awaiting_target = cmd
        out = self.planner.handle(cmd, self._q_cmd(), self._grip_cmd())
        self._apply(out)
        if not self.planner.active:                       # nothing more will be produced for this command
            self._awaiting_target = None

    # ------------------------------------------------------------------ applying planner output safely
    def _apply(self, out: PlanOutput) -> None:
        t = self.sim_time
        for code, msg, sev in out.events:
            self.safety.report(t, code, msg, sev)
        if out.stop_now:
            self.arm.stop()
        if out.gripper is not None:
            self.arm.gripper.set_position(out.gripper)
            self._record_target_latency()
        if out.joint_targets is not None:
            q = np.asarray(out.joint_targets, dtype=float)[:5]
            lo, hi = self.kin.lower, self.kin.upper
            for i in np.nonzero((q < lo - 1e-9) | (q > hi + 1e-9))[0]:
                self.safety.report(t, JOINT_LIMIT, f"J{i+1} joint limit reached", Severity.WARNING, key=f"J{i+1}")
            q = np.clip(q, lo, hi)
            rep = self.collision_checker.check(q, self.arm.gripper.target_opening)
            if rep.colliding and not self.collision_checker.check(self._q_cmd(), self.arm.gripper.target_opening).colliding:
                # Veto: never command a pose that collides. Keep the old target and drop the planner's path.
                self._set_collision(str(rep), t)
                self.safety.report(t, COLLISION, "Collision detected: target rejected (" + ", ".join(rep.pairs) + ")",
                                   Severity.WARNING)
                self.planner.cancel()
                return
            for i in range(5):
                self.arm.set_joint_angle(i + 1, float(q[i]))
            self.arm.drain_warnings()
            self._record_target_latency()

    def _record_target_latency(self) -> None:
        if self._awaiting_target is not None:
            self.latency.record("command_to_target_ms",
                                (time.perf_counter() - self._awaiting_target.created_at) * 1000.0)
            self._awaiting_target = None

    def _set_collision(self, text: str, t: float) -> None:
        self._collision_text, self._collision_t = text, t

    # ------------------------------------------------------------------ the control step
    def step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else dt
        with self._lock:
            while self._pending:
                self._process(self._pending.popleft())

            if not self.blocked and self.planner.active:
                self._planner_accum += dt
                if self._planner_accum >= self.planner_period - 1e-9:
                    out = self.planner.update(self._planner_accum, self._q_cmd(), self._grip_cmd())
                    self._planner_accum = 0.0
                    self._apply(out)
            else:
                self._planner_accum = 0.0

            self.arm.update(dt)
            self._sync_backend()
            self.backend.step()
            self._actual_q, _ = self.backend.read_state()
            self.sim_time += dt
            self._step_count += 1

            status = self.safety.evaluate(self.sim_time, dt, self.arm, self._actual_q if self.backend.supports_physics else None)
            if self.backend.supports_physics and self._step_count % 8 == 0:
                contacts = self.backend.get_contacts()
                if contacts:
                    self._set_collision("CONTACT: " + ", ".join(contacts), self.sim_time)
                    self.safety.report(self.sim_time, CONTACT, "Collision detected: " + ", ".join(contacts))
            if self.sim_time - self._collision_t > 1.0:
                self._collision_text = "NONE"
            if self.safety.faulted and not self.estopped:
                self._hold_after_fault()
            if self.logger:
                self._log_row()

    def _hold_after_fault(self) -> None:
        self.arm.stop()
        self.planner.cancel()
        self._pending.clear()
        log.error("Safety FAULT latched: %s - reset required", self.safety.status(self.sim_time))

    def _log_row(self) -> None:
        pose = self.arm.get_end_effector_pose()
        self.logger.log(self.sim_time, self._gesture if self._gesture != "-" else "", self._confidence,
                        self.current_command.describe() if self.current_command else "",
                        self.arm.joint_angles, pose.position, self._collision_text,
                        str(self.safety.status(self.sim_time)))

    # ------------------------------------------------------------------ waiting helpers
    def run_for(self, seconds: float) -> None:
        """Synchronously advance simulation time (deterministic; for scripts and tests)."""
        for _ in range(int(round(seconds / self.dt))):
            self.step()

    @property
    def settled(self) -> bool:
        return (not self._pending and not self.planner.active
                and all(j.servo.at_target for j in self.arm.joints))

    def wait_until_settled(self, timeout: float = 20.0) -> bool:
        if self.external_stepper:
            t0 = time.perf_counter()
            while not self.settled and time.perf_counter() - t0 < timeout:
                time.sleep(0.005)
            return self.settled
        for _ in range(int(timeout / self.dt)):
            if self.settled:
                return True
            self.step()
        return self.settled

    def pause(self, seconds: float) -> None:
        """Let time pass (steps the simulation itself unless a runtime thread is doing it)."""
        if self.external_stepper:
            time.sleep(seconds)
        else:
            self.run_for(seconds)

    # ------------------------------------------------------------------ RobotController API
    def move_joint(self, joint_id: int, angle: float, wait: bool = False) -> None:
        self.submit(MotionCommand.joint_target(angle, joint=joint_id, source="api"))
        if wait:
            self.wait_until_settled()

    def move_joints(self, angles, wait: bool = False) -> None:
        self.submit(MotionCommand.joint_target(angles, source="api"))
        if wait:
            self.wait_until_settled()

    def move_cartesian(self, position, orientation=None, tool_pitch=None, speed=None, wait: bool = False) -> bool:
        """Straight-line TCP move. Returns False (and raises a safety warning) if the target is unreachable."""
        with self._lock:
            res = self.arm.inverse_kinematics(position, orientation, seed=self._q_cmd(), tool_pitch=tool_pitch)
            if not res.success and orientation is None and tool_pitch is not None:
                res = self.arm.inverse_kinematics(position, seed=self._q_cmd())
            if not res.success:
                self.safety.report(self.sim_time, UNREACHABLE, res.message)
                log.warning(res.message)
                return False
        ok = self.submit(MotionCommand.move_to(position, speed, orientation, tool_pitch, source="api"))
        if ok and wait:
            self.wait_until_settled()
        return ok

    def set_gripper_position(self, value: float, wait: bool = False) -> None:
        self.submit(MotionCommand.gripper("SET", value, source="api"))
        if wait:
            self.wait_until_settled()

    def open_gripper(self, wait: bool = False) -> None:
        self.set_gripper_position(1.0, wait)

    def close_gripper(self, wait: bool = False) -> None:
        self.set_gripper_position(0.0, wait)

    def home(self, wait: bool = False) -> None:
        self.submit(MotionCommand.home(source="api"))
        if wait:
            self.wait_until_settled()

    def stop(self) -> None:
        """Immediate stop of all motion (targets := current angles). Does not latch."""
        with self._lock:
            self._pending.clear()
            self.planner.cancel()
            self.arm.stop()

    def emergency_stop(self) -> None:
        """Latching stop: halts every joint at once; commands are refused until reset()."""
        with self._lock:
            self.estopped = True
            self._pending.clear()
            self.planner.cancel()
            self.arm.stop()
            self.safety.report(self.sim_time, ESTOP, "EMERGENCY STOP - reset required", Severity.FAULT)
        log.warning("EMERGENCY STOP")

    def reset(self, to_home: bool = False) -> None:
        """Explicitly clear e-stop and latched faults. Holds position (or snaps home in simulation)."""
        with self._lock:
            self.estopped = False
            self.safety.clear()
            self.planner.cancel()
            self._pending.clear()
            self._collision_text = "NONE"
            if to_home:
                self.arm.reset()
                if hasattr(self.backend, "reset_state"):
                    self.backend.reset_state(self.arm.joint_angles[:5], self.arm.gripper.opening)
            else:
                self.arm.stop()
            self._sync_backend()
        log.info("reset: controller re-armed")

    # ------------------------------------------------------------------ telemetry
    def snapshot(self) -> Telemetry:
        with self._lock:
            pose = self.arm.get_end_effector_pose()
            return Telemetry(
                sim_time=self.sim_time,
                joint_angles=self.arm.joint_angles.copy(),
                joint_targets=self.arm.joint_targets.copy(),
                tcp_position=pose.position.copy(), tcp_rpy=pose.rpy.copy(),
                gripper_opening=self.arm.gripper.opening,
                gesture=self._gesture, confidence=self._confidence,
                command=self.current_command.describe() if self.current_command else "-",
                collision=self._collision_text,
                safety=self.safety.status(self.sim_time),
                estopped=self.estopped, control_hz=self.control_hz_measured,
                torques=self.arm.gravity_torques(self.safety.payload_kg))


# ====================================================================================================
# PICK AND PLACE   (from control/pick_and_place.py)
# ====================================================================================================
# pick_and_place(): a scripted sequence built only from RobotController calls (works on any controller).

def pick_and_place(controller, object_id, pick_position, place_position, hover: float = 0.06,
                   speed: float = 0.08, tool_pitch: float = math.pi, grasp_time: float = 0.6,
                   arrive_tol: float = 0.004) -> bool:
    """
    Pick the object whose grip point is at ``pick_position`` (TCP target, m) and put it down at
    ``place_position``. ``object_id`` is only used for logging / verification (a name or id).
    Returns False as soon as a waypoint is refused, vetoed by the safety layer, or not reached within
    ``arrive_tol`` metres (unreachable target, collision veto, e-stop, fault).
    """
    pick = np.asarray(pick_position, dtype=float)
    place = np.asarray(place_position, dtype=float)
    up = np.array([0.0, 0.0, hover])

    def go(target) -> bool:
        ok = controller.move_cartesian(target, tool_pitch=tool_pitch, speed=speed, wait=True)
        err = np.linalg.norm(controller.snapshot().tcp_position - target) if ok else np.inf
        if not ok or controller.blocked or err > arrive_tol:
            log.warning("pick_and_place[%s]: could not reach %s (error %.1f mm, safety: %s)", object_id,
                        np.round(target, 3), err * 1000, controller.snapshot().safety)
            return False
        return True

    controller.open_gripper(wait=True)
    steps = [("hover over pick", pick + up), ("descend", pick)]
    for name, target in steps:
        if not go(target):
            return False
    controller.close_gripper(wait=True)
    controller.pause(grasp_time)                          # let the fingers squeeze
    for name, target in (("lift", pick + up), ("transfer", place + up), ("lower", place)):
        if not go(target):
            return False
    controller.open_gripper(wait=True)
    controller.pause(0.3)
    ok = go(place + up)                                  # retreat
    log.info("pick_and_place[%s]: %s", object_id, "done" if ok else "failed on retreat")
    return ok


# ====================================================================================================
# HARDWARE CONTROLLER (PLACEHOLDER)   (from control/hardware_controller.py)
# ====================================================================================================
# HardwareRobotController - PLACEHOLDER for the real LewanSoul/STM32 arm (Phase 11).
#
# It implements the same ``RobotController`` interface as the simulator, so the EMG layer,
# planner and keyboard code run unchanged. What is missing is only the transport:
#
#   * open the serial/USB link to the 6-channel bus-servo controller board,
#   * encode a "move servo N to position P over T ms" frame for the board's protocol,
#   * (optionally) read back servo positions/voltage for telemetry and the low-voltage alarm.
#
# TODO: take the exact frame format from the controller's protocol document - nothing about it
# is assumed here. ``joint_to_ticks`` shows how the calibrated direction / zero offset from
# config/robot_config.py are applied; the tick scale is a PLACEHOLDER.
#
# Keep the same safety rules as the simulator: clamp to joint limits, reject commands while
# e-stopped, and rate-limit using the servo velocity limits (reuse SimulatedServo as the
# "desired motion generator" and send its angle to the hardware each tick).

class HardwareRobotController(RobotController):
    def __init__(self, port: str, baudrate: int = 115200, config=None):
        self.port, self.baudrate = port, baudrate
        self.arm = RobotArm(config)           # kinematic model reused for IK / limits / calibration

    def joint_to_ticks(self, joint_angles) -> list:
        """Joint angles (rad, J1..J6) -> per-servo position counts using the calibration in the config."""
        return [j.servo.to_ticks(a) for j, a in zip(self.arm.joints, joint_angles)]

    def _todo(self, what: str):
        raise NotImplementedError(f"HardwareRobotController.{what}: implement the LewanSoul controller transport")

    def move_joint(self, joint_id, angle, wait=False): self._todo("move_joint")
    def move_joints(self, angles, wait=False): self._todo("move_joints")
    def move_cartesian(self, position, orientation=None, tool_pitch=None, speed=None, wait=False): self._todo("move_cartesian")
    def open_gripper(self, wait=False): self._todo("open_gripper")
    def close_gripper(self, wait=False): self._todo("close_gripper")
    def set_gripper_position(self, value, wait=False): self._todo("set_gripper_position")
    def stop(self): self._todo("stop")
    def emergency_stop(self): self._todo("emergency_stop")
    def reset(self): self._todo("reset")
    def home(self, wait=False): self._todo("home")
    def submit(self, command: MotionCommand) -> bool: self._todo("submit")
    def snapshot(self): self._todo("snapshot")


# ====================================================================================================
# EMG: GESTURE MAPPING & SMOOTHING   (from emg/gesture_mapper.py)
# ====================================================================================================
# Gesture normalisation, temporal smoothing and gesture -> MotionCommand mapping.
#
# Nothing here knows about servos, IK or the simulator: the output is a MotionCommand.

def normalize_gesture(name: str) -> str:
    g = str(name).strip().lower().replace("-", "_")
    g = GESTURE_ALIASES.get(g, g)
    if g not in GESTURES:
        raise ValueError(f"unknown gesture '{name}' (expected one of {GESTURES})")
    return g


class GestureSmoother:
    """Confidence gate + sliding-window majority vote.

    * prediction with confidence < threshold  -> counted as REST
    * winner needs at least ceil(min_agreement * window) votes of the last ``window`` samples,
      otherwise the output is REST (so LEFT,RIGHT,LEFT,REST,RIGHT stays REST).
    Until ``window`` samples have arrived the missing slots count as "no vote", so a single
    sample can never start a motion.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW, min_agreement: float = SMOOTHING_MIN_AGREEMENT,
                 confidence_threshold: float = EMG_CONFIDENCE_THRESHOLD):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.min_agreement = min_agreement
        self.threshold = confidence_threshold
        self.needed = max(1, math.ceil(min_agreement * window - 1e-9))
        self._buf: deque = deque(maxlen=window)

    def update(self, gesture: str, confidence: float = 1.0) -> str:
        g = normalize_gesture(gesture)
        if confidence is None or confidence < self.threshold:
            g = "rest"
        self._buf.append(g)
        counts = Counter(self._buf)
        recency = {v: i for i, v in enumerate(self._buf)}          # later index = more recent
        best, n = max(counts.items(), key=lambda kv: (kv[1], recency[kv[0]]))
        return best if n >= self.needed else "rest"

    def reset(self) -> None:
        self._buf.clear()


class GestureMapper:
    """smoothed gesture -> MotionCommand (continuous Cartesian jog / gripper / hold)."""

    def __init__(self, speed: float = EMG_SPEED_M_S, mapping: dict | None = None,
                 fist_toggles: bool = FIST_CLOSE_TOGGLES):
        self.speed = speed
        self.mapping = dict(mapping or GESTURE_TO_COMMAND)
        self.fist_toggles = fist_toggles
        self._last = "rest"

    def map(self, gesture: str, confidence: float | None = None, source: str = "emg") -> MotionCommand:
        g = normalize_gesture(gesture)
        ctype, direction = self.mapping[g]
        prev, self._last = self._last, g
        kw = dict(source=source, gesture=g, confidence=confidence)
        if ctype == CARTESIAN:
            return MotionCommand(CARTESIAN, direction, magnitude=None, speed=self.speed, **kw)
        if ctype == GRIPPER:
            if self.fist_toggles:
                if prev != g:
                    return MotionCommand(GRIPPER, "TOGGLE", **kw)
                return MotionCommand(HOLD, "NONE", **kw)
            return MotionCommand(GRIPPER, direction, **kw)
        return MotionCommand(HOLD, "NONE", **kw)


# ====================================================================================================
# EMG: CONTROLLER   (from emg/emg_controller.py)
# ====================================================================================================
# EMGController: the ONLY place where gesture predictions enter the system.
#
#     EMGSource.read() -> {"gesture","confidence","timestamp"}
#         -> confidence gate + smoothing -> GestureMapper -> MotionCommand -> sink (controller.submit)
#
# To use your real Random-Forest pipeline, implement an ``EMGSource`` (or wrap a function with
# ``CallbackEMGSource``); nothing in the robot/simulation core changes.

@dataclass
class GesturePrediction:
    gesture: str
    confidence: float = 1.0
    timestamp: float | None = None            # wall-clock epoch seconds from the classifier

    @classmethod
    def from_dict(cls, d) -> "GesturePrediction":
        return cls(d["gesture"], float(d.get("confidence", 1.0)), d.get("timestamp"))


class EMGSource(ABC):
    """A producer of gesture predictions (real device + classifier, keyboard, script ...)."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @abstractmethod
    def read(self) -> Optional[dict | GesturePrediction]:
        """Return the newest prediction or None if nothing new. Must not block for long."""


class CallbackEMGSource(EMGSource):
    """Wrap any callable returning {"gesture":..,"confidence":..,"timestamp":..} (or None)."""

    def __init__(self, fn: Callable[[], Optional[dict]], connect_fn=None, disconnect_fn=None):
        self.fn, self._c, self._d = fn, connect_fn, disconnect_fn

    def connect(self):
        if self._c:
            self._c()

    def disconnect(self):
        if self._d:
            self._d()

    def read(self):
        return self.fn()


class EMGController:
    def __init__(self, source: EMGSource | None = None, sink: Callable[[MotionCommand], bool] | None = None,
                 status_callback: Callable[[str, float | None], None] | None = None,
                 confidence_threshold: float = EMG_CONFIDENCE_THRESHOLD,
                 window: int = SMOOTHING_WINDOW, min_agreement: float = SMOOTHING_MIN_AGREEMENT,
                 speed: float = EMG_SPEED_M_S, poll_hz: float = EMG_POLL_HZ,
                 fist_toggles: bool = FIST_CLOSE_TOGGLES, latency: LatencyTracker | None = None):
        self.source = source
        self.sink = sink
        self.status_callback = status_callback
        self.smoother = GestureSmoother(window, min_agreement, confidence_threshold)
        self.mapper = GestureMapper(speed, fist_toggles=fist_toggles)
        self.poll_period = 1.0 / poll_hz
        self.latency = latency or LatencyTracker()
        self.last_prediction: GesturePrediction | None = None
        self.last_smoothed = "rest"
        self.last_command: MotionCommand | None = None
        self.connected = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ API required by the spec
    def connect(self) -> None:
        if self.source is not None:
            self.source.connect()
        self.connected = True

    def disconnect(self) -> None:
        self.stop()
        if self.source is not None:
            self.source.disconnect()
        self.connected = False

    def receive_gesture(self) -> Optional[GesturePrediction]:
        """Poll the source once."""
        if self.source is None:
            return None
        raw = self.source.read()
        if raw is None:
            return None
        return raw if isinstance(raw, GesturePrediction) else GesturePrediction.from_dict(raw)

    def process_gesture(self, gesture: str, confidence: float = 1.0, timestamp: float | None = None) -> MotionCommand:
        """One prediction in -> one MotionCommand out (and submitted to the sink, if any)."""
        t0 = time.perf_counter()
        try:
            smoothed = self.smoother.update(gesture, confidence)
            raw_name = normalize_gesture(gesture)
        except ValueError as exc:                       # unknown label: fail safe
            log.warning("%s - treating as REST", exc)
            smoothed = self.smoother.update("rest", 1.0)
            raw_name = "rest"
        cmd = self.mapper.map(smoothed, confidence)
        cmd.created_at = t0                              # latency clock starts when the prediction arrived
        cmd.timestamp = timestamp
        self.last_prediction = GesturePrediction(raw_name, confidence, timestamp)
        self.last_smoothed, self.last_command = smoothed, cmd
        self.latency.record("emg_to_command_ms", (time.perf_counter() - t0) * 1000.0)
        if timestamp is not None:
            self.latency.record("classifier_age_ms", max(0.0, (time.time() - timestamp) * 1000.0))
        if self.status_callback:
            label = smoothed.upper() if smoothed == raw_name else f"{smoothed.upper()} (raw {raw_name.upper()})"
            self.status_callback(label, confidence)
        if self.sink is not None:
            self.sink(cmd)
        return cmd

    def process_prediction(self, pred: GesturePrediction | dict) -> MotionCommand:
        if isinstance(pred, dict):
            pred = GesturePrediction.from_dict(pred)
        return self.process_gesture(pred.gesture, pred.confidence, pred.timestamp)

    # ------------------------------------------------------------------ input thread
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="emg-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                pred = self.receive_gesture()
                if pred is not None:
                    self.process_prediction(pred)
            except Exception:                           # never let the input thread die silently
                log.exception("EMG input error")
            nxt += self.poll_period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()


# ====================================================================================================
# EMG: MOCK SOURCES   (from emg/mock_emg.py)
# ====================================================================================================
# Simulated EMG sources for development without the armband.

# Keyboard 1..5 / 0 -> gesture (the UI feeds key presses in via KeyboardEMGSource.set_gesture)
KEY_TO_GESTURE = {"1": "left", "2": "right", "3": "up", "4": "down", "5": "fist_close", "0": "rest"}


class KeyboardEMGSource(EMGSource):
    """While a gesture key is held it streams that gesture at ``confidence``; otherwise REST.

    ``quiet_when_idle``: after a key is released, send ``quiet_after`` REST samples (enough to flush the
    smoothing window and stop the arm) and then nothing at all, so manual jogging is not cancelled by a
    constant stream of REST -> HOLD commands.
    """

    def __init__(self, confidence: float = 0.95, quiet_when_idle: bool = False, quiet_after: int = 5):
        self.confidence = confidence
        self.quiet_when_idle = quiet_when_idle
        self.quiet_after = max(1, quiet_after)
        self._gesture = "rest"
        self._rest_left = 0

    def set_gesture(self, gesture: str) -> None:
        self._gesture = gesture

    def release(self) -> None:
        if self._gesture != "rest":
            self._rest_left = self.quiet_after + 2
        self._gesture = "rest"

    def read(self) -> Optional[GesturePrediction]:
        if self.quiet_when_idle and self._gesture == "rest":
            if self._rest_left <= 0:
                return None
            self._rest_left -= 1
        return GesturePrediction(self._gesture, self.confidence, time.time())


class ScriptedEMGSource(EMGSource):
    """Plays [(gesture, confidence, n_samples), ...] one sample per read(); REST afterwards."""

    def __init__(self, script: Iterable[tuple]):
        self._samples = [(g, c) for g, c, n in script for _ in range(n)]
        self._i = 0

    @property
    def finished(self) -> bool:
        return self._i >= len(self._samples)

    def read(self) -> Optional[GesturePrediction]:
        g, c = self._samples[self._i] if self._i < len(self._samples) else ("rest", 1.0)
        self._i += 1
        return GesturePrediction(g, c, time.time())


class NoisyEMGSource(EMGSource):
    """Wraps another source; randomly flips labels / drops confidence to mimic a poor classifier."""

    def __init__(self, inner: EMGSource, flip_prob: float = 0.15, low_conf_prob: float = 0.1, seed: int = 0):
        self.inner, self.flip, self.low = inner, flip_prob, low_conf_prob
        self.rng = np.random.default_rng(seed)

    def connect(self):
        self.inner.connect()

    def disconnect(self):
        self.inner.disconnect()

    def read(self):
        p = self.inner.read()
        if p is None:
            return None
        g, c = p.gesture, p.confidence
        if self.rng.random() < self.flip:
            g = str(self.rng.choice([x for x in GESTURES if x != g]))
            c = float(self.rng.uniform(0.5, 0.95))
        if self.rng.random() < self.low:
            c = float(self.rng.uniform(0.3, 0.7))
        return GesturePrediction(g, c, p.timestamp)


def demo_script() -> list:
    """A short, deterministic demo: move left, up, close gripper, down, right; with some noise samples."""
    return [("rest", 0.9, 10), ("left", 0.95, 60), ("rest", 0.9, 10), ("up", 0.93, 40), ("left", 0.6, 3),
            ("up", 0.93, 10), ("fist_close", 0.9, 30), ("rest", 0.9, 10), ("down", 0.94, 40),
            ("right", 0.95, 80), ("rest", 0.9, 20)]


# ====================================================================================================
# UI: KEYBOARD CONTROLLER   (from ui/controls.py)
# ====================================================================================================
# Input handling that does not depend on PyBullet: key map + KeyboardController.
#
# The GUI feeds in the set of currently held key NAMES (and the edge-triggered ones);
# this module turns them into MotionCommands. Only ONE motion key is honoured at a time
# (the planner executes one motion at a time), preferring the key pressed first.

# key -> (joint id, sign).  Q/A J1, W/S J2, E/D J3, R/F J4, T/G J5, Y/H gripper (J6)
JOINT_KEYS = {
    "q": (1, +1), "a": (1, -1), "w": (2, +1), "s": (2, -1), "e": (3, +1), "d": (3, -1),
    "r": (4, +1), "f": (4, -1), "t": (5, +1), "g": (5, -1), "y": (6, +1), "h": (6, -1),
}
# arrow keys (+ I/K for depth) -> world-frame Cartesian jog
CARTESIAN_KEYS = {"left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN", "i": "FORWARD", "k": "BACKWARD"}

KEY_HELP = """\
KEYBOARD (full / manual mode)
  Q/A J1 base   W/S J2 shoulder   E/D J3 elbow   R/F J4 wrist pitch   T/G J5 wrist roll   Y/H gripper open/close
  Arrows: end effector LEFT(-X) RIGHT(+X) UP(+Z) DOWN(-Z)     I/K: forward/back (+Y/-Y)
  O open gripper   C close gripper   SPACE stop   Z home   X reset (re-arm after stop)   ESC emergency stop
  P pick-and-place demo
MOCK EMG GESTURES (full / emg mode)
  1 left  2 right  3 up  4 down  5 fist_close  0 rest (release = rest)"""


class KeyboardController:
    def __init__(self, sink: Callable[[MotionCommand], bool], mode: str = "manual",
                 emg_source: KeyboardEMGSource | None = None, demo: Callable[[], None] | None = None):
        self.sink = sink
        self.mode = mode
        self.emg_source = emg_source
        self.demo = demo
        self._active: str | None = None

    def update(self, held: set, triggered: set) -> None:
        # --- system keys (always available), edge-triggered
        if "esc" in triggered:
            self.sink(MotionCommand(ESTOP, source="keyboard"))
        if "x" in triggered:
            self.sink(MotionCommand(RESET, source="keyboard"))
        if "z" in triggered and self.mode != "emg":
            self.sink(MotionCommand.home(source="keyboard"))
        if "p" in triggered and self.demo is not None:
            self.demo()
        if " " in triggered:
            self.sink(MotionCommand.stop(source="keyboard"))
        if "o" in triggered:
            self.sink(MotionCommand.gripper("OPEN", source="keyboard"))
        if "c" in triggered:
            self.sink(MotionCommand.gripper("CLOSE", source="keyboard"))

        if self.emg_source is not None:                       # modes "emg" and "full": keys 0-5 = mock gestures
            digits = [k for k in KEY_TO_GESTURE if k in held]
            if digits:
                self.emg_source.set_gesture(KEY_TO_GESTURE[digits[-1]])
            else:
                self.emg_source.release()
        if self.mode == "emg":
            return

        # --- manual jogging: pick one active key
        motion = [k for k in held if k in JOINT_KEYS or k in CARTESIAN_KEYS]
        if self._active not in motion:
            self._active = sorted(motion)[0] if motion else None
        k = self._active
        if k is None:
            if self._was_jogging:
                self.sink(MotionCommand.hold(source="keyboard"))
            self._was_jogging = False
            return
        self._was_jogging = True
        if k in JOINT_KEYS:
            j, s = JOINT_KEYS[k]
            self.sink(MotionCommand.joint_jog(j, s, source="keyboard"))
        else:
            self.sink(MotionCommand(CARTESIAN, CARTESIAN_KEYS[k], source="keyboard"))

    _was_jogging = False


# ====================================================================================================
# UI: PYBULLET GUI VIEWER (SEPARATE PROCESS)   (from ui/gui.py)
# ====================================================================================================
# PyBullet GUI viewer. Runs in its OWN PROCESS (see ui/viewer_link.py).
#
# Why a separate process: any call into PyBullet's GUI client can block 100+ ms when the window is idle,
# and on macOS the GUI must own a main thread. Keeping it away from the control process means the 240 Hz
# control/physics loop can never be stalled by rendering or window events.
#
# Parent -> viewer : ("state", {"joints", "objects", "telemetry"})  /  ("quit",)
# Viewer -> parent : ("cmd", MotionCommand) / ("gesture", name | None) / ("demo",) / ("closed",)
#
# Window layout (PyBullet's own): sidebar = joint + Cartesian sliders and buttons, centre = 3D scene
# (frames drawn on every link, TCP frame longer) with one status line; full telemetry is shown by the parent
# in the terminal.

class _RemoteGestureKeys:
    """Stands in for KeyboardEMGSource inside the viewer: forwards mock-EMG key state to the parent."""

    def __init__(self, send):
        self.send = send
        self._cur = None

    def set_gesture(self, g):
        if g != self._cur:
            self._cur = g
            self.send(("gesture", g))

    def release(self):
        self.set_gesture(None)


class PyBulletGUI:
    def __init__(self, cfg, env, conn, mode: str = "manual", show_frames: bool = True, dt: float = 1 / 240,
                 panel: bool = False):
        import pybullet as p
        self.p, self.cfg, self.conn, self.mode = p, cfg, conn, mode
        self.panel = panel
        self.client = p.connect(p.GUI, options="--width=1280 --height=760")
        self.world = make_world(self.client, cfg, env, dt, gui=True, show_frames=show_frames)
        self.kb = KeyboardController(self._send_cmd, mode,
                                     _RemoteGestureKeys(self.conn.send) if mode in ("emg", "full") else None,
                                     demo=lambda: self.conn.send(("demo",)))
        self.fps = 0.0
        self.running = True
        self._slider_last, self._button_last = {}, {}
        self._text_id, self._last_status = None, None
        self.telemetry = None
        self.joint_params, self.cart_params, self.buttons = [], {}, {}
        if panel:
            self._build_widgets()
        else:
            # PyBullet's sidebar is redrawn on EVERY render and costs ~8 ms per widget on the M1's OpenGL
            # driver (measured: 21 widgets -> ~160 ms/frame = 5 fps; no panel -> ~3 ms/frame). The lean view
            # keeps the 3D window smooth; every action has a key (see KEY_HELP). --panel brings the sliders back.
            self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)

    def _send_cmd(self, cmd: MotionCommand) -> bool:
        self.conn.send(("cmd", cmd))
        return True

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self) -> None:
        p, cl, cfg = self.p, self.client, self.cfg
        for j in cfg.joints:
            lo, hi = math.degrees(j.limit.min_angle), math.degrees(j.limit.max_angle)
            pid = p.addUserDebugParameter(f"J{j.joint_id} {j.role} (deg)", lo, hi, math.degrees(j.home), physicsClientId=cl)
            self.joint_params.append(pid)
            self._slider_last[pid] = math.degrees(j.home)
        for name, lo, hi, val in (("X (mm)", -300, 300, 0.0), ("Y (mm)", -300, 320, 136.0), ("Z (mm)", 0, 420, 64.0),
                                  ("Roll (deg)", -180, 180, 180.0), ("Pitch (deg)", -90, 90, 0.0),
                                  ("Yaw (deg)", -180, 180, 0.0)):
            self.cart_params[name] = p.addUserDebugParameter("Cartesian " + name, lo, hi, val, physicsClientId=cl)
        for name in ("GO to XYZ (tool down)", "GO to full pose (X,Y,Z,R,P,Y)", "HOME", "RESET", "STOP",
                     "OPEN GRIPPER", "CLOSE GRIPPER", "PICK & PLACE DEMO", "E-STOP"):
            pid = p.addUserDebugParameter(name, 1, 0, 0, physicsClientId=cl)      # rangeMin > rangeMax => button
            self.buttons[name] = pid
            self._button_last[pid] = 0

    def _poll_widgets(self) -> None:
        if not self.panel:
            return
        p, cl, send = self.p, self.client, self._send_cmd
        rd = lambda pid: p.readUserDebugParameter(pid, physicsClientId=cl)
        if self.mode in ("manual", "full"):
            for i, pid in enumerate(self.joint_params):
                v = rd(pid)
                if abs(v - self._slider_last[pid]) > 1e-6:
                    self._slider_last[pid] = v
                    send(MotionCommand.joint_target(math.radians(v), joint=i + 1, source="slider"))
        pressed = set()
        for n, pid in self.buttons.items():
            v = rd(pid)
            if v != self._button_last[pid]:
                self._button_last[pid] = v
                pressed.add(n)
        if "E-STOP" in pressed:
            send(MotionCommand(ESTOP, source="button"))
        if "RESET" in pressed:
            send(MotionCommand(RESET, source="button"))
        if "STOP" in pressed:
            send(MotionCommand.stop(source="button"))
        if "HOME" in pressed:
            send(MotionCommand.home(source="button"))
        if "OPEN GRIPPER" in pressed:
            send(MotionCommand.gripper("OPEN", source="button"))
        if "CLOSE GRIPPER" in pressed:
            send(MotionCommand.gripper("CLOSE", source="button"))
        if "PICK & PLACE DEMO" in pressed:
            self.conn.send(("demo",))
        if pressed & {"GO to XYZ (tool down)", "GO to full pose (X,Y,Z,R,P,Y)"}:
            c = {n: rd(pid) for n, pid in self.cart_params.items()}
            pos = (c["X (mm)"] / 1000, c["Y (mm)"] / 1000, c["Z (mm)"] / 1000)
            if "GO to XYZ (tool down)" in pressed:
                send(MotionCommand.move_to(pos, tool_pitch=math.pi, source="slider"))
            else:
                rpy = tuple(math.radians(c[k]) for k in ("Roll (deg)", "Pitch (deg)", "Yaw (deg)"))
                send(MotionCommand.move_to(pos, orientation=rpy, source="slider"))

    # ------------------------------------------------------------------ keyboard
    def _key_name(self, key: int):
        p = self.p
        special = {p.B3G_LEFT_ARROW: "left", p.B3G_RIGHT_ARROW: "right", p.B3G_UP_ARROW: "up",
                   p.B3G_DOWN_ARROW: "down", 27: "esc"}
        if key in special:
            return special[key]
        return chr(key).lower() if 32 <= key < 127 else None

    def _poll_keys(self) -> None:
        p = self.p
        held, trig = set(), set()
        for key, st in p.getKeyboardEvents(physicsClientId=self.client).items():
            name = self._key_name(key)
            if name is None:
                continue
            if st & p.KEY_IS_DOWN:
                held.add(name)
            if st & p.KEY_WAS_TRIGGERED:
                trig.add(name)
        self.kb.update(held, trig)

    # ------------------------------------------------------------------ state from the control process
    def _apply_state(self, st: dict) -> None:
        p, cl, w = self.p, self.client, self.world
        for ji, pos in zip(w.arm_idx + w.finger_idx, st["joints"]):
            p.resetJointState(w.robot, ji, pos, physicsClientId=cl)
        for name, (pos, orn) in st["objects"].items():
            p.resetBasePositionAndOrientation(w.objects[name], pos, orn, physicsClientId=cl)
        self.telemetry = st["telemetry"]

    def _draw_status(self) -> None:
        """ONE in-scene text line, rewritten only when it changes (debug text costs ~75 ms/update)."""
        s = self.telemetry
        if s is None:
            return
        conf = "-" if s.confidence is None else f"{s.confidence*100:.0f}%"
        status = f"{s.gesture} {conf} | {s.command} | " + ("ESTOP" if s.estopped else "SAFETY " + s.safety.level.name)
        if status == self._last_status:
            return
        color = (1, 0.3, 0.3) if (s.estopped or s.safety.level >= 2) else (1, 0.6, 0.1) if not s.safety.ok else (0.1, 0.5, 0.1)
        kw = dict(textColorRGB=list(color), textSize=1.3, physicsClientId=self.client)
        if self._text_id is not None:
            kw["replaceItemUniqueId"] = self._text_id
        self._text_id = self.p.addUserDebugText(status, [-0.05, 0.0, 0.50], **kw)
        self._last_status = status

    # ------------------------------------------------------------------ main loop (viewer process main thread)
    def run(self, target_fps: float = 30.0) -> None:
        period = 1.0 / target_fps
        last = time.perf_counter()
        try:
            while self.running and self.p.isConnected(self.client):
                latest = None
                while self.conn.poll():
                    msg = self.conn.recv()
                    if msg[0] == "quit":
                        return
                    if msg[0] == "state":
                        latest = msg[1]
                if latest is not None:
                    self._apply_state(latest)            # pose update first: it also wakes the GUI's render loop
                self._poll_keys()
                self._poll_widgets()
                self._draw_status()
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
                last = now
                time.sleep(max(0.0, period - 0.002 - (time.perf_counter() - now)))
        except self.p.error:                             # window closed
            pass
        finally:
            try:
                self.conn.send(("closed",))
            except Exception:
                pass


def viewer_main(conn, cfg, env, mode, show_frames, dt, fps: float = 30.0, panel: bool = False) -> None:
    """Entry point of the viewer process."""
    gui = PyBulletGUI(cfg, env, conn, mode, show_frames, dt, panel)
    gui.run(fps)


# ====================================================================================================
# UI: VIEWER LINK & TERMINAL DASHBOARD   (from ui/viewer_link.py)
# ====================================================================================================
# Parent-process side of the GUI: spawns the viewer process and bridges it to the controller.
#
#     publisher thread (30 Hz):  backend.view_state() + controller.snapshot()  --pipe-->  viewer
#                                viewer --pipe--> ("cmd", MotionCommand) -> controller.submit
#                                                 ("gesture", g)         -> KeyboardEMGSource
# Also prints a live telemetry dashboard in the terminal (when stdout is a TTY).

def telemetry_lines(s, mode: str) -> list:
    j = "  ".join(f"J{i+1}:{math.degrees(a):7.1f} deg" for i, a in enumerate(s.joint_angles))
    conf = "-" if s.confidence is None else f"{s.confidence*100:.0f}%"
    safety = "ESTOP - press X / RESET" if s.estopped else str(s.safety)
    return [
        f"SIM {s.sim_time:8.2f} s | CONTROL {s.control_hz:4.0f} Hz | CONNECTED: {s.connected} | MODE: {mode.upper()}",
        j,
        f"X {s.tcp_position[0]*1000:7.1f}  Y {s.tcp_position[1]*1000:7.1f}  Z {s.tcp_position[2]*1000:7.1f} mm"
        f"   Roll {math.degrees(s.tcp_rpy[0]):7.1f}  Pitch {math.degrees(s.tcp_rpy[1]):7.1f}  Yaw {math.degrees(s.tcp_rpy[2]):7.1f} deg",
        f"GRIPPER {s.gripper_opening*100:3.0f}% open | GESTURE: {s.gesture} | CONFIDENCE: {conf} | COMMAND: {s.command}",
        f"COLLISION: {s.collision} | SAFETY: {safety}",
    ]


class ViewerLink:
    def __init__(self, cfg, env, backend, controller, mode: str = "manual", show_frames: bool = True,
                 gesture_source=None, dashboard: bool = True, rate_hz: float = 30.0, on_demo=None,
                 gui_fps: float = 30.0, panel: bool = False):
        self.cfg, self.env, self.backend, self.ctl = cfg, env, backend, controller
        self.panel = panel
        self.on_demo = on_demo
        self.gui_fps = gui_fps
        self.mode, self.show_frames = mode, show_frames
        self.gesture_source = gesture_source
        self.dashboard = dashboard and sys.stdout.isatty()
        self.period = 1.0 / rate_hz
        self.closed = threading.Event()
        self._proc = None
        self._conn = None
        self._thread = None
        self._stop = threading.Event()
        self._dash_drawn = False

    def start(self) -> None:
        ctx = mp.get_context("spawn")                 # fresh process: owns its own main thread for the GUI
        self._conn, child = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=viewer_main, name="viewer",
                                 args=(child, self.cfg, self.env, self.mode, self.show_frames, self.backend.dt, self.gui_fps, self.panel),
                                 daemon=True)
        self._proc.start()
        self._thread = threading.Thread(target=self._loop, name="viewer-link", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._conn:
                self._conn.send(("quit",))
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._proc:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()

    def _handle(self, msg) -> None:
        kind = msg[0]
        if kind == "cmd":
            self.ctl.submit(msg[1])
        elif kind == "gesture" and self.gesture_source is not None:
            if msg[1] is None:
                self.gesture_source.release()
            else:
                self.gesture_source.set_gesture(msg[1])
        elif kind == "demo" and self.on_demo is not None:
            self.on_demo()
        elif kind == "closed":
            self.closed.set()

    def _loop(self) -> None:
        nxt = time.perf_counter()
        n = 0
        while not self._stop.is_set() and not self.closed.is_set():
            try:
                while self._conn.poll():
                    self._handle(self._conn.recv())
                if not self._proc.is_alive():
                    self.closed.set()
                    break
                st = self.backend.view_state()
                snap = self.ctl.snapshot()
                st["telemetry"] = snap
                self._conn.send(("state", st))
                n += 1
                if self.dashboard and n % 3 == 0:
                    self._print_dashboard(snap)
            except (EOFError, BrokenPipeError, OSError):
                self.closed.set()
                break
            nxt += self.period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def _print_dashboard(self, snap) -> None:
        lines = telemetry_lines(snap, self.mode)
        up = "\x1b[%dA" % len(lines) if self._dash_drawn else ""
        sys.stdout.write(up + "".join(l[:200].ljust(150) + "\n" for l in lines))
        sys.stdout.flush()
        self._dash_drawn = True


# ====================================================================================================
# RUNTIME: wires everything together and owns the threads   (from runtime.py)
# ====================================================================================================
#     thread "emg-input"   : EMGController polls the EMG source, smooths, maps -> MotionCommand queue
#     thread "control"     : fixed-rate loop (default 240 Hz): planner/IK, servos, physics (DIRECT client), safety, log
#     thread "viewer-link" : 30 Hz bridge to the GUI process (state out, MotionCommands in) + terminal dashboard
#     thread "demo"        : optional pick-and-place sequence (key P / button)
#     process "viewer"     : PyBullet GUI window - isolated because GUI calls can block 100+ ms and must own a
#                            main thread on macOS
# The only coupling between them is the thread-safe ``controller.submit(MotionCommand)`` queue and
# ``controller.snapshot()``.
log = get_logger("arm")

MODES = ("full", "manual", "emg")
# Physics cost is per step. 120 Hz halves the CPU/heat of the original 240 Hz with no visible change in the arm's
# behaviour (planner runs at 100 Hz, EMG at 50 Hz, GUI at 30 Hz anyway) - good for a fanless MacBook Air.
DEFAULT_CONTROL_HZ = 120.0


class KeepAwake:
    """macOS: stop App Nap / idle sleep from throttling the 240 Hz timers while the sim runs.

    ``caffeinate -i -w <pid>`` exits by itself as soon as this process does, so nothing can be left behind.
    """

    def __init__(self):
        self._proc = None

    def start(self) -> None:
        if IS_MACOS and shutil.which("caffeinate"):
            try:
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self._proc = None

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


class SimulationRuntime:
    def __init__(self, config_path=None, mode: str = "full", emg_kind: str = "keyboard", headless: bool = False,
                 physics: bool = True, control_hz: float = DEFAULT_CONTROL_HZ, log_dir: str = "logs",
                 objects: bool = True, show_frames: bool = True, threshold: float = EMG_CONFIDENCE_THRESHOLD,
                 window: int = SMOOTHING_WINDOW, save_log: bool = True, gui_fps: float = 30.0,
                 panel: bool = False):
        self.mode, self.headless = mode, headless
        self.gui_fps, self.panel = gui_fps, panel
        self.cfg = load_config(config_path)
        self.env = Environment.default_scene() if objects else Environment()
        use_pb = physics and pybullet_available()
        if not headless and not use_pb:
            raise SystemExit("The 3D view needs PyBullet:  pip install pybullet\n"
                             "(or run with --headless for a console-only simulation)")
        if physics and not use_pb:
            log.warning("PyBullet not installed: running with the kinematic backend (no objects/contacts)")
        self.backend = (PyBulletBackend(self.cfg, dt=1.0 / control_hz) if use_pb else KinematicBackend())
        self.logger = SessionLogger(log_dir) if save_log else None
        self.controller = SimulatedRobotController(self.cfg, self.backend, control_hz=control_hz,
                                                   session_logger=self.logger)
        self.controller.external_stepper = False

        self.emg_source: EMGSource | None = None
        self.kb_source: KeyboardEMGSource | None = None
        if mode in ("emg", "full"):
            if emg_kind == "scripted":
                self.emg_source = ScriptedEMGSource(demo_script())
            elif emg_kind == "noisy":
                self.emg_source = NoisyEMGSource(ScriptedEMGSource(demo_script()))
            else:
                # In "full" mode the keyboard source stays silent while no gesture key is held, so the
                # manual keys / sliders are not cancelled by a constant stream of REST -> HOLD commands.
                self.kb_source = self.emg_source = KeyboardEMGSource(quiet_when_idle=(mode == "full"),
                                                                     quiet_after=window)
        self.emg = EMGController(self.emg_source, sink=self.controller.submit,
                                 status_callback=self.controller.set_emg_status, confidence_threshold=threshold,
                                 window=window, latency=self.controller.latency)
        self.keep_awake = KeepAwake()
        self.viewer = None
        self.show_frames = show_frames
        self._stop = threading.Event()
        self._ctl_thread: threading.Thread | None = None
        self._demo_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.keep_awake.start()
        self.controller.connect(self.env)
        self.controller.external_stepper = True
        self._ctl_thread = threading.Thread(target=self._control_loop, name="control", daemon=True)
        self._ctl_thread.start()
        if self.emg_source is not None:
            self.emg.connect()
            self.emg.start()
        if not self.headless:
            self.viewer = ViewerLink(self.cfg, self.env, self.backend, self.controller, self.mode,
                                     self.show_frames, gesture_source=self.kb_source, on_demo=self.start_demo,
                                     gui_fps=self.gui_fps, panel=self.panel)
            self.viewer.start()
            print(KEY_HELP + "\n")
        log.info("runtime started (mode=%s, headless=%s, backend=%s, log=%s)", self.mode, self.headless,
                 type(self.backend).__name__, self.logger.path if self.logger else "off")

    def stop(self) -> None:
        self._stop.set()
        if self.viewer:
            self.viewer.stop()
        self.emg.disconnect()
        if self._ctl_thread:
            self._ctl_thread.join(timeout=2.0)
        if self._demo_thread and self._demo_thread.is_alive():
            self._demo_thread.join(timeout=2.0)
        self.controller.external_stepper = False
        print("\nLatency summary:\n" + self.controller.latency.summary())
        self.controller.shutdown()
        self.keep_awake.stop()

    def _control_loop(self) -> None:
        c = self.controller
        dt = c.dt
        nxt = time.perf_counter()
        t_rate, n = nxt, 0
        while not self._stop.is_set():
            c.step(dt)
            n += 1
            now = time.perf_counter()
            if now - t_rate >= 1.0:
                c.control_hz_measured = n / (now - t_rate)
                t_rate, n = now, 0
            nxt += dt
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:                  # fell far behind: do not spiral, resync
                nxt = time.perf_counter()

    # ------------------------------------------------------------------ pick-and-place demo (key P / button)
    def start_demo(self) -> None:
        if "cube" not in {o.name for o in self.env.objects}:
            log.warning("pick-and-place demo needs the default scene (do not use --no-objects)")
            return
        if self._demo_thread and self._demo_thread.is_alive():
            return
        self._demo_thread = threading.Thread(target=self._demo, name="demo", daemon=True)
        self._demo_thread.start()

    def _demo(self) -> None:
        c = self.controller
        try:
            if c.blocked:
                c.reset()
            cube = self.env.get("cube")
            pos = self.backend.object_position("cube") or cube.position
            grip_z = max(cube.half_height, 0.022)               # keep the fingertips above the table
            dx = 0.08 if pos[0] + 0.08 <= 0.25 else -0.08
            c.home(wait=True)
            ok = pick_and_place(c, "cube", np.array([pos[0], pos[1], grip_z]),
                                np.array([pos[0] + dx, pos[1], grip_z + 0.002]))
            log.info("pick-and-place demo: %s", "done" if ok else "failed (see warnings above)")
        except Exception:
            log.exception("pick-and-place demo crashed")

    # ------------------------------------------------------------------ run modes
    def run(self, duration: float | None = None) -> None:
        """Real-time run: control thread + EMG thread (+ GUI viewer process). Ctrl-C or closing the window ends it."""
        self.start()
        t_end = None if duration is None else time.perf_counter() + duration
        try:
            last = 0.0
            while (t_end is None or time.perf_counter() < t_end) and not (self.viewer and self.viewer.closed.is_set()):
                time.sleep(0.1)
                if self.viewer is None:                      # headless: console status once per sim second
                    s = self.controller.snapshot()
                    if s.sim_time - last >= 1.0:
                        last = s.sim_time
                        self.print_status(s)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def run_fast_scripted(self, duration: float) -> None:
        """Deterministic, unpaced headless run driven by simulated time (no threads)."""
        c = self.controller
        c.connect(self.env)
        if self.emg_source is not None:
            self.emg.connect()
        poll_every = max(1, int(round(1.0 / (EMG_POLL_HZ * c.dt))))
        steps = int(round(duration / c.dt))
        for i in range(steps):
            if self.emg_source is not None and i % poll_every == 0:
                pred = self.emg.receive_gesture()
                if pred is not None:
                    self.emg.process_prediction(pred)
            c.step()
            if i % int(round(1.0 / c.dt)) == 0:
                self.print_status(c.snapshot())
        print("\nLatency summary:\n" + c.latency.summary())
        c.shutdown()

    @staticmethod
    def print_status(s) -> None:
        j = " ".join(f"{math.degrees(a):6.1f}" for a in s.joint_angles)
        x, y, z = (v * 1000 for v in s.tcp_position)
        print(f"t={s.sim_time:6.2f}s  J[deg]= {j}  TCP[mm]= {x:6.1f} {y:6.1f} {z:6.1f}  "
              f"grip={s.gripper_opening*100:3.0f}%  gest={s.gesture:<14} cmd={s.command:<12} "
              f"safety={'ESTOP' if s.estopped else s.safety}")


# ====================================================================================================
# COMMAND LINE ENTRY POINT   (from main.py)
# ====================================================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="6-DOF robotic arm simulator (EMG-ready), single file. "
                                             "Run with no arguments to start everything.")
    ap.add_argument("--mode", choices=MODES, default="full",
                    help="full (default): manual keys/sliders AND EMG gestures; manual: no EMG; "
                         "emg: only gestures drive the arm")
    ap.add_argument("--emg", choices=["keyboard", "scripted", "noisy"], default="keyboard",
                    help="EMG source (replace with your classifier, see CallbackEMGSource)")
    ap.add_argument("--config", help="JSON calibration file")
    ap.add_argument("--headless", action="store_true", help="no 3D window; console telemetry")
    ap.add_argument("--fast", action="store_true", help="headless only: simulate as fast as possible, deterministic")
    ap.add_argument("--duration", type=float, help="seconds to run (default: until closed)")
    ap.add_argument("--no-physics", action="store_true", help="kinematic backend (no PyBullet dynamics)")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="do not draw coordinate frames (lighter GUI)")
    ap.add_argument("--no-log", action="store_true", help="do not write logs/session_NNN.csv")
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ,
                    help="control + physics rate (default %(default).0f; the original project used 240)")
    ap.add_argument("--gui-fps", type=float, default=30.0, help="viewer window loop rate (lower = cooler)")
    ap.add_argument("--panel", action="store_true",
                    help="show PyBullet's slider/button sidebar (slow on the M1: ~5 fps; default is the lean view)")
    ap.add_argument("--threshold", type=float, default=EMG_CONFIDENCE_THRESHOLD)
    ap.add_argument("--window", type=int, default=SMOOTHING_WINDOW)
    ap.add_argument("--export-urdf", metavar="PATH", help="write the generated URDF and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.export_urdf:
        print("wrote", export_urdf(load_config(a.config), a.export_urdf))
        return 0
    if running_under_rosetta():
        log.warning("This Python is x86_64 running under Rosetta (2-3x slower). Use a native arm64 Python: "
                    "https://www.python.org/downloads/macos/ (universal2) or `brew install python`, then "
                    "re-create your venv.")
    rt = SimulationRuntime(config_path=a.config, mode=a.mode, emg_kind=a.emg, headless=a.headless,
                           physics=not a.no_physics, control_hz=a.control_hz, log_dir=a.log_dir,
                           objects=not a.no_objects, show_frames=not a.no_frames, threshold=a.threshold,
                           window=a.window, save_log=not a.no_log, gui_fps=a.gui_fps, panel=a.panel)
    if a.headless and a.fast:
        rt.run_fast_scripted(a.duration or 12.0)
    else:
        rt.run(a.duration)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
