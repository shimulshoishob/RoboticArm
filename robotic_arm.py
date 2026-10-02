#!/usr/bin/env python3
"""6-DOF robotic arm simulator driven by EMG gestures (MuJoCo physics) - everything in ONE file.

Just run it (no arguments needed):

    python robotic_arm.py

That opens one BioWave-style window with everything in it:

    * 3D view of the arm          (MuJoCo, rendered off-screen into the window - no extra process, no mjpython)
    * connect / model / calibrate (ESP32-S3 armband over Wi-Fi or USB, trained .joblib model, REST/FLEX)
    * gesture -> arm-action mapping, live EMG plot, confidence, signal quality, arm telemetry
    * E-STOP, home, gripper, jog pad, "test without device" gesture buttons

and these threads: control loop (planner + IK + servos + safety + collision veto + CSV log in ./logs), EMG input
(mock source), and the MuJoCo physics inside the control loop.

    pip install numpy mujoco PyQt5 pyqtgraph pyserial joblib scikit-learn

Keys (window focused, no text box selected):
    Q/A W/S E/D R/F T/G Y/H   jog joints J1..J6        arrows + I/K   jog the tool (X/Z, Y)
    1 left  2 right  3 up  4 down  5 fist(close)  0 rest      <- mock EMG gestures (hold the key)
    O open gripper   C close gripper   SPACE stop   Z home   X reset after a stop   ESC emergency stop
    P  pick-and-place demo
3D view: drag = orbit, right-drag (or shift-drag) = pan, wheel = zoom, double-click = reset view.

Other ways to run it:

    python robotic_arm.py --model path/to/rf_realtime_model.joblib   # pre-load a trained model
    python robotic_arm.py --emg scripted                       # scripted EMG gesture demo
    python robotic_arm.py --mode emg                           # only EMG drives the arm (no manual jogging)
    python robotic_arm.py --headless --emg noisy --fast --duration 12   # console-only, deterministic
    python robotic_arm.py --export-urdf arm.urdf               # also: --export-mjcf arm.xml

MacBook Air M1 notes (what was tuned, and why):
    * BLAS/OpenMP pinned to 1 thread (set before numpy loads): the maths is tiny, thread hand-offs only cost.
    * MuJoCo: ~4 physics sub-steps per 120 Hz control tick cost well under 1 ms; nothing sleeps or spins.
    * Rendering is off-screen via CGL (MUJOCO_GL=cgl), capped to ~0.7 megapixel and --gui-fps (default 30);
      frames are only drawn while the window is visible. Shadows/reflections can be switched off (--no-shadows).
    * Kinematics: cached forward kinematics, closed-form rotations, analytic pitch Jacobian (IK ~2x faster).
    * caffeinate keeps App Nap from throttling the real-time timers; warns if Python runs under Rosetta.

Setup (Apple Silicon, NATIVE arm64 Python 3.10-3.13):   pip install numpy mujoco
Without MuJoCo the arm still runs headless on the pure-NumPy kinematic backend (no 3D view).
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
import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple
from urllib.parse import quote

import numpy as np

IS_MACOS = sys.platform == "darwin"

# Optional dependencies. The simulator itself only needs numpy (+ mujoco for physics and the 3D view).
if IS_MACOS:
    os.environ.setdefault("MUJOCO_GL", "cgl")        # headless OpenGL on Apple Silicon: no GLFW window needed
try:
    import mujoco
    HAS_MUJOCO = True
except Exception:                                    # pragma: no cover
    mujoco, HAS_MUJOCO = None, False
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    serial, HAS_SERIAL = None, False
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    joblib, HAS_JOBLIB = None, False

# A missing PyQt5 / pyqtgraph only disables the dashboard; the simulator itself keeps working.
HAS_QT = HAS_PYQTGRAPH = False
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
try:
    from PyQt5.QtCore import QEvent, QSize, QSettings, Qt, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont, QImage, QPixmap
    from PyQt5.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                                 QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                                 QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                                 QSizePolicy, QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)
    HAS_QT = True
except ImportError:
    pass
if HAS_QT:
    try:
        import pyqtgraph as pg
        HAS_PYQTGRAPH = True
    except ImportError:
        pass


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
# ====================================================================================================
# Physics backends. The controller talks only to ``PhysicsBackend``.
#
# * KinematicBackend - no dependencies; joints track their commands perfectly, no objects/contacts.
# * MuJoCoBackend    - gravity, contacts, position actuators with torque caps, friction grasping.
#
# MuJoCo is a plain library (no GUI process, no OpenGL window): the dashboard renders the scene off-screen into
# a Qt widget (see SceneRenderer / ViewWidget), so there is nothing to launch with `mjpython` and nothing that can
# stall the control loop. All MuJoCo calls are serialised through ``backend.lock``.

SUBSTEP_MAX_S = 0.0025          # physics sub-step upper bound; MuJoCo is so cheap here that 4 sub-steps per
                                # 120 Hz control tick cost < 0.1 ms on an M1
ARM_KP, ARM_KV = 60.0, 1.0      # position-actuator gains of the arm joints (N*m/rad, N*m*s/rad)
FINGER_KP, FINGER_KV = 2000.0, 20.0
JOINT_ARMATURE = 0.001          # servo gear-train inertia: keeps the stiff position loop well conditioned


def mujoco_available() -> bool:
    return HAS_MUJOCO


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
    """Perfect tracking, no physics. Useful for tests and machines without MuJoCo."""

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


# ----------------------------------------------------------------------------------------------------
# MJCF generation: the model comes from RobotConfig, so geometry/mass/limits live in ONE place.
# Link frames follow robot/kinematics.py: link i's frame sits on joint i's axis and the link extends along
# +Z by ``length`` to the next joint origin. Collision filtering: robot geoms never collide with each other
# (like the old URDF without self-collision); they collide with the table, floor and objects.
# ----------------------------------------------------------------------------------------------------
def _v(vals) -> str:
    return " ".join(f"{float(x):.6g}" for x in vals)


def _half(size) -> str:
    return _v([s / 2 for s in size])


def _rgba(color) -> str:
    return _v(color)


def build_mjcf(cfg: RobotConfig, env: Environment | None = None, timestep: float = 1.0 / 480.0) -> str:
    env = env or Environment()
    L, g = cfg.links, cfg.gripper
    tx, ty = env.table_size
    th = env.table_thickness
    bw, bd = cfg.base_size

    def inertial(link) -> str:
        ixx, iyy, izz = link.inertia_diag()
        return f'<inertial pos="{_v(link.com)}" mass="{link.mass:.6g}" diaginertia="{_v((ixx, iyy, izz))}"/>'

    x = [f'<mujoco model="{cfg.name}">',
         '<compiler angle="radian" autolimits="true"/>',
         f'<option timestep="{timestep:.8g}" gravity="0 0 -9.81" integrator="implicitfast"/>',
         '<visual><global offwidth="1920" offheight="1200"/><quality shadowsize="2048" offsamples="2"/>'
         '<headlight ambient="0.42 0.42 0.45" diffuse="0.5 0.5 0.5" specular="0 0 0"/>'
         '<map znear="0.01" zfar="20"/><rgba haze="0.07 0.14 0.25 1"/></visual>',
         '<asset>'
         '<texture type="skybox" builtin="gradient" rgb1="0.20 0.32 0.52" rgb2="0.05 0.09 0.16" width="512" height="512"/>'
         '<texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.22 0.32" rgb2="0.12 0.17 0.26" width="256" height="256" '
         'mark="edge" markrgb="0.25 0.35 0.45"/>'
         '<material name="floor" texture="grid" texrepeat="12 12" reflectance="0.12"/>'
         '<material name="alu" rgba="0.78 0.80 0.83 1" specular="0.6" shininess="0.5"/>'
         '<material name="dark" rgba="0.10 0.10 0.12 1"/>'
         '<material name="finger" rgba="0.62 0.65 0.70 1" specular="0.4"/>'
         '<material name="table" rgba="0.55 0.42 0.30 1" specular="0.1"/>'
         '</asset>',
         '<default>'
         '<joint damping="0.02" frictionloss="0.01"/>'
         f'<default class="robot"><geom contype="1" conaffinity="2" friction="0.8 0.01 0.001"/></default>'
         '<default class="world"><geom contype="2" conaffinity="3"/></default>'
         '<default class="visual"><geom contype="0" conaffinity="0" group="2"/></default>'
         '</default>',
         '<worldbody>',
         '<light name="sun" pos="0.4 -0.6 1.4" dir="-0.25 0.45 -1" directional="true" diffuse="0.75 0.75 0.75" '
         'specular="0.1 0.1 0.1" castshadow="true"/>',
         f'<geom name="floor" class="world" type="plane" pos="0 0 -0.02" size="3 3 0.1" material="floor"/>',
         f'<geom name="table" class="world" type="box" pos="0 {ty / 2 - 0.2:.6g} {-th / 2:.6g}" '
         f'size="{_v((tx / 2, ty / 2, th / 2))}" material="table" friction="0.9 0.01 0.001"/>']

    # ---- arm (nested bodies)
    x.append(f'<body name="base" pos="0 0 0">'
             f'<geom class="robot" type="box" pos="0 0 {L[0].length / 2:.6g}" size="{_half((bw, bd, L[0].length))}" material="alu"/>')
    for i in range(1, 6):
        link, parent = L[i], L[i - 1]
        j = cfg.joints[i - 1]
        lw, ld = link.size
        x.append(f'<body name="{link.name}" pos="0 0 {parent.length:.6g}">{inertial(link)}'
                 f'<joint name="j{i}" type="hinge" axis="{_v(j.axis)}" range="{j.limit.min_angle:.6g} {j.limit.max_angle:.6g}" '
                 f'armature="{JOINT_ARMATURE}"/>')
        if i == 5:
            x.append(f'<geom class="robot" type="box" pos="0 0 {g.palm_length / 2:.6g}" '
                     f'size="{_half((lw, ld, g.palm_length))}" material="alu"/>')
        else:
            x.append(f'<geom class="visual" type="box" pos="0 0 {link.length - 0.018:.6g}" '
                     f'size="{_half((lw * 1.15, ld * 1.5, 0.036))}" material="dark"/>')
            x.append(f'<geom class="robot" type="box" pos="0 0 {link.length / 2:.6g}" '
                     f'size="{_half((lw, ld, link.length))}" material="alu"/>')
    # ---- gripper fingers + TCP, children of the end-effector body
    fx = g.finger_thickness / 2
    for side, sgn in (("l", 1), ("r", -1)):
        m = g.finger_mass
        ix = m / 12 * (g.finger_depth ** 2 + g.finger_length ** 2)
        iy = m / 12 * (g.finger_thickness ** 2 + g.finger_length ** 2)
        iz = m / 12 * (g.finger_thickness ** 2 + g.finger_depth ** 2)
        x.append(f'<body name="finger_{side}" pos="{sgn * fx:.6g} 0 {g.palm_length:.6g}">'
                 f'<inertial pos="0 0 {g.finger_length / 2:.6g}" mass="{m}" diaginertia="{_v((ix, iy, iz))}"/>'
                 f'<joint name="j6_finger_{side}" type="slide" axis="{sgn} 0 0" range="0 {g.finger_travel:.6g}" '
                 f'armature="0.0002" damping="0.5"/>'
                 f'<geom class="robot" type="box" pos="0 0 {g.finger_length / 2:.6g}" '
                 f'size="{_half((g.finger_thickness, g.finger_depth, g.finger_length))}" material="finger" '
                 f'friction="1.4 0.02 0.002"/></body>')
    x.append(f'<site name="tcp" pos="0 0 {cfg.tcp_offset:.6g}" size="0.006" rgba="1 0.25 0.25 0.9"/>')
    x.append("</body>" * 5 + "</body>")             # close link5(ee) ... link1 (5 links) and base

    # ---- objects (free bodies)
    for o in env.objects:
        if o.kind == "cube":
            shape = f'type="box" size="{_v([o.dims[0] / 2] * 3)}"'
        elif o.kind == "sphere":
            shape = f'type="sphere" size="{o.dims[0]:.6g}"'
        else:
            shape = f'type="cylinder" size="{o.dims[0]:.6g} {o.dims[1] / 2:.6g}"'
        x.append(f'<body name="{o.name}" pos="{_v(o.position)}"><freejoint name="{o.name}_free"/>'
                 f'<geom class="world" {shape} mass="{o.mass:.6g}" rgba="{_rgba(o.color)}" '
                 f'friction="1.0 0.005 0.001" condim="4"/></body>')
    x.append('</worldbody>')
    x.append('<contact><exclude body1="finger_l" body2="finger_r"/></contact>')
    # ---- actuators: position servos with torque caps
    x.append('<actuator>')
    for i in range(1, 6):
        t = cfg.joints[i - 1].servo.torque_limit
        x.append(f'<position name="a{i}" joint="j{i}" kp="{ARM_KP}" kv="{ARM_KV}" forcerange="{-t:.6g} {t:.6g}" '
                 f'ctrlrange="{cfg.joints[i - 1].limit.min_angle:.6g} {cfg.joints[i - 1].limit.max_angle:.6g}"/>')
    for side in ("l", "r"):
        x.append(f'<position name="a6{side}" joint="j6_finger_{side}" kp="{FINGER_KP}" kv="{FINGER_KV}" '
                 f'forcerange="{-g.max_grip_force} {g.max_grip_force}" ctrlrange="0 {g.finger_travel:.6g}"/>')
    x.append('</actuator></mujoco>')
    return "\n".join(x)


def export_mjcf(cfg: RobotConfig, path: str | Path, env: Environment | None = None) -> Path:
    path = Path(path)
    path.write_text(build_mjcf(cfg, env), encoding="utf-8")
    return path


class MuJoCoBackend(PhysicsBackend):
    supports_physics = True

    def __init__(self, config: RobotConfig, dt: float = 1.0 / 120.0):
        if not HAS_MUJOCO:
            raise RuntimeError("mujoco is not installed (pip install mujoco)")
        self.cfg = config
        self.dt = dt                                    # one control tick
        self.substeps = max(1, math.ceil(dt / SUBSTEP_MAX_S - 1e-9))
        self.lock = threading.RLock()
        self.model = None
        self.data = None
        self.objects: dict[str, SimObject] = {}

    # ------------------------------------------------------------------ lifecycle
    def connect(self, env: Environment | None = None) -> None:
        self._build(env or Environment())

    def rebuild(self, env: Environment) -> None:
        """Apply a changed object list. MuJoCo models are immutable, so compile a new one and carry over the arm
        pose and the current pose of every object that is still in the scene."""
        with self.lock:
            q5, opening = self.read_state()
            carried = {}
            for name, bid in self._body_id.items():
                j = int(self.model.body_jntadr[bid])
                qa = int(self.model.jnt_qposadr[j])
                carried[name] = self.data.qpos[qa:qa + 7].copy()
            self._build(env)
            for name, qpos in carried.items():
                if name in self._body_id:
                    j = int(self.model.body_jntadr[self._body_id[name]])
                    qa = int(self.model.jnt_qposadr[j])
                    self.data.qpos[qa:qa + 7] = qpos
            self.reset_state(q5, opening)
            for name, o in self.objects.items():                 # keep Environment in sync with where things are now
                o.position = tuple(float(v) for v in self.data.xpos[self._body_id[name]])

    def _build(self, env: Environment) -> None:
        mj = mujoco
        with self.lock:
            self.model_version = getattr(self, "model_version", 0) + 1
            self.__dict__.pop("_link_set", None)
            self.model = mj.MjModel.from_xml_string(build_mjcf(self.cfg, env, self.dt / self.substeps))
            self.data = mj.MjData(self.model)
            m = self.model
            jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
            self._arm_qadr = [int(m.jnt_qposadr[jid(f"j{i}")]) for i in range(1, 6)]
            self._fing_qadr = [int(m.jnt_qposadr[jid(f"j6_finger_{s}")]) for s in ("l", "r")]
            self._tcp_site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "tcp")
            self._body_id = {o.name: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, o.name) for o in env.objects}
            self._body_name = {i: (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "world") for i in range(m.nbody)}
            self._table_geom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "table")
            self.objects = {o.name: o for o in env.objects}
            self._finger_bodies = {mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"finger_{s}") for s in ("l", "r")}
            self._obj_bodies = set(self._body_id.values())
            self._base_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")
            self.reset_state(np.array(self.cfg.home_angles[:5]), 1.0)

    def disconnect(self) -> None:
        with self.lock:
            self.model = self.data = None

    # ------------------------------------------------------------------ objects
    def object_position(self, name: str):
        with self.lock:
            return tuple(float(v) for v in self.data.xpos[self._body_id[name]])

    def reset_object(self, name: str, position) -> None:
        with self.lock:
            m, d = self.model, self.data
            j = int(m.body_jntadr[self._body_id[name]])
            qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            d.qpos[qa:qa + 3] = position
            d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
            d.qvel[va:va + 6] = 0
            mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ control / stepping
    def reset_state(self, q5, opening: float) -> None:
        with self.lock:
            d = self.data
            for a, q in zip(self._arm_qadr, q5):
                d.qpos[a] = float(q)
            s = float(np.clip(opening, 0, 1)) * self.cfg.gripper.finger_travel
            for a in self._fing_qadr:
                d.qpos[a] = s
            d.qvel[:] = 0
            self._write_ctrl(q5, opening)
            mujoco.mj_forward(self.model, d)

    def _write_ctrl(self, q5, opening: float) -> None:
        d = self.data
        d.ctrl[:5] = np.asarray(q5, float)[:5]
        d.ctrl[5:7] = float(np.clip(opening, 0, 1)) * self.cfg.gripper.finger_travel

    def set_targets(self, q5, opening: float) -> None:
        with self.lock:
            self._write_ctrl(q5, opening)

    def step(self) -> None:
        with self.lock:
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def read_state(self):
        with self.lock:
            q = self.data.qpos[self._arm_qadr].copy()
            travel = float(np.mean(self.data.qpos[self._fing_qadr]))
        return q, travel / self.cfg.gripper.finger_travel

    def tcp_position(self) -> np.ndarray:
        with self.lock:
            return self.data.site_xpos[self._tcp_site].copy()

    def view_qpos(self) -> np.ndarray:
        """Snapshot of the generalized positions: all a renderer needs to pose its own copy of the scene."""
        with self.lock:
            return self.data.qpos.copy()

    def view_snapshot(self):
        """(model version, model, qpos) taken atomically, so a renderer notices when the scene was rebuilt."""
        with self.lock:
            return self.model_version, self.model, self.data.qpos.copy()

    def get_contacts(self) -> list:
        """Unwanted contacts: robot vs table/floor/objects, excluding base-table and finger-object (grasp)."""
        found = set()
        with self.lock:
            m, d = self.model, self.data
            for k in range(d.ncon):
                c = d.contact[k]
                if c.dist > 0:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                g1, g2 = int(c.geom1), int(c.geom2)
                for link, other, og in ((b1, b2, g2), (b2, b1, g1)):
                    if link not in self._link_bodies or other in self._link_bodies:
                        continue
                    if link == self._base_body and og == self._table_geom:
                        continue
                    if link in self._finger_bodies and other in self._obj_bodies:
                        continue
                    tag = "table" if og == self._table_geom else "floor" if other == 0 else self._body_name[other]
                    found.add(f"{self._body_name[link]}-{tag}")
        return sorted(found)

    @property
    def _link_bodies(self) -> set:
        if not hasattr(self, "_link_set"):
            self._link_set = {i for i, n in self._body_name.items()
                              if n in {l.name for l in self.cfg.links} | {"finger_l", "finger_r"}}
        return self._link_set


# ----------------------------------------------------------------------------------------------------
# Off-screen scene renderer (Qt-free). Poses its OWN MjData from ``backend.view_qpos()``, so rendering never
# touches the physics state and can never stall the control loop. ``MUJOCO_GL=cgl`` (set on macOS at import)
# gives a headless OpenGL context on Apple Silicon: no GLFW window, no mjpython.
# ----------------------------------------------------------------------------------------------------
class SceneRenderer:
    def __init__(self, backend: "MuJoCoBackend", width: int = 960, height: int = 600, shadows: bool = True):
        self.backend = backend
        self.version, self.model, _ = backend.view_snapshot()
        self.data = mujoco.MjData(self.model)
        self.size = (int(width), int(height))
        self.renderer = mujoco.Renderer(self.model, self.size[1], self.size[0])
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.shadows = shadows
        self.model.vis.scale.framelength = 0.35
        self.model.vis.scale.framewidth = 0.04
        self.reset_view()

    def reset_view(self) -> None:
        c = self.cam
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = (0.0, 0.14, 0.09)
        c.distance, c.azimuth, c.elevation = 0.95, -128.0, -24.0

    def set_frames(self, on: bool) -> None:
        self.opt.frame = mujoco.mjtFrame.mjFRAME_BODY if on else mujoco.mjtFrame.mjFRAME_NONE

    def resize(self, width: int, height: int) -> None:
        if (int(width), int(height)) == self.size:
            return
        self.renderer.close()
        self.size = (int(width), int(height))
        self.renderer = mujoco.Renderer(self.model, self.size[1], self.size[0])

    def _reload(self, version: int, model) -> None:
        """The backend compiled a new model (objects added/removed): rebuild our copy and the GL renderer."""
        self.renderer.close()
        self.version, self.model = version, model
        self.data = mujoco.MjData(model)
        self.model.vis.scale.framelength = 0.35
        self.model.vis.scale.framewidth = 0.04
        self.renderer = mujoco.Renderer(model, self.size[1], self.size[0])

    def render(self) -> np.ndarray:
        version, model, qpos = self.backend.view_snapshot()
        if version != self.version:
            self._reload(version, model)
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.shadows)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(self.shadows)
        return self.renderer.render()

    # ---- camera interaction (pixels in, camera out)
    def orbit(self, dx: float, dy: float) -> None:
        self.cam.azimuth -= dx * 0.4
        self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.4, -89.0, 5.0))

    def pan(self, dx: float, dy: float) -> None:
        az, el = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        right = np.cross(fwd, [0, 0, 1.0])
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        k = self.cam.distance * 0.0016
        self.cam.lookat[:] = np.clip(self.cam.lookat - right * dx * k + up * dy * k, -1.0, 1.0)

    def zoom(self, steps: float) -> None:
        self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** steps), 0.25, 3.0))

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass


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
# Input handling that does not depend on any GUI: key map + KeyboardController.
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
KEYBOARD (dashboard window focused; full / manual mode)
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
# TERMINAL TELEMETRY   (console runs: --no-dashboard / --headless)
# ====================================================================================================

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


# ====================================================================================================
# BIOWAVE EMG DEVICE LAYER   (from BioWaveEMG_ArmBand: rf_features, emg_v4_core, realtime_pipeline, mouse_controller)
# ====================================================================================================
# Feature extraction, calibration, signal-quality gating, decision engine, ring buffer and the ESP32-S3 wireless / USB
# protocol, copied verbatim so live inference here matches the trainer in BioWave exactly.

FEATURE_SCHEMA_VERSION = "emg-rf-15+rms-ratio+corr.v1"


FFT_MIN_HZ = 20.0


FFT_MAX_HZ = 220.0


BANDS = [(20.0, 60.0), (60.0, 120.0), (120.0, 220.0)]


def _ensure_window_shape(window):
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("window must be 2D")
    # Prefer (samples, channels). If likely transposed, flip.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] <= 0:
        raise ValueError("window must have at least 1 channel")
    return arr


def _spectral_1d(x, sample_rate):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n < 8:
        return {
            "mean_hz": 0.0,
            "median_hz": 0.0,
            "peak_hz": 0.0,
            "spec_entropy": 0.0,
            "band_power_pct": [0.0, 0.0, 0.0],
        }

    xc = x - np.mean(x)
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(xc * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))

    mask = (freqs >= FFT_MIN_HZ) & (freqs <= FFT_MAX_HZ)
    if not np.any(mask):
        return {
            "mean_hz": 0.0,
            "median_hz": 0.0,
            "peak_hz": 0.0,
            "spec_entropy": 0.0,
            "band_power_pct": [0.0, 0.0, 0.0],
        }

    sv = spec[mask]
    fv = freqs[mask]
    total = float(np.sum(sv) + 1e-9)

    peak_hz = float(fv[int(np.argmax(sv))])
    mean_hz = float(np.sum(sv * fv) / total)
    csum = np.cumsum(sv)
    med_hz = float(fv[int(np.argmax(csum >= (0.5 * total)))])

    p = sv / total
    spec_entropy = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-9))

    band_power = []
    for lo, hi in BANDS:
        bmask = (fv >= lo) & (fv < hi)
        if np.any(bmask):
            band_power.append(float(np.sum(sv[bmask]) / total * 100.0))
        else:
            band_power.append(0.0)

    return {
        "mean_hz": mean_hz,
        "median_hz": med_hz,
        "peak_hz": peak_hz,
        "spec_entropy": spec_entropy,
        "band_power_pct": band_power,
    }


def extract_window_features_legacy(window, sample_rate=500):
    """Canonical legacy RF implementation used by deployed joblib artifacts.

    Keep its ordering and numerical operations stable. Optimized implementations
    are deliberately opt-in and tested against this reference.
    """
    arr = _ensure_window_shape(window)
    n_samples = arr.shape[0]
    n_ch = arr.shape[1]
    arr_centered = arr - np.mean(arr, axis=0, keepdims=True)

    zc_thresh = 10.0
    ssc_thresh = 8.0
    wamp_thresh = 12.0

    feats = []
    rms_vals = []
    for ch in range(n_ch):
        x = arr_centered[:, ch]
        abs_x = np.abs(x)
        dx = np.diff(x) if n_samples > 1 else np.array([], dtype=np.float32)

        mav = float(np.mean(abs_x))
        rms = float(np.sqrt(np.mean(np.square(x))))
        iemg = float(np.sum(abs_x))
        var = float(np.var(x))
        wl = float(np.sum(np.abs(dx))) if dx.size else 0.0

        if n_samples > 1:
            zc = int(np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) >= zc_thresh)))
            wamp = int(np.sum(np.abs(x[1:] - x[:-1]) >= wamp_thresh))
        else:
            zc = 0
            wamp = 0

        if n_samples > 2:
            s1 = x[1:-1] - x[:-2]
            s2 = x[1:-1] - x[2:]
            ssc = int(np.sum(((s1 * s2) > 0) & ((np.abs(s1) + np.abs(s2)) >= ssc_thresh)))
        else:
            ssc = 0

        sp = _spectral_1d(x, sample_rate)
        feats.extend(
            [
                mav,
                rms,
                iemg,
                var,
                wl,
                float(zc),
                float(ssc),
                float(wamp),
                sp["mean_hz"],
                sp["median_hz"],
                sp["peak_hz"],
                sp["spec_entropy"],
                sp["band_power_pct"][0],
                sp["band_power_pct"][1],
                sp["band_power_pct"][2],
            ]
        )
        rms_vals.append(rms)

    rms_vals = np.asarray(rms_vals, dtype=np.float32)
    mean_rms = float(np.mean(rms_vals) + 1e-9)
    feats.extend((rms_vals / mean_rms).tolist())

    # Pairwise channel correlation features.
    std = np.std(arr_centered, axis=0)
    valid = np.isfinite(std) & (std > 1e-8)
    if np.any(valid):
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(arr_centered.T)
    else:
        corr = np.eye(n_ch, dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(valid):
        corr[~valid, :] = 0.0
        corr[:, ~valid] = 0.0
        np.fill_diagonal(corr, 1.0)
    for a in range(n_ch):
        for b in range(a + 1, n_ch):
            feats.append(float(corr[a, b]))

    return np.asarray(feats, dtype=np.float32)


PREPROCESSING_VERSION = "v2"


LEGACY_PREPROCESSING_VERSION = "legacy-baseline-v1"


FEATURE_EXTRACTOR_VERSION = "rf_features.v1"


@dataclass
class SampleBatch:
    """Samples plus transport metadata. Values are never silently discarded."""
    samples: np.ndarray
    packet_sequences: Optional[np.ndarray] = None
    frame_ids: Optional[np.ndarray] = None
    emg_timestamps: Optional[np.ndarray] = None
    imu_ids: Optional[np.ndarray] = None
    imu_timestamps: Optional[np.ndarray] = None
    host_received_monotonic: float = field(default_factory=time.monotonic)
    gap_before: int = 0
    invalid: bool = False


@dataclass
class WirelessStats:
    packets_received: int = 0
    packets_missing: int = 0
    packets_out_of_order: int = 0
    invalid_packets: int = 0
    previous_sequence: Optional[int] = None
    last_arrival: Optional[float] = None
    arrival_intervals: deque = field(default_factory=lambda: deque(maxlen=200))

    def observe(self, sequence: int, arrival: Optional[float] = None) -> int:
        """Record one UDP packet and return the number missing before it.

        UDP sequence numbers are unsigned 32-bit; wrap-around is handled.  A
        duplicate/out-of-order packet is retained in diagnostics but never used
        to make a gap look like valid continuous data.
        """
        arrival = time.monotonic() if arrival is None else arrival
        self.packets_received += 1
        gap = 0
        if self.previous_sequence is not None:
            delta = (int(sequence) - self.previous_sequence) & 0xFFFFFFFF
            if delta == 0 or delta > 0x7FFFFFFF:
                self.packets_out_of_order += 1
            elif delta > 1:
                gap = delta - 1
                self.packets_missing += gap
        if self.last_arrival is not None:
            self.arrival_intervals.append(arrival - self.last_arrival)
        self.previous_sequence = int(sequence)
        self.last_arrival = arrival
        return gap

    @property
    def packet_loss_percent(self) -> float:
        total = self.packets_received + self.packets_missing
        return 100.0 * self.packets_missing / total if total else 0.0

    @property
    def jitter_ms(self) -> float:
        return float(np.std(self.arrival_intervals) * 1000.0) if len(self.arrival_intervals) > 1 else 0.0


@dataclass
class CalibrationProfile:
    rest_baseline: np.ndarray
    rest_std: np.ndarray
    rest_rms: np.ndarray
    flex_rms: np.ndarray
    flex_peak: np.ndarray
    activation_scale: np.ndarray
    quality: list[str]
    valid: bool
    reasons: list[str]
    normalization: str = "activation_rms"


def compute_calibration(rest: np.ndarray, flex: np.ndarray, *, min_samples: int = 250,
                        dead_std: float = 1e-3, rest_noise_limit: float = 80.0,
                        min_activation_ratio: float = 1.25,
                        saturation_limit: Optional[float] = 65534.0) -> CalibrationProfile:
    """Quantify calibration, rejecting unsafe captures rather than guessing."""
    rest = np.asarray(rest, dtype=np.float32)
    flex = np.asarray(flex, dtype=np.float32)
    if rest.ndim != 2 or flex.ndim != 2 or rest.shape[1] != flex.shape[1]:
        raise ValueError("REST and FLEX must be 2-D with identical channel count")
    channels = rest.shape[1]
    if rest.shape[0] < min_samples or flex.shape[0] < min_samples:
        zeros = np.zeros(channels, dtype=np.float32)
        return CalibrationProfile(zeros, zeros, zeros, zeros, zeros, np.ones(channels),
                                  ["INSUFFICIENT"] * channels, False,
                                  ["Insufficient REST or FLEX samples"])
    baseline = np.median(rest, axis=0)
    rest_centered = rest - baseline
    flex_centered = flex - baseline
    rest_std = np.std(rest_centered, axis=0)
    rest_rms = np.sqrt(np.mean(rest_centered ** 2, axis=0))
    flex_rms = np.sqrt(np.mean(flex_centered ** 2, axis=0))
    flex_peak = np.max(np.abs(flex_centered), axis=0)
    # RMS activation is robust against one transient; epsilon prevents divide-by-zero.
    scale = np.maximum(flex_rms, 1e-6)
    quality, reasons = [], []
    finite = np.isfinite(rest).all(axis=0) & np.isfinite(flex).all(axis=0)
    for ch in range(channels):
        if not finite[ch]:
            quality.append("INVALID"); reasons.append(f"CH{ch + 1}: NaN/Inf")
        elif saturation_limit is not None and ((np.abs(rest[:, ch]) >= saturation_limit).any() or (np.abs(flex[:, ch]) >= saturation_limit).any()):
            quality.append("SATURATED"); reasons.append(f"CH{ch + 1}: saturated")
        elif rest_std[ch] <= dead_std and flex_rms[ch] <= dead_std:
            quality.append("DEAD"); reasons.append(f"CH{ch + 1}: no measurable signal")
        elif rest_std[ch] > rest_noise_limit:
            quality.append("NOISY"); reasons.append(f"CH{ch + 1}: unstable REST")
        elif flex_rms[ch] < max(rest_rms[ch] * min_activation_ratio, dead_std * 2):
            quality.append("WEAK"); reasons.append(f"CH{ch + 1}: insufficient FLEX activation")
        else:
            quality.append("GOOD")
    valid = all(q == "GOOD" for q in quality)
    return CalibrationProfile(baseline.astype(np.float32), rest_std.astype(np.float32), rest_rms.astype(np.float32),
                              flex_rms.astype(np.float32), flex_peak.astype(np.float32), scale.astype(np.float32),
                              quality, valid, reasons)


@dataclass
class PreprocessingConfig:
    version: str = PREPROCESSING_VERSION
    sample_rate: float = 500.0
    highpass_hz: float = 20.0
    lowpass_hz: float = 220.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    normalization: str = "activation_rms"  # "none" retains centered ADC scale.


class RealTimePreprocessor:
    """Causal, stateful band-pass + notch processor.

    Cascaded one-pole high/low pass filters and a normalized RBJ notch are
    stable for the configured 500 Hz stream.  States persist across batches,
    avoiding the non-causal look-ahead and edge artifacts of ``filtfilt``.
    """
    def __init__(self, channels: int, config: PreprocessingConfig, profile: CalibrationProfile):
        self.channels, self.config, self.profile = int(channels), config, profile
        if not 0 < config.highpass_hz < config.lowpass_hz < config.sample_rate / 2:
            raise ValueError("invalid causal band-pass configuration")
        self._hp_x = np.zeros(channels, np.float64); self._hp_y = np.zeros(channels, np.float64)
        self._lp_y = np.zeros(channels, np.float64)
        self._z1 = np.zeros(channels, np.float64); self._z2 = np.zeros(channels, np.float64)
        dt = 1.0 / config.sample_rate
        self._hp_a = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * config.highpass_hz * dt))
        self._lp_a = dt / ((1.0 / (2.0 * math.pi * config.lowpass_hz)) + dt)
        w0 = 2.0 * math.pi * config.notch_hz / config.sample_rate
        alpha = math.sin(w0) / (2.0 * config.notch_q)
        b0, b1, b2, a0, a1, a2 = 1, -2 * math.cos(w0), 1, 1 + alpha, -2 * math.cos(w0), 1 - alpha
        self._b0, self._b1, self._b2 = b0 / a0, b1 / a0, b2 / a0
        self._a1, self._a2 = a1 / a0, a2 / a0

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.channels or not np.isfinite(x).all():
            raise ValueError("invalid sample batch for preprocessing")
        out = np.empty_like(x)
        baseline = self.profile.rest_baseline
        scale = self.profile.activation_scale
        for i, row in enumerate(x):
            centered = row.astype(np.float64) - baseline
            hp = self._hp_a * (self._hp_y + centered - self._hp_x)
            self._hp_x, self._hp_y = centered, hp
            lp = self._lp_y + self._lp_a * (hp - self._lp_y)
            self._lp_y = lp
            y = self._b0 * lp + self._z1
            self._z1 = self._b1 * lp - self._a1 * y + self._z2
            self._z2 = self._b2 * lp - self._a2 * y
            out[i] = y / scale if self.config.normalization != "none" else y
        return out


@dataclass
class SignalQuality:
    state: str
    channel_states: list[str]
    reason: str = ""


def assess_signal_quality(window: np.ndarray, profile: CalibrationProfile, *, saturation_limit: float = 65534.0,
                          max_noise_multiplier: float = 8.0, min_active_std: float = 1e-4) -> SignalQuality:
    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(profile.quality) or not np.isfinite(x).all():
        return SignalQuality("SIGNAL_POOR", ["INVALID"] * len(profile.quality), "missing or non-finite samples")
    states = []
    for ch in range(x.shape[1]):
        std, peak = float(np.std(x[:, ch])), float(np.max(np.abs(x[:, ch])))
        if peak >= saturation_limit: states.append("SATURATED")
        elif std <= min_active_std: states.append("DEAD")
        elif profile.rest_std[ch] > 0 and std > max_noise_multiplier * max(profile.flex_rms[ch], profile.rest_std[ch]): states.append("NOISY")
        else: states.append(profile.quality[ch] if profile.quality[ch] != "GOOD" else "GOOD")
    bad = [s for s in states if s != "GOOD"]
    return SignalQuality("GOOD" if not bad else "SIGNAL_POOR", states, ", ".join(bad))


@dataclass
class ModelCompatibility:
    compatible: bool
    warnings: list[str]
    errors: list[str]
    preprocessing_version: str


def expected_feature_count(channels: int) -> int:
    return 15 * channels + channels + channels * (channels - 1) // 2


def validate_model_artifact(artifact: Any, acquisition_rate: Optional[float] = None,
                            acquisition_channels: Optional[int] = None) -> ModelCompatibility:
    warnings, errors = [], []
    if not isinstance(artifact, dict) or "model" not in artifact:
        return ModelCompatibility(False, warnings, ["Joblib artifact must contain a model"], "unknown")
    model = artifact["model"]
    if not hasattr(model, "predict"):
        errors.append("Model has no predict()")
    classes = artifact.get("class_names", artifact.get("classes", getattr(model, "classes_", [])))
    if not list(classes): errors.append("Model classes are missing")
    version = artifact.get("preprocessing_version", LEGACY_PREPROCESSING_VERSION)
    if "preprocessing_version" not in artifact: warnings.append("Model preprocessing metadata is missing; using legacy baseline compatibility mode.")
    if "input_channels" not in artifact: warnings.append("Model input channel metadata is missing; feature count will be used when available.")
    rate = artifact.get("sample_rate")
    if rate is None: warnings.append("Model sample-rate metadata is missing; compatibility cannot be fully verified.")
    elif acquisition_rate is not None and not math.isclose(float(rate), float(acquisition_rate), rel_tol=0, abs_tol=0.01):
        errors.append(f"Model sample rate ({rate} Hz) differs from acquisition ({acquisition_rate} Hz); resampling is not enabled.")
    channels = artifact.get("input_channels")
    if channels is not None and acquisition_channels is not None and int(channels) != int(acquisition_channels):
        # "input_channels" is the total column count the model was trained on
        # (EMG plus any IMU columns logged alongside them), not an EMG-only count -
        # callers must pass the acquisition's matching total, not its EMG-only count.
        errors.append(f"Model expects {channels} input channel(s) (as trained), device supplies {acquisition_channels}.")
    count = getattr(model, "n_features_in_", None)
    if count is not None and channels is not None and int(count) != expected_feature_count(int(channels)):
        errors.append(f"Model has {count} features; expected {expected_feature_count(int(channels))} for declared channel order.")
    return ModelCompatibility(not errors, warnings, errors, version)


@dataclass
class Decision:
    state: str
    action_allowed: bool
    click_triggered: bool = False


class GestureDecisionEngine:
    """Temporal gate: actions require sustained, confident, quality-approved labels."""
    def __init__(self, *, vote_windows: int = 5, consecutive_required: int = 3,
                 min_confidence: float = .65, min_margin: float = .10, refractory_s: float = .8):
        self.history = deque(maxlen=vote_windows); self.consecutive_required = consecutive_required
        self.min_confidence, self.min_margin, self.refractory_s = min_confidence, min_margin, refractory_s
        self.active: Optional[str] = None; self.candidate: Optional[str] = None; self.candidate_count = 0
        self.last_click = -float("inf")

    def update(self, label: Optional[str], confidence: Optional[float], margin: Optional[float], quality_good: bool,
               now: Optional[float] = None) -> Decision:
        now = time.monotonic() if now is None else now
        valid = quality_good and label not in (None, "", "REST", "UNKNOWN") and confidence is not None and confidence >= self.min_confidence and (margin is None or margin >= self.min_margin)
        if not valid:
            self.history.clear(); self.candidate = None; self.candidate_count = 0; self.active = None
            return Decision("REST" if label == "REST" and quality_good else "UNKNOWN", False)
        self.history.append(label)
        voted = Counter(self.history).most_common(1)[0][0]
        if voted != self.candidate:
            self.candidate, self.candidate_count = voted, 1
        else: self.candidate_count += 1
        if self.candidate_count < self.consecutive_required:
            return Decision("GESTURE_CANDIDATE", False)
        if self.active != voted:
            self.active = voted
            return Decision(f"{voted}_ACTIVE", True)
        return Decision(f"{voted}_HELD", True)

    def trigger_click_once(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self.last_click < self.refractory_s: return False
        self.last_click = now
        return True


class SampleRingBuffer:
    """Fixed-size, single-producer ring buffer with explicit gap recovery.

    The caller obtains one chronological copy only when an inference/analysis
    window is actually required; ingest never shifts the complete history.
    """
    def __init__(self, channels: int, capacity: int, dtype=np.float32) -> None:
        if channels <= 0 or capacity <= 0:
            raise ValueError("channels and capacity must be positive")
        self.channels, self.capacity = int(channels), int(capacity)
        self.data = np.empty((self.capacity, self.channels), dtype=dtype)
        self.write_index = 0
        self.sample_count = 0
        self.generation = 0
        self.invalid_until_generation = 0

    def reset(self) -> None:
        self.write_index = self.sample_count = self.generation = self.invalid_until_generation = 0

    def append(self, samples: np.ndarray, *, discontinuity: bool = False) -> int:
        x = np.asarray(samples, dtype=self.data.dtype)
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError("sample shape does not match ring channels")
        n = len(x)
        if not n: return 0
        if discontinuity:
            # Never synthesize data; require a clean subsequent window.
            self.sample_count = 0
            self.invalid_until_generation = self.generation + self.capacity
        if n >= self.capacity:
            self.data[:, :] = x[-self.capacity:]
            self.write_index = 0
            self.sample_count = self.capacity
        else:
            first = min(n, self.capacity - self.write_index)
            self.data[self.write_index:self.write_index + first] = x[:first]
            remaining = n - first
            if remaining: self.data[:remaining] = x[first:]
            self.write_index = (self.write_index + n) % self.capacity
            self.sample_count = min(self.capacity, self.sample_count + n)
        self.generation += n
        return n

    def has_window(self, size: int) -> bool:
        return 0 < size <= self.capacity and self.sample_count >= size and self.generation >= self.invalid_until_generation

    def latest(self, size: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.has_window(size):
            raise ValueError("no continuous window available")
        if out is None:
            out = np.empty((size, self.channels), dtype=self.data.dtype)
        if out.shape != (size, self.channels):
            raise ValueError("output window shape mismatch")
        start = (self.write_index - size) % self.capacity
        first = min(size, self.capacity - start)
        out[:first] = self.data[start:start + first]
        if first < size: out[first:] = self.data[:size - first]
        return out


@dataclass
class LatencySnapshot:
    count: int = 0
    min_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


class StageProfiler:
    """Bounded high-resolution stage timings; no I/O on the hot path."""
    def __init__(self, history: int = 2048) -> None:
        self._samples: dict[str, deque[int]] = {}
        self._history = history

    @staticmethod
    def now_ns() -> int: return time.perf_counter_ns()

    def record_ns(self, stage: str, start_ns: int, end_ns: Optional[int] = None) -> None:
        elapsed = (self.now_ns() if end_ns is None else end_ns) - start_ns
        self._samples.setdefault(stage, deque(maxlen=self._history)).append(max(0, elapsed))

    def snapshot(self) -> dict[str, LatencySnapshot]:
        answer = {}
        for stage, values in self._samples.items():
            a = np.asarray(values, dtype=np.float64) / 1_000_000.0
            answer[stage] = LatencySnapshot(len(a), float(a.min()), float(a.mean()), float(np.percentile(a, 50)),
                                            float(np.percentile(a, 95)), float(np.percentile(a, 99)), float(a.max()))
        return answer



DEFAULT_BAUD_RATE = 921600         # Wired EMG stream baud rate.
USB_SERIAL_BAUD = 115200           # Baud rate used only for USB provisioning handshakes.
SERIAL_BOOT_WAIT_S = 3.5
SERIAL_RESPONSE_TIMEOUT_S = 15.0
SAMPLE_RATE = 500                  # Hz, matches the ESP32 firmware.
WINDOW_SIZE = 1000                 # Rolling sample buffer length (channels x samples).
DEFAULT_WIRED_CHANNELS = 4

DEFAULT_DEVICE_ACCESS_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"
WIFI_STREAM_PORT = 5000
WIFI_CONTROL_PORT = 5001
DISCOVERY_ADDRESS = "255.255.255.255"
DISCOVERY_TIMEOUT = 1.2
CONTROL_TIMEOUT = 2.0
KEEPALIVE_INTERVAL_MS = 2000       # Matches main app: periodic PING keeps the ESP32 stream alive.
KEEPALIVE_MAX_FAILURES = 3         # Consecutive failed pings before we flag the link as lost.

WIRELESS_EMG_CHANNELS = 8
WIRELESS_IMU_CHANNELS = 3
WIRELESS_TOTAL_CHANNELS = WIRELESS_EMG_CHANNELS + WIRELESS_IMU_CHANNELS
WIFI_PACKET_HEADER_FORMAT = "<4sBBHI"
WIFI_PACKET_HEADER_SIZE = struct.calcsize(WIFI_PACKET_HEADER_FORMAT)
WIRELESS_FRAME_FORMAT = "<IIII8HfffB3x"
WIRELESS_FRAME_SIZE = struct.calcsize(WIRELESS_FRAME_FORMAT)
WIRELESS_FRAMES_PER_PACKET = 5
WIRELESS_PACKET_SIZE = WIFI_PACKET_HEADER_SIZE + (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET)

CAL_TICK_MS = 100
CAL_REST_MS = 3000
CAL_FLEX_MS = 3000
CAL_DURATION_MIN_S = 3
CAL_DURATION_MAX_S = 10
BASE_ADAPT_ALPHA = 0.001           # Slow baseline drift compensation.
BASE_ADAPT_GUARD = 80.0            # Only adapt baseline while signal is near rest.
RF_LABEL_SMOOTH_WINDOW = 5         # Majority-vote smoothing window for displayed label.
GESTURE_MIN_CONFIDENCE_DEFAULT = 65.0
MAX_INFERENCE_PACKET_GAP = 1        # A larger UDP loss invalidates a window; never classify it.

LOG = logging.getLogger("biowave.emg")
extract_window_features = extract_window_features_legacy      # the ONE feature extractor (matches the trainer)
HAS_RF_FEATURES = True
PLOT_SAMPLES = 2500                 # 5 s of live EMG at 500 Hz
PLOT_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]
STREAM_STALL_S = 1.5                # no samples for this long -> arm control is switched off
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "BioWaveEMG_ArmBand" / "trained_model"


# ---- wireless protocol
def sign_message(secret, *parts):
    message = "|".join(str(part) for part in parts)
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def get_local_ip_for_target(target_ip):
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((target_ip, 1))
        return probe.getsockname()[0]
    finally:
        probe.close()


@dataclass
class DeviceInfo:
    ip: str
    device_id: str
    device_name: str
    wifi_mode: str
    reported_ip: str
    imu_ready: bool
    streaming: bool
    firmware: str

    @property
    def summary(self):
        return f"{self.device_name} @ {self.ip} ({self.wifi_mode})"


@dataclass
class SerialDeviceInfo:
    port_name: str
    device_id: str
    device_name: str
    imu_ready: bool
    wifi_saved: bool
    firmware: str


class ControlProtocol:
    """UDP control-plane protocol for discovering and driving the wireless
    BioWave EMG device (mirrors the ESP32 firmware's HELLO/CHALLENGE/START/STOP)."""

    @staticmethod
    def parse_device_info(message, source_ip):
        parts = message.strip().split("|")
        if len(parts) != 8 or parts[0] != "HELLO":
            raise ValueError("Unexpected device response.")
        return DeviceInfo(
            ip=source_ip,
            device_id=parts[1],
            device_name=parts[2],
            wifi_mode=parts[3],
            reported_ip=parts[4],
            imu_ready=parts[5] == "1",
            streaming=parts[6] == "1",
            firmware=parts[7],
        )

    @staticmethod
    def send_and_receive(message, target_ip, expect_multiple=False, timeout=CONTROL_TIMEOUT, broadcast=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2 if expect_multiple else timeout)
        if broadcast:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        responses = []
        deadline = time.monotonic() + timeout
        try:
            sock.sendto(message.encode("utf-8"), (target_ip, WIFI_CONTROL_PORT))
            if expect_multiple:
                while time.monotonic() < deadline:
                    try:
                        data, addr = sock.recvfrom(2048)
                        responses.append((data.decode("utf-8", errors="replace"), addr[0]))
                    except socket.timeout:
                        continue
                return responses

            data, addr = sock.recvfrom(2048)
            return data.decode("utf-8", errors="replace"), addr[0]
        finally:
            sock.close()

    @staticmethod
    def discover():
        devices = {}
        responses = ControlProtocol.send_and_receive(
            "DISCOVER", DISCOVERY_ADDRESS, expect_multiple=True,
            timeout=DISCOVERY_TIMEOUT, broadcast=True,
        )
        for response, source_ip in responses:
            try:
                device = ControlProtocol.parse_device_info(response, source_ip)
                devices[device.ip] = device
            except ValueError:
                continue
        return list(devices.values())

    @staticmethod
    def get_challenge(target_ip):
        response, _ = ControlProtocol.send_and_receive("CHALLENGE", target_ip)
        parts = response.strip().split("|")
        if len(parts) != 2 or parts[0] != "CHALLENGE":
            raise RuntimeError("Device did not return a valid challenge.")
        return parts[1]

    @staticmethod
    def authenticated_command(target_ip, secret, command, *payload):
        if not secret:
            raise RuntimeError("Device access key is required.")
        challenge = ControlProtocol.get_challenge(target_ip)
        auth = sign_message(secret, command, challenge, *payload)
        message = "|".join([command, challenge, *payload, auth])
        response, _ = ControlProtocol.send_and_receive(message, target_ip)

        parts = response.strip().split("|")
        if not parts:
            raise RuntimeError("Device returned an empty response.")
        if parts[0] == "ERR":
            detail = parts[1] if len(parts) > 1 else "UNKNOWN"
            raise RuntimeError(f"Device rejected command: {detail}")
        if parts[0] != "ACK":
            raise RuntimeError("Unexpected device acknowledgement.")
        return parts[1:]

    @staticmethod
    def start_stream(target_ip, secret, client_ip, client_port):
        return ControlProtocol.authenticated_command(target_ip, secret, "START", client_ip, str(client_port))

    @staticmethod
    def stop_stream(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "STOP")

    @staticmethod
    def ping(target_ip, secret):
        return ControlProtocol.authenticated_command(target_ip, secret, "PING")


class WiFiSerialProvisionProtocol:
    """One-time USB handshake used to hand Wi-Fi credentials to a fresh ESP32."""

    @staticmethod
    def available_ports():
        return list(serial.tools.list_ports.comports())

    @staticmethod
    def _exchange_line(port_name, command, expected_prefixes, timeout=SERIAL_RESPONSE_TIMEOUT_S):
        try:
            with serial.Serial(port_name, USB_SERIAL_BAUD, timeout=0.3, write_timeout=1) as ser:
                ser.setDTR(False)
                ser.setRTS(False)
                time.sleep(0.15)
                time.sleep(SERIAL_BOOT_WAIT_S)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write((command + "\n").encode("utf-8"))
                ser.flush()

                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    raw_line = ser.readline()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if any(line.startswith(prefix) for prefix in expected_prefixes):
                        return line
        except serial.SerialException as exc:
            raise RuntimeError(f"Serial communication failed on {port_name}: {exc}") from exc
        raise RuntimeError("The ESP32 did not return a serial response in time.")

    @staticmethod
    def query_info(port_name):
        response = WiFiSerialProvisionProtocol._exchange_line(port_name, "INFO", expected_prefixes=("INFO|", "ERR|"))
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 returned an error: {parts[1]}")
        if len(parts) != 6 or parts[0] != "INFO":
            raise RuntimeError(f"Unexpected serial response: {response}")
        return SerialDeviceInfo(
            port_name=port_name, device_id=parts[1], device_name=parts[2],
            imu_ready=parts[3] == "1", wifi_saved=parts[4] == "1", firmware=parts[5],
        )

    @staticmethod
    def provision(port_name, ssid, password):
        encoded_ssid = quote(ssid, safe="")
        encoded_password = quote(password, safe="")
        response = WiFiSerialProvisionProtocol._exchange_line(
            port_name, f"PROVISION|{encoded_ssid}|{encoded_password}",
            expected_prefixes=("ACK|", "ERR|"), timeout=8.0,
        )
        parts = response.split("|")
        if len(parts) >= 2 and parts[0] == "ACK" and parts[1] == "PROVISIONED":
            return
        if len(parts) >= 2 and parts[0] == "ERR":
            raise RuntimeError(f"ESP32 rejected provisioning: {parts[1]}")
        raise RuntimeError(f"Unexpected serial response: {response}")


# ====================================================================================================
# THEME   (from app_theme.py: same palette / stylesheet as the BioWave apps)
# ====================================================================================================

THEME_COLORS = {
    "special": "#BF092F",
    "bg": "#132440",
    "title_bar": "#0D1B33",
    "panel": "#16476A",
    "accent": "#3B9797",
    "success": "#3B9797",
    "graph_bg": "#132440",
    "text": "#E8EEF0",
    "muted": "#A9C2CF",
    "disabled": "#6F8A99",
}


def apply_dark_title_bar(window):
    """Request a dark native title bar on Windows where supported."""
    try:
        import ctypes
        import sys
        from ctypes import wintypes
    except Exception:
        return False

    if sys.platform != "win32":
        return False

    try:
        hwnd = int(window.winId())
    except Exception:
        return False

    def _hex_to_colorref(hex_color):
        val = (hex_color or "").strip().lstrip("#")
        if len(val) != 6:
            return None
        try:
            r = int(val[0:2], 16)
            g = int(val[2:4], 16)
            b = int(val[4:6], 16)
        except ValueError:
            return None
        return (b << 16) | (g << 8) | r

    use_dark = ctypes.c_int(1)
    use_dark_size = ctypes.sizeof(use_dark)
    attrs = (20, 19)  # Win10 20H1+, then legacy fallback
    enabled_dark = False
    for attr in attrs:
        try:
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(attr),
                ctypes.byref(use_dark),
                wintypes.DWORD(use_dark_size),
            )
            if result == 0:
                enabled_dark = True
                break
        except Exception:
            continue

    caption_color = _hex_to_colorref(THEME_COLORS.get("title_bar", THEME_COLORS["bg"]))
    text_color = _hex_to_colorref(THEME_COLORS["text"])
    if caption_color is not None:
        try:
            caption_val = ctypes.c_int(caption_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(35),  # DWMWA_CAPTION_COLOR
                ctypes.byref(caption_val),
                wintypes.DWORD(ctypes.sizeof(caption_val)),
            )
        except Exception:
            pass
    if text_color is not None:
        try:
            text_val = ctypes.c_int(text_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(36),  # DWMWA_TEXT_COLOR
                ctypes.byref(text_val),
                wintypes.DWORD(ctypes.sizeof(text_val)),
            )
        except Exception:
            pass
    return enabled_dark


def app_stylesheet(font_size=16):
    c = THEME_COLORS
    default_font = "Bahnschrift" if os.name == "nt" else "Sans Serif"
    return f"""
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: "{default_font}";
    font-size: {int(font_size)}px;
}}
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
QLabel {{
    color: {c['text']};
}}
QPushButton {{
    background-color: {c['accent']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {c['panel']};
    color: {c['text']};
}}
QPushButton:pressed {{
    background-color: {c['special']};
    border-color: {c['special']};
}}
QPushButton:disabled {{
    background-color: {c['panel']};
    color: {c['disabled']};
    border-color: {c['accent']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: {c['accent']};
}}
QComboBox QAbstractItemView {{
    background-color: {c['panel']};
    color: {c['text']};
    selection-background-color: {c['accent']};
    border: 1px solid {c['accent']};
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QTabWidget::pane {{
    border: 1px solid {c['accent']};
    background: {c['panel']};
}}
QTabBar::tab {{
    background: {c['panel']};
    color: {c['muted']};
    border: 1px solid {c['accent']};
    padding: 6px 12px;
}}
QTabBar::tab:selected {{
    background: {c['accent']};
    color: {c['text']};
}}
QHeaderView::section {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    padding: 4px;
}}
QTableWidget {{
    background-color: {c['panel']};
    color: {c['text']};
    gridline-color: {c['accent']};
    border: 1px solid {c['accent']};
}}
QProgressBar {{
    border: 1px solid {c['accent']};
    border-radius: 5px;
    text-align: center;
    background-color: {c['panel']};
    color: {c['text']};
}}
QProgressBar::chunk {{
    background-color: {c['success']};
}}
QScrollArea {{
    border: 1px solid {c['accent']};
    background-color: {c['panel']};
}}
QScrollBar:vertical {{
    border: none;
    background: {c['bg']};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c['accent']};
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {c['bg']};
}}
QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
    background: transparent;
    width: 0px;
    height: 0px;
}}
QScrollBar:horizontal {{
    border: none;
    background: {c['bg']};
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c['accent']};
    min-width: 24px;
    border-radius: 5px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: {c['bg']};
}}
QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{
    background: transparent;
    width: 0px;
    height: 0px;
}}
"""


def configure_high_dpi():
    """Call once, before constructing QApplication.

    Without this, PyQt5 can report window/widget sizes in physical pixels on
    a Retina/HiDPI display (every MacBook since ~2012, plus most Windows
    laptops today) instead of logical points, which is what makes a window
    sized for a "normal" screen come out oversized and clip its own buttons.
    """
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


# ====================================================================================================
# EMG -> ARM BRIDGE   (Qt-free, unit-tested: tests/test_emg_arm_bridge.py)
# ====================================================================================================
# Model class name  --(user mapping)-->  arm action  --(decision engine gate)-->  MotionCommand.
#
# The BioWave decision engine (confidence + margin + debounce + signal-quality gate) already decides whether a
# gesture may act, so the bridge does not smooth a second time: an allowed gesture is sent as a continuous jog
# every inference tick (~20 Hz), and the planner's command timeout stops the arm by itself if predictions stop.

ARM_ACTIONS = {
    "Ignore (hold)":        None,
    "Move Left  (-X)":      ("CARTESIAN", "LEFT"),
    "Move Right (+X)":      ("CARTESIAN", "RIGHT"),
    "Move Up    (+Z)":      ("CARTESIAN", "UP"),
    "Move Down  (-Z)":      ("CARTESIAN", "DOWN"),
    "Move Forward (+Y)":    ("CARTESIAN", "FORWARD"),
    "Move Backward (-Y)":   ("CARTESIAN", "BACKWARD"),
    "Close Gripper":        ("GRIPPER", "CLOSE"),
    "Open Gripper":         ("GRIPPER", "OPEN"),
}
IGNORE_ACTION = "Ignore (hold)"
REST_LABELS = {"rest", "idle", "none", "neutral", "relax", "relaxed", "hold"}


def is_rest_label(label) -> bool:
    return str(label).strip().lower().replace("-", "_") in REST_LABELS


def default_arm_action(class_name: str) -> str:
    """Sensible first guess for a trained class name (the user can change it in the mapping panel)."""
    c = str(class_name).strip().lower().replace("-", "_").replace(" ", "_")
    if is_rest_label(c):
        return IGNORE_ACTION
    if "open" in c:
        return "Open Gripper"
    if "fist" in c or "close" in c or "click" in c or "grip" in c or "grab" in c:
        return "Close Gripper"
    for key, action in (("left", "Move Left  (-X)"), ("right", "Move Right (+X)"), ("up", "Move Up    (+Z)"),
                        ("down", "Move Down  (-Z)"), ("forward", "Move Forward (+Y)"),
                        ("back", "Move Backward (-Y)")):
        if key in c:
            return action
    return IGNORE_ACTION


class EMGArmBridge:
    def __init__(self, sink: Callable[[MotionCommand], bool], status_callback=None,
                 speed_m_s: float = EMG_SPEED_M_S, engine: "GestureDecisionEngine | None" = None):
        self.sink = sink
        self.status_callback = status_callback
        self.speed = speed_m_s
        self.engine = engine or GestureDecisionEngine()
        self.action_map: dict[str, str] = {}
        self.enabled = False
        self.last_action = IGNORE_ACTION
        self.last_decision = "UNKNOWN"
        self._moving = False

    def set_mapping(self, class_names) -> None:
        self.action_map = {str(c): default_arm_action(c) for c in class_names}

    def set_action(self, class_name: str, action: str) -> None:
        if action not in ARM_ACTIONS:
            raise ValueError(f"unknown arm action {action!r}")
        self.action_map[str(class_name)] = action

    def enable(self, on: bool) -> None:
        if not on:
            self.release("control disabled")
        self.enabled = bool(on)

    def _stop_motion(self) -> None:
        if self._moving:
            self.sink(MotionCommand.hold(source="emg"))
        self._moving = False
        self.last_action = IGNORE_ACTION

    def release(self, reason: str = "") -> None:
        """Fail-safe: stop any jog we started AND forget the gesture history. Safe to call repeatedly."""
        self._stop_motion()
        self.engine.history.clear()
        self.engine.candidate, self.engine.candidate_count, self.engine.active = None, 0, None

    def on_prediction(self, label, confidence, margin, quality_good: bool) -> str:
        """Feed one classifier output. Returns the arm action that was executed (or IGNORE_ACTION)."""
        label = str(label)
        engine_label = "REST" if is_rest_label(label) else label
        decision = self.engine.update(engine_label, confidence, margin, quality_good)
        self.last_decision = decision.state
        action = self.action_map.get(label, IGNORE_ACTION)
        spec = ARM_ACTIONS.get(action)
        if self.status_callback:
            self.status_callback(label, confidence)
        if not self.enabled:
            return IGNORE_ACTION
        if not decision.action_allowed or spec is None:
            self._stop_motion()
            return IGNORE_ACTION
        kind, direction = spec
        kw = dict(source="emg", gesture=label, confidence=confidence)
        if kind == "CARTESIAN":
            self.sink(MotionCommand(CARTESIAN, direction, magnitude=None, speed=self.speed, **kw))
            self._moving = True
        else:                                    # gripper: once per refractory period, not every tick
            if self.engine.trigger_click_once():
                self.sink(MotionCommand.gripper(direction, **kw))
        self.last_action = action
        return action


# ====================================================================================================
# RUNTIME: wires everything together and owns the threads   (from runtime.py)
# ====================================================================================================
#     thread "emg-input"   : EMGController polls the EMG source, smooths, maps -> MotionCommand queue
#     thread "control"     : fixed-rate loop (default 120 Hz): planner/IK, servos, MuJoCo physics, safety, log
#     thread "demo"        : optional pick-and-place sequence (key P / button)
#     main thread (Qt)     : dashboard + off-screen MuJoCo rendering (30 fps), reads controller.snapshot()
# The only coupling between them is the thread-safe ``controller.submit(MotionCommand)`` queue and
# ``controller.snapshot()``.
log = get_logger("arm")

MODES = ("full", "manual", "emg")
# Physics cost is per step. 120 Hz halves the CPU/heat of the original 240 Hz with no visible change in the arm's
# behaviour (planner runs at 100 Hz, EMG at 50 Hz, GUI at 30 Hz anyway) - good for a fanless MacBook Air.
DEFAULT_CONTROL_HZ = 120.0
MAX_OBJECTS = 10                      # most objects allowed on the table at once
OBJECT_KINDS = ("cube", "sphere", "cylinder")
OBJECT_COLORS = ((0.85, 0.25, 0.20, 1), (0.20, 0.50, 0.85, 1), (0.25, 0.70, 0.35, 1), (0.95, 0.75, 0.15, 1),
                 (0.65, 0.35, 0.85, 1), (0.95, 0.50, 0.15, 1), (0.15, 0.75, 0.75, 1), (0.90, 0.40, 0.65, 1))


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
                 objects: bool = True, threshold: float = EMG_CONFIDENCE_THRESHOLD,
                 window: int = SMOOTHING_WINDOW, save_log: bool = True):
        self.mode, self.headless = mode, headless
        self.cfg = load_config(config_path)
        self.env = Environment.default_scene() if objects else Environment()
        use_mj = physics and mujoco_available()
        if physics and not use_mj:
            log.warning("MuJoCo not installed (pip install mujoco): running with the kinematic backend "
                        "(no 3D view, objects or contacts)")
        self.backend = MuJoCoBackend(self.cfg, dt=1.0 / control_hz) if use_mj else KinematicBackend()
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
            print(KEY_HELP + "\n")
        log.info("runtime started (mode=%s, headless=%s, backend=%s, log=%s)", self.mode, self.headless,
                 type(self.backend).__name__, self.logger.path if self.logger else "off")

    def stop(self) -> None:
        self._stop.set()
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

    # ------------------------------------------------------------------ objects on the table (max MAX_OBJECTS)
    def _apply_objects(self) -> None:
        if hasattr(self.backend, "rebuild"):
            self.backend.rebuild(self.env)

    def random_reachable_position(self, kind: str, rng=None):
        """A free spot on the table that the gripper can really reach (checked with the arm's own IK), or None."""
        rng = rng or np.random.default_rng()
        proto = {"cube": make_cube, "sphere": make_sphere, "cylinder": make_cylinder}[kind]("probe", 0.0, 0.0)
        radius = proto.width / 2
        ik = self.controller.arm.ik_solver
        home = np.asarray(self.cfg.home_angles, dtype=float)
        for _ in range(800):
            r = rng.uniform(0.12, 0.22)
            th = math.radians(rng.uniform(35.0, 145.0))              # in front of the arm, away from its base
            x, y = r * math.cos(th), r * math.sin(th)
            if any(math.hypot(x - o.position[0], y - o.position[1]) < radius + o.width / 2 + 0.012
                   for o in self.env.objects):
                continue
            grip_z = max(proto.half_height, 0.022)
            if ik.solve(np.array([x, y, grip_z]), tool_pitch=math.pi, seed=home).success:
                return x, y
        return None

    def add_random_object(self, kind: str):
        """Add one object of ``kind`` at a random reachable spot. Returns the SimObject, or None (limit / no room)."""
        if kind not in OBJECT_KINDS:
            raise ValueError(f"kind must be one of {OBJECT_KINDS}")
        if len(self.env.objects) >= MAX_OBJECTS:
            return None
        spot = self.random_reachable_position(kind)
        if spot is None:
            return None
        n = 1 + max([int(o.name.rsplit("_", 1)[1]) for o in self.env.objects if o.name.rsplit("_", 1)[-1].isdigit()] or [0])
        color = OBJECT_COLORS[len(self.env.objects) % len(OBJECT_COLORS)]
        maker = {"cube": make_cube, "sphere": make_sphere, "cylinder": make_cylinder}[kind]
        obj = self.env.add(maker(f"{kind}_{n}", spot[0], spot[1], color=color))
        self._apply_objects()
        return obj

    def clear_objects(self) -> None:
        self.env.objects.clear()
        self._apply_objects()

    def reset_objects(self) -> None:
        self.env.objects[:] = Environment.default_scene().objects
        self._apply_objects()

    # ------------------------------------------------------------------ pick-and-place demo (key P / button)
    def _demo_target(self):
        cubes = [o for o in self.env.objects if o.kind == "cube"]
        pool = cubes or list(self.env.objects)
        return pool[0] if pool else None

    def start_demo(self) -> None:
        if self._demo_target() is None:
            log.warning("pick-and-place demo needs an object on the table (add one from the full-screen view)")
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
            cube = self._demo_target()
            if cube is None:
                return
            pos = self.backend.object_position(cube.name) or cube.position
            grip_z = max(cube.half_height, 0.022)               # keep the fingertips above the table
            dx = 0.08 if pos[0] + 0.08 <= 0.25 else -0.08
            c.home(wait=True)
            ok = pick_and_place(c, cube.name, np.array([pos[0], pos[1], grip_z]),
                                np.array([pos[0] + dx, pos[1], grip_z + 0.002]))
            log.info("pick-and-place demo: %s", "done" if ok else "failed (see warnings above)")
        except Exception:
            log.exception("pick-and-place demo crashed")

    # ------------------------------------------------------------------ run modes
    def run(self, duration: float | None = None) -> None:
        """Console run: control thread + EMG thread, status once per sim second. Ctrl-C ends it."""
        self.start()
        t_end = None if duration is None else time.perf_counter() + duration
        try:
            last = 0.0
            while t_end is None or time.perf_counter() < t_end:
                time.sleep(0.1)
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
# UI: BIOWAVE-STYLE DASHBOARD   (PyQt5; main process only - the PyBullet viewer child never loads Qt)
# ====================================================================================================
# Needs:  pip install PyQt5 pyqtgraph pyserial joblib scikit-learn
# Without PyQt5 the simulator still runs exactly as before (keyboard + terminal dashboard).

if HAS_QT:
    class SerialWorker(QThread):
        """Reads a live EMG stream over USB/COM serial (or a socket:// simulator)."""
        batch_received = pyqtSignal(object)
        error_occurred = pyqtSignal(str)

        def __init__(self, port_name, baud_rate, num_channels, batch_size=25):
            super().__init__()
            self.port_name = port_name
            self.baud_rate = baud_rate
            self.num_channels = num_channels
            self.batch_size = batch_size
            self.is_socket_url = str(port_name).strip().lower().startswith("socket://")
            self._running = True
            self._serial = None

        def _close_serial(self):
            try:
                if self._serial and self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
            self._serial = None

        def run(self):
            partial_line = ""
            batch = []
            try:
                while self._running:
                    try:
                        if self._serial is None or not self._serial.is_open:
                            self._serial = serial.serial_for_url(self.port_name, self.baud_rate, timeout=0.02)
                            try:
                                self._serial.reset_input_buffer()
                            except Exception:
                                pass
                            partial_line = ""

                        waiting = self._serial.in_waiting
                        chunk = self._serial.read(waiting if waiting else 1)
                        if not chunk:
                            time.sleep(0.001)
                            continue

                        partial_line += chunk.decode("utf-8", errors="ignore")
                        lines = partial_line.split("\n")
                        partial_line = lines.pop()

                        for raw_line in lines:
                            line = raw_line.strip()
                            if not line:
                                continue
                            parts = line.replace(",", " ").split()
                            if len(parts) < self.num_channels:
                                continue
                            try:
                                vals = [float(parts[i]) for i in range(self.num_channels)]
                            except ValueError:
                                continue
                            batch.append(vals)
                            if len(batch) >= self.batch_size:
                                self.batch_received.emit(np.asarray(batch, dtype=np.float32))
                                batch = []
                    except Exception as e:
                        if not self._running:
                            break
                        self._close_serial()
                        partial_line = ""
                        if self.is_socket_url:
                            time.sleep(0.3)
                            continue
                        self.error_occurred.emit(str(e))
                        break
            finally:
                self._close_serial()

        def stop(self):
            self._running = False
            self.wait()


    class WirelessStreamWorker(QThread):
        """Receives the UDP EMG+IMU packet stream from a wireless BioWave device."""
        batch_received = pyqtSignal(object)
        error_occurred = pyqtSignal(str)

        def __init__(self, port=WIFI_STREAM_PORT):
            super().__init__()
            self.port = int(port)
            self._running = True
            self._sock = None
            self._fallback_packet_sequence = 0
            self.stats = WirelessStats()

        def _decode_frames(self, payload, count):
            rows, frame_ids, frame_ts, imu_ids, imu_ts = [], [], [], [], []
            for offset in range(0, count * WIRELESS_FRAME_SIZE, WIRELESS_FRAME_SIZE):
                frame = payload[offset: offset + WIRELESS_FRAME_SIZE]
                frame_id, emg_ts, imu_id, imu_ts_value, *frame_fields = struct.unpack(WIRELESS_FRAME_FORMAT, frame)
                row = [float(v) for v in frame_fields[:WIRELESS_EMG_CHANNELS]]
                row.extend([float(frame_fields[8]), float(frame_fields[9]), float(frame_fields[10])])
                rows.append(row)
                frame_ids.append(frame_id); frame_ts.append(emg_ts)
                imu_ids.append(imu_id); imu_ts.append(imu_ts_value)
            return rows, frame_ids, frame_ts, imu_ids, imu_ts

        def _parse_datagram(self, data):
            if len(data) == WIRELESS_PACKET_SIZE:
                magic, version, frame_count, frame_size, packet_sequence = struct.unpack(
                    WIFI_PACKET_HEADER_FORMAT, data[:WIFI_PACKET_HEADER_SIZE]
                )
                if magic != b"BWIM" or version != 1 or frame_size != WIRELESS_FRAME_SIZE or frame_count != WIRELESS_FRAMES_PER_PACKET:
                    self.stats.invalid_packets += 1
                    return None
                payload = data[WIFI_PACKET_HEADER_SIZE:]
                rows, frame_ids, frame_ts, imu_ids, imu_ts = self._decode_frames(payload, WIRELESS_FRAMES_PER_PACKET)
                batch = np.asarray(rows, dtype=np.float32)
                packet_numbers = np.full(batch.shape[0], int(packet_sequence), dtype=np.int64)
                arrival = time.monotonic()
                return SampleBatch(batch, packet_numbers, np.asarray(frame_ids), np.asarray(frame_ts),
                                   np.asarray(imu_ids), np.asarray(imu_ts), arrival,
                                   gap_before=self.stats.observe(packet_sequence, arrival))

            if len(data) == (WIRELESS_FRAME_SIZE * WIRELESS_FRAMES_PER_PACKET):
                rows, frame_ids, frame_ts, imu_ids, imu_ts = self._decode_frames(data, WIRELESS_FRAMES_PER_PACKET)
                batch = np.asarray(rows, dtype=np.float32)
                packet_no = int(self._fallback_packet_sequence)
                self._fallback_packet_sequence += 1
                packet_numbers = np.full(batch.shape[0], packet_no, dtype=np.int64)
                return SampleBatch(batch, packet_numbers, np.asarray(frame_ids), np.asarray(frame_ts),
                                   np.asarray(imu_ids), np.asarray(imu_ts), time.monotonic())

            if len(data) == WIRELESS_FRAME_SIZE:
                rows, frame_ids, frame_ts, imu_ids, imu_ts = self._decode_frames(data, 1)
                batch = np.asarray(rows, dtype=np.float32)
                packet_no = int(self._fallback_packet_sequence)
                self._fallback_packet_sequence += 1
                packet_numbers = np.full(batch.shape[0], packet_no, dtype=np.int64)
                return SampleBatch(batch, packet_numbers, np.asarray(frame_ids), np.asarray(frame_ts),
                                   np.asarray(imu_ids), np.asarray(imu_ts), time.monotonic())

            return None

        def run(self):
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self.port))
            self._sock.settimeout(1.0)
            try:
                while self._running:
                    try:
                        data, _addr = self._sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    payload = self._parse_datagram(data)
                    if payload is None:
                        continue
                    batch = np.asarray(payload.samples, dtype=np.float32)
                    if batch.size > 0:
                        self.batch_received.emit(payload)
            except Exception as exc:
                if self._running:
                    self.error_occurred.emit(f"Wireless stream error: {exc}")
            finally:
                try:
                    if self._sock is not None:
                        self._sock.close()
                except Exception:
                    pass
                self._sock = None

        def stop(self):
            self._running = False
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            self.wait()


    class InferenceWorker(QThread):
        """Extracts features and runs the pretrained Random Forest prediction."""
        # label, confidence (None if unavailable), top-1/top-2 margin, inference ms
        prediction_ready = pyqtSignal(str, object, object, float)
        inference_error = pyqtSignal(str)

        def __init__(self, sample_rate):
            super().__init__()
            self.sample_rate = int(sample_rate)
            self.model = None
            self.class_names = []
            self._window = None
            self._running = True
            self._lock = threading.Lock()
            self._event = threading.Event()

        def load_model(self, model, class_names, sample_rate=None):
            with self._lock:
                self.model = model
                self.class_names = [str(x) for x in list(class_names or [])]
                if sample_rate is not None:
                    self.sample_rate = int(max(1, sample_rate))
                self._window = None
            self._event.clear()

        def clear_model(self):
            with self._lock:
                self.model = None
                self.class_names = []
                self._window = None
            self._event.clear()

        def submit_window(self, window):
            with self._lock:
                self._window = window
            self._event.set()

        def run(self):
            while self._running:
                self._event.wait(0.1)
                if not self._running:
                    break
                if not self._event.is_set():
                    continue
                self._event.clear()

                with self._lock:
                    win = self._window
                    model = self.model
                    classes = list(self.class_names)
                    self._window = None

                if win is None or model is None or not HAS_RF_FEATURES:
                    continue

                try:
                    started = time.monotonic()
                    feats = extract_window_features(win, sample_rate=self.sample_rate).reshape(1, -1)
                    pred_label = "N/A"
                    conf = None
                    margin = None

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(feats)[0]
                        confidences = np.zeros(len(classes), dtype=np.float32)
                        model_classes = list(getattr(model, "classes_", []))
                        if len(model_classes) == len(proba) and len(classes) > 0:
                            for i, cls_id in enumerate(model_classes):
                                idx = -1
                                try:
                                    idx = int(cls_id)
                                except Exception:
                                    cls_text = str(cls_id)
                                    if cls_text in classes:
                                        idx = classes.index(cls_text)
                                if 0 <= idx < len(confidences):
                                    confidences[idx] = float(proba[i])
                            pred_idx = int(np.argmax(confidences)) if np.max(confidences) > 0 else int(np.argmax(proba))
                            pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(model_classes[int(np.argmax(proba))])
                            conf = float(np.max(confidences)) if np.max(confidences) > 0 else float(np.max(proba))
                            sorted_p = np.sort(proba)
                            margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else float(sorted_p[-1])
                        else:
                            pred_idx = int(np.argmax(proba))
                            pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
                            conf = float(proba[pred_idx])
                            sorted_p = np.sort(proba)
                            margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else float(sorted_p[-1])
                    else:
                        pred_raw = model.predict(feats)[0]
                        if isinstance(pred_raw, (int, np.integer)) and 0 <= int(pred_raw) < len(classes):
                            pred_label = classes[int(pred_raw)]
                        else:
                            pred_label = str(pred_raw)
                        # A classifier without probabilities does not provide a confidence.
                        # Safety gating consequently holds it in UNKNOWN rather than inventing 100%.
                        conf = None

                    self.prediction_ready.emit(pred_label, conf, margin, (time.monotonic() - started) * 1000.0)
                except Exception as e:
                    LOG.exception("Inference error")
                    self.inference_error.emit(str(e))

        def stop(self):
            self._running = False
            self._event.set()
            self.wait()


    class ProvisionDialog(QDialog):
        """One-off USB step: hand Wi-Fi credentials to a fresh ESP32 device."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Provision Wireless Device (USB)")
            self.resize(480, 300)
            self.setModal(True)

            layout = QVBoxLayout(self)
            intro = QLabel("Connect the ESP32 over USB, pick its port, then send your Wi-Fi credentials.")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            port_row = QHBoxLayout()
            port_row.addWidget(QLabel("USB Port:"))
            self.combo_port = QComboBox()
            port_row.addWidget(self.combo_port, 1)
            btn_refresh = QPushButton("Refresh")
            btn_refresh.clicked.connect(self.refresh_ports)
            port_row.addWidget(btn_refresh)
            layout.addLayout(port_row)

            btn_info = QPushButton("Query Device Info")
            btn_info.clicked.connect(self.query_info)
            layout.addWidget(btn_info)

            self.lbl_info = QLabel("No device queried yet.")
            self.lbl_info.setWordWrap(True)
            layout.addWidget(self.lbl_info)

            form = QFormLayout()
            self.txt_ssid = QLineEdit()
            form.addRow("Wi-Fi SSID:", self.txt_ssid)
            self.txt_password = QLineEdit()
            self.txt_password.setEchoMode(QLineEdit.Password)
            form.addRow("Wi-Fi Password:", self.txt_password)
            layout.addLayout(form)

            btn_row = QHBoxLayout()
            self.btn_send = QPushButton("Send Credentials")
            self.btn_send.clicked.connect(self.send_credentials)
            btn_row.addWidget(self.btn_send)
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            btn_row.addWidget(btn_close)
            layout.addLayout(btn_row)

            self.refresh_ports()

        def refresh_ports(self):
            self.combo_port.clear()
            for p in WiFiSerialProvisionProtocol.available_ports():
                self.combo_port.addItem(f"{p.device} - {p.description}", p.device)

        def _selected_port(self):
            data = self.combo_port.currentData()
            if data:
                return data
            text = self.combo_port.currentText()
            return text.split()[0] if text else ""

        def query_info(self):
            port = self._selected_port()
            if not port:
                QMessageBox.warning(self, "No Port", "Select a USB port first.")
                return
            try:
                info = WiFiSerialProvisionProtocol.query_info(port)
                self.lbl_info.setText(
                    f"{info.device_name} | FW={info.firmware} | IMU ready={info.imu_ready} | Wi-Fi saved={info.wifi_saved}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Query Failed", str(e))

        def send_credentials(self):
            port = self._selected_port()
            ssid = self.txt_ssid.text().strip()
            password = self.txt_password.text()
            if not port or not ssid:
                QMessageBox.warning(self, "Missing Info", "Select a port and enter an SSID.")
                return
            try:
                WiFiSerialProvisionProtocol.provision(port, ssid, password)
                QMessageBox.information(self, "Provisioned", "Wi-Fi credentials sent. The device will reboot onto your network.")
            except Exception as e:
                QMessageBox.critical(self, "Provisioning Failed", str(e))


    class CalibrationDialog(QDialog):
        """Guides the user through a REST -> FLEX capture used to zero the baseline."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Calibration")
            self.resize(420, 220)
            self.setModal(True)
            self.setWindowFlag(Qt.WindowCloseButtonHint, False)

            layout = QVBoxLayout(self)
            self.lbl_phase = QLabel("REST")
            self.lbl_phase.setAlignment(Qt.AlignCenter)
            f = QFont()
            f.setPointSize(22)
            f.setBold(True)
            self.lbl_phase.setFont(f)
            layout.addWidget(self.lbl_phase)

            self.lbl_instruction = QLabel("")
            self.lbl_instruction.setWordWrap(True)
            self.lbl_instruction.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.lbl_instruction)

            self.lbl_countdown = QLabel("")
            self.lbl_countdown.setAlignment(Qt.AlignCenter)
            cf = QFont()
            cf.setPointSize(16)
            self.lbl_countdown.setFont(cf)
            layout.addWidget(self.lbl_countdown)

            self.btn_cancel = QPushButton("Cancel")
            layout.addWidget(self.btn_cancel)

        def set_phase(self, name, instruction, remaining_ms, total_ms):
            self.lbl_phase.setText(name)
            self.lbl_instruction.setText(instruction)
            self.lbl_countdown.setText(f"{max(0, remaining_ms) / 1000.0:0.1f}s remaining")

        def set_finished(self, summary):
            self.lbl_phase.setText("Done")
            self.lbl_instruction.setText(summary)
            self.lbl_countdown.setText("")
            self.btn_cancel.setText("Close")



    class HoldButton(QPushButton):
        """Button that calls ``on_tick`` every 100 ms while it is held down (on-screen jog pad)."""

        def __init__(self, text, on_tick, on_release=None, parent=None):
            super().__init__(text, parent)
            self._tick, self._release = on_tick, on_release
            self._timer = QTimer(self)
            self._timer.setInterval(100)
            self._timer.timeout.connect(self._tick)
            self.pressed.connect(self._start)
            self.released.connect(self._stop)

        def _start(self):
            self._tick()
            self._timer.start()

        def _stop(self):
            self._timer.stop()
            if self._release:
                self._release()


    class _NoHfwLayout(QVBoxLayout):
        """A layout that never asks its scroll area for 'height for width': otherwise QScrollArea inflates the content
        to the height all wrapped labels would need at the current width and shows a needless scroll bar."""

        def hasHeightForWidth(self):
            return False

    class ViewWidget(QLabel):
        """The MuJoCo scene, rendered off-screen (SceneRenderer) and shown in a QLabel.

        drag = orbit, right-drag / shift-drag = pan, wheel = zoom, double-click = reset view.
        Frames are only rendered while the widget is visible, at most ``max_mpix`` megapixels (cooler on a fanless Air).
        """

        def __init__(self, backend, max_mpix: float = 0.70, shadows: bool = True):
            super().__init__()
            self.backend, self.max_mpix, self.shadows, self.frames = backend, max_mpix, shadows, False
            self.renderer = None
            self.initial_distance = None                    # None = default camera distance
            self.error = ""
            self.ms = 0.0                                   # EMA of render time (ms)
            self._last = None
            self.setMinimumSize(320, 200)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet(f"background:{THEME_COLORS['graph_bg']}; border-radius: 6px; color:{THEME_COLORS['muted']};")
            if getattr(backend, "model", None) is None:
                self.error = "3D view needs the MuJoCo physics backend (pip install mujoco)."
                self.setText(self.error)

        # ---- options
        def set_frames(self, on: bool) -> None:
            self.frames = bool(on)
            if self.renderer:
                self.renderer.set_frames(self.frames)

        def set_shadows(self, on: bool) -> None:
            self.shadows = bool(on)
            if self.renderer:
                self.renderer.shadows = self.shadows

        def reset_view(self) -> None:
            if self.renderer:
                self.renderer.reset_view()
                if self.initial_distance:
                    self.renderer.cam.distance = self.initial_distance

        # ---- rendering
        def minimumSizeHint(self):
            return QSize(320, 200)

        def sizeHint(self):
            return QSize(640, 420)

        def _target_size(self):
            w, h = max(64, self.width()), max(64, self.height())
            k = min(1.0, math.sqrt(self.max_mpix * 1e6 / (w * h)), 1920.0 / w, 1200.0 / h)   # MJCF offscreen buffer is 1920x1200
            return int(w * k) // 2 * 2, int(h * k) // 2 * 2

        def render_frame(self) -> None:
            if self.error or not self.isVisible() or getattr(self.backend, "model", None) is None:
                return
            try:
                tw, th = self._target_size()
                if self.renderer is None:
                    self.renderer = SceneRenderer(self.backend, tw, th, self.shadows)
                    self.renderer.set_frames(self.frames)
                    if self.initial_distance:
                        self.renderer.cam.distance = self.initial_distance
                elif abs(tw - self.renderer.size[0]) > 16 or abs(th - self.renderer.size[1]) > 16:
                    self.renderer.resize(tw, th)
                t0 = time.perf_counter()
                img = self.renderer.render()
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
                self._last = QPixmap.fromImage(qimg)
                self.setPixmap(self._last.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.ms = 0.9 * self.ms + 0.1 * (time.perf_counter() - t0) * 1000.0 if self.ms else (time.perf_counter() - t0) * 1000.0
            except Exception as exc:                        # no OpenGL context etc.: keep the simulator running
                self.error = f"3D view unavailable: {exc}"
                self.setText(self.error)
                LOG.error(self.error)

        # ---- mouse
        def mousePressEvent(self, e):
            self._p = e.pos()
            self._pan = bool(e.buttons() & (Qt.RightButton | Qt.MiddleButton)) or bool(e.modifiers() & Qt.ShiftModifier)

        def mouseMoveEvent(self, e):
            if not self.renderer or not hasattr(self, "_p"):
                return
            d = e.pos() - self._p
            self._p = e.pos()
            (self.renderer.pan if self._pan else self.renderer.orbit)(d.x(), d.y())

        def wheelEvent(self, e):
            if self.renderer:
                self.renderer.zoom(e.angleDelta().y() / 120.0)

        def mouseDoubleClickEvent(self, e):
            self.reset_view()

        def close_renderer(self) -> None:
            if self.renderer:
                self.renderer.close()
                self.renderer = None


    KEY_GUIDE = [
        ("Move the gripper (end effector)", [
            ("←  →", "left / right  (X axis)"),
            ("↑  ↓", "up / down  (Z axis)"),
            ("I   K", "forward / back  (Y axis)")]),
        ("Move a single joint  (hold the key)", [
            ("Q   A", "J1 base: turn one way / the other"),
            ("W   S", "J2 shoulder"),
            ("E   D", "J3 elbow"),
            ("R   F", "J4 wrist pitch"),
            ("T   G", "J5 wrist roll")]),
        ("Gripper", [
            ("O", "open  (press once)"),
            ("C", "close  (press once)"),
            ("Y   H", "open / close slowly  (hold)")]),
        ("Safety and system", [
            ("Esc", "EMERGENCY STOP  (then X to reset)"),
            ("X", "reset after an emergency stop"),
            ("Space", "stop moving"),
            ("Z", "go to the Home pose"),
            ("P", "pick-and-place demo")]),
        ("Test gestures without the armband  (hold)", [
            ("1  2  3  4", "left, right, up, down"),
            ("5", "fist = close gripper"),
            ("0", "rest = hold still")]),
        ("Window", [
            ("F11", "full-screen 3D view  (F11 or the Exit button to leave)"),
            ("F1", "open this help")]),
    ]

    KEY_ACTIONS = {
        "left": "move left (-X)", "right": "move right (+X)", "up": "move up (+Z)", "down": "move down (-Z)",
        "i": "move forward (+Y)", "k": "move back (-Y)",
        "q": "J1 base +", "a": "J1 base -", "w": "J2 shoulder +", "s": "J2 shoulder -", "e": "J3 elbow +",
        "d": "J3 elbow -", "r": "J4 wrist pitch +", "f": "J4 wrist pitch -", "t": "J5 wrist roll +",
        "g": "J5 wrist roll -", "y": "open gripper (slow)", "h": "close gripper (slow)",
        "o": "open gripper", "c": "close gripper", "esc": "EMERGENCY STOP", "x": "reset", "z": "home",
        " ": "stop", "p": "pick-and-place", "1": "test gesture: left", "2": "test gesture: right",
        "3": "test gesture: up", "4": "test gesture: down", "5": "test gesture: fist", "0": "test gesture: rest",
    }
    QUALITY_COLORS = {"GOOD": "#4cc38a", "WEAK": "#f5a524", "NOISY": "#f5a524", "SATURATED": "#ff5c5c",
                      "DEAD": "#ff5c5c", "INVALID": "#ff5c5c"}


    def key_guide_html(compact: bool = False) -> str:
        size = "12px" if compact else "13px"
        rows = ""
        for title, items in KEY_GUIDE:
            rows += f'<tr><td colspan="2" style="padding-top:8px;color:{THEME_COLORS["accent"]};font-weight:700">{title}</td></tr>'
            for keys, what in items:
                rows += (f'<tr><td style="padding:2px 10px 2px 0;white-space:nowrap"><span style="background:{THEME_COLORS["panel"]};'
                         f'border:1px solid {THEME_COLORS["muted"]};border-radius:4px;padding:1px 6px;font-weight:700">{keys}</span></td>'
                         f'<td>{what}</td></tr>')
        return f'<table style="font-size:{size}" cellspacing="0">{rows}</table>'


    def help_html() -> str:
        c = THEME_COLORS
        h = lambda t: f'<h3 style="color:{c["accent"]};margin-bottom:2px">{t}</h3>'
        return f"""
    <h2>BioWave Robotic Arm - quick guide</h2>
    {h("A. Control the arm with your EMG armband")}
    <ol>
    <li><b>Connect</b> (left panel, step 1). Wireless: switch the armband on, make sure this computer is on the
    same Wi-Fi, click <i>Discover</i>, pick the device, type its access key, click <i>Connect Wireless</i>.
    New armband? <i>Provision (USB)</i> first. Wired: choose the serial port and click <i>Connect</i>.</li>
    <li><b>Load the model</b> (step 2): <i>Browse .joblib</i> and choose the model you trained in BioWave
    (<code>rf_realtime_model.joblib</code>). Its gestures appear in the mapping list.</li>
    <li><b>Calibrate</b> (step 3): click <i>Calibrate</i>. First <b>REST</b> - relax your arm completely.
    Then <b>FLEX</b> - squeeze steadily until the timer ends. Channels with bad contact are reported.</li>
    <li><b>Check the mapping</b>: click <i>Show / edit gesture &rarr; arm action mapping</i> (left panel) and choose which
    arm action each gesture triggers. <i>Rest</i> should be &quot;Ignore (hold)&quot;. Click the button again to hide the list.</li>
    <li>Press <b>ENABLE ARM CONTROL</b>. Make a gesture and hold it: the arm moves while you hold it and stops
    when you relax. Relax = the arm holds still.</li>
    </ol>
    {h("B. No armband? Try it anyway")}
    <p>Use the keyboard (see below) or the <i>Test</i> tab: hold a button and the simulated
    gesture goes through the same filtering and mapping as a real armband.</p>
    {h("C. Keyboard")}
    <p>Click the window first. Keys do nothing while you are typing in a text box. Hold <b>one</b> movement key
    at a time.</p>
    {key_guide_html()}
    {h("D. 3D view")}
    <p><b>Drag</b> = rotate, <b>right-drag</b> (or Shift+drag) = pan, <b>wheel</b> = zoom, <b>double-click</b> = reset.
    <b>Full screen</b>: the button under the view or <b>F11</b>. The full-screen bar keeps Emergency Stop and Home within reach.
    <i>Link frames</i> shows the coordinate axes of every joint; the red dot is the gripper tip (TCP).</p>
    <p><b>Objects (full screen only)</b>: pick a shape (cube, sphere or cylinder) and click <i>Add object</i>: it appears at a
    random spot the arm can reach. Up to {MAX_OBJECTS} objects. <i>Clear all</i> empties the table, <i>Reset objects</i>
    restores the starting three. The pick-and-place demo (P) uses the first cube, or the first object if there is no cube.</p>
    {h("E. EMG signals")}
    <p>Click <b>Show EMG signals</b> to open a large window with one row per channel. A flat line means no signal
    (check skin contact); a huge noisy trace means poor contact or movement. When you flex, the channels over that
    muscle should grow. The colour of each title shows its quality (green = GOOD). Choose the time span, give all
    channels the same scale, or pause the display.</p>
    {h("F. Settings that matter")}
    <ul>
    <li><b>Minimum confidence</b>: how sure the model must be before the arm acts (raise it if the arm twitches).</li>
    <li><b>Confidence margin</b>: the winner must beat the runner-up by this much.</li>
    <li><b>Debounce</b>: how many consecutive predictions must agree.</li>
    <li><b>EMG arm speed</b>: speed of gesture-driven movement. <b>Gripper cooldown</b>: minimum time between open/close.</li>
    </ul>
    {h("G. Safety")}
    <ul>
    <li><b>EMERGENCY STOP</b> (button or Esc) stops everything and latches until you press <i>Reset after stop</i> (X).</li>
    <li>Arm control switches itself off and the arm stops if: the signal quality drops, packets are lost, the
    stream stalls, the device link is lost, or the model fails.</li>
    <li>The arm will not move into the table, into itself, or beyond joint limits.</li>
    </ul>
    {h("H. Troubleshooting")}
    <ul>
    <li><b>Device not found</b>: same Wi-Fi? firewall allowing UDP 5000/5001? Try <i>Discover</i> again.</li>
    <li><b>Calibration rejected</b>: the message lists the bad channels. Moisten/reseat the electrodes, relax fully
    during REST, flex harder during FLEX.</li>
    <li><b>ENABLE ARM CONTROL is greyed out</b>: you need all of: connected, model loaded, calibrated.</li>
    <li><b>Arm does not move</b>: is the gesture mapped to an action (not &quot;Ignore&quot;)? Is the status strip
    showing E-STOP? Press X. Try lowering Minimum confidence.</li>
    <li><b>Keys do nothing</b>: click the window; close any text box focus; check the line under the 3D view.</li>
    </ul>
    """


    class HelpDialog(QDialog):
        """Scrollable user guide (F1)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("BioWave Robotic Arm - Help")
            self.resize(780, 700)
            self.setStyleSheet(ux_stylesheet(13))
            lay = QVBoxLayout(self)
            view = QTextBrowser()
            view.setOpenExternalLinks(False)
            view.setHtml(help_html())
            lay.addWidget(view)
            btn = QPushButton("Close")
            btn.clicked.connect(self.close)
            lay.addWidget(btn)


    class EMGWindow(QWidget):
        """Large, readable per-channel EMG view, opened on demand from the dashboard."""

        SPANS = (("1 second", 1.0), ("2 seconds", 2.0), ("5 seconds", 5.0))

        def __init__(self, dash):
            super().__init__(None, Qt.Window)
            self.dash, self.paused = dash, False
            self.setWindowTitle("BioWave - Live EMG signals")
            self.resize(1000, 780)
            self.setStyleSheet(ux_stylesheet(12))
            apply_dark_title_bar(self)
            lay = QVBoxLayout(self)
            intro = QLabel("One row per EMG channel, centred on its baseline. A flat line = no signal (check skin contact). "
                           "A huge noisy trace = poor contact or movement. Flex your muscle: the matching channels should grow. "
                           "Title colour: green = good quality.")
            intro.setWordWrap(True)
            intro.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(intro)
            row = QHBoxLayout()
            row.addWidget(QLabel("Time span:"))
            self.combo_span = QComboBox()
            for text, sec in self.SPANS:
                self.combo_span.addItem(text, sec)
            self.combo_span.setCurrentIndex(1)
            row.addWidget(self.combo_span)
            self.chk_same = QCheckBox("Same scale on all channels")
            self.chk_same.setToolTip("Off: every row auto-scales (best for checking contact). "
                                     "On: rows share one scale (best for comparing channel strength).")
            row.addWidget(self.chk_same)
            self.btn_pause = QPushButton("Pause")
            self.btn_pause.setCheckable(True)
            self.btn_pause.toggled.connect(self._toggle_pause)
            row.addWidget(self.btn_pause)
            row.addStretch(1)
            self.lbl_status = QLabel("")
            self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            row.addWidget(self.lbl_status)
            lay.addLayout(row)
            self.glw = pg.GraphicsLayoutWidget()
            self.glw.setBackground(THEME_COLORS["graph_bg"])
            self.plots, self.curves = [], []
            for i in range(WIRELESS_EMG_CHANNELS):
                p = self.glw.addPlot(row=i, col=0)
                p.setMouseEnabled(False, False)
                p.hideButtons()
                p.showGrid(y=True, x=False, alpha=0.12)
                p.getAxis("left").setWidth(14)
                p.getAxis("left").setStyle(showValues=False, tickLength=0)      # the range is printed in the title instead
                p.setYRange(-1, 1, padding=0)
                p.setXRange(-2, 0, padding=0)
                p.getAxis("left").setPen(pg.mkPen(THEME_COLORS["muted"]))
                p.getAxis("left").setTextPen(pg.mkPen(THEME_COLORS["text"]))
                p.getAxis("bottom").setPen(pg.mkPen(THEME_COLORS["muted"]))
                p.getAxis("bottom").setTextPen(pg.mkPen(THEME_COLORS["text"]))
                if i < WIRELESS_EMG_CHANNELS - 1:
                    p.getAxis("bottom").setStyle(showValues=False, tickLength=0)
                else:
                    p.setLabel("bottom", "seconds  (0 = now)")
                    p.getAxis("bottom").setStyle(tickFont=QFont("Arial", 9))
                p.setTitle(f"CH{i + 1}   waiting for signal", color=THEME_COLORS["muted"], size="9pt")
                self.plots.append(p)
                self.curves.append(p.plot(pen=pg.mkPen(PLOT_COLORS[i % len(PLOT_COLORS)], width=1.2)))
            self.glw.setMinimumHeight(WIRELESS_EMG_CHANNELS * 56)
            lay.addWidget(self.glw, 1)
            # shown over the empty plots until samples arrive
            self.lbl_wait = QLabel("No EMG signal yet.\nConnect the armband (step 1 in the main window)\nor wait for samples to arrive.",
                                   self.glw)
            self.lbl_wait.setAlignment(Qt.AlignCenter)
            self.lbl_wait.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.lbl_wait.setStyleSheet(f"background: rgba(13,27,51,215); color: {THEME_COLORS['text']}; border-radius: 10px;"
                                        " padding: 14px; font-size: 14px;")
            self._place_wait()

        def _place_wait(self):
            self.lbl_wait.adjustSize()
            self.lbl_wait.move(max(0, (self.glw.width() - self.lbl_wait.width()) // 2),
                               max(0, (self.glw.height() - self.lbl_wait.height()) // 2))

        def resizeEvent(self, ev):
            super().resizeEvent(ev)
            if hasattr(self, "lbl_wait"):
                self._place_wait()

        def _toggle_pause(self, on):
            self.paused = on
            self.btn_pause.setText("Resume" if on else "Pause")

        def _states(self):
            d = self.dash
            if d.last_signal_quality is not None and d.last_signal_quality.channel_states:
                return list(d.last_signal_quality.channel_states)
            if d.calibration_profile is not None:
                return list(d.calibration_profile.quality)
            return []

        def closeEvent(self, ev):
            self.dash.btn_emg.setChecked(False)
            ev.accept()

        def refresh(self):
            if not self.isVisible() or self.paused:
                return
            d = self.dash
            span = float(self.combo_span.currentData())
            n = min(d.plot_filled, int(span * SAMPLE_RATE))
            states = self._states()
            self.lbl_wait.setVisible(n < 10)
            if n < 10:
                self._place_wait()
                self.lbl_status.setText("Waiting for samples...")
                return
            data = d.plot_buf[:WIRELESS_EMG_CHANNELS, PLOT_SAMPLES - n:]
            centred = data - data.mean(axis=1, keepdims=True)
            step = max(1, n // 700)
            xs = (np.arange(n)[::step] - n) / float(SAMPLE_RATE)
            amp = np.maximum(np.percentile(np.abs(centred), 99, axis=1), 1.0)
            if self.chk_same.isChecked():
                amp[:] = amp.max()
            for i in range(WIRELESS_EMG_CHANNELS):
                self.curves[i].setData(xs, centred[i][::step])
                self.plots[i].setYRange(-float(amp[i]) * 1.25, float(amp[i]) * 1.25, padding=0)
                self.plots[i].setXRange(-float(span), 0, padding=0)
                st = states[i] if i < len(states) else ("not calibrated yet" if not d.is_calibrated else "?")
                self.plots[i].setTitle(f"CH{i + 1}   {st}   (range +/-{amp[i]:.0f})",
                                       color=QUALITY_COLORS.get(st, THEME_COLORS["muted"]), size="9pt")
            self.lbl_status.setText(f"{d.sample_rate_measured:.0f} samples/s per channel (expected {SAMPLE_RATE})")


    def keycap(text: str) -> str:
        return (f'<span style="background:{THEME_COLORS["panel"]};border:1px solid {THEME_COLORS["muted"]};'
                f'border-radius:4px;padding:1px 6px;font-weight:700">{text}</span>')


    def ux_stylesheet(size: int = 12) -> str:
        """app_stylesheet + visible check boxes (the base theme draws an unchecked box invisibly dark)."""
        c = THEME_COLORS
        return app_stylesheet(size) + f"""
    QCheckBox {{ spacing: 6px; }}
    QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {c['muted']}; border-radius: 3px;
                           background: {c['title_bar']}; }}
    QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['text']}; }}
    QToolTip {{ background: {c['title_bar']}; color: {c['text']}; border: 1px solid {c['accent']}; padding: 4px; }}
    """


    class FullscreenView(QWidget):
        """Full-screen 3D view: short on-screen operating instructions, object controls (max 10) and a control bar."""

        SHAPES = (("Cube", "cube"), ("Sphere", "sphere"), ("Cylinder", "cylinder"))

        def __init__(self, dash):
            super().__init__(None, Qt.Window)
            self.dash = dash
            self.setWindowTitle("BioWave - 3D view")
            self.setStyleSheet(ux_stylesheet(12))
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            self.view = ViewWidget(dash.rt.backend, max(dash.view_max_mpix, 1.4), dash.view.shadows)
            self.view.frames = dash.view.frames
            self.view.initial_distance = 0.70                 # a little closer than the small view: objects are the point
            lay.addWidget(self.view, 1)

            # ---- operating instructions, drawn over the 3D scene (mouse passes through to the camera controls)
            self.tips = QLabel(self.view)
            self.tips.setTextFormat(Qt.RichText)
            self.tips.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.tips.setStyleSheet(f"background: rgba(13,27,51,210); color: {THEME_COLORS['text']}; border-radius: 10px;"
                                    " padding: 10px 14px; font-size: 13px;")
            self.tips.setText(
                "<b>How to drive the arm</b><br>"
                f"{keycap('&larr;')} {keycap('&rarr;')} {keycap('&uarr;')} {keycap('&darr;')} &nbsp;move the gripper<br>"
                f"{keycap('I')} {keycap('K')} &nbsp;move forward / back<br>"
                f"{keycap('O')} open gripper &nbsp;&nbsp;{keycap('C')} close gripper<br>"
                f"{keycap('Z')} home &nbsp;&nbsp;{keycap('Space')} stop<br>"
                f"<span style='color:#ff8a8a'>{keycap('Esc')} <b>EMERGENCY STOP</b></span><br>"
                f"{keycap('F11')} leave full screen &nbsp;&nbsp;{keycap('F1')} full help")
            self.tips.adjustSize()
            self.tips.move(16, 16)

            bar = QFrame()
            bar.setStyleSheet(f"background:{THEME_COLORS['title_bar']};")
            vb = QVBoxLayout(bar)
            vb.setContentsMargins(8, 6, 8, 6)
            vb.setSpacing(6)
            # ---- row 1: arm
            row = QHBoxLayout()
            btn = QPushButton("Exit full screen  (F11)")
            btn.clicked.connect(dash.toggle_fullscreen)
            row.addWidget(btn)
            estop = QPushButton("EMERGENCY STOP  (Esc)")
            estop.setStyleSheet(f"background-color: {THEME_COLORS['special']}; font-weight: 700; padding: 6px 14px;")
            estop.clicked.connect(dash.emergency_stop)
            row.addWidget(estop)
            for text, fn in (("Home  (Z)", lambda: dash.send(MotionCommand.home(source="fullscreen"))),
                             ("Reset  (X)", lambda: dash.send(MotionCommand(RESET, source="fullscreen")))):
                b = QPushButton(text)
                b.clicked.connect(fn)
                row.addWidget(b)
            self.btn_tips = QPushButton("Tips")
            self.btn_tips.setCheckable(True)
            self.btn_tips.setChecked(True)
            self.btn_tips.setToolTip("Show / hide the on-screen operating instructions")
            self.btn_tips.toggled.connect(self.tips.setVisible)
            row.addWidget(self.btn_tips)
            self.lbl = QLabel("")
            self.lbl.setStyleSheet("font-weight: 600;")
            row.addWidget(self.lbl, 1)
            vb.addLayout(row)
            # ---- row 2: objects
            row = QHBoxLayout()
            row.addWidget(QLabel("Objects:"))
            self.combo_shape = QComboBox()
            for text, kind in self.SHAPES:
                self.combo_shape.addItem(text, kind)
            self.combo_shape.setToolTip("Shape of the next object")
            row.addWidget(self.combo_shape)
            self.btn_add = QPushButton("Add object")
            self.btn_add.setToolTip(f"Drop one object of this shape at a random spot the arm can reach (max {MAX_OBJECTS})")
            self.btn_add.clicked.connect(self.add_object)
            row.addWidget(self.btn_add)
            for text, tip, fn in (("Clear all", "Remove every object from the table", self.clear_objects),
                                  ("Reset objects", "Back to the starting cube, sphere and cylinder", self.reset_objects)):
                b = QPushButton(text)
                b.setToolTip(tip)
                b.clicked.connect(fn)
                row.addWidget(b)
            self.lbl_count = QLabel("")
            self.lbl_count.setStyleSheet("font-weight: 600;")
            row.addWidget(self.lbl_count)
            self.lbl_msg = QLabel("")
            self.lbl_msg.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            row.addWidget(self.lbl_msg, 1)
            vb.addLayout(row)
            lay.addWidget(bar)
            self.update_count()

        # ---- objects
        def update_count(self):
            n = len(self.dash.rt.env.objects)
            self.lbl_count.setText(f"{n} / {MAX_OBJECTS}")
            self.btn_add.setEnabled(n < MAX_OBJECTS)

        def add_object(self):
            kind = self.combo_shape.currentData()
            obj = self.dash.rt.add_random_object(kind)
            if obj is not None:
                self.lbl_msg.setText(f"Added a {kind} where the arm can reach it.")
            elif len(self.dash.rt.env.objects) >= MAX_OBJECTS:
                self.lbl_msg.setText(f"The table is full ({MAX_OBJECTS} objects). Clear some first.")
            else:
                self.lbl_msg.setText("No free spot within reach - clear an object first.")
            self.update_count()

        def clear_objects(self):
            self.dash.rt.clear_objects()
            self.lbl_msg.setText("Table cleared.")
            self.update_count()

        def reset_objects(self):
            self.dash.rt.reset_objects()
            self.lbl_msg.setText("Back to the starting objects.")
            self.update_count()

        def closeEvent(self, ev):
            self.view.close_renderer()
            ev.accept()
            if self.dash.fs is self:
                self.dash.fs = None


    class ArmDashboard(QMainWindow):
        """BioWave-style control centre for the simulated arm.

        connect device -> load model -> calibrate -> map gestures to arm actions -> enable arm control,
        with live EMG, prediction, signal-quality and arm telemetry on one screen.
        """

        PILL_OFF = f"background:{THEME_COLORS['title_bar']}; color:{THEME_COLORS['muted']};"
        PILL_OK = f"background:{THEME_COLORS['accent']}; color:{THEME_COLORS['text']};"
        PILL_BAD = f"background:{THEME_COLORS['special']}; color:{THEME_COLORS['text']};"
        PILL_WARN = "background:#f57c00; color:#ffffff;"

        def __init__(self, runtime: "SimulationRuntime", fps: float = 30.0, max_mpix: float = 0.70,
                     shadows: bool = True, frames: bool = False):
            super().__init__()
            self.rt = runtime
            self.view_fps, self.view_max_mpix, self.view_shadows, self.view_frames = fps, max_mpix, shadows, frames
            self.setWindowTitle("BioWave - Robotic Arm Control Centre")
            self.setMinimumSize(980, 620)
            screen = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
            w, h = (1360, 860) if screen is None else (min(1360, int(screen.width() * .96)), min(860, int(screen.height() * .94)))
            self.resize(w, h)
            if screen is not None:                       # open fully on-screen (macOS would otherwise shrink the window)
                self.move(screen.x() + max(0, (screen.width() - w) // 2), screen.y())
            self.setStyleSheet(ux_stylesheet(12))
            apply_dark_title_bar(self)
            self._settings = QSettings("BioWave", "RoboticArm")

            # ---- connection / stream state
            self.worker = None
            self.connection_medium = ""
            self.is_connected = False
            self.current_device = None
            self.discovered_devices = []
            self.wireless_access_key = DEFAULT_DEVICE_ACCESS_KEY
            self.num_channels = WIRELESS_TOTAL_CHANNELS
            self.emg_channel_count = WIRELESS_EMG_CHANNELS
            self.keepalive_failures = 0
            self.last_batch_received_monotonic = 0.0
            self.stream_invalid_until = 0.0
            # ---- model state
            self.model_loaded = False
            self.rf_class_names = []
            self.rf_window_samples = 100
            self.rf_stride_samples = 25
            self.rf_model_input_channels = WIRELESS_TOTAL_CHANNELS
            self.rf_model_sample_rate = SAMPLE_RATE
            self.rf_preprocessing_version = LEGACY_PREPROCESSING_VERSION
            self.model_compatible = False
            self.model_compatibility_message = "No model loaded."
            self.model_expected_feature_count = None
            # ---- calibration state
            self.is_calibrated = False
            self.calibration_active = False
            self.calibration_dialog = None
            self.cal_rest_seconds, self.cal_flex_seconds = CAL_REST_MS // 1000, CAL_FLEX_MS // 1000
            self.calibration_phases, self.current_cal_phase_idx = [], -1
            self.current_phase_key, self.current_phase_remaining_ms, self.current_phase_total_ms = "", 0, 0
            self.rest_capture, self.flex_capture = [], []
            self.baseline_offsets = np.zeros(1, dtype=np.float32)
            self.calibration_profile = None
            self.preprocessor = None
            self.calibration_timer = QTimer(self)
            self.calibration_timer.timeout.connect(self.on_calibration_tick)
            # ---- streaming buffers
            self.sample_ring = None
            self.profiler = StageProfiler()
            self._buffer_lock = threading.RLock()
            self.rf_valid_sample_count = 0
            self.rf_samples_since_submit = 0
            self.last_signal_quality = None
            self.plot_buf = np.zeros((WIRELESS_TOTAL_CHANNELS, PLOT_SAMPLES), dtype=np.float32)
            self.plot_filled = 0
            self.packets_in_last_second, self._rate_t, self._rate_n, self.sample_rate_measured = 0, time.monotonic(), 0, 0.0
            # ---- arm control
            self.bridge = EMGArmBridge(self.rt.controller.submit, self.rt.controller.set_emg_status,
                                       speed_m_s=EMG_SPEED_M_S)
            self.mapping_combos = {}
            # ---- keyboard (same keys as before, now handled by this window)
            self.kb = KeyboardController(self._send_key, self.rt.mode, None, demo=self.rt.start_demo)
            self.fs = None                                  # FullscreenView while open
            self.emg_win = None                             # EMGWindow once opened
            self.help_dlg = None
            self._held, self._trig = set(), set()
            self._last_key_action, self._last_key_t = "", 0.0

            self.init_ui()

            self.inference_worker = InferenceWorker(SAMPLE_RATE)
            self.inference_worker.prediction_ready.connect(self.on_prediction_ready)
            self.inference_worker.inference_error.connect(self.on_inference_error)
            self.inference_worker.start()

            self.keepalive_timer = QTimer(self)
            self.keepalive_timer.timeout.connect(self.send_wireless_keepalive)
            self.keepalive_timer.start(KEEPALIVE_INTERVAL_MS)
            self.ui_timer = QTimer(self)                    # 15 Hz: arm telemetry, stream watchdog, plot
            self.ui_timer.timeout.connect(self.refresh_ui)
            self.ui_timer.start(66)
            self.render_timer = QTimer(self)                # 3D view
            self.render_timer.timeout.connect(self._render_tick)
            self.render_timer.start(int(1000 / max(1.0, self.view_fps)))
            self.key_timer = QTimer(self)                   # 30 Hz: held keys -> jog commands
            self.key_timer.timeout.connect(self.poll_keys)
            self.key_timer.start(33)
            QApplication.instance().installEventFilter(self)

        # ================================================================== UI construction
        def _group(self, title):
            g = QGroupBox(title)
            lay = QVBoxLayout(g)
            lay.setSpacing(8)
            return g, lay

        def _pill(self, text="-"):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumWidth(100)
            lbl.setStyleSheet(self.PILL_OFF + " border-radius: 11px; padding: 3px 8px; font-weight: 600;")
            return lbl

        @staticmethod
        def _pred_style(color):
            return f"color: {color}; font-size: 24px; font-weight: 700;"

        def _set_pill(self, pill, text, style):
            pill.setText(text)
            pill.setStyleSheet(style + " border-radius: 11px; padding: 3px 8px; font-weight: 600;")

        def init_ui(self):
            central = QWidget()
            root = _NoHfwLayout(central)
            root.setSpacing(10)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(central)
            self.setCentralWidget(scroll)

            # ---- status strip
            strip = QHBoxLayout()
            title = QLabel("BioWave  |  6-DOF Robotic Arm")
            title.setStyleSheet("font-size: 16px; font-weight: 700;")
            strip.addWidget(title)
            strip.addStretch(1)
            self.pill_device, self.pill_model = self._pill("Device: off"), self._pill("Model: none")
            self.pill_cal, self.pill_control = self._pill("Calibration: no"), self._pill("Arm control: OFF")
            self.pill_arm = self._pill("Arm: ready")
            for p in (self.pill_device, self.pill_model, self.pill_cal, self.pill_control, self.pill_arm):
                strip.addWidget(p)
            self.btn_help = QPushButton("Help  (F1)")
            self.btn_help.setToolTip("Quick start, keyboard map, EMG tips, safety and troubleshooting")
            self.btn_help.clicked.connect(self.show_help)
            strip.addWidget(self.btn_help)
            root.addLayout(strip)
            self.lbl_next = QLabel("")
            self.lbl_next.setWordWrap(True)
            self.lbl_next.setStyleSheet(f"background:{THEME_COLORS['panel']}; border-left: 4px solid {THEME_COLORS['accent']};"
                                        " padding: 6px 10px; font-weight: 600;")
            root.addWidget(self.lbl_next)

            body = QHBoxLayout()
            body.setSpacing(14)
            left, middle, right = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
            for c in (left, middle, right):
                c.setSpacing(10)
            body.addLayout(left, 4)
            body.addLayout(middle, 6)
            body.addLayout(right, 4)
            root.addLayout(body, 1)

            # ============ LEFT: setup (connect -> model -> calibrate)
            g, lay = self._group("1  Connect Device")
            self.tabs = QTabWidget()
            self.tabs.addTab(self._build_wireless_tab(), "Wi-Fi")
            self.tabs.addTab(self._build_wired_tab(), "Wired USB")
            lay.addWidget(self.tabs)
            self.lbl_conn_status = QLabel("Status: Disconnected")
            self.lbl_conn_status.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.lbl_conn_status)
            left.addWidget(g)

            g, lay = self._group("2  Load Pretrained Model")
            row = QHBoxLayout()
            self.txt_model_path = QLineEdit()
            self.txt_model_path.setReadOnly(True)
            self.txt_model_path.setPlaceholderText("rf_realtime_model.joblib")
            row.addWidget(self.txt_model_path, 1)
            btn = QPushButton("Browse .joblib")
            btn.clicked.connect(self.browse_model)
            row.addWidget(btn)
            lay.addLayout(row)
            self.lbl_model_info = QLabel("No model loaded.")
            self.lbl_model_info.setWordWrap(True)
            self.lbl_model_info.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(self.lbl_model_info)
            left.addWidget(g)

            g, lay = self._group("3  Calibrate")
            form = QFormLayout()
            self.spin_rest_sec, self.spin_flex_sec = QSpinBox(), QSpinBox()
            durations = QHBoxLayout()
            for sp, val, lab in ((self.spin_rest_sec, self.cal_rest_seconds, "Rest:"),
                                 (self.spin_flex_sec, self.cal_flex_seconds, "Flex:")):
                sp.setRange(CAL_DURATION_MIN_S, CAL_DURATION_MAX_S)
                sp.setValue(val)
                sp.setSuffix(" sec")
                sp.setToolTip(f"How long the {lab[:-1].upper()} phase of the calibration lasts")
                durations.addWidget(QLabel(lab))
                durations.addWidget(sp, 1)
            lay.addLayout(durations)
            self.btn_calibrate = QPushButton("Calibrate")
            self.btn_calibrate.setEnabled(False)
            self.btn_calibrate.clicked.connect(self.start_calibration_sequence)
            lay.addWidget(self.btn_calibrate)
            self.lbl_cal_status = QLabel("Connect a device and load a model to calibrate.")
            self.lbl_cal_status.setWordWrap(True)
            self.lbl_cal_status.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(self.lbl_cal_status)
            left.addWidget(g)

            self.btn_map = QPushButton("")
            self.btn_map.setCheckable(True)
            self.btn_map.setToolTip("Choose which arm action each gesture of your model triggers")
            self.btn_map.toggled.connect(self._toggle_mapping)
            left.addWidget(self.btn_map)
            g, lay = self._group("Gesture to Arm-Action Mapping")
            self.grp_map = g
            g.setVisible(False)
            self.scroll_map = QScrollArea()
            self.scroll_map.setWidgetResizable(True)
            self.scroll_map.setMinimumHeight(90)
            self.map_content = QWidget()
            self.map_form = QFormLayout(self.map_content)
            self.scroll_map.setWidget(self.map_content)
            lay.addWidget(self.scroll_map)
            self.lbl_map_hint = QLabel("Load a model to see its gestures.")
            self.lbl_map_hint.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            self.map_form.addRow(self.lbl_map_hint)
            left.addWidget(g, 1)
            left.addStretch(1)
            self._update_map_button()

            # ============ MIDDLE: 3D view + live gesture + EMG
            g, lay = self._group("3D View")
            g.setToolTip("drag: orbit   |   right-drag or shift-drag: pan   |   wheel: zoom   |   double-click: reset view")
            self.view = ViewWidget(self.rt.backend, self.view_max_mpix, self.view_shadows)
            self.view.frames = self.view_frames
            lay.addWidget(self.view, 1)
            row = QHBoxLayout()
            self.chk_frames = QCheckBox("Frames")
            self.chk_frames.setChecked(self.view_frames)
            self.chk_frames.toggled.connect(self.view.set_frames)
            self.chk_shadows = QCheckBox("Shadows")
            self.chk_shadows.setChecked(self.view_shadows)
            self.chk_shadows.toggled.connect(self.view.set_shadows)
            btn = QPushButton("Reset")
            btn.setToolTip("Back to the default camera angle (also: double-click the view)")
            btn.clicked.connect(self.view.reset_view)
            self.btn_fullscreen = QPushButton("Full screen")
            self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
            self.lbl_fps = QLabel("")
            self.lbl_fps.setToolTip("3D view: drag = orbit, right-drag = pan, wheel = zoom, double-click = reset")
            self.lbl_fps.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            for w_ in (self.chk_frames, self.chk_shadows, btn, self.btn_fullscreen):
                row.addWidget(w_)
            row.addStretch(1)
            row.addWidget(self.lbl_fps)
            lay.addLayout(row)
            self.lbl_keys = QLabel("")
            self.lbl_keys.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            self.lbl_keys.setWordWrap(True)
            self.lbl_keys.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            lay.addWidget(self.lbl_keys)
            middle.addWidget(g, 5)

            g, lay = self._group("Live Gesture")
            row = QHBoxLayout()
            self.lbl_prediction = QLabel("REST")
            self.lbl_prediction.setMinimumWidth(170)
            self.lbl_prediction.setStyleSheet(self._pred_style(THEME_COLORS['disabled']))
            row.addWidget(self.lbl_prediction)
            col = QVBoxLayout()
            self.bar_conf = QProgressBar()
            self.bar_conf.setRange(0, 100)
            self.bar_conf.setTextVisible(False)
            col.addWidget(self.bar_conf)
            self.lbl_action = QLabel("Arm action: -")
            col.addWidget(self.lbl_action)
            row.addLayout(col, 1)
            lay.addLayout(row)
            self.lbl_diagnostics = QLabel("Signal: WAITING | Calibration: NOT VALID | Packet loss: n/a")
            self.lbl_diagnostics.setWordWrap(True)
            self.lbl_diagnostics.setAlignment(Qt.AlignCenter)
            self.lbl_diagnostics.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(self.lbl_diagnostics)
            self.btn_control = QPushButton("ENABLE ARM CONTROL")
            self.btn_control.setCheckable(True)
            self.btn_control.setEnabled(False)
            self.btn_control.setStyleSheet("background-color: #2e7d32; font-size: 16px; padding: 10px;")
            self.btn_control.toggled.connect(self.toggle_arm_control)
            lay.addWidget(self.btn_control)
            self.lbl_status = QLabel("Status: Idle")
            self.lbl_status.setAlignment(Qt.AlignCenter)
            self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(self.lbl_status)
            row = QHBoxLayout()
            self.btn_emg = QPushButton("Show EMG signals")
            self.btn_emg.setCheckable(True)
            self.btn_emg.setToolTip("Open a large window with the live signal of every EMG channel")
            self.btn_emg.toggled.connect(self.toggle_emg_window)
            row.addWidget(self.btn_emg)
            self.lbl_stream = QLabel("No stream.")
            self.lbl_stream.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            row.addWidget(self.lbl_stream, 1)
            lay.addLayout(row)
            middle.addWidget(g)

            # ============ RIGHT: arm state, commands, jog / test / settings
            g, lay = self._group("Arm State")
            self.lbl_tcp = QLabel("X 0.0   Y 0.0   Z 0.0 mm")
            self.lbl_tcp.setStyleSheet("font-size: 16px; font-weight: 600;")
            lay.addWidget(self.lbl_tcp)
            self.lbl_joints = QLabel("J1..J5: -")
            self.lbl_joints.setWordWrap(True)
            lay.addWidget(self.lbl_joints)
            self.bar_gripper = QProgressBar()
            self.bar_gripper.setRange(0, 100)
            self.bar_gripper.setFormat("Gripper %p% open")
            lay.addWidget(self.bar_gripper)
            self.lbl_safety = QLabel("Safety: OK")
            self.lbl_safety.setWordWrap(True)
            lay.addWidget(self.lbl_safety)
            self.lbl_loop = QLabel("")
            self.lbl_loop.setWordWrap(True)
            self.lbl_loop.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(self.lbl_loop)
            right.addWidget(g)

            g, lay = self._group("Arm Commands")
            self.btn_estop = QPushButton("EMERGENCY STOP   (Esc)")
            self.btn_estop.setStyleSheet(f"background-color: {THEME_COLORS['special']}; font-size: 16px; padding: 12px;")
            self.btn_estop.clicked.connect(self.emergency_stop)
            lay.addWidget(self.btn_estop)
            grid = QGridLayout()
            for i, (text, fn) in enumerate((("Home  (Z)", lambda: self.send(MotionCommand.home(source="dashboard"))),
                                            ("Stop  (Space)", lambda: self.send(MotionCommand.stop(source="dashboard"))),
                                            ("Reset  (X)", lambda: self.send(MotionCommand(RESET, source="dashboard"))),
                                            ("Pick/place  (P)", self.rt.start_demo),
                                            ("Open  (O)", lambda: self.send(MotionCommand.gripper("OPEN", source="dashboard"))),
                                            ("Close  (C)", lambda: self.send(MotionCommand.gripper("CLOSE", source="dashboard"))))):
                b = QPushButton(text)
                b.clicked.connect(fn)
                grid.addWidget(b, i // 2, i % 2)
            lay.addLayout(grid)
            right.addWidget(g)

            tabs = QTabWidget()
            self.tabs_right = tabs
            # -- keyboard guide
            w_ = QWidget()
            kl = QVBoxLayout(w_)
            kl.setContentsMargins(4, 4, 4, 4)
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            lab = QLabel(key_guide_html(compact=True))
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            sc.setWidget(lab)
            kl.addWidget(sc, 1)
            tip = QLabel("Click the window first. Hold ONE movement key at a time. Esc = emergency stop.")
            tip.setWordWrap(True)
            tip.setStyleSheet(f"color: {THEME_COLORS['muted']}; font-size: 11px;")
            kl.addWidget(tip)
            tabs.addTab(w_, "Keys")
            # -- manual jog
            w_ = QWidget()
            pad = QGridLayout(w_)
            jog = {"Up +Z": ("UP", 0, 1), "Left -X": ("LEFT", 1, 0), "Right +X": ("RIGHT", 1, 2),
                   "Down -Z": ("DOWN", 2, 1), "Fwd +Y": ("FORWARD", 0, 2), "Back -Y": ("BACKWARD", 2, 0)}
            for text, (direction, r, c) in jog.items():
                b = HoldButton(text, lambda d=direction: self.send(MotionCommand(
                    CARTESIAN, d, speed=self.spin_speed_jog.value() / 1000.0, source="dashboard")),
                    lambda: self.send(MotionCommand.hold(source="dashboard")))
                pad.addWidget(b, r, c)
            self.spin_speed_jog = QSpinBox()
            self.spin_speed_jog.setRange(5, 120)
            self.spin_speed_jog.setValue(60)
            self.spin_speed_jog.setSuffix(" mm/s")
            pad.addWidget(QLabel("Jog speed:"), 3, 0)
            pad.addWidget(self.spin_speed_jog, 3, 1, 1, 2)
            note = QLabel("Hold a button to move the gripper (same as the arrow keys and I / K).")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {THEME_COLORS['muted']}; font-size: 11px;")
            pad.addWidget(note, 4, 0, 1, 3)
            tabs.addTab(w_, "Jog")
            # -- test without device
            w_ = QWidget()
            row = QGridLayout(w_)
            for i, (text, gesture) in enumerate((("Left", "left"), ("Right", "right"), ("Up", "up"),
                                                 ("Down", "down"), ("Fist", "fist_close"), ("Rest", "rest"))):
                b = QPushButton(text)
                b.pressed.connect(lambda g_=gesture: self.mock_gesture(g_))
                b.released.connect(lambda: self.mock_gesture(None))
                row.addWidget(b, i // 3, i % 3)
            note = QLabel("Hold a button (or keys 1-5, 0): simulated gestures run through the same smoothing + mapping "
                          "path as the armband.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {THEME_COLORS['muted']}; font-size: 11px;")
            row.addWidget(note, 2, 0, 1, 3)
            tabs.addTab(w_, "Test")
            # -- control settings
            w_ = QWidget()
            form = QFormLayout(w_)
            eng = self.bridge.engine
            self.spin_conf = QDoubleSpinBox()
            self.spin_conf.setRange(10.0, 99.9)
            self.spin_conf.setValue(GESTURE_MIN_CONFIDENCE_DEFAULT)
            self.spin_conf.setSuffix("%")
            self.spin_conf.valueChanged.connect(lambda v: setattr(self.bridge.engine, "min_confidence", v / 100.0))
            eng.min_confidence = GESTURE_MIN_CONFIDENCE_DEFAULT / 100.0
            form.addRow("Minimum Confidence:", self.spin_conf)
            self.spin_margin = QDoubleSpinBox()
            self.spin_margin.setRange(0.0, 50.0)
            self.spin_margin.setValue(eng.min_margin * 100.0)
            self.spin_margin.setSuffix(" pts")
            self.spin_margin.setToolTip("Gap required between the top prediction and the runner-up. "
                                        "Raise it if Rest keeps triggering moves.")
            self.spin_margin.valueChanged.connect(lambda v: setattr(self.bridge.engine, "min_margin", v / 100.0))
            form.addRow("Confidence Margin:", self.spin_margin)
            self.spin_debounce = QSpinBox()
            self.spin_debounce.setRange(1, 10)
            self.spin_debounce.setValue(eng.consecutive_required)
            self.spin_debounce.setSuffix(" frames")
            self.spin_debounce.valueChanged.connect(lambda v: setattr(self.bridge.engine, "consecutive_required", v))
            form.addRow("Debounce Count:", self.spin_debounce)
            self.spin_speed = QSpinBox()
            self.spin_speed.setRange(5, 120)
            self.spin_speed.setValue(int(round(EMG_SPEED_M_S * 1000)))
            self.spin_speed.setSuffix(" mm/s")
            self.spin_speed.valueChanged.connect(lambda v: setattr(self.bridge, "speed", v / 1000.0))
            form.addRow("EMG Arm Speed:", self.spin_speed)
            self.spin_grip_refr = QDoubleSpinBox()
            self.spin_grip_refr.setRange(0.1, 5.0)
            self.spin_grip_refr.setValue(eng.refractory_s)
            self.spin_grip_refr.setSuffix(" sec")
            self.spin_grip_refr.valueChanged.connect(lambda v: setattr(self.bridge.engine, "refractory_s", v))
            form.addRow("Gripper Cooldown:", self.spin_grip_refr)
            tabs.addTab(w_, "Settings")
            right.addWidget(tabs)
            right.addStretch(1)

            if not HAS_PYQTGRAPH:
                self.btn_emg.setEnabled(False)
                self.btn_emg.setToolTip("pyqtgraph is missing: pip install pyqtgraph")
            self._apply_tooltips()
            self.refresh_guidance()

        def _build_wireless_tab(self):
            w = QWidget()
            lay = QVBoxLayout(w)
            row = QHBoxLayout()
            row.addWidget(QLabel("Access Key:"))
            self.txt_access_key = QLineEdit(self.wireless_access_key)
            self.txt_access_key.setMinimumWidth(60)
            self.txt_access_key.setEchoMode(QLineEdit.Password)
            row.addWidget(self.txt_access_key, 1)
            lay.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel("Device:"))
            self.combo_devices = QComboBox()
            self.combo_devices.setMinimumContentsLength(10)
            self.combo_devices.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            row.addWidget(self.combo_devices, 1)
            btn = QPushButton("Discover")
            btn.clicked.connect(self.discover_wireless_devices)
            row.addWidget(btn)
            lay.addLayout(row)
            self.lbl_device_info = QLabel("No wireless device discovered yet. Make sure this computer is on the same Wi-Fi as the armband.")
            self.lbl_device_info.setWordWrap(True)
            self.lbl_device_info.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            lay.addWidget(self.lbl_device_info)
            self.combo_devices.currentIndexChanged.connect(self._refresh_device_info)
            row = QHBoxLayout()
            btn = QPushButton("Provision (USB)")
            btn.setToolTip("First-time setup of a new armband: send it your Wi-Fi name and password over USB")
            btn.clicked.connect(lambda: ProvisionDialog(self).exec_())
            row.addWidget(btn)
            self.btn_wireless_connect = QPushButton("Connect Wireless")
            self.btn_wireless_connect.clicked.connect(self.toggle_wireless_connection)
            row.addWidget(self.btn_wireless_connect, 1)
            lay.addLayout(row)
            return w

        def _build_wired_tab(self):
            w = QWidget()
            lay = QVBoxLayout(w)
            row = QHBoxLayout()
            row.addWidget(QLabel("Port:"))
            self.combo_ports = QComboBox()
            row.addWidget(self.combo_ports, 1)
            btn = QPushButton("Refresh")
            btn.clicked.connect(self.refresh_wired_ports)
            row.addWidget(btn)
            lay.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel("Channels (EMG + IMU):"))
            self.spin_wired_channels = QSpinBox()
            self.spin_wired_channels.setRange(2, WIRELESS_TOTAL_CHANNELS)
            self.spin_wired_channels.setValue(WIRELESS_TOTAL_CHANNELS)
            row.addWidget(self.spin_wired_channels)
            row.addStretch()
            lay.addLayout(row)
            self.btn_wired_connect = QPushButton("Connect")
            self.btn_wired_connect.clicked.connect(self.toggle_wired_connection)
            lay.addWidget(self.btn_wired_connect)
            self.refresh_wired_ports()
            return w

        # ================================================================== keyboard
        @staticmethod
        def _key_name(ev):
            special = {Qt.Key_Left: "left", Qt.Key_Right: "right", Qt.Key_Up: "up", Qt.Key_Down: "down",
                       Qt.Key_Escape: "esc", Qt.Key_Space: " "}
            if ev.key() in special:
                return special[ev.key()]
            t = ev.text().lower()
            return t if len(t) == 1 and t.isprintable() else None

        def _active(self) -> bool:
            w = QApplication.activeWindow()
            return self.isActiveWindow() or (w is not None and w is self.fs)

        def _send_key(self, cmd) -> bool:
            """Keyboard jogs follow the 'Jog speed' setting (the planner default is very slow)."""
            if cmd.command_type == CARTESIAN and cmd.speed is None:
                cmd.speed = self.spin_speed_jog.value() / 1000.0
            return self.send(cmd)

        def eventFilter(self, obj, ev):
            if ev.type() not in (QEvent.KeyPress, QEvent.KeyRelease) or not self._active() or ev.isAutoRepeat():
                return False
            if ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_F11, Qt.Key_F1):
                (self.toggle_fullscreen if ev.key() == Qt.Key_F11 else self.show_help)()
                return True
            if isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox)):
                return False                                # typing in a text box / spin box: not a robot key
            name = self._key_name(ev)
            if name is None or ev.modifiers() & (Qt.ControlModifier | Qt.MetaModifier | Qt.AltModifier):
                return False
            gesture = KEY_TO_GESTURE.get(name) if self.rt.mode in ("emg", "full") else None
            known = name in KEY_TO_GESTURE or name in JOINT_KEYS or name in CARTESIAN_KEYS or name in ("esc", " ", "o", "c", "z", "x", "p")
            if not known:
                return False
            if ev.type() == QEvent.KeyPress:
                self._held.add(name)
                self._trig.add(name)
                self._last_key_action, self._last_key_t = KEY_ACTIONS.get(name, name), time.monotonic()
                if gesture:
                    self.mock_gesture(gesture)
            else:
                self._held.discard(name)
                if gesture and not any(k in KEY_TO_GESTURE for k in self._held):
                    self.mock_gesture(None)
            return True

        def poll_keys(self):
            if not self._active():
                if self._held:
                    self._held.clear()
                    self.mock_gesture(None)
                return
            trig, self._trig = self._trig, set()
            self.kb.update({k for k in self._held if k not in KEY_TO_GESTURE}, {k for k in trig if k not in KEY_TO_GESTURE})

        def _key_hint_text(self) -> str:
            if self._held:
                return "Keyboard: " + " + ".join(KEY_ACTIONS.get(k, k) for k in sorted(self._held))
            if self._last_key_t and time.monotonic() - self._last_key_t < 2.0:
                return f"Keyboard: {self._last_key_action}"
            return "Keys: arrows move  |  O / C gripper  |  Esc E-STOP  |  F11 full screen  |  F1 help"

        # ================================================================== helpers
        def send(self, cmd) -> bool:
            return self.rt.controller.submit(cmd)

        def mock_gesture(self, gesture):
            src = self.rt.kb_source
            if src is None:
                self.lbl_status.setText("Status: mock gestures need --mode full or emg")
                return
            src.set_gesture(gesture) if gesture else src.release()

        def emergency_stop(self):
            self.btn_control.setChecked(False)
            self.send(MotionCommand(ESTOP, source="dashboard"))
            self.lbl_status.setText("Status: EMERGENCY STOP - press 'Reset after stop' (X) to continue")

        # ---- full screen / EMG window / help
        def toggle_fullscreen(self):
            if self.fs is not None:
                fs, self.fs = self.fs, None
                fs.close()
                self.activateWindow()
                return
            self.fs = FullscreenView(self)
            self.fs.show()
            self.fs.showFullScreen()
            self.fs.activateWindow()

        def _render_tick(self):
            (self.fs.view if self.fs is not None else self.view).render_frame()

        def toggle_emg_window(self, on):
            if not on:
                if self.emg_win is not None:
                    self.emg_win.hide()
                return
            if self.emg_win is None:
                self.emg_win = EMGWindow(self)
            self.emg_win.show()
            self.emg_win.raise_()
            self.emg_win.refresh()

        def show_help(self):
            if self.help_dlg is None:
                self.help_dlg = HelpDialog(self)
            self.help_dlg.show()
            self.help_dlg.raise_()

        def refresh_guidance(self):
            """One plain-language 'what to do next' line, always visible under the title."""
            if self.rt.controller.estopped:
                msg = "EMERGENCY STOP is active. Press 'Reset after stop' (X) to re-arm the arm."
            elif self.calibration_active:
                msg = "Calibrating: follow the REST / FLEX prompts. Keep the armband still on your arm."
            elif self.bridge.enabled:
                msg = "Arm control is ON: hold a gesture to move the arm, relax to hold still. Esc = emergency stop."
            elif not self.is_connected and not self.model_loaded:
                msg = ("Step 1: connect your armband (left panel).   No armband? Use the keyboard (Keys tab) "
                       "or the 'Test' tab. Press F1 for the full guide.")
            elif not self.is_connected:
                msg = "Step 1: connect your armband (Wireless: Discover, access key, Connect)."
            elif not self.model_loaded:
                msg = "Step 2: load your trained model (Browse .joblib)."
            elif not self.model_compatible:
                msg = "The model does not match this device: " + self.model_compatibility_message.splitlines()[0]
            elif not self.is_calibrated:
                msg = "Step 3: click Calibrate. Relax your arm during REST, then squeeze steadily during FLEX."
            else:
                msg = "Step 4: check the gesture mapping (button in the left panel), then press ENABLE ARM CONTROL."
            if self.lbl_next.text() != msg:
                self.lbl_next.setText(msg)

        def _apply_tooltips(self):
            tips = {
                self.btn_calibrate: "Records your relaxed (REST) and squeezed (FLEX) signal to set the baseline and check channel quality.",
                self.btn_control: "Lets recognised gestures move the arm. Switches itself off if the signal or link fails.",
                self.btn_estop: "Stops everything immediately. Latches until 'Reset after stop'. Shortcut: Esc.",
                self.spin_conf: "How sure the model must be before the arm acts. Raise it if the arm twitches.",
                self.spin_margin: "The best gesture must beat the second-best by this many points.",
                self.spin_debounce: "How many predictions in a row must agree before the arm moves.",
                self.spin_speed: "Arm speed when driven by gestures.",
                self.spin_grip_refr: "Minimum time between gripper open/close actions.",
                self.spin_speed_jog: "Speed of the jog buttons and the keyboard arrow keys.",
                self.chk_frames: "Draw the coordinate axes of every joint.",
                self.chk_shadows: "Turn off for a faster, cooler 3D view on a fanless laptop.",
                self.btn_fullscreen: "Show the 3D view full screen. Leave with F11 or the Exit button.",
                self.txt_access_key: "The device access key set in the armband firmware (DEVICE_ACCESS_KEY).",
                self.btn_wireless_connect: "Start streaming from the selected armband over Wi-Fi.",
                self.btn_wired_connect: "Start streaming from the selected serial port.",
                self.combo_devices: "Armbands found on this Wi-Fi network. Click Discover to search.",
            }
            for w_, t in tips.items():
                w_.setToolTip(t)

        # ================================================================== wired
        def refresh_wired_ports(self):
            self.combo_ports.clear()
            self.combo_ports.addItem("socket://127.0.0.1:7000 (Simulator)", "socket://127.0.0.1:7000")
            if HAS_SERIAL:
                for p in serial.tools.list_ports.comports():
                    self.combo_ports.addItem(f"{p.device} - {p.description}", p.device)

        def toggle_wired_connection(self):
            self.disconnect_stream() if self.is_connected else self.connect_wired()

        def connect_wired(self):
            if not HAS_SERIAL:
                QMessageBox.warning(self, "Missing Library", "pyserial not found. Run: pip install pyserial")
                return
            port = self.combo_ports.currentData() or (self.combo_ports.currentText().split() or [""])[0]
            if not port:
                QMessageBox.warning(self, "No Port", "Please select a valid serial port.")
                return
            self.num_channels = int(self.spin_wired_channels.value())
            self.emg_channel_count = min(WIRELESS_EMG_CHANNELS, self.num_channels)
            self._reset_stream_state()
            self.worker = SerialWorker(port, DEFAULT_BAUD_RATE, self.num_channels, batch_size=25)
            self.worker.batch_received.connect(self.on_stream_batch)
            self.worker.error_occurred.connect(self.on_stream_error)
            self.worker.start()
            self.connection_medium, self.is_connected = "wired", True
            self.btn_wired_connect.setText("Disconnect")
            self.btn_wireless_connect.setEnabled(False)
            self._set_connected_text(f"Connected (wired, {self.num_channels} ch)")

        # ================================================================== wireless
        def discover_wireless_devices(self):
            self.lbl_device_info.setText("Searching the network...")
            QApplication.processEvents()
            try:
                self.discovered_devices = ControlProtocol.discover()
            except Exception as exc:
                QMessageBox.warning(self, "Discovery Failed", str(exc))
                return
            self.combo_devices.clear()
            for d in self.discovered_devices:
                self.combo_devices.addItem(d.summary, d)
            if not self.discovered_devices:
                self.lbl_device_info.setText("No BioWave wireless devices replied on the current Wi-Fi.")
            self._refresh_device_info()

        def _refresh_device_info(self):
            d = self.combo_devices.currentData()
            if isinstance(d, DeviceInfo):
                self.lbl_device_info.setText(f"{d.device_name} | IP={d.ip} | Mode={d.wifi_mode} | FW={d.firmware} | IMU ready={d.imu_ready}")

        def toggle_wireless_connection(self):
            self.disconnect_stream() if self.is_connected else self.connect_wireless()

        def connect_wireless(self):
            device = self.combo_devices.currentData()
            if not isinstance(device, DeviceInfo):
                QMessageBox.warning(self, "No Device", "Discover and select a wireless device first.")
                return
            key = self.txt_access_key.text().strip()
            if not key:
                QMessageBox.warning(self, "Missing Access Key", "Enter the device access key before connecting.")
                return
            self.wireless_access_key = key
            self.num_channels, self.emg_channel_count = WIRELESS_TOTAL_CHANNELS, WIRELESS_EMG_CHANNELS
            self._reset_stream_state()
            try:
                self.worker = WirelessStreamWorker(WIFI_STREAM_PORT)
                self.worker.batch_received.connect(self.on_stream_batch)
                self.worker.error_occurred.connect(self.on_stream_error)
                self.worker.start()
                ControlProtocol.start_stream(device.ip, key, get_local_ip_for_target(device.ip), WIFI_STREAM_PORT)
                self.current_device, self.connection_medium, self.is_connected = device, "wireless", True
                self.keepalive_failures = 0
                self.btn_wireless_connect.setText("Disconnect")
                self.btn_wired_connect.setEnabled(False)
                self._set_connected_text(f"Connected (wireless, {device.summary})")
            except Exception as exc:
                if self.worker:
                    self.worker.stop()
                    self.worker = None
                QMessageBox.critical(self, "Wireless Connection Error", str(exc))

        def _set_connected_text(self, text):
            self.lbl_conn_status.setText("Status: " + text)
            self.lbl_conn_status.setStyleSheet(f"color: {THEME_COLORS['accent']};")
            self._set_pill(self.pill_device, "Device: streaming", self.PILL_OK)
            self.check_ready_state()

        def send_wireless_keepalive(self):
            """Ping the ESP32 so its firmware does not time the stream out; flag the link if pings keep failing."""
            if not (self.is_connected and self.connection_medium == "wireless" and self.current_device):
                return
            try:
                ControlProtocol.ping(self.current_device.ip, self.wireless_access_key)
                self.keepalive_failures = 0
            except Exception:
                self.keepalive_failures += 1
                if self.keepalive_failures >= KEEPALIVE_MAX_FAILURES:
                    self.lbl_conn_status.setText("Status: Wireless keepalive lost")
                    self.lbl_conn_status.setStyleSheet("color: #f57c00;")
                    self._set_pill(self.pill_device, "Device: link lost", self.PILL_BAD)
                    self.disable_arm_control("Wireless connection lost")

        def on_stream_error(self, message):
            LOG.error("Stream error: %s", message)
            self.disable_arm_control("Connection error")
            QMessageBox.warning(self, "Stream Error", message)

        def disconnect_stream(self):
            self.stop_calibration_if_running()
            if self.connection_medium == "wireless" and self.current_device and self.wireless_access_key:
                try:
                    ControlProtocol.stop_stream(self.current_device.ip, self.wireless_access_key)
                except Exception:
                    pass
            if self.worker:
                try:
                    self.worker.batch_received.disconnect()
                    self.worker.error_occurred.disconnect()
                except Exception:
                    pass
                self.worker.stop()
                self.worker = None
            self.is_connected = self.is_calibrated = False
            self.preprocessor = self.calibration_profile = self.current_device = None
            self.connection_medium = ""
            self.keepalive_failures = 0
            self.disable_arm_control("Disconnected")
            self.btn_wired_connect.setText("Connect")
            self.btn_wired_connect.setEnabled(True)
            self.btn_wireless_connect.setText("Connect Wireless")
            self.btn_wireless_connect.setEnabled(True)
            self.lbl_conn_status.setText("Status: Disconnected")
            self.lbl_conn_status.setStyleSheet(f"color: {THEME_COLORS['muted']};")
            self._set_pill(self.pill_device, "Device: off", self.PILL_OFF)
            self.check_ready_state()

        def _reset_stream_state(self):
            with self._buffer_lock:
                self.sample_ring = SampleRingBuffer(self.num_channels, WINDOW_SIZE)
            self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
            self.plot_buf[:] = 0
            self.plot_filled = 0
            self.rf_valid_sample_count = self.rf_samples_since_submit = 0
            self.is_calibrated = self.calibration_active = False
            self.rest_capture, self.flex_capture = [], []
            self.last_batch_received_monotonic = 0.0

        # ================================================================== model
        def browse_model(self):
            start = self._settings.value("model_dir", "") or str(DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else Path.home())
            path, _ = QFileDialog.getOpenFileName(self, "Select RF Model", start, "Joblib Files (*.joblib)")
            if path:
                self.load_model(path)

        def load_model(self, path: str) -> bool:
            try:
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    artifact = joblib.load(path)
                lib_notes = sorted({("saved with another scikit-learn version (normally fine)" if "unpickle" in str(w_.message)
                                 else str(w_.message).split("\n")[0][:110]) for w_ in wlist})
                model = artifact["model"]
                names = list(artifact.get("class_names", artifact.get("classes", [])))
                if not names and hasattr(model, "classes_"):
                    names = [str(x) for x in model.classes_]
                self.rf_window_samples = int(max(8, artifact.get("window_samples", 100)))
                self.rf_stride_samples = int(max(1, artifact.get("stride_samples", self.rf_window_samples)))
                self.rf_model_input_channels = int(max(1, artifact.get("input_channels", self.num_channels)))
                self.rf_model_sample_rate = int(artifact.get("sample_rate", SAMPLE_RATE))
                self.rf_class_names = [str(x) for x in names]
                # "input_channels" is the trained total (8 EMG + 3 IMU = 11), so compare with num_channels.
                comp = validate_model_artifact(artifact, SAMPLE_RATE, self.num_channels)
                self.rf_preprocessing_version = comp.preprocessing_version
                self.model_compatible = comp.compatible
                self.model_compatibility_message = "\n".join(comp.errors + comp.warnings) or "Model compatibility verified."
                self.model_expected_feature_count = getattr(model, "n_features_in_", None)
                if not comp.compatible:
                    raise ValueError("Model is incompatible:\n" + "\n".join(comp.errors))
                self.inference_worker.load_model(model, self.rf_class_names, sample_rate=self.rf_model_sample_rate)
                self.txt_model_path.setText(path)
                self._settings.setValue("model_dir", str(Path(path).parent))
                self.build_mapping_ui(self.rf_class_names)
                self.model_loaded = True
                self.lbl_model_info.setText(f"{len(self.rf_class_names)} gestures: {', '.join(self.rf_class_names)}\n"
                                            f"window {self.rf_window_samples} samples, {self.rf_model_input_channels} input channels"
                                            + ("\nNote: " + " ".join(lib_notes) if lib_notes else ""))
                self._set_pill(self.pill_model, "Model: loaded", self.PILL_OK)
                if comp.warnings:                  # legacy models lack metadata: note it, do not block with a dialog
                    self.lbl_model_info.setText(self.lbl_model_info.text() + "\nNote: older model without metadata - legacy mode.")
                self.check_ready_state()
                return True
            except Exception as exc:
                self.model_loaded = False
                self._set_pill(self.pill_model, "Model: error", self.PILL_BAD)
                QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{exc}")
                return False

        def build_mapping_ui(self, class_names):
            while self.map_form.count():
                item = self.map_form.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.mapping_combos = {}
            self.bridge.set_mapping(class_names)
            for cls in class_names:
                combo = QComboBox()
                combo.addItems(list(ARM_ACTIONS))
                combo.setCurrentText(self.bridge.action_map[cls])
                combo.currentTextChanged.connect(lambda text, c=cls: self.bridge.set_action(c, text))
                self.map_form.addRow(f"Gesture: {cls}", combo)
                self.mapping_combos[cls] = combo
            self._update_map_button()

        def _toggle_mapping(self, on):
            self.grp_map.setVisible(on)
            self._update_map_button()

        def _update_map_button(self):
            n = len(self.mapping_combos)
            what = f"{n} gestures" if n else "load a model first"
            self.btn_map.setText(("Hide" if self.btn_map.isChecked() else "Show / edit") + f" gesture \u2192 arm action mapping  ({what})")

        def check_ready_state(self):
            if self.model_loaded:
                expected = self.model_expected_feature_count
                if expected is not None and expected != expected_feature_count(self.num_channels):
                    self.model_compatible = False
                    self.model_compatibility_message = "Feature count does not match the connected device's channel count."
                elif expected is not None:
                    self.model_compatible = True
                    self.model_compatibility_message = "Model feature count matches the connected device's channels."
            ready = self.model_loaded and self.is_connected and self.model_compatible
            self.btn_calibrate.setEnabled(ready and not self.calibration_active)
            control_ready = ready and self.is_calibrated
            self.btn_control.setEnabled(control_ready)
            if not control_ready and self.btn_control.isChecked():
                self.btn_control.setChecked(False)
            if ready and not self.is_calibrated:
                self.lbl_cal_status.setText("Ready to calibrate. Click Calibrate and follow the prompts.")
            elif not ready:
                self.lbl_cal_status.setText(self.model_compatibility_message if self.model_loaded and not self.model_compatible
                                            else "Connect a device and load a model to calibrate.")
            self._set_pill(self.pill_cal, "Calibration: valid" if self.is_calibrated else "Calibration: no",
                           self.PILL_OK if self.is_calibrated else self.PILL_OFF)

        # ================================================================== calibration
        def start_calibration_sequence(self):
            if not self.is_connected or self.calibration_active:
                return
            self.cal_rest_seconds, self.cal_flex_seconds = self.spin_rest_sec.value(), self.spin_flex_sec.value()
            self.calibration_phases = [
                {"key": "rest", "name": "REST", "duration_ms": self.cal_rest_seconds * 1000,
                 "instruction": "Keep your arm fully relaxed. Do not move."},
                {"key": "flex", "name": "FLEX", "duration_ms": self.cal_flex_seconds * 1000,
                 "instruction": "Flex the target muscle steadily until this phase ends."},
            ]
            self.bridge.enable(False)
            self.calibration_active, self.is_calibrated = True, False
            self.rest_capture, self.flex_capture = [], []
            self.current_cal_phase_idx = -1
            self.rf_valid_sample_count = self.rf_samples_since_submit = 0
            self.btn_calibrate.setEnabled(False)
            self.lbl_cal_status.setText("Calibrating - follow the on-screen prompts.")
            self.calibration_dialog = CalibrationDialog(self)
            self.calibration_dialog.btn_cancel.clicked.connect(self.cancel_calibration_sequence)
            self.calibration_dialog.show()
            self.begin_next_calibration_phase()

        def begin_next_calibration_phase(self):
            self.current_cal_phase_idx += 1
            if self.current_cal_phase_idx >= len(self.calibration_phases):
                self.finish_calibration_sequence()
                return
            ph = self.calibration_phases[self.current_cal_phase_idx]
            self.current_phase_key = ph["key"]
            self.current_phase_total_ms = self.current_phase_remaining_ms = ph["duration_ms"]
            if self.calibration_dialog:
                self.calibration_dialog.set_phase(ph["name"], ph["instruction"], self.current_phase_remaining_ms, self.current_phase_total_ms)
            self.calibration_timer.start(CAL_TICK_MS)

        def on_calibration_tick(self):
            if not self.calibration_active:
                self.calibration_timer.stop()
                return
            self.current_phase_remaining_ms -= CAL_TICK_MS
            ph = self.calibration_phases[self.current_cal_phase_idx]
            if self.calibration_dialog:
                self.calibration_dialog.set_phase(ph["name"], ph["instruction"], self.current_phase_remaining_ms, self.current_phase_total_ms)
            if self.current_phase_remaining_ms <= 0:
                self.calibration_timer.stop()
                self.begin_next_calibration_phase()

        def finish_calibration_sequence(self):
            self.calibration_timer.stop()
            self.calibration_active = False
            self.current_phase_key = ""
            if not self.rest_capture or not self.flex_capture:
                self.lbl_cal_status.setText("Calibration failed: no samples captured.")
                QMessageBox.warning(self, "Calibration Failed", "No REST/FLEX samples were captured. Is the stream running?")
                self._close_calibration_dialog()
                self.check_ready_state()
                return
            rest = np.vstack(self.rest_capture).astype(np.float32)
            flex = np.vstack(self.flex_capture).astype(np.float32)
            n_emg = int(min(self.emg_channel_count, rest.shape[1]))
            try:
                self.calibration_profile = compute_calibration(rest[:, :n_emg], flex[:, :n_emg])
            except ValueError as exc:
                self.lbl_cal_status.setText(f"Calibration failed: {exc}")
                self._close_calibration_dialog()
                self.check_ready_state()
                return
            if not self.calibration_profile.valid:
                report = ", ".join(f"CH{i + 1} {q}" for i, q in enumerate(self.calibration_profile.quality))
                self.lbl_cal_status.setText("Calibration rejected: " + report)
                QMessageBox.warning(self, "Calibration Rejected", "Unsafe channel quality:\n" + "\n".join(self.calibration_profile.reasons))
                self._close_calibration_dialog()
                self.check_ready_state()
                return
            self.baseline_offsets = np.zeros(self.num_channels, dtype=np.float32)
            if n_emg > 0:
                self.baseline_offsets[:n_emg] = np.median(rest[:, :n_emg], axis=0).astype(np.float32)
            with self._buffer_lock:
                if self.sample_ring is not None:
                    self.sample_ring.reset()
            self.rf_valid_sample_count = self.rf_samples_since_submit = 0
            # Fresh vote history, but every user-tuned setting carries over.
            old = self.bridge.engine
            self.bridge.engine = GestureDecisionEngine(min_confidence=old.min_confidence, min_margin=old.min_margin,
                                                       consecutive_required=old.consecutive_required, refractory_s=old.refractory_s)
            self.preprocessor = None
            if self.rf_preprocessing_version == PREPROCESSING_VERSION:
                self.preprocessor = RealTimePreprocessor(n_emg, PreprocessingConfig(sample_rate=SAMPLE_RATE), self.calibration_profile)
            self.is_calibrated = True
            summary = (f"Calibration complete.\nBaseline (ADC): {np.array2string(self.baseline_offsets, precision=1)}\n"
                       f"Channel quality: {', '.join(f'CH{i + 1} {q}' for i, q in enumerate(self.calibration_profile.quality))}")
            self.lbl_cal_status.setText("Calibrated - live inference running. You can enable arm control.")
            if self.calibration_dialog:
                self.calibration_dialog.set_finished(summary)
                QTimer.singleShot(900, self._close_calibration_dialog)
            self.check_ready_state()

        def _close_calibration_dialog(self):
            if self.calibration_dialog:
                self.calibration_dialog.close()
                self.calibration_dialog = None

        def stop_calibration_if_running(self):
            if self.calibration_active:
                self.calibration_active = False
                self.calibration_timer.stop()
                self.current_phase_key = ""
            self._close_calibration_dialog()
            self.check_ready_state()

        def cancel_calibration_sequence(self):
            self.stop_calibration_if_running()
            self.lbl_cal_status.setText("Calibration canceled.")

        # ================================================================== streaming
        def apply_baseline(self, raw_T):
            adjusted = np.ascontiguousarray(raw_T, dtype=np.float32)
            n = int(min(self.emg_channel_count, adjusted.shape[0]))
            if n <= 0:
                return adjusted
            centered = adjusted[:n] - self.baseline_offsets[:n, np.newaxis]
            for ch in range(n):
                near = np.abs(centered[ch]) < BASE_ADAPT_GUARD
                if np.any(near):
                    self.baseline_offsets[ch] += BASE_ADAPT_ALPHA * float(np.mean(centered[ch, near]))
            adjusted[:n] -= self.baseline_offsets[:n, np.newaxis]
            return adjusted

        def _push_plot(self, batch):
            n = min(batch.shape[0], PLOT_SAMPLES)
            k = min(batch.shape[1], self.plot_buf.shape[0])
            self.plot_buf[:k] = np.roll(self.plot_buf[:k], -n, axis=1)
            self.plot_buf[:k, -n:] = batch[-n:, :k].T
            self.plot_filled = min(PLOT_SAMPLES, self.plot_filled + n)

        def on_stream_batch(self, batch):
            meta = batch if isinstance(batch, SampleBatch) else None
            batch = np.asarray(meta.samples if meta is not None else batch, dtype=np.float32)
            if batch.ndim != 2 or batch.shape[1] != self.num_channels or self.sample_ring is None:
                return
            now = meta.host_received_monotonic if meta else time.monotonic()
            self.last_batch_received_monotonic = now
            self._rate_n += batch.shape[0]
            self._push_plot(batch)
            if meta and meta.gap_before:
                self.rf_valid_sample_count = self.rf_samples_since_submit = 0
                if meta.gap_before > MAX_INFERENCE_PACKET_GAP:
                    self.stream_invalid_until = time.monotonic() + self.rf_window_samples / SAMPLE_RATE
                LOG.warning("UDP gap: %d packet(s); inference window invalidated", meta.gap_before)
            if self.calibration_active:
                if self.current_phase_key == "rest":
                    self.rest_capture.append(batch.copy())
                elif self.current_phase_key == "flex":
                    self.flex_capture.append(batch.copy())
                return
            if not self.is_calibrated:
                return
            raw_T = batch.T
            if self.preprocessor is not None:
                centered = raw_T.copy()
                centered[:self.emg_channel_count] = self.preprocessor.process(batch[:, :self.emg_channel_count]).T
            else:
                centered = self.apply_baseline(raw_T)
            t0 = time.perf_counter_ns()
            with self._buffer_lock:
                self.sample_ring.append(centered.T, discontinuity=bool(meta and meta.gap_before))
            self.profiler.record_ns("buffer_append", t0)
            self.rf_valid_sample_count = self.sample_ring.sample_count
            self.rf_samples_since_submit += centered.shape[1]
            model_ch = int(max(1, self.rf_model_input_channels))
            if (self.rf_samples_since_submit >= self.rf_stride_samples
                    and self.sample_ring.has_window(self.rf_window_samples) and self.num_channels >= model_ch):
                self.rf_samples_since_submit = 0
                with self._buffer_lock:
                    win = self.sample_ring.latest(self.rf_window_samples)[:, :model_ch]
                profile = self.calibration_profile          # profile covers the EMG channels only (IMU columns excluded)
                quality = assess_signal_quality(win[:, :len(profile.quality)], profile) if profile is not None else SignalQuality("SIGNAL_POOR", [], "uncalibrated")
                self.last_signal_quality = quality
                self.update_diagnostics(quality)
                if quality.state != "GOOD" or time.monotonic() < self.stream_invalid_until:
                    self.disable_arm_control("Signal quality or packet continuity failed")
                    return
                self.inference_worker.submit_window(win)

        # ================================================================== prediction -> arm
        def update_diagnostics(self, quality=None, inference_ms=None):
            quality = quality or self.last_signal_quality
            q = quality.state if quality else "WAITING"
            ch = ""
            if quality and quality.channel_states:
                ch = " | " + ", ".join(f"CH{i + 1} {s}" for i, s in enumerate(quality.channel_states) if s != "GOOD") if q != "GOOD" else ""
            packet = "n/a"
            if isinstance(self.worker, WirelessStreamWorker):
                packet = f"{self.worker.stats.packet_loss_percent:.2f}% (jitter {self.worker.stats.jitter_ms:.1f} ms)"
            lat = f" | inference {inference_ms:.1f} ms" if inference_ms is not None else ""
            self.lbl_diagnostics.setText(f"Signal: {q} | Calibration: {'VALID' if self.is_calibrated else 'NOT VALID'} | Packet loss: {packet}{lat}{ch}")

        def on_inference_error(self, message):
            LOG.error("Model inference failed: %s", message)
            self.disable_arm_control("Model inference error")
            self.lbl_prediction.setText("UNKNOWN")

        def on_prediction_ready(self, label, conf, margin, inference_ms):
            pct = None if conf is None else conf * 100.0
            self.lbl_prediction.setText(str(label).upper())
            self.bar_conf.setValue(int(pct or 0))
            self._conf_text = "Confidence: n/a" if pct is None else f"Confidence {pct:.0f}%"
            quality_good = self.last_signal_quality is not None and self.last_signal_quality.state == "GOOD"
            action = self.bridge.on_prediction(label, conf, margin, quality_good)
            self.update_diagnostics(inference_ms=inference_ms)
            active = action != IGNORE_ACTION
            self.lbl_prediction.setStyleSheet(self._pred_style(THEME_COLORS['accent'] if active else THEME_COLORS['disabled']))
            if self.bridge.enabled:
                self.lbl_action.setText(f"{self._conf_text}  |  Arm action: {action.strip()}" if active else
                                        f"{self._conf_text}  |  holding ({self.bridge.last_decision.lower().replace('_', ' ')})")
            else:
                self.lbl_action.setText(f"{self._conf_text}  |  arm control OFF: gesture shown, not sent")

        def toggle_arm_control(self, checked):
            ok = checked and self.is_connected and self.is_calibrated and self.model_compatible
            self.bridge.enable(bool(ok))
            if checked and ok:
                self.btn_control.setText("STOP ARM CONTROL")
                self.btn_control.setStyleSheet(f"background-color: {THEME_COLORS['special']}; font-size: 16px; padding: 12px;")
                self._set_pill(self.pill_control, "Arm control: ON", self.PILL_OK)
            else:
                self.btn_control.setText("ENABLE ARM CONTROL")
                self.btn_control.setStyleSheet("background-color: #2e7d32; font-size: 16px; padding: 12px;")
                self._set_pill(self.pill_control, "Arm control: OFF", self.PILL_OFF)

        def disable_arm_control(self, reason):
            """Single fail-safe path for link, packet, quality and model failures: stop the arm and switch control off."""
            was_on = self.bridge.enabled
            self.bridge.enable(False)
            if self.btn_control.isChecked():
                self.btn_control.setChecked(False)
            if was_on:
                LOG.warning("Arm control disabled: %s", reason)
            self.lbl_status.setText(f"Status: {reason}")

        # ================================================================== periodic refresh
        def refresh_ui(self):
            s = self.rt.controller.snapshot()
            x, y, z = (v * 1000 for v in s.tcp_position)
            self.lbl_tcp.setText(f"X {x:6.1f}   Y {y:6.1f}   Z {z:6.1f} mm")
            self.lbl_joints.setText("  ".join(f"J{i + 1} {math.degrees(a):6.1f}°" for i, a in enumerate(s.joint_angles)))
            self.bar_gripper.setValue(int(round(s.gripper_opening * 100)))
            if s.estopped:
                self.lbl_safety.setText("Safety: EMERGENCY STOP latched")
                self._set_pill(self.pill_arm, "Arm: E-STOP", self.PILL_BAD)
            else:
                self.lbl_safety.setText(f"Safety: {s.safety}  |  Collision: {s.collision}")
                warn = not s.safety.ok
                self._set_pill(self.pill_arm, "Arm: warning" if warn else "Arm: ready", self.PILL_WARN if warn else self.PILL_OK)
            self.lbl_loop.setText(f"Control loop {s.control_hz:.0f} Hz  |  Command: {s.command}  |  Gesture: {s.gesture}")

            now = time.monotonic()
            if now - self._rate_t >= 1.0:
                self.sample_rate_measured, self._rate_n, self._rate_t = self._rate_n / (now - self._rate_t), 0, now
            if self.is_connected:
                age = now - self.last_batch_received_monotonic if self.last_batch_received_monotonic else None
                if age is not None and age > STREAM_STALL_S:
                    self._set_pill(self.pill_device, "Device: stalled", self.PILL_WARN)
                    self.disable_arm_control("EMG stream stalled")
                    self.lbl_stream.setText(f"No samples for {age:.1f} s")
                else:
                    self.lbl_stream.setText(f"Stream: {self.sample_rate_measured:.0f} samples/s per channel (expected {SAMPLE_RATE})")
            elif not self.worker:
                self.lbl_stream.setText("No stream.")
            if self.emg_win is not None and self.emg_win.isVisible():
                self.emg_win.refresh()
            view = self.fs.view if self.fs is not None else self.view
            if view.renderer is not None and view.isVisible():
                self.lbl_fps.setText(f"{view.ms:.1f} ms")
            if self.fs is not None:
                self.fs.update_count()
                self.fs.lbl.setText(f"X {x:6.1f}  Y {y:6.1f}  Z {z:6.1f} mm   |   gripper {s.gripper_opening * 100:3.0f}%   |   "
                                    + ("E-STOP" if s.estopped else f"safety {s.safety}")
                                + ("   |   " + hint if (hint := self._key_hint_text()) and not hint.startswith("Keys:") else ""))
            self.lbl_keys.setText(self._key_hint_text())
            self.refresh_guidance()

        # ================================================================== window
        def closeEvent(self, event):
            for t in (self.keepalive_timer, self.ui_timer, self.render_timer, self.key_timer):
                t.stop()
            for w_ in (self.fs, self.emg_win, self.help_dlg):
                if w_ is not None:
                    w_.close()
            self.fs = None
            QApplication.instance().removeEventFilter(self)
            self.view.close_renderer()
            self.bridge.enable(False)
            self.disconnect_stream()
            self.inference_worker.stop()
            event.accept()
            QApplication.quit()


    def run_dashboard(rt: "SimulationRuntime", model_path: str | None = None, duration: float | None = None,
                      fps: float = 30.0, max_mpix: float = 0.70, shadows: bool = True, frames: bool = False) -> int:
        configure_high_dpi()
        app = QApplication.instance() or QApplication(sys.argv)
        rt.start()
        win = ArmDashboard(rt, fps=fps, max_mpix=max_mpix, shadows=shadows, frames=frames)
        win.show()
        if model_path:
            win.load_model(model_path)
        if duration is not None:
            QTimer.singleShot(int(duration * 1000), win.close)
        try:
            return app.exec_()
        finally:
            rt.stop()






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
    ap.add_argument("--no-physics", action="store_true", help="kinematic backend (no MuJoCo dynamics, no 3D view)")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--frames", action="store_true", help="start with link coordinate frames drawn in the 3D view")
    ap.add_argument("--no-shadows", action="store_true", help="3D view without shadows/reflections (cooler, faster)")
    ap.add_argument("--no-log", action="store_true", help="do not write logs/session_NNN.csv")
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ,
                    help="control + physics rate (default %(default).0f; the original project used 240)")
    ap.add_argument("--gui-fps", type=float, default=30.0, help="3D view frame rate (lower = cooler, default %(default).0f)")
    ap.add_argument("--view-mpix", type=float, default=0.70,
                    help="max 3D render size in megapixels (lower = cooler on a fanless Air, default %(default).2f)")
    ap.add_argument("--threshold", type=float, default=EMG_CONFIDENCE_THRESHOLD)
    ap.add_argument("--window", type=int, default=SMOOTHING_WINDOW)
    ap.add_argument("--no-dashboard", action="store_true",
                    help="console only: no window, no 3D view (use with --duration or Ctrl-C)")
    ap.add_argument("--model", metavar="JOBLIB", help="pre-load this trained RF model in the dashboard")
    ap.add_argument("--export-urdf", metavar="PATH", help="write the generated URDF and exit")
    ap.add_argument("--export-mjcf", metavar="PATH", help="write the generated MuJoCo model (MJCF) and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.export_urdf:
        print("wrote", export_urdf(load_config(a.config), a.export_urdf))
        return 0
    if a.export_mjcf:
        print("wrote", export_mjcf(load_config(a.config), a.export_mjcf, Environment.default_scene()))
        return 0
    if running_under_rosetta():
        log.warning("This Python is x86_64 running under Rosetta (2-3x slower). Use a native arm64 Python: "
                    "https://www.python.org/downloads/macos/ (universal2) or `brew install python`, then "
                    "re-create your venv.")
    rt = SimulationRuntime(config_path=a.config, mode=a.mode, emg_kind=a.emg, headless=a.headless,
                           physics=not a.no_physics, control_hz=a.control_hz, log_dir=a.log_dir,
                           objects=not a.no_objects, threshold=a.threshold, window=a.window,
                           save_log=not a.no_log)
    if a.headless and a.fast:
        rt.run_fast_scripted(a.duration or 12.0)
    elif not a.headless and not a.no_dashboard:
        if HAS_QT and HAS_JOBLIB:
            return run_dashboard(rt, a.model, duration=a.duration, fps=a.gui_fps, max_mpix=a.view_mpix,
                                 shadows=not a.no_shadows, frames=a.frames)
        log.warning("Dashboard unavailable (pip install PyQt5 pyqtgraph pyserial joblib scikit-learn); "
                    "falling back to the console.")
        rt.run(a.duration)
    else:
        rt.run(a.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
