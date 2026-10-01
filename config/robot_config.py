"""Robot description + calibration parameters (SI units: m, kg, s, rad, N*m).

Only the figures in ``PUBLISHED_SPEC`` come from the kit's published description.
EVERYTHING ELSE below (link lengths, masses, joint limits, servo speeds, torque
per joint ...) is a PLACEHOLDER chosen to be plausible and to add up to the
published overall height/weight. Each is tagged ``# TODO: calibrate using
physical Hiwonder arm``.  Override them without touching code via a JSON file
(see ``load_config`` and ``config/calibration_example.json``).

Kinematic frame convention (world == base frame, right-handed):
    +X : operator's right      +Y : forward (arm reach direction at J1 = 0)
    +Z : up                    origin: centre of the base plate on the table
Zero pose = every joint at 0 -> arm points straight up.
    J1 : rotation about +Z  (positive = counter-clockwise seen from above, i.e. toward -X / "left")
    J2-J4 : pitch about local -X (positive = lean FORWARD toward +Y)
    J5 : roll about the link axis
    J6 : gripper (not part of the pose chain)
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

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
