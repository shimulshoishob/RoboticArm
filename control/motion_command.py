"""MotionCommand: the single, input-agnostic command format.

Keyboard, GUI sliders, EMG and (future) ROS all emit these; nothing downstream knows
which one produced a command. SI units: magnitude/target in metres or radians, speed in m/s
(or rad/s for joints, opening/s for the gripper).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

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
