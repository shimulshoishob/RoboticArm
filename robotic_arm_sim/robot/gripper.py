"""Two-finger parallel gripper driven by servo J6.

opening: 0.0 = completely closed, 1.0 = completely open (J6 angle limits map linearly).
"""
from __future__ import annotations

from config.robot_config import GripperConfig
from robot.joints import Joint


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
