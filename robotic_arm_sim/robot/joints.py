"""Joint = static description (JointConfig) + its simulated servo."""
from __future__ import annotations

from config.robot_config import JointConfig
from robot.servo import SimulatedServo


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
