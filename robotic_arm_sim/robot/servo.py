"""Simulated digital servo with velocity/acceleration-limited motion.

The servo works in JOINT space (radians). ``direction`` / ``zero_offset`` only matter
when converting to/from hardware angles or counts (``to_hardware`` / ``from_hardware``),
which is what a future HardwareRobotController will use.
"""
from __future__ import annotations

import math

from config.robot_config import JointLimit, ServoConfig


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
