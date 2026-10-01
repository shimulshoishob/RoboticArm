"""Safety monitor inspired by the real controller's protections (limits, over-current, stall).

Events carry a severity. WARNING events expire if they stop being re-reported; FAULT events
are latched and require ``SimulatedRobotController.reset()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from config.robot_config import RobotConfig


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
