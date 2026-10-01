"""MotionPlanner: MotionCommand -> Cartesian/joint targets -> IK -> joint targets.

Pure logic (no threads, no simulation, no clock): time only advances through ``update(dt)``,
so behaviour is deterministic and unit-testable. Continuous commands (CARTESIAN / JOINT with
no magnitude) are kept alive by being re-sent; if they stop arriving for ``command_timeout``
seconds (e.g. EMG dropout) the planner stops generating motion by itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from control import motion_command as mc
from control.safety import Severity, UNREACHABLE
from robot.kinematics import Kinematics, NumericalIKSolver


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
    def handle(self, cmd: mc.MotionCommand, q_cmd, gripper_cmd: float) -> PlanOutput:
        out = PlanOutput()
        q_cmd = np.asarray(q_cmd, dtype=float)[:5]
        ct = cmd.command_type

        if ct == mc.CARTESIAN:
            self._start_cartesian(cmd, q_cmd, out)
        elif ct == mc.JOINT:
            self._start_jog(cmd)
        elif ct == mc.MOVE_TO:
            self._start_move_to(cmd, q_cmd, out)
        elif ct == mc.JOINT_TARGET:
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
        elif ct == mc.GRIPPER:
            d = cmd.direction
            if d == "OPEN":
                out.gripper = 1.0
            elif d == "CLOSE":
                out.gripper = 0.0
            elif d == "TOGGLE":
                out.gripper = 0.0 if gripper_cmd > 0.5 else 1.0
            elif d in ("SET", "PARTIAL_OPEN") and cmd.target:
                out.gripper = float(np.clip(cmd.target[0], 0.0, 1.0))
        elif ct == mc.HOME:
            self.cancel()
            out.joint_targets = self.home_angles.copy()
            out.gripper = self.home_gripper
        elif ct == mc.HOLD:
            self.cancel()
        elif ct == mc.STOP:
            self.cancel()
            out.stop_now = True
        return out

    def _angle_to_opening(self, angle: float) -> float:
        lo, hi = self.gripper_range
        return float(np.clip((angle - lo) / (hi - lo), 0.0, 1.0))

    def _lifetime(self, cmd) -> float:
        return self.t + (cmd.duration if cmd.duration is not None else self.command_timeout)

    def _start_cartesian(self, cmd, q_cmd, out: PlanOutput) -> None:
        if cmd.direction not in mc.DIRECTIONS:
            out.event(UNREACHABLE, f"Unknown direction '{cmd.direction}'")
            return
        d = np.array(mc.DIRECTIONS[cmd.direction])
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
