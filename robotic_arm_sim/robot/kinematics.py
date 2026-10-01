"""Forward kinematics (homogeneous transforms), Jacobian, statics and numerical IK.

Chain (5 pose joints; J6 is the gripper and does not move the TCP):

    T_tcp = prod_{i=1..5} [ Trans(0,0,len(link_{i-1})) * Rot(axis_i, q_i) ] * Trans(0,0,tcp_offset)

The arm has 5 pose DOF, so a general 6-D pose is NOT always reachable. IK therefore has
three modes:  "position" (3 constraints),  "pitch" (position + tool pitch, 4),
"full" (position + full orientation, 6 - succeeds only for reachable orientations).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from config.robot_config import GRAVITY, RobotConfig
from utils.math_utils import (matrix_to_rpy, rotation_error, rotation_transform,
                              rpy_to_matrix, translation, wrap_angle)

N_POSE_JOINTS = 5


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
        self.lower = np.array([j.limit.min_angle for j in config.joints[:N_POSE_JOINTS]])
        self.upper = np.array([j.limit.max_angle for j in config.joints[:N_POSE_JOINTS]])

    # ------------------------------------------------------------------ FK
    def frames(self, q) -> list:
        """[T_base, T_1, ..., T_5]; T_i is the frame of link i (origin on joint i axis)."""
        T = np.eye(4)
        out = [T]
        for i in range(N_POSE_JOINTS):
            T = T @ translation(*self._offsets[i]) @ rotation_transform(self._axes[i], q[i])
            out.append(T)
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
            J[:3, i] = np.cross(a, p_tcp - T[:3, 3])
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
                t += a @ np.cross(coms[k] - p, links[k].mass * g)
            if payload_kg:
                t += a @ np.cross(tcp - p, payload_kg * g)
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

    def _pitch_row(self, q, pitch_t, h=1e-5):
        base = self.kin.tool_pitch(self.kin.forward(q), q[0])
        row = np.zeros(N_POSE_JOINTS)
        for i in range(N_POSE_JOINTS):
            qq = q.copy()
            qq[i] += h
            row[i] = wrap_angle(self.kin.tool_pitch(self.kin.forward(qq), qq[0]) - base) / h
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
