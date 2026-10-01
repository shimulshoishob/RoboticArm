"""Small SO(3)/SE(3) helpers. SI units (metres, radians) throughout.

Euler convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (extrinsic x-y-z, a.k.a. ZYX).
"""
from __future__ import annotations

import math

import numpy as np


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
