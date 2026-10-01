"""Gesture normalisation, temporal smoothing and gesture -> MotionCommand mapping.

Nothing here knows about servos, IK or the simulator: the output is a MotionCommand.
"""
from __future__ import annotations

import math
from collections import Counter, deque

from config import emg_config as cfg
from control import motion_command as mc


def normalize_gesture(name: str) -> str:
    g = str(name).strip().lower().replace("-", "_")
    g = cfg.GESTURE_ALIASES.get(g, g)
    if g not in cfg.GESTURES:
        raise ValueError(f"unknown gesture '{name}' (expected one of {cfg.GESTURES})")
    return g


class GestureSmoother:
    """Confidence gate + sliding-window majority vote.

    * prediction with confidence < threshold  -> counted as REST
    * winner needs at least ceil(min_agreement * window) votes of the last ``window`` samples,
      otherwise the output is REST (so LEFT,RIGHT,LEFT,REST,RIGHT stays REST).
    Until ``window`` samples have arrived the missing slots count as "no vote", so a single
    sample can never start a motion.
    """

    def __init__(self, window: int = cfg.SMOOTHING_WINDOW, min_agreement: float = cfg.SMOOTHING_MIN_AGREEMENT,
                 confidence_threshold: float = cfg.EMG_CONFIDENCE_THRESHOLD):
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

    def __init__(self, speed: float = cfg.EMG_SPEED_M_S, mapping: dict | None = None,
                 fist_toggles: bool = cfg.FIST_CLOSE_TOGGLES):
        self.speed = speed
        self.mapping = dict(mapping or cfg.GESTURE_TO_COMMAND)
        self.fist_toggles = fist_toggles
        self._last = "rest"

    def map(self, gesture: str, confidence: float | None = None, source: str = "emg") -> mc.MotionCommand:
        g = normalize_gesture(gesture)
        ctype, direction = self.mapping[g]
        prev, self._last = self._last, g
        kw = dict(source=source, gesture=g, confidence=confidence)
        if ctype == mc.CARTESIAN:
            return mc.MotionCommand(mc.CARTESIAN, direction, magnitude=None, speed=self.speed, **kw)
        if ctype == mc.GRIPPER:
            if self.fist_toggles:
                if prev != g:
                    return mc.MotionCommand(mc.GRIPPER, "TOGGLE", **kw)
                return mc.MotionCommand(mc.HOLD, "NONE", **kw)
            return mc.MotionCommand(mc.GRIPPER, direction, **kw)
        return mc.MotionCommand(mc.HOLD, "NONE", **kw)
