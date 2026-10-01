"""Simulated EMG sources for development without the armband."""
from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np

from config import emg_config as cfg
from emg.emg_controller import EMGSource, GesturePrediction

# Keyboard 1..5 / 0 -> gesture (the UI feeds key presses in via KeyboardEMGSource.set_gesture)
KEY_TO_GESTURE = {"1": "left", "2": "right", "3": "up", "4": "down", "5": "fist_close", "0": "rest"}


class KeyboardEMGSource(EMGSource):
    """While a gesture key is held it streams that gesture at ``confidence``; otherwise REST."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self._gesture = "rest"

    def set_gesture(self, gesture: str) -> None:
        self._gesture = gesture

    def release(self) -> None:
        self._gesture = "rest"

    def read(self) -> Optional[GesturePrediction]:
        return GesturePrediction(self._gesture, self.confidence, time.time())


class ScriptedEMGSource(EMGSource):
    """Plays [(gesture, confidence, n_samples), ...] one sample per read(); REST afterwards."""

    def __init__(self, script: Iterable[tuple]):
        self._samples = [(g, c) for g, c, n in script for _ in range(n)]
        self._i = 0

    @property
    def finished(self) -> bool:
        return self._i >= len(self._samples)

    def read(self) -> Optional[GesturePrediction]:
        g, c = self._samples[self._i] if self._i < len(self._samples) else ("rest", 1.0)
        self._i += 1
        return GesturePrediction(g, c, time.time())


class NoisyEMGSource(EMGSource):
    """Wraps another source; randomly flips labels / drops confidence to mimic a poor classifier."""

    def __init__(self, inner: EMGSource, flip_prob: float = 0.15, low_conf_prob: float = 0.1, seed: int = 0):
        self.inner, self.flip, self.low = inner, flip_prob, low_conf_prob
        self.rng = np.random.default_rng(seed)

    def connect(self):
        self.inner.connect()

    def disconnect(self):
        self.inner.disconnect()

    def read(self):
        p = self.inner.read()
        if p is None:
            return None
        g, c = p.gesture, p.confidence
        if self.rng.random() < self.flip:
            g = str(self.rng.choice([x for x in cfg.GESTURES if x != g]))
            c = float(self.rng.uniform(0.5, 0.95))
        if self.rng.random() < self.low:
            c = float(self.rng.uniform(0.3, 0.7))
        return GesturePrediction(g, c, p.timestamp)


def demo_script() -> list:
    """A short, deterministic demo: move left, up, close gripper, down, right; with some noise samples."""
    return [("rest", 0.9, 10), ("left", 0.95, 60), ("rest", 0.9, 10), ("up", 0.93, 40), ("left", 0.6, 3),
            ("up", 0.93, 10), ("fist_close", 0.9, 30), ("rest", 0.9, 10), ("down", 0.94, 40),
            ("right", 0.95, 80), ("rest", 0.9, 20)]
