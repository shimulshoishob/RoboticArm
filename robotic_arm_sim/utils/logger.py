"""Console logging, CSV session logging and latency statistics."""
from __future__ import annotations

import csv
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
        _CONFIGURED = True
    return logging.getLogger(name)


CSV_FIELDS = (["wall_time", "sim_time", "gesture", "confidence", "command"]
              + [f"j{i}_rad" for i in range(1, 7)]
              + ["x_m", "y_m", "z_m", "collision", "safety"])


class SessionLogger:
    """Writes logs/session_NNN.csv (SI units: rad, m, s). Thread-safe, rate-limited."""

    def __init__(self, directory: str | Path = "logs", rate_hz: float = 50.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        nums = [int(m.group(1)) for p in self.dir.glob("session_*.csv")
                if (m := re.match(r"session_(\d+)\.csv", p.name))]
        self.path = self.dir / f"session_{(max(nums) + 1 if nums else 1):03d}.csv"
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(CSV_FIELDS)
        self._lock = threading.Lock()
        self._period = 1.0 / rate_hz
        self._last = -1e9

    def log(self, sim_time: float, gesture, confidence, command, joint_angles, tcp, collision, safety,
            force: bool = False) -> bool:
        if not force and sim_time - self._last < self._period:
            return False
        self._last = sim_time
        row = [f"{time.time():.4f}", f"{sim_time:.4f}", gesture or "", "" if confidence is None else f"{confidence:.3f}",
               command or ""] + [f"{a:.5f}" for a in joint_angles] + [f"{v:.5f}" for v in tcp] + [collision, safety]
        with self._lock:
            if not self._fh.closed:
                self._w.writerow(row)
        return True

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class LatencyTracker:
    """Rolling latency samples in milliseconds, keyed by stage name."""

    def __init__(self, maxlen: int = 2000):
        self._d: dict = {}
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def record(self, name: str, ms: float) -> None:
        with self._lock:
            self._d.setdefault(name, deque(maxlen=self._maxlen)).append(ms)

    def stats(self, name: str) -> dict:
        with self._lock:
            v = np.array(self._d.get(name, ()), dtype=float)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    def summary(self) -> str:
        lines = []
        for k in sorted(self._d):
            s = self.stats(k)
            lines.append(f"{k}: n={s['n']} mean={s['mean']:.2f} ms p95={s['p95']:.2f} ms max={s['max']:.2f} ms")
        return "\n".join(lines) or "no latency samples"
