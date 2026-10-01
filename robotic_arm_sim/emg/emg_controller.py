"""EMGController: the ONLY place where gesture predictions enter the system.

    EMGSource.read() -> {"gesture","confidence","timestamp"}
        -> confidence gate + smoothing -> GestureMapper -> MotionCommand -> sink (controller.submit)

To use your real Random-Forest pipeline, implement an ``EMGSource`` (or wrap a function with
``CallbackEMGSource``); nothing in the robot/simulation core changes.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from config import emg_config as cfg
from control import motion_command as mc
from emg.gesture_mapper import GestureMapper, GestureSmoother, normalize_gesture
from utils.logger import LatencyTracker, get_logger

log = get_logger("emg")


@dataclass
class GesturePrediction:
    gesture: str
    confidence: float = 1.0
    timestamp: float | None = None            # wall-clock epoch seconds from the classifier

    @classmethod
    def from_dict(cls, d) -> "GesturePrediction":
        return cls(d["gesture"], float(d.get("confidence", 1.0)), d.get("timestamp"))


class EMGSource(ABC):
    """A producer of gesture predictions (real device + classifier, keyboard, script ...)."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @abstractmethod
    def read(self) -> Optional[dict | GesturePrediction]:
        """Return the newest prediction or None if nothing new. Must not block for long."""


class CallbackEMGSource(EMGSource):
    """Wrap any callable returning {"gesture":..,"confidence":..,"timestamp":..} (or None)."""

    def __init__(self, fn: Callable[[], Optional[dict]], connect_fn=None, disconnect_fn=None):
        self.fn, self._c, self._d = fn, connect_fn, disconnect_fn

    def connect(self):
        if self._c:
            self._c()

    def disconnect(self):
        if self._d:
            self._d()

    def read(self):
        return self.fn()


class EMGController:
    def __init__(self, source: EMGSource | None = None, sink: Callable[[mc.MotionCommand], bool] | None = None,
                 status_callback: Callable[[str, float | None], None] | None = None,
                 confidence_threshold: float = cfg.EMG_CONFIDENCE_THRESHOLD,
                 window: int = cfg.SMOOTHING_WINDOW, min_agreement: float = cfg.SMOOTHING_MIN_AGREEMENT,
                 speed: float = cfg.EMG_SPEED_M_S, poll_hz: float = cfg.EMG_POLL_HZ,
                 fist_toggles: bool = cfg.FIST_CLOSE_TOGGLES, latency: LatencyTracker | None = None):
        self.source = source
        self.sink = sink
        self.status_callback = status_callback
        self.smoother = GestureSmoother(window, min_agreement, confidence_threshold)
        self.mapper = GestureMapper(speed, fist_toggles=fist_toggles)
        self.poll_period = 1.0 / poll_hz
        self.latency = latency or LatencyTracker()
        self.last_prediction: GesturePrediction | None = None
        self.last_smoothed = "rest"
        self.last_command: mc.MotionCommand | None = None
        self.connected = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ API required by the spec
    def connect(self) -> None:
        if self.source is not None:
            self.source.connect()
        self.connected = True

    def disconnect(self) -> None:
        self.stop()
        if self.source is not None:
            self.source.disconnect()
        self.connected = False

    def receive_gesture(self) -> Optional[GesturePrediction]:
        """Poll the source once."""
        if self.source is None:
            return None
        raw = self.source.read()
        if raw is None:
            return None
        return raw if isinstance(raw, GesturePrediction) else GesturePrediction.from_dict(raw)

    def process_gesture(self, gesture: str, confidence: float = 1.0, timestamp: float | None = None) -> mc.MotionCommand:
        """One prediction in -> one MotionCommand out (and submitted to the sink, if any)."""
        t0 = time.perf_counter()
        try:
            smoothed = self.smoother.update(gesture, confidence)
            raw_name = normalize_gesture(gesture)
        except ValueError as exc:                       # unknown label: fail safe
            log.warning("%s - treating as REST", exc)
            smoothed = self.smoother.update("rest", 1.0)
            raw_name = "rest"
        cmd = self.mapper.map(smoothed, confidence)
        cmd.created_at = t0                              # latency clock starts when the prediction arrived
        cmd.timestamp = timestamp
        self.last_prediction = GesturePrediction(raw_name, confidence, timestamp)
        self.last_smoothed, self.last_command = smoothed, cmd
        self.latency.record("emg_to_command_ms", (time.perf_counter() - t0) * 1000.0)
        if timestamp is not None:
            self.latency.record("classifier_age_ms", max(0.0, (time.time() - timestamp) * 1000.0))
        if self.status_callback:
            label = smoothed.upper() if smoothed == raw_name else f"{smoothed.upper()} (raw {raw_name.upper()})"
            self.status_callback(label, confidence)
        if self.sink is not None:
            self.sink(cmd)
        return cmd

    def process_prediction(self, pred: GesturePrediction | dict) -> mc.MotionCommand:
        if isinstance(pred, dict):
            pred = GesturePrediction.from_dict(pred)
        return self.process_gesture(pred.gesture, pred.confidence, pred.timestamp)

    # ------------------------------------------------------------------ input thread
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="emg-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                pred = self.receive_gesture()
                if pred is not None:
                    self.process_prediction(pred)
            except Exception:                           # never let the input thread die silently
                log.exception("EMG input error")
            nxt += self.poll_period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()
