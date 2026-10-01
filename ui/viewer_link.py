"""Parent-process side of the GUI: spawns the viewer process and bridges it to the controller.

    publisher thread (30 Hz):  backend.view_state() + controller.snapshot()  --pipe-->  viewer
                               viewer --pipe--> ("cmd", MotionCommand) -> controller.submit
                                                ("gesture", g)         -> KeyboardEMGSource
Also prints a live telemetry dashboard in the terminal (when stdout is a TTY).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
import threading
import time

from ui.gui import viewer_main
from utils.logger import get_logger

log = get_logger("viewer")


def telemetry_lines(s, mode: str) -> list:
    j = "  ".join(f"J{i+1}:{math.degrees(a):7.1f} deg" for i, a in enumerate(s.joint_angles))
    conf = "-" if s.confidence is None else f"{s.confidence*100:.0f}%"
    safety = "ESTOP - press X / RESET" if s.estopped else str(s.safety)
    return [
        f"SIM {s.sim_time:8.2f} s | CONTROL {s.control_hz:4.0f} Hz | CONNECTED: {s.connected} | MODE: {mode.upper()}",
        j,
        f"X {s.tcp_position[0]*1000:7.1f}  Y {s.tcp_position[1]*1000:7.1f}  Z {s.tcp_position[2]*1000:7.1f} mm"
        f"   Roll {math.degrees(s.tcp_rpy[0]):7.1f}  Pitch {math.degrees(s.tcp_rpy[1]):7.1f}  Yaw {math.degrees(s.tcp_rpy[2]):7.1f} deg",
        f"GRIPPER {s.gripper_opening*100:3.0f}% open | GESTURE: {s.gesture} | CONFIDENCE: {conf} | COMMAND: {s.command}",
        f"COLLISION: {s.collision} | SAFETY: {safety}",
    ]


class ViewerLink:
    def __init__(self, cfg, env, backend, controller, mode: str = "manual", show_frames: bool = True,
                 gesture_source=None, dashboard: bool = True, rate_hz: float = 30.0):
        self.cfg, self.env, self.backend, self.ctl = cfg, env, backend, controller
        self.mode, self.show_frames = mode, show_frames
        self.gesture_source = gesture_source
        self.dashboard = dashboard and sys.stdout.isatty()
        self.period = 1.0 / rate_hz
        self.closed = threading.Event()
        self._proc = None
        self._conn = None
        self._thread = None
        self._stop = threading.Event()
        self._dash_drawn = False

    def start(self) -> None:
        ctx = mp.get_context("spawn")                 # fresh process: owns its own main thread for the GUI
        self._conn, child = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=viewer_main, name="viewer",
                                 args=(child, self.cfg, self.env, self.mode, self.show_frames, self.backend.dt),
                                 daemon=True)
        self._proc.start()
        self._thread = threading.Thread(target=self._loop, name="viewer-link", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._conn:
                self._conn.send(("quit",))
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._proc:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()

    def _handle(self, msg) -> None:
        kind = msg[0]
        if kind == "cmd":
            self.ctl.submit(msg[1])
        elif kind == "gesture" and self.gesture_source is not None:
            if msg[1] is None:
                self.gesture_source.release()
            else:
                self.gesture_source.set_gesture(msg[1])
        elif kind == "closed":
            self.closed.set()

    def _loop(self) -> None:
        nxt = time.perf_counter()
        n = 0
        while not self._stop.is_set() and not self.closed.is_set():
            try:
                while self._conn.poll():
                    self._handle(self._conn.recv())
                if not self._proc.is_alive():
                    self.closed.set()
                    break
                st = self.backend.view_state()
                snap = self.ctl.snapshot()
                st["telemetry"] = snap
                self._conn.send(("state", st))
                n += 1
                if self.dashboard and n % 3 == 0:
                    self._print_dashboard(snap)
            except (EOFError, BrokenPipeError, OSError):
                self.closed.set()
                break
            nxt += self.period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def _print_dashboard(self, snap) -> None:
        lines = telemetry_lines(snap, self.mode)
        up = "\x1b[%dA" % len(lines) if self._dash_drawn else ""
        sys.stdout.write(up + "".join(l[:200].ljust(150) + "\n" for l in lines))
        sys.stdout.flush()
        self._dash_drawn = True
