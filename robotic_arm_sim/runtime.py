"""SimulationRuntime: wires everything together and owns the threads.

    thread "emg-input"   : EMGController polls the EMG source, smooths, maps -> MotionCommand queue
    thread "control"     : fixed-rate loop (default 240 Hz): planner/IK, servos, physics (DIRECT client), safety, log
    thread "viewer-link" : 30 Hz bridge to the GUI process (state out, MotionCommands in) + terminal dashboard
    process "viewer"     : PyBullet GUI window (sliders, buttons, keyboard, 3D view) - isolated because GUI calls
                           can block 100+ ms and must own a main thread on macOS

The only coupling between them is the thread-safe ``controller.submit(MotionCommand)`` queue and
``controller.snapshot()``.
"""
from __future__ import annotations

import threading
import time

from config import emg_config
from config.robot_config import load_config
from control.controller import SimulatedRobotController
from emg.emg_controller import EMGController, EMGSource
from emg.mock_emg import KeyboardEMGSource, NoisyEMGSource, ScriptedEMGSource, demo_script
from simulation.environment import Environment
from simulation.physics import KinematicBackend, PyBulletBackend, pybullet_available
from ui.controls import KEY_HELP, KeyboardController
from utils.logger import SessionLogger, get_logger

log = get_logger("runtime")


class SimulationRuntime:
    def __init__(self, config_path=None, mode: str = "manual", emg_kind: str = "keyboard", headless: bool = False,
                 physics: bool = True, control_hz: float = 240.0, log_dir: str = "logs", objects: bool = True,
                 show_frames: bool = True, threshold: float = emg_config.EMG_CONFIDENCE_THRESHOLD,
                 window: int = emg_config.SMOOTHING_WINDOW, save_log: bool = True):
        self.mode, self.headless = mode, headless
        self.cfg = load_config(config_path)
        self.env = Environment.default_scene() if objects else Environment()
        use_pb = physics and pybullet_available()
        if not headless and not use_pb:
            raise SystemExit("The 3D view needs PyBullet (pip install pybullet). "
                             "Use --headless for a console-only run.")
        if physics and not use_pb:
            log.warning("PyBullet not installed: running with the kinematic backend (no objects/contacts)")
        self.backend = (PyBulletBackend(self.cfg, dt=1.0 / control_hz) if use_pb else KinematicBackend())
        self.logger = SessionLogger(log_dir) if save_log else None
        self.controller = SimulatedRobotController(self.cfg, self.backend, control_hz=control_hz,
                                                   session_logger=self.logger)
        self.controller.external_stepper = False

        self.emg_source: EMGSource | None = None
        self.kb_source: KeyboardEMGSource | None = None
        if mode == "emg":
            if emg_kind == "scripted":
                self.emg_source = ScriptedEMGSource(demo_script())
            elif emg_kind == "noisy":
                self.emg_source = NoisyEMGSource(ScriptedEMGSource(demo_script()))
            else:
                self.kb_source = self.emg_source = KeyboardEMGSource()
        self.emg = EMGController(self.emg_source, sink=self.controller.submit,
                                 status_callback=self.controller.set_emg_status, confidence_threshold=threshold,
                                 window=window, latency=self.controller.latency)
        self.keyboard = KeyboardController(self.controller.submit, mode, self.kb_source)
        self.viewer = None
        self.show_frames = show_frames
        self._stop = threading.Event()
        self._ctl_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.controller.connect(self.env)
        self.controller.external_stepper = True
        self._ctl_thread = threading.Thread(target=self._control_loop, name="control", daemon=True)
        self._ctl_thread.start()
        if self.emg_source is not None:
            self.emg.connect()
            self.emg.start()
        if not self.headless:
            from ui.viewer_link import ViewerLink
            self.viewer = ViewerLink(self.cfg, self.env, self.backend, self.controller, self.mode, self.show_frames,
                                     gesture_source=self.kb_source)
            self.viewer.start()
            print(KEY_HELP + "\n")
        log.info("runtime started (mode=%s, headless=%s, backend=%s, log=%s)", self.mode, self.headless,
                 type(self.backend).__name__, self.logger.path if self.logger else "off")

    def stop(self) -> None:
        self._stop.set()
        if self.viewer:
            self.viewer.stop()
        self.emg.disconnect()
        if self._ctl_thread:
            self._ctl_thread.join(timeout=2.0)
        self.controller.external_stepper = False
        print("\nLatency summary:\n" + self.controller.latency.summary())
        self.controller.shutdown()

    def _control_loop(self) -> None:
        c = self.controller
        dt = c.dt
        nxt = time.perf_counter()
        t_rate, n = nxt, 0
        while not self._stop.is_set():
            c.step(dt)
            n += 1
            now = time.perf_counter()
            if now - t_rate >= 1.0:
                c.control_hz_measured = n / (now - t_rate)
                t_rate, n = now, 0
            nxt += dt
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:                  # fell far behind: do not spiral, resync
                nxt = time.perf_counter()

    def run(self, duration: float | None = None) -> None:
        """Real-time run: control thread + EMG thread (+ GUI viewer process). Ctrl-C or closing the window ends it."""
        self.start()
        t_end = None if duration is None else time.perf_counter() + duration
        try:
            last = 0.0
            while (t_end is None or time.perf_counter() < t_end) and not (self.viewer and self.viewer.closed.is_set()):
                time.sleep(0.1)
                if self.viewer is None:                      # headless: console status once per sim second
                    s = self.controller.snapshot()
                    if s.sim_time - last >= 1.0:
                        last = s.sim_time
                        self.print_status(s)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def run_fast_scripted(self, duration: float) -> None:
        """Deterministic, unpaced headless run driven by simulated time (no threads)."""
        c = self.controller
        c.connect(self.env)
        if self.emg_source is not None:
            self.emg.connect()
        poll_every = max(1, int(round(1.0 / (emg_config.EMG_POLL_HZ * c.dt))))
        steps = int(round(duration / c.dt))
        for i in range(steps):
            if self.emg_source is not None and i % poll_every == 0:
                pred = self.emg.receive_gesture()
                if pred is not None:
                    self.emg.process_prediction(pred)
            c.step()
            if i % int(round(1.0 / c.dt)) == 0:
                self.print_status(c.snapshot())
        print("\nLatency summary:\n" + c.latency.summary())
        c.shutdown()

    @staticmethod
    def print_status(s) -> None:
        import math
        j = " ".join(f"{math.degrees(a):6.1f}" for a in s.joint_angles)
        x, y, z = (v * 1000 for v in s.tcp_position)
        print(f"t={s.sim_time:6.2f}s  J[deg]= {j}  TCP[mm]= {x:6.1f} {y:6.1f} {z:6.1f}  "
              f"grip={s.gripper_opening*100:3.0f}%  gest={s.gesture:<14} cmd={s.command:<12} "
              f"safety={'ESTOP' if s.estopped else s.safety}")
