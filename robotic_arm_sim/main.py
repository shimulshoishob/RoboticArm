"""Entry point.  Examples:

    python main.py                       # 3D GUI, manual keyboard/sliders
    python main.py --mode emg            # 3D GUI, keys 1-5/0 act as mock EMG gestures
    python main.py --mode emg --emg scripted        # scripted gesture demo in the 3D view
    python main.py --headless --mode emg --emg noisy --fast --duration 12   # console demo, no window
    python main.py --export-urdf arm.urdf
"""
from __future__ import annotations

import argparse
import sys

from config import emg_config


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="6-DOF robotic arm simulator (EMG-ready)")
    ap.add_argument("--mode", choices=["manual", "emg"], default="manual",
                    help="manual: keyboard/sliders/buttons; emg: gestures drive the arm")
    ap.add_argument("--emg", choices=["keyboard", "scripted", "noisy"], default="keyboard",
                    help="EMG source in emg mode (replace with your classifier, see README)")
    ap.add_argument("--config", help="JSON calibration file (see config/calibration_example.json)")
    ap.add_argument("--headless", action="store_true", help="no 3D window; console telemetry")
    ap.add_argument("--fast", action="store_true", help="headless only: simulate as fast as possible, deterministic")
    ap.add_argument("--duration", type=float, help="seconds to run (default: until closed)")
    ap.add_argument("--no-physics", action="store_true", help="kinematic backend (no PyBullet dynamics)")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="do not draw coordinate frames")
    ap.add_argument("--no-log", action="store_true", help="do not write logs/session_NNN.csv")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--control-hz", type=float, default=240.0)
    ap.add_argument("--threshold", type=float, default=emg_config.EMG_CONFIDENCE_THRESHOLD)
    ap.add_argument("--window", type=int, default=emg_config.SMOOTHING_WINDOW)
    ap.add_argument("--export-urdf", metavar="PATH", help="write the generated URDF and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.export_urdf:
        from config.robot_config import load_config
        from simulation.urdf import export_urdf
        print("wrote", export_urdf(load_config(a.config), a.export_urdf))
        return 0
    from runtime import SimulationRuntime
    rt = SimulationRuntime(config_path=a.config, mode=a.mode, emg_kind=a.emg, headless=a.headless,
                           physics=not a.no_physics, control_hz=a.control_hz, log_dir=a.log_dir,
                           objects=not a.no_objects, show_frames=not a.no_frames, threshold=a.threshold,
                           window=a.window, save_log=not a.no_log)
    if a.headless and a.fast:
        rt.run_fast_scripted(a.duration or 12.0)
    else:
        rt.run(a.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
