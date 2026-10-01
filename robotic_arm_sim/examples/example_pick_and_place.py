"""Pick the cube and put it down 80 mm to the right, using only RobotController calls.

    python examples/example_pick_and_place.py            # headless, prints the result
    python examples/example_pick_and_place.py --gui      # watch it in the 3D window
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from control.controller import SimulatedRobotController
from control.pick_and_place import pick_and_place
from simulation.environment import Environment
from simulation.objects import make_cube
from simulation.physics import KinematicBackend, PyBulletBackend, pybullet_available
from config.robot_config import default_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    a = ap.parse_args()
    cfg = default_config()
    env = Environment()
    cube = env.add(make_cube("cube", 0.0, 0.15))
    backend = PyBulletBackend(cfg) if pybullet_available() else KinematicBackend()
    ctl = SimulatedRobotController(cfg, backend)
    ctl.connect(env)
    viewer = None
    if a.gui:                                  # GUI = separate viewer process fed by a 30 Hz bridge thread
        from ui.viewer_link import ViewerLink
        import threading
        ctl.external_stepper = True
        threading.Thread(target=lambda: [ctl.step() or time.sleep(ctl.dt) for _ in iter(int, 1)], daemon=True).start()
        viewer = ViewerLink(cfg, env, backend, ctl, 'manual', dashboard=False)
        viewer.start()
        time.sleep(4)
    ctl.home(wait=True)

    grip_z = max(cube.half_height, 0.022)                       # keep the fingertips above the table
    pick = np.array([0.0, 0.15, grip_z])
    place = np.array([0.08, 0.15, grip_z + 0.002])
    ok = pick_and_place(ctl, "cube", pick, place)

    if a.gui:
        time.sleep(2)
    final = backend.object_position("cube")
    print("pick_and_place success:", ok)
    if final is not None:
        err = np.linalg.norm(np.array(final[:2]) - place[:2]) * 1000
        print(f"cube final position (mm): {np.round(np.array(final)*1000, 1)}   error to target: {err:.1f} mm")
    print("safety:", ctl.snapshot().safety)
    if viewer:
        viewer.stop()
    ctl.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
