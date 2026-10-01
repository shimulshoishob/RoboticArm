"""pick_and_place(): a scripted sequence built only from RobotController calls (works on any controller)."""
from __future__ import annotations

import math

import numpy as np

from utils.logger import get_logger

log = get_logger("pick_and_place")


def pick_and_place(controller, object_id, pick_position, place_position, hover: float = 0.06,
                   speed: float = 0.08, tool_pitch: float = math.pi, grasp_time: float = 0.6,
                   arrive_tol: float = 0.004) -> bool:
    """
    Pick the object whose grip point is at ``pick_position`` (TCP target, m) and put it down at
    ``place_position``. ``object_id`` is only used for logging / verification (a name or id).
    Returns False as soon as a waypoint is refused, vetoed by the safety layer, or not reached within
    ``arrive_tol`` metres (unreachable target, collision veto, e-stop, fault).
    """
    pick = np.asarray(pick_position, dtype=float)
    place = np.asarray(place_position, dtype=float)
    up = np.array([0.0, 0.0, hover])

    def go(target) -> bool:
        ok = controller.move_cartesian(target, tool_pitch=tool_pitch, speed=speed, wait=True)
        err = np.linalg.norm(controller.snapshot().tcp_position - target) if ok else np.inf
        if not ok or controller.blocked or err > arrive_tol:
            log.warning("pick_and_place[%s]: could not reach %s (error %.1f mm, safety: %s)", object_id,
                        np.round(target, 3), err * 1000, controller.snapshot().safety)
            return False
        return True

    controller.open_gripper(wait=True)
    steps = [("hover over pick", pick + up), ("descend", pick)]
    for name, target in steps:
        if not go(target):
            return False
    controller.close_gripper(wait=True)
    controller.pause(grasp_time)                          # let the fingers squeeze
    for name, target in (("lift", pick + up), ("transfer", place + up), ("lower", place)):
        if not go(target):
            return False
    controller.open_gripper(wait=True)
    controller.pause(0.3)
    ok = go(place + up)                                  # retreat
    log.info("pick_and_place[%s]: %s", object_id, "done" if ok else "failed on retreat")
    return ok
