"""Input handling that does not depend on PyBullet: key map + KeyboardController.

The GUI feeds in the set of currently held key NAMES (and the edge-triggered ones);
this module turns them into MotionCommands. Only ONE motion key is honoured at a time
(the planner executes one motion at a time), preferring the key pressed first.
"""
from __future__ import annotations

from typing import Callable

from control import motion_command as mc
from emg.mock_emg import KEY_TO_GESTURE, KeyboardEMGSource

# key -> (joint id, sign).  Q/A J1, W/S J2, E/D J3, R/F J4, T/G J5, Y/H gripper (J6)
JOINT_KEYS = {
    "q": (1, +1), "a": (1, -1), "w": (2, +1), "s": (2, -1), "e": (3, +1), "d": (3, -1),
    "r": (4, +1), "f": (4, -1), "t": (5, +1), "g": (5, -1), "y": (6, +1), "h": (6, -1),
}
# arrow keys (+ I/K for depth) -> world-frame Cartesian jog
CARTESIAN_KEYS = {"left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN", "i": "FORWARD", "k": "BACKWARD"}

KEY_HELP = """\
KEYBOARD (manual mode)
  Q/A J1 base   W/S J2 shoulder   E/D J3 elbow   R/F J4 wrist pitch   T/G J5 wrist roll   Y/H gripper open/close
  Arrows: end effector LEFT(-X) RIGHT(+X) UP(+Z) DOWN(-Z)     I/K: forward/back (+Y/-Y)
  Z home   X reset (re-arm after stop)   ESC emergency stop
KEYBOARD (emg mode, mock gestures)
  1 left  2 right  3 up  4 down  5 fist_close  0 rest (release = rest)"""


class KeyboardController:
    def __init__(self, sink: Callable[[mc.MotionCommand], bool], mode: str = "manual",
                 emg_source: KeyboardEMGSource | None = None):
        self.sink = sink
        self.mode = mode
        self.emg_source = emg_source
        self._active: str | None = None

    def update(self, held: set, triggered: set) -> None:
        # --- system keys (always available), edge-triggered
        if "esc" in triggered:
            self.sink(mc.MotionCommand(mc.ESTOP, source="keyboard"))
        if "x" in triggered:
            self.sink(mc.MotionCommand(mc.RESET, source="keyboard"))
        if "z" in triggered and self.mode == "manual":
            self.sink(mc.MotionCommand.home(source="keyboard"))

        if self.mode == "emg":
            if self.emg_source is not None:
                digits = [k for k in KEY_TO_GESTURE if k in held]
                if digits:
                    self.emg_source.set_gesture(KEY_TO_GESTURE[digits[-1]])
                else:
                    self.emg_source.release()
            return

        # --- manual jogging: pick one active key
        motion = [k for k in held if k in JOINT_KEYS or k in CARTESIAN_KEYS]
        if self._active not in motion:
            self._active = sorted(motion)[0] if motion else None
        k = self._active
        if k is None:
            if self._was_jogging:
                self.sink(mc.MotionCommand.hold(source="keyboard"))
            self._was_jogging = False
            return
        self._was_jogging = True
        if k in JOINT_KEYS:
            j, s = JOINT_KEYS[k]
            self.sink(mc.MotionCommand.joint_jog(j, s, source="keyboard"))
        else:
            self.sink(mc.MotionCommand(mc.CARTESIAN, CARTESIAN_KEYS[k], source="keyboard"))

    _was_jogging = False
