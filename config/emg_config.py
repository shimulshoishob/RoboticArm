"""EMG-layer tunables. Nothing here touches the robot core."""
from __future__ import annotations

GESTURES = ("left", "right", "up", "down", "fist_close", "rest")

# Predictions below this confidence are replaced by REST before smoothing.
EMG_CONFIDENCE_THRESHOLD = 0.75

# Sliding-window majority vote.
SMOOTHING_WINDOW = 5            # number of recent predictions considered
SMOOTHING_MIN_AGREEMENT = 0.6   # fraction of the window the winner needs, else REST

# Cartesian jog performed while a direction gesture is held.
EMG_SPEED_M_S = 0.040           # 40 mm/s
# Continuous commands expire if not refreshed (EMG dropout -> robot stops itself).
COMMAND_TIMEOUT_S = 0.30
EMG_POLL_HZ = 50

# gesture -> (command_type, direction).  REST -> HOLD (stop moving, keep position).
GESTURE_TO_COMMAND = {
    "left": ("CARTESIAN", "LEFT"),       # -X
    "right": ("CARTESIAN", "RIGHT"),     # +X
    "up": ("CARTESIAN", "UP"),           # +Z
    "down": ("CARTESIAN", "DOWN"),       # -Z
    "fist_close": ("GRIPPER", "CLOSE"),
    "rest": ("HOLD", "NONE"),
}

# Accepted spellings coming from a classifier.
GESTURE_ALIASES = {
    "fist": "fist_close", "close": "fist_close", "fistclose": "fist_close", "fist close": "fist_close",
    "idle": "rest", "none": "rest", "neutral": "rest", "hold": "rest",
}

# If True, a fist while the gripper is closed re-opens it (toggle). Default follows the spec:
# FIST_CLOSE only closes; opening is done from keyboard / GUI.
FIST_CLOSE_TOGGLES = False
