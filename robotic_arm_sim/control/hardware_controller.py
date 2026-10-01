"""HardwareRobotController - PLACEHOLDER for the real LewanSoul/STM32 arm (Phase 11).

It implements the same ``RobotController`` interface as the simulator, so the EMG layer,
planner and keyboard code run unchanged. What is missing is only the transport:

  * open the serial/USB link to the 6-channel bus-servo controller board,
  * encode a "move servo N to position P over T ms" frame for the board's protocol,
  * (optionally) read back servo positions/voltage for telemetry and the low-voltage alarm.

TODO: take the exact frame format from the controller's protocol document - nothing about it
is assumed here. ``joint_to_ticks`` shows how the calibrated direction / zero offset from
config/robot_config.py are applied; the tick scale is a PLACEHOLDER.

Keep the same safety rules as the simulator: clamp to joint limits, reject commands while
e-stopped, and rate-limit using the servo velocity limits (reuse SimulatedServo as the
"desired motion generator" and send its angle to the hardware each tick).
"""
from __future__ import annotations

from control.controller import RobotController
from control.motion_command import MotionCommand
from robot.robot_arm import RobotArm


class HardwareRobotController(RobotController):
    def __init__(self, port: str, baudrate: int = 115200, config=None):
        self.port, self.baudrate = port, baudrate
        self.arm = RobotArm(config)           # kinematic model reused for IK / limits / calibration

    def joint_to_ticks(self, joint_angles) -> list:
        """Joint angles (rad, J1..J6) -> per-servo position counts using the calibration in the config."""
        return [j.servo.to_ticks(a) for j, a in zip(self.arm.joints, joint_angles)]

    def _todo(self, what: str):
        raise NotImplementedError(f"HardwareRobotController.{what}: implement the LewanSoul controller transport")

    def move_joint(self, joint_id, angle, wait=False): self._todo("move_joint")
    def move_joints(self, angles, wait=False): self._todo("move_joints")
    def move_cartesian(self, position, orientation=None, tool_pitch=None, speed=None, wait=False): self._todo("move_cartesian")
    def open_gripper(self, wait=False): self._todo("open_gripper")
    def close_gripper(self, wait=False): self._todo("close_gripper")
    def set_gripper_position(self, value, wait=False): self._todo("set_gripper_position")
    def stop(self): self._todo("stop")
    def emergency_stop(self): self._todo("emergency_stop")
    def reset(self): self._todo("reset")
    def home(self, wait=False): self._todo("home")
    def submit(self, command: MotionCommand) -> bool: self._todo("submit")
    def snapshot(self): self._todo("snapshot")
