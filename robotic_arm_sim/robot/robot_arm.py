"""RobotArm: the kinematic + servo model of the 6-DOF arm. Knows nothing about EMG or physics."""
from __future__ import annotations

import numpy as np

from config.robot_config import RobotConfig, default_config
from robot.gripper import Gripper
from robot.joints import Joint
from robot.kinematics import IKResult, Kinematics, NumericalIKSolver, Pose, N_POSE_JOINTS
from robot.links import Link


class RobotArm:
    def __init__(self, config: RobotConfig | None = None, ik_solver=None):
        self.config = config or default_config()
        self.joints = [Joint(c) for c in self.config.joints]          # J1..J6
        self.links = [Link(i, c) for i, c in enumerate(self.config.links)]
        self.kinematics = Kinematics(self.config)
        self.ik_solver = ik_solver or NumericalIKSolver(self.kinematics)
        self.gripper = Gripper(self.joints[5], self.config.gripper)
        self.servo_parameters = {j.id: j.servo.cfg for j in self.joints}
        self.joint_limits = {j.id: j.limit for j in self.joints}
        self._warnings: list[str] = []
        self.reset()

    # ------------------------------------------------------------------ state access
    @property
    def joint_angles(self) -> np.ndarray:
        """Current servo angles of J1..J6 (rad)."""
        return np.array([j.angle for j in self.joints])

    @property
    def joint_targets(self) -> np.ndarray:
        return np.array([j.target for j in self.joints])

    @property
    def end_effector(self) -> Pose:
        return self.get_end_effector_pose()

    def forward_kinematics(self, joint_angles) -> Pose:
        """Pose (position, roll/pitch/yaw, 4x4 matrix) of the TCP for J1..J5 (a 6th value is ignored)."""
        return self.kinematics.pose(np.asarray(joint_angles, dtype=float)[:N_POSE_JOINTS])

    def get_end_effector_pose(self) -> Pose:
        return self.forward_kinematics(self.joint_angles)

    def get_joint_positions(self) -> np.ndarray:
        """(6,3) world positions of J1..J6."""
        return self.kinematics.joint_positions(self.joint_angles[:N_POSE_JOINTS])

    def inverse_kinematics(self, target_position, target_orientation=None, seed=None, tool_pitch=None) -> IKResult:
        seed = self.joint_targets if seed is None else seed
        return self.ik_solver.solve(target_position, target_orientation, seed=seed, tool_pitch=tool_pitch)

    # ------------------------------------------------------------------ commands (set servo TARGETS)
    def set_joint_angle(self, joint_id: int, angle: float) -> bool:
        """Target a single joint (1-based id). Clamps at the limit; returns True if clamped."""
        j = self.joints[joint_id - 1]
        clamped = j.servo.set_target(angle)
        if clamped:
            self._warnings.append(f"J{joint_id} joint limit reached")
        return clamped

    def set_joint_angles(self, joint_angles, immediate: bool = False) -> list:
        """Target J1..J5 (or J1..J6). Returns the list of joint ids that were clamped."""
        clamped = []
        for i, a in enumerate(joint_angles[:6]):
            if self.set_joint_angle(i + 1, float(a)):
                clamped.append(i + 1)
        if immediate:
            for j in self.joints:
                j.servo.reset(j.servo.target_angle)
        return clamped

    def home(self) -> None:
        for j in self.joints:
            j.servo.set_target(j.cfg.home)

    def stop(self) -> None:
        for j in self.joints:
            j.servo.stop()

    def reset(self) -> None:
        """Snap every servo to its home angle and clear warnings (simulation reset)."""
        for j in self.joints:
            j.servo.reset()
        self._warnings.clear()

    # ------------------------------------------------------------------ simulation step
    def update(self, dt: float) -> None:
        for j in self.joints:
            j.servo.update(dt)

    def drain_warnings(self) -> list:
        w, self._warnings = self._warnings, []
        return w

    def gravity_torques(self, payload_kg: float = 0.0) -> np.ndarray:
        return self.kinematics.gravity_torques(self.joint_angles[:N_POSE_JOINTS], payload_kg)
