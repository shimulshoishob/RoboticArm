import math

import numpy as np
import pytest

from config.robot_config import default_config, load_config
from robot.robot_arm import RobotArm
from utils.math_utils import matrix_to_rpy, rpy_to_matrix, rot_axis, so3_log, matrix_to_quat


@pytest.fixture
def arm():
    return RobotArm()


def test_dimensions_and_mass_match_published_spec():
    cfg = default_config()
    assert cfg.total_height == pytest.approx(0.465, abs=1e-9)
    assert cfg.total_mass == pytest.approx(1.24, abs=1e-9)
    assert len(cfg.joints) == 6


def test_zero_pose_points_straight_up(arm):
    """All joints 0: TCP directly above the base, height = sum of links up to the TCP."""
    p = arm.forward_kinematics(np.zeros(6))
    cfg = arm.config
    expected_z = sum(l.length for l in cfg.links[:5]) + cfg.tcp_offset
    assert p.position == pytest.approx([0, 0, expected_z], abs=1e-9)
    assert p.matrix[:3, :3] == pytest.approx(np.eye(3), abs=1e-9)


def test_fk_returns_4x4_rigid_transform(arm):
    q = np.radians([20, 30, -40, 50, 60, 0])
    T = arm.forward_kinematics(q).matrix
    assert T.shape == (4, 4)
    R = T[:3, :3]
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert T[3] == pytest.approx([0, 0, 0, 1])


def test_base_rotation_moves_tool_left_with_positive_j1(arm):
    """+J1 (CCW from above) swings an arm reaching along +Y toward -X ("left")."""
    q = np.radians([0, 40, 40, 0, 0, 0])
    p0 = arm.forward_kinematics(q).position
    q[0] = math.radians(30)
    p1 = arm.forward_kinematics(q).position
    assert p0[0] == pytest.approx(0, abs=1e-9) and p0[1] > 0
    assert p1[0] < 0
    assert np.hypot(*p1[:2]) == pytest.approx(np.hypot(*p0[:2]))      # same radius
    assert p1[2] == pytest.approx(p0[2])


def test_positive_pitch_leans_forward(arm):
    p = arm.forward_kinematics(np.radians([0, 90, 0, 0, 0, 0])).position
    assert p[1] > 0.3 and p[0] == pytest.approx(0, abs=1e-9)


def test_planar_two_link_closed_form(arm):
    """J2=J3 chain: compare with hand-computed planar geometry (J4=J5=0)."""
    c = arm.config
    L0 = c.links[0].length + c.links[1].length
    L2, L3 = c.links[2].length, c.links[3].length
    L4 = c.links[4].length + c.tcp_offset
    a2, a3 = math.radians(30), math.radians(45)
    y = L2 * math.sin(a2) + (L3 + L4) * math.sin(a2 + a3)
    z = L0 + L2 * math.cos(a2) + (L3 + L4) * math.cos(a2 + a3)
    p = arm.forward_kinematics(np.array([0, a2, a3, 0, 0, 0])).position
    assert p == pytest.approx([0, y, z], abs=1e-9)


def test_end_effector_pose_uses_current_servo_state(arm):
    arm.reset()
    assert arm.get_end_effector_pose().position == pytest.approx(
        arm.forward_kinematics(arm.joint_angles).position)
    assert arm.get_joint_positions().shape == (6, 3)


def test_jacobian_matches_finite_differences(arm):
    kin = arm.kinematics
    q = np.radians([15, 35, 50, 20, 10])
    J = kin.jacobian(q)
    for i in range(5):
        dq = np.zeros(5)
        dq[i] = 1e-6
        num = (kin.forward(q + dq)[:3, 3] - kin.forward(q - dq)[:3, 3]) / 2e-6
        assert J[:3, i] == pytest.approx(num, abs=1e-6)


def test_rpy_roundtrip_and_so3_log():
    for rpy in [(0.1, 0.2, 0.3), (-1.0, 0.5, 2.0), (3.0, -0.7, -2.5)]:
        R = rpy_to_matrix(*rpy)
        assert rpy_to_matrix(*matrix_to_rpy(R)) == pytest.approx(R, abs=1e-9)
    R = rot_axis([0, 0, 1], 0.7)
    assert so3_log(R) == pytest.approx([0, 0, 0.7])
    q = matrix_to_quat(R)
    assert q == pytest.approx((0, 0, math.sin(0.35), math.cos(0.35)))


def test_gravity_torques_within_servo_limits_at_home(arm):
    tau = arm.gravity_torques()
    for i in range(5):
        assert tau[i] < arm.joints[i].servo.torque_limit
    assert tau[0] == pytest.approx(0, abs=1e-9)          # base yaw axis is vertical: no gravity torque


def test_calibration_json_changes_model_without_code(tmp_path):
    f = tmp_path / "cal.json"
    f.write_text('{"joints":{"joint_1":{"min_angle":-45,"max_angle":45,"direction":-1,"zero_offset":10},'
                 '"joint_2":{"torque_limit_kgcm":12}},"links":{"link2":{"length_mm":120,"mass_kg":0.2}},'
                 '"tcp_offset_mm":100}')
    cfg = load_config(f)
    assert cfg.joint(1).limit.min_angle == pytest.approx(math.radians(-45))
    assert cfg.joint(1).servo.direction == -1
    assert cfg.joint(1).servo.zero_offset == pytest.approx(math.radians(10))
    assert cfg.joint(2).servo.torque_limit == pytest.approx(12 * 0.0980665)
    assert cfg.link("link2").length == pytest.approx(0.12) and cfg.link("link2").mass == 0.2
    assert RobotArm(cfg).forward_kinematics(np.zeros(6)).position[2] == pytest.approx(
        0.012 + 0.068 + 0.12 + 0.105 + 0.065 + 0.100)
    with pytest.raises(KeyError):
        f.write_text('{"joints":{"1":{"bogus":1}}}')
        load_config(f)
