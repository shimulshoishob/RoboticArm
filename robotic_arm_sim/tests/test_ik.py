import math

import numpy as np
import pytest

from robot.robot_arm import RobotArm


@pytest.fixture
def arm():
    return RobotArm()


def test_ik_roundtrip_position_only(arm):
    rng = np.random.default_rng(1)
    k = arm.kinematics
    ok = 0
    for _ in range(25):
        q = rng.uniform(k.lower * 0.7, k.upper * 0.7)
        target = k.forward(q)[:3, 3]
        r = arm.inverse_kinematics(target, seed=arm.config.home_angles)
        if r.success:
            ok += 1
            assert np.linalg.norm(k.forward(r.joint_angles)[:3, 3] - target) < 1.1e-3
            assert np.all(r.joint_angles >= k.lower - 1e-9) and np.all(r.joint_angles <= k.upper + 1e-9)
    assert ok >= 23          # numerical solver with restarts: allow rare local-minimum misses


def test_ik_respects_joint_limits_by_construction(arm):
    k = arm.kinematics
    r = arm.inverse_kinematics([0.0, 0.14, 0.06], tool_pitch=math.pi)
    assert r.success
    assert np.all(r.joint_angles >= k.lower) and np.all(r.joint_angles <= k.upper)


def test_ik_pitch_mode_holds_tool_pitch(arm):
    k = arm.kinematics
    r = arm.inverse_kinematics([0.04, 0.14, 0.064], tool_pitch=math.pi)
    assert r.success
    T = k.forward(r.joint_angles)
    pitch = k.tool_pitch(T, r.joint_angles[0])
    assert abs(math.remainder(pitch - math.pi, 2 * math.pi)) < math.radians(1.1)


def test_ik_full_pose_reachable_orientation(arm):
    q = np.radians([10, 25, 50, 90, 20])
    pose = arm.forward_kinematics(q)
    r = arm.inverse_kinematics(pose.position, pose.matrix[:3, :3], seed=q + 0.1)
    assert r.success and r.mode == "full"
    assert np.linalg.norm(arm.forward_kinematics(r.joint_angles).position - pose.position) < 1.1e-3


def test_unreachable_position_reports_clear_error(arm):
    r = arm.inverse_kinematics([0.0, 0.8, 0.1])
    assert not r.success
    assert "unreachable" in r.message.lower() and "reach" in r.message.lower()


def test_unreachable_orientation_reports_clear_error(arm):
    # tool pointing straight up at a low position is not achievable with the wrist limits
    r = arm.inverse_kinematics([0.0, 0.14, 0.04], (0.0, 0.0, 0.0))
    assert not r.success
    assert "orientation" in r.message.lower() or "unreachable" in r.message.lower()


def test_target_outside_joint_limits_fails_not_clamps():
    """Within geometric reach, but J1 is limited to +-10 deg by calibration -> must fail, not silently clamp."""
    from config.robot_config import default_config
    cfg = default_config()
    cfg.joints[0].limit.min_angle, cfg.joints[0].limit.max_angle = math.radians(-10), math.radians(10)
    a = RobotArm(cfg)
    r = a.inverse_kinematics([0.15, 0.05, 0.1])                       # needs J1 ~ 71 deg
    assert not r.success and "limits" in r.message.lower()


def test_ik_is_deterministic_and_cached(arm):
    a = arm.inverse_kinematics([0.05, 0.16, 0.07], tool_pitch=math.pi)
    b = arm.inverse_kinematics([0.05, 0.16, 0.07], tool_pitch=math.pi)
    assert a is b                                           # cache hit returns the stored result
    arm2 = RobotArm()
    c = arm2.inverse_kinematics([0.05, 0.16, 0.07], tool_pitch=math.pi)
    assert c.joint_angles == pytest.approx(a.joint_angles)
