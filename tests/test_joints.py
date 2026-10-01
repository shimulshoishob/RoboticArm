import math

import numpy as np
import pytest

from control import motion_command as mc
from control.controller import SimulatedRobotController
from control.safety import Severity
from robot.robot_arm import RobotArm
from simulation.collision import CollisionChecker


@pytest.fixture
def arm():
    return RobotArm()


@pytest.fixture
def ctl():
    c = SimulatedRobotController()
    c.connect()
    return c


# ---------------------------------------------------------------- joint limits
def test_joint_limit_is_clamped_and_warned(arm):
    j2 = arm.joints[1]
    clamped = arm.set_joint_angle(2, j2.limit.max_angle + 0.5)
    assert clamped
    assert j2.servo.target_angle == pytest.approx(j2.limit.max_angle)
    assert "J2 joint limit reached" in arm.drain_warnings()


def test_within_limit_not_clamped(arm):
    assert not arm.set_joint_angle(1, 0.3)
    assert arm.drain_warnings() == []


def test_controller_reports_joint_limit_warning(ctl):
    ctl.move_joint(2, math.radians(170))
    ctl.run_for(0.05)
    msgs = ctl.snapshot().safety.messages
    assert any("J2 joint limit reached" in m for m in msgs)
    assert ctl.arm.joints[1].target <= ctl.arm.joints[1].limit.max_angle + 1e-9


# ---------------------------------------------------------------- servo
def test_servo_does_not_teleport_and_respects_limits(arm):
    s = arm.joints[0].servo
    s.set_target(1.0)
    dt, prev, vmax_seen, amax_seen, prev_v = 1 / 240, s.angle, 0.0, 0.0, 0.0
    for _ in range(2000):
        s.update(dt)
        assert abs(s.angle - prev) <= s.max_velocity * dt + 1e-9
        vmax_seen = max(vmax_seen, abs(s.velocity))
        if not s.at_target:                                           # the final snap-to-target step is exempt
            amax_seen = max(amax_seen, abs(s.velocity - prev_v) / dt)
        prev, prev_v = s.angle, s.velocity
        if s.at_target:
            assert abs(prev_v) <= 2 * s.limit.decel * dt + 1e-9       # ...but arrives nearly stopped
            break
    assert s.angle == pytest.approx(1.0, abs=1e-4)                  # arrives
    assert vmax_seen <= s.max_velocity + 1e-9
    assert amax_seen <= s.limit.max_acceleration * 1.01
    assert vmax_seen > 0.5 * s.max_velocity                          # actually accelerates


def test_servo_no_overshoot_and_reversal(arm):
    s = arm.joints[2].servo
    for target in (0.8, -0.5, 0.2):
        s.set_target(target)
        for _ in range(4000):
            s.update(1 / 240)
            if s.at_target:
                break
        assert s.angle == pytest.approx(target, abs=1e-4)


def test_servo_stop_and_reset(arm):
    s = arm.joints[0].servo
    s.set_target(1.0)
    for _ in range(30):
        s.update(1 / 240)
    s.stop()
    a = s.angle
    s.update(1 / 240)
    assert s.angle == a and s.velocity == 0
    s.reset()
    assert s.angle == s.home_angle


def test_servo_hardware_mapping_roundtrip(arm):
    s = arm.joints[0].servo
    s.direction, s.offset = -1, math.radians(5)
    assert s.from_hardware(s.to_hardware(0.4)) == pytest.approx(0.4)
    assert isinstance(s.to_ticks(0.4), int)


# ---------------------------------------------------------------- gripper
def test_gripper_open_close_partial(arm):
    g = arm.gripper
    g.close()
    for _ in range(600):
        arm.update(1 / 240)
    assert g.opening == pytest.approx(0.0, abs=1e-3) and g.width == pytest.approx(0, abs=1e-4)
    g.open()
    for _ in range(600):
        arm.update(1 / 240)
    assert g.opening == pytest.approx(1.0, abs=1e-3)
    assert g.width == pytest.approx(2 * arm.config.gripper.finger_travel, abs=1e-4)
    g.set_position(0.5)
    for _ in range(600):
        arm.update(1 / 240)
    assert g.opening == pytest.approx(0.5, abs=1e-3)
    g.set_position(7.0)                                              # clamped to 1.0
    assert g.target_opening == pytest.approx(1.0)


def test_gripper_via_controller_api(ctl):
    ctl.close_gripper(wait=True)
    assert ctl.arm.gripper.opening == pytest.approx(0.0, abs=1e-3)
    ctl.set_gripper_position(0.4, wait=True)
    assert ctl.arm.gripper.opening == pytest.approx(0.4, abs=1e-3)
    ctl.open_gripper(wait=True)
    assert ctl.arm.gripper.opening == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- collision
def test_home_pose_is_collision_free(arm):
    chk = CollisionChecker(arm.kinematics, arm.config)
    assert not chk.check(arm.config.home_angles[:5], 1.0).colliding


def test_floor_collision_detected(arm):
    chk = CollisionChecker(arm.kinematics, arm.config)
    q = np.radians([0, 60, 60, 60, 0])                               # arm swung down into the table
    rep = chk.check(q, 1.0)
    assert rep.colliding and any("table" in p for p in rep.pairs)


def test_self_collision_detected(arm):
    chk = CollisionChecker(arm.kinematics, arm.config)
    q = np.radians([0, 85, 118, 118, 0])                             # folded back onto itself
    assert chk.check(q, 1.0).colliding


def test_controller_vetoes_colliding_target(ctl):
    before = ctl.arm.joint_targets.copy()
    ctl.move_joints(np.radians([0, 60, 60, 60, 0]))
    ctl.run_for(0.05)
    assert ctl.arm.joint_targets[:5] == pytest.approx(before[:5])    # target was NOT accepted
    snap = ctl.snapshot()
    assert snap.collision != "NONE" and not snap.safety.ok


def test_cartesian_down_stops_above_table(ctl):
    ctl.submit(mc.MotionCommand(mc.CARTESIAN, "DOWN", speed=0.06))
    for _ in range(240 * 4):
        ctl.submit(mc.MotionCommand(mc.CARTESIAN, "DOWN", speed=0.06))
        ctl.step()
    z_tip = ctl.snapshot().tcp_position[2] - (0.105 - 0.090)
    assert z_tip >= -1e-3                                            # fingertips never go below the table


# ---------------------------------------------------------------- emergency stop
def test_emergency_stop_halts_immediately_and_requires_reset(ctl):
    ctl.move_joints(np.radians([40, 30, 40, 60, 0]))
    ctl.run_for(0.2)
    assert any(abs(j.servo.velocity) > 0 for j in ctl.arm.joints)
    ctl.emergency_stop()
    assert all(j.servo.velocity == 0 for j in ctl.arm.joints)
    snap_angles = ctl.arm.joint_angles.copy()
    ctl.run_for(0.5)
    assert ctl.arm.joint_angles == pytest.approx(snap_angles)        # no creeping after stop
    assert ctl.snapshot().estopped and ctl.snapshot().safety.level == Severity.FAULT

    assert ctl.submit(mc.MotionCommand.cartesian("LEFT")) is False   # refused until reset
    ctl.move_joint(1, 0.5)
    ctl.run_for(0.5)
    assert ctl.arm.joint_angles == pytest.approx(snap_angles)

    ctl.reset()
    assert not ctl.snapshot().estopped
    ctl.move_joint(1, 0.5, wait=True)
    assert ctl.arm.joints[0].angle == pytest.approx(0.5, abs=1e-3)


def test_estop_via_motion_command_and_stop_is_not_latching(ctl):
    ctl.move_joint(1, 1.0)
    ctl.run_for(0.1)
    ctl.stop()
    a = ctl.arm.joint_angles.copy()
    ctl.run_for(0.2)
    assert ctl.arm.joint_angles == pytest.approx(a)
    assert ctl.submit(mc.MotionCommand.cartesian("LEFT"))            # still accepting commands
    ctl.submit(mc.MotionCommand(mc.ESTOP))
    assert ctl.estopped


def test_unreachable_cartesian_target_raises_warning(ctl):
    assert ctl.move_cartesian([0.0, 0.9, 0.1]) is False
    assert any("unreachable" in m.lower() for m in ctl.snapshot().safety.messages)
