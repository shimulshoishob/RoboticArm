import math

import numpy as np
import pytest

from control import motion_command as mc
from control.controller import SimulatedRobotController
from control.pick_and_place import pick_and_place
from control.safety import SafetyMonitor, Severity
from simulation.environment import Environment
from simulation.objects import make_cube
from simulation.physics import PyBulletBackend, pybullet_available
from simulation.urdf import build_urdf
import xml.dom.minidom as minidom


def test_step_command_moves_exact_distance():
    ctl = SimulatedRobotController()
    ctl.connect()
    p0 = ctl.snapshot().tcp_position.copy()
    ctl.submit(mc.MotionCommand.cartesian("UP", magnitude_mm=20, speed_mm_s=50))
    assert ctl.wait_until_settled()
    d = ctl.snapshot().tcp_position - p0
    assert d[2] == pytest.approx(0.020, abs=2e-3) and abs(d[0]) < 1e-3


def test_move_to_straight_line_and_pitch_hold():
    ctl = SimulatedRobotController()
    ctl.connect()
    target = np.array([0.05, 0.16, 0.09])
    assert ctl.move_cartesian(target, tool_pitch=math.pi, speed=0.08, wait=True)
    assert np.linalg.norm(ctl.snapshot().tcp_position - target) < 2e-3


def test_move_to_outside_workspace_refused():
    ctl = SimulatedRobotController()
    ctl.connect()
    assert ctl.move_cartesian([0.5, 0.2, 0.1]) is False


def test_joint_target_and_home():
    ctl = SimulatedRobotController()
    ctl.connect()
    ctl.move_joints(np.radians([30, 25, 40, 90, 10, 0]), wait=True)
    assert np.degrees(ctl.arm.joint_angles[:5]) == pytest.approx([30, 25, 40, 90, 10], abs=0.1)
    assert ctl.arm.gripper.opening == pytest.approx(0, abs=1e-3)
    ctl.home(wait=True)
    assert ctl.arm.joint_angles == pytest.approx(ctl.config.home_angles, abs=1e-3)


def test_overload_and_stall_detection():
    ctl = SimulatedRobotController()
    ctl.connect()
    ctl.arm.joints[1].servo.torque_limit = 0.05          # pretend a weak servo: gravity load exceeds it
    ctl.run_for(0.2)
    assert any("overload" in m.lower() for m in ctl.snapshot().safety.messages)
    ctl.run_for(0.6)                                      # sustained -> latched fault, refuses commands
    assert ctl.faulted
    assert ctl.submit(mc.MotionCommand.cartesian("LEFT")) is False
    ctl.arm.joints[1].servo.torque_limit = 1.6
    ctl.reset()
    assert not ctl.faulted

    sm = SafetyMonitor(ctl.config)
    for _ in range(300):                                  # commanded vs actual differ for > stall_time
        st = sm.evaluate(0.0, 1 / 240, ctl.arm, actual_angles=ctl.arm.joint_angles[:5] + np.array([0.5, 0, 0, 0, 0]))
    assert st.level == Severity.FAULT and any("Locked-rotor" in m for m in st.messages)


def test_urdf_is_well_formed_and_has_all_joints():
    from config.robot_config import default_config
    for axes in (False, True):
        doc = minidom.parseString(build_urdf(default_config(), axes=axes))
        names = {j.getAttribute("name") for j in doc.getElementsByTagName("joint")}
        assert {"j1", "j2", "j3", "j4", "j5", "j6_finger_l", "j6_finger_r", "tcp_fixed"} <= names


def test_hardware_controller_shares_interface():
    from control.controller import RobotController
    from control.hardware_controller import HardwareRobotController
    hw = HardwareRobotController("/dev/null")
    assert isinstance(hw, RobotController) and isinstance(SimulatedRobotController(), RobotController)
    assert all(isinstance(t, int) for t in hw.joint_to_ticks(np.zeros(6)))
    with pytest.raises(NotImplementedError):
        hw.close_gripper()


pb = pytest.mark.skipif(not pybullet_available(), reason="pybullet not installed")


@pb
def test_pybullet_tracks_servo_targets_and_holds_against_gravity():
    ctl = SimulatedRobotController(backend=PyBulletBackend(__import__("config.robot_config", fromlist=["x"]).default_config()))
    ctl.connect(Environment())
    ctl.move_joints(np.radians([0, 40, 50, 80, 0]), wait=True)
    ctl.run_for(1.0)
    q_act, _ = ctl.backend.read_state()
    assert q_act == pytest.approx(ctl.arm.joint_angles[:5], abs=0.03)     # physics follows the servo model
    assert ctl.snapshot().safety.ok
    ctl.shutdown()


@pb
def test_pick_and_place_in_physics():
    from config.robot_config import default_config
    env = Environment()
    cube = env.add(make_cube("cube", 0.0, 0.15))
    ctl = SimulatedRobotController(backend=PyBulletBackend(default_config()))
    ctl.connect(env)
    ctl.home(wait=True)
    pick = np.array([0.0, 0.15, 0.022])
    place = np.array([0.08, 0.15, 0.024])
    assert pick_and_place(ctl, "cube", pick, place)
    pos = np.array(ctl.backend.object_position("cube"))
    assert np.linalg.norm(pos[:2] - place[:2]) < 0.012                     # carried 80 mm and released
    assert pos[2] == pytest.approx(0.015, abs=0.004)                       # resting on the table again
    ctl.shutdown()


@pb
def test_gripper_blocked_by_object_does_not_trigger_stall_fault():
    from config.robot_config import default_config
    env = Environment()
    env.add(make_cube("cube", 0.0, 0.15))
    ctl = SimulatedRobotController(backend=PyBulletBackend(default_config()))
    ctl.connect(env)
    ctl.move_cartesian([0.0, 0.15, 0.022], tool_pitch=math.pi, wait=True)
    ctl.close_gripper(wait=True)
    ctl.run_for(1.5)
    assert not ctl.faulted and ctl.backend.read_state()[1] > 0.3           # fingers held open by the cube
    ctl.shutdown()
