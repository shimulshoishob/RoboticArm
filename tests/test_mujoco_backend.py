"""MuJoCo physics backend (single-file robotic_arm.py)."""
import math
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import robotic_arm as ra

mj = pytest.mark.skipif(not ra.HAS_MUJOCO, reason="mujoco not installed")
HZ = 240.0


def make(env=None, hz=HZ):
    cfg = ra.default_config()
    ctl = ra.SimulatedRobotController(cfg, ra.MuJoCoBackend(cfg, dt=1.0 / hz), control_hz=hz)
    ctl.connect(env or ra.Environment())
    return ctl


@mj
def test_mjcf_matches_kinematics_for_random_poses():
    """The MuJoCo TCP site must sit exactly where robot/kinematics.py says the TCP is."""
    cfg = ra.default_config()
    be = ra.MuJoCoBackend(cfg, dt=1 / 120)
    be.connect(ra.Environment())
    kin = ra.RobotArm(cfg).kinematics
    rng = np.random.default_rng(3)
    lo = np.array([j.limit.min_angle for j in cfg.joints[:5]])
    hi = np.array([j.limit.max_angle for j in cfg.joints[:5]])
    for _ in range(8):
        q = rng.uniform(lo, hi)
        be.reset_state(q, 1.0)
        assert be.tcp_position() == pytest.approx(kin.forward(q)[:3, 3], abs=1e-5)


@mj
def test_physics_tracks_servo_targets_and_holds_against_gravity():
    ctl = make()
    ctl.move_joints(np.radians([0, 40, 50, 80, 0]), wait=True)
    ctl.run_for(1.0)
    q_act, _ = ctl.backend.read_state()
    assert q_act == pytest.approx(ctl.arm.joint_angles[:5], abs=0.03)
    assert ctl.snapshot().safety.ok
    ctl.shutdown()


@mj
def test_pick_and_place_in_physics():
    env = ra.Environment()
    env.add(ra.make_cube("cube", 0.0, 0.15))
    ctl = make(env)
    ctl.home(wait=True)
    pick, place = np.array([0.0, 0.15, 0.022]), np.array([0.08, 0.15, 0.024])
    assert ra.pick_and_place(ctl, "cube", pick, place)
    pos = np.array(ctl.backend.object_position("cube"))
    assert np.linalg.norm(pos[:2] - place[:2]) < 0.012          # carried 80 mm and released
    assert pos[2] == pytest.approx(0.015, abs=0.004)            # resting on the table again
    ctl.shutdown()


@mj
def test_gripper_blocked_by_object_does_not_trigger_stall_fault():
    env = ra.Environment()
    env.add(ra.make_cube("cube", 0.0, 0.15))
    ctl = make(env)
    ctl.move_cartesian([0.0, 0.15, 0.022], tool_pitch=math.pi, wait=True)
    ctl.close_gripper(wait=True)
    ctl.run_for(1.5)
    assert not ctl.faulted and ctl.backend.read_state()[1] > 0.3     # fingers held open by the cube
    ctl.shutdown()


@mj
def test_unwanted_contacts_are_reported_but_grasp_and_base_are_not():
    env = ra.Environment()
    env.add(ra.make_cube("cube", 0.0, 0.15))
    ctl = make(env)
    ctl.run_for(0.3)
    assert ctl.backend.get_contacts() == []                    # base sits on the table, nothing else touches
    ctl.backend.reset_state(np.radians([0, 90, 90, 0, 0]), 1.0)   # fold the arm down into the table
    ctl.backend.set_targets(np.radians([0, 90, 90, 0, 0]), 1.0)
    for _ in range(60):
        ctl.backend.step()
    assert ctl.backend.get_contacts()
    ctl.shutdown()


@mj
def test_object_reset_and_view_qpos():
    env = ra.Environment.default_scene()
    be = ra.MuJoCoBackend(ra.default_config(), dt=1 / 120)
    be.connect(env)
    be.reset_object("cube", (0.05, 0.2, 0.3))
    assert be.object_position("cube") == pytest.approx((0.05, 0.2, 0.3), abs=1e-6)
    assert be.view_qpos().shape == (be.model.nq,)


@mj
def test_mjcf_export(tmp_path):
    p = ra.export_mjcf(ra.default_config(), tmp_path / "arm.xml", ra.Environment.default_scene())
    import mujoco
    assert mujoco.MjModel.from_xml_path(str(p)).nu == 7


@mj
def test_physics_step_is_cheap_enough_for_a_fanless_air():
    ctl = make(hz=120.0)
    t0 = time.perf_counter()
    for _ in range(600):
        ctl.step()
    per_tick_ms = (time.perf_counter() - t0) / 600 * 1000
    assert per_tick_ms < 3.0, f"{per_tick_ms:.2f} ms per 120 Hz control tick"
    ctl.shutdown()


@mj
@pytest.mark.skipif(not ra.HAS_QT, reason="PyQt5 not installed")
def test_dashboard_renders_the_mujoco_scene_and_keys_drive_the_arm():
    rt = ra.SimulationRuntime(headless=True, physics=True, save_log=False, mode="full", control_hz=120.0)
    app = ra.QApplication.instance() or ra.QApplication([])
    rt.start()
    win = ra.ArmDashboard(rt, fps=30, max_mpix=0.3)
    try:
        win.show()
        app.processEvents()
        win.view.render_frame()
        assert not win.view.error, win.view.error
        assert win.view.renderer is not None and win.view.ms > 0
        frame = win.view.renderer.render()
        assert frame.shape[2] == 3 and frame.std() > 5               # not a blank image
        win.view.renderer.orbit(30, 10); win.view.renderer.pan(5, 5); win.view.renderer.zoom(2)
        win.view.set_frames(True); win.view.set_shadows(False)
        assert win.view.renderer.render().shape == frame.shape
        # keys: the window's own key path (held set -> 30 Hz poll_keys -> KeyboardController -> arm)
        win.isActiveWindow = lambda: True                           # offscreen windows may never become "active"
        z0 = rt.controller.snapshot().tcp_position[2]
        win._held, win._trig = {"up"}, {"up"}
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            app.processEvents()
            time.sleep(0.02)
        win._held.clear()
        assert rt.controller.snapshot().tcp_position[2] > z0 + 0.005
        t_end = time.monotonic() + 0.4
        while time.monotonic() < t_end:                              # released -> HOLD, arm stops
            app.processEvents()
            time.sleep(0.02)
        z1 = rt.controller.snapshot().tcp_position[2]
        time.sleep(0.2)
        app.processEvents()
        assert abs(rt.controller.snapshot().tcp_position[2] - z1) < 0.002
    finally:
        win.close()
        rt.stop()


def _key(app, widget, key, text="", press=True):
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QKeyEvent
    ev = QKeyEvent(QEvent.KeyPress if press else QEvent.KeyRelease, key, ra.Qt.NoModifier, text)
    app.sendEvent(widget, ev)


@mj
@pytest.mark.skipif(not (ra.HAS_QT and ra.HAS_PYQTGRAPH), reason="PyQt5 / pyqtgraph not installed")
def test_fullscreen_emg_window_help_keys_and_guidance(monkeypatch):
    rt = ra.SimulationRuntime(headless=True, physics=True, save_log=False, mode="full", control_hz=120.0)
    app = ra.QApplication.instance() or ra.QApplication([])
    rt.start()
    win = ra.ArmDashboard(rt, fps=30, max_mpix=0.3)
    try:
        win.show()
        monkeypatch.setattr(win, "isActiveWindow", lambda: True)
        app.processEvents()
        # ---- guidance changes with state, and starts with something actionable
        win.refresh_guidance()
        assert "Step 1" in win.lbl_next.text()
        win.is_connected = True
        win.refresh_guidance()
        assert "Step 2" in win.lbl_next.text()
        win.is_connected = False
        # ---- real key events through the app-wide filter
        _key(app, win, ra.Qt.Key_Up, "")
        assert "up" in win._held and "move up" in win._key_hint_text()
        _key(app, win, ra.Qt.Key_Up, "", press=False)
        assert "up" not in win._held
        _key(app, win, ra.Qt.Key_F1)
        assert win.help_dlg is not None and win.help_dlg.isVisible()
        win.help_dlg.close()
        # ---- full screen: F11 opens, F11 closes, esc still reaches the e-stop path
        _key(app, win, ra.Qt.Key_F11)
        assert win.fs is not None
        win.fs.view.render_frame()
        assert not win.fs.view.error
        monkeypatch.setattr(ra.QApplication, "activeWindow", staticmethod(lambda: win.fs))
        win.refresh_ui()
        assert "Keyboard" in win.fs.lbl.text() or "mm" in win.fs.lbl.text()
        _key(app, win.fs, ra.Qt.Key_Escape)
        win.poll_keys()
        assert rt.controller.estopped
        win.send(ra.MotionCommand(ra.RESET, source="test"))
        _key(app, win.fs, ra.Qt.Key_Escape, press=False)
        _key(app, win.fs, ra.Qt.Key_F11)
        assert win.fs is None
        monkeypatch.undo()
        monkeypatch.setattr(win, "isActiveWindow", lambda: True)
        # ---- EMG window: hidden until asked, shows 8 labelled channels when data flows
        assert win.emg_win is None
        rng = np.random.default_rng(0)
        win.num_channels, win.emg_channel_count, win.is_connected = 11, 8, True
        win._reset_stream_state()
        for i in range(60):
            b = rng.normal(0, 20, (25, 11)).astype(np.float32)
            b[:, 2] += 200 * np.sin(np.arange(25) / 3 + i)
            win.on_stream_batch(b)
        win.btn_emg.setChecked(True)
        assert win.emg_win is not None and win.emg_win.isVisible()
        win.emg_win.refresh()
        assert len(win.emg_win.curves) == 8
        title = win.emg_win.plots[2].titleLabel.text
        assert "CH3" in title
        win.emg_win.chk_same.setChecked(True)
        win.emg_win.btn_pause.setChecked(True)
        win.emg_win.close()
        assert not win.btn_emg.isChecked()
    finally:
        win.close()
        rt.stop()
