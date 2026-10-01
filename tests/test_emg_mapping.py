import numpy as np
import pytest

from control import motion_command as mc
from control.controller import SimulatedRobotController
from emg.emg_controller import CallbackEMGSource, EMGController
from emg.gesture_mapper import GestureMapper, GestureSmoother, normalize_gesture
from emg.mock_emg import KeyboardEMGSource, ScriptedEMGSource, NoisyEMGSource


def fresh(window=5, **kw):
    ctl = SimulatedRobotController()
    ctl.connect()
    emg = EMGController(sink=ctl.submit, status_callback=ctl.set_emg_status, window=window, **kw)
    return ctl, emg


def feed(emg, ctl, gesture, n, conf=0.95, run=0.02):
    cmd = None
    for _ in range(n):
        cmd = emg.process_gesture(gesture, conf)
        ctl.run_for(run)
    return cmd


# ---------------------------------------------------------------- mapping
@pytest.mark.parametrize("gesture,direction", [("left", "LEFT"), ("right", "RIGHT"), ("up", "UP"), ("down", "DOWN")])
def test_direction_gestures_map_to_cartesian_commands(gesture, direction):
    cmd = GestureMapper().map(gesture, 0.9)
    assert cmd.command_type == mc.CARTESIAN and cmd.direction == direction
    assert cmd.source == "emg" and cmd.gesture == gesture and cmd.confidence == 0.9


def test_fist_and_rest_mapping():
    m = GestureMapper()
    assert (m.map("fist_close").command_type, m.map("fist_close").direction) == (mc.GRIPPER, "CLOSE")
    assert m.map("rest").command_type == mc.HOLD


def test_aliases_and_unknown_gesture():
    assert normalize_gesture("Fist") == "fist_close" and normalize_gesture("FIST-CLOSE") == "fist_close"
    with pytest.raises(ValueError):
        normalize_gesture("wave")
    ctl, emg = fresh()
    assert emg.process_gesture("wave", 0.99).command_type == mc.HOLD      # unknown label fails safe


def test_left_gesture_decreases_x():
    ctl, emg = fresh()
    x0 = ctl.snapshot().tcp_position[0]
    feed(emg, ctl, "left", 40)
    assert ctl.snapshot().tcp_position[0] < x0 - 0.01


def test_right_up_down_directions():
    for g, axis, sign in (("right", 0, +1), ("up", 2, +1), ("down", 2, -1)):
        ctl, emg = fresh()
        p0 = ctl.snapshot().tcp_position.copy()
        feed(emg, ctl, g, 40)
        d = ctl.snapshot().tcp_position - p0
        assert sign * d[axis] > 0.01, (g, d)
        others = [i for i in range(3) if i != axis]
        assert max(abs(d[i]) for i in others) < 0.004                      # moves along one world axis only


def test_fist_close_closes_gripper():
    ctl, emg = fresh()
    assert ctl.arm.gripper.opening == pytest.approx(1.0)
    feed(emg, ctl, "fist_close", 20)
    ctl.run_for(1.0)
    assert ctl.arm.gripper.opening == pytest.approx(0.0, abs=1e-3)


def test_rest_holds_position():
    ctl, emg = fresh()
    feed(emg, ctl, "up", 30)
    feed(emg, ctl, "rest", 8)
    ctl.run_for(0.5)
    p = ctl.snapshot().tcp_position.copy()
    feed(emg, ctl, "rest", 50)
    assert ctl.snapshot().tcp_position == pytest.approx(p, abs=1e-6)
    assert not ctl.planner.active


def test_emg_dropout_stops_robot_by_timeout():
    ctl, emg = fresh()
    feed(emg, ctl, "left", 30)
    p = ctl.snapshot().tcp_position.copy()
    ctl.run_for(1.0)                                                       # no more gestures arrive
    p2 = ctl.snapshot().tcp_position.copy()
    ctl.run_for(1.0)
    assert ctl.snapshot().tcp_position == pytest.approx(p2, abs=1e-6)      # stopped (keep-alive lapsed)
    assert np.linalg.norm(p2 - p) < 0.03                                   # only coasted briefly


def test_toggle_mode_fist_opens_and_closes():
    ctl, emg = fresh(fist_toggles=True)
    feed(emg, ctl, "fist_close", 12)
    ctl.run_for(1.0)
    assert ctl.arm.gripper.opening < 0.05
    feed(emg, ctl, "rest", 8)
    feed(emg, ctl, "fist_close", 12)
    ctl.run_for(1.0)
    assert ctl.arm.gripper.opening > 0.95


# ---------------------------------------------------------------- confidence threshold
def test_low_confidence_becomes_rest():
    ctl, emg = fresh()
    x0 = ctl.snapshot().tcp_position[0]
    feed(emg, ctl, "left", 40, conf=0.60)                                  # below 0.75
    assert emg.last_smoothed == "rest"
    assert ctl.snapshot().tcp_position[0] == pytest.approx(x0, abs=1e-6)


def test_threshold_boundary_and_configurable():
    s = GestureSmoother(window=1, confidence_threshold=0.75)
    assert s.update("left", 0.75) == "left" and s.update("left", 0.7499) == "rest"
    s2 = GestureSmoother(window=1, confidence_threshold=0.5)
    assert s2.update("left", 0.6) == "left"


# ---------------------------------------------------------------- smoothing
def run_smoother(seq, window=5, **kw):
    s = GestureSmoother(window, **kw)
    return [s.update(g, 0.95) for g in seq]


def test_stable_majority_survives_single_glitch():
    out = run_smoother(["left"] * 4 + ["rest"])
    assert out[-1] == "left"
    out = run_smoother(["left"] * 4 + ["right"])
    assert out[-1] == "left"


def test_noisy_sequence_does_not_move():
    out = run_smoother(["left", "right", "left", "rest", "right"])
    assert out[-1] == "rest"
    ctl, emg = fresh()
    x0 = ctl.snapshot().tcp_position.copy()
    for g in ["left", "right", "left", "rest", "right"]:
        emg.process_gesture(g, 0.95)
        ctl.run_for(0.02)
    assert np.linalg.norm(ctl.snapshot().tcp_position - x0) < 1e-6


def test_no_motion_until_window_fills():
    out = run_smoother(["left"] * 5)
    assert out[:2] == ["rest", "rest"] and out[2:] == ["left"] * 3          # needs ceil(0.6*5)=3 votes


def test_window_size_configurable():
    out = run_smoother(["left"] * 3 + ["rest"] * 2, window=3)
    assert out[2] == "left"
    out = run_smoother(["left"] * 3, window=9, min_agreement=0.6)
    assert out[-1] == "rest"                                               # 3/9 < 0.6
    with pytest.raises(ValueError):
        GestureSmoother(window=0)


def test_smoother_recovers_after_switch():
    out = run_smoother(["left"] * 6 + ["right"] * 6)
    assert out[5] == "left" and out[-1] == "right"


# ---------------------------------------------------------------- API / sources
def test_prediction_dict_api_and_latency_recorded():
    ctl, emg = fresh()
    cmd = emg.process_prediction({"gesture": "left", "confidence": 0.94, "timestamp": 1720000000.123})
    assert emg.last_prediction.confidence == 0.94 and emg.last_prediction.timestamp == 1720000000.123
    assert emg.latency.stats("emg_to_command_ms")["n"] == 1
    assert emg.latency.stats("emg_to_command_ms")["mean"] < 20.0           # < 20 ms budget


def test_command_to_target_latency_under_budget():
    ctl, emg = fresh()
    feed(emg, ctl, "left", 30)
    st = ctl.latency.stats("command_to_target_ms")
    assert st["n"] > 0 and st["p95"] < 20.0


def test_sources_are_interchangeable():
    k = KeyboardEMGSource()
    k.set_gesture("up")
    assert k.read().gesture == "up"
    k.release()
    assert k.read().gesture == "rest"
    scripted = ScriptedEMGSource([("left", 0.9, 2), ("rest", 0.9, 1)])
    assert [scripted.read().gesture for _ in range(4)] == ["left", "left", "rest", "rest"]
    assert scripted.finished
    noisy = NoisyEMGSource(ScriptedEMGSource([("left", 0.95, 200)]), flip_prob=0.5, seed=1)
    assert len({noisy.read().gesture for _ in range(100)}) > 1
    ctl, _ = fresh()
    emg = EMGController(CallbackEMGSource(lambda: {"gesture": "up", "confidence": 0.9}), sink=ctl.submit)
    emg.connect()
    for _ in range(3):                                                    # smoother needs 3 of 5 votes
        emg.process_prediction(emg.receive_gesture())
    assert emg.last_command.direction == "UP"
    emg.disconnect()


def test_emg_input_thread_drives_the_robot():
    """Real input thread (50 Hz) feeding a controller that is stepped in (roughly) real time alongside it."""
    import time
    ctl, _ = fresh()
    src = ScriptedEMGSource([("left", 0.95, 60), ("rest", 0.95, 1000)])
    emg = EMGController(src, sink=ctl.submit, poll_hz=50)
    emg.connect()
    emg.start()
    x_min, t0 = 0.0, time.time()
    while time.time() - t0 < 2.5:                         # 60 samples @ 50 Hz = 1.2 s of LEFT, then REST
        ctl.run_for(0.01)
        x_min = min(x_min, ctl.snapshot().tcp_position[0])
        time.sleep(0.01)
    emg.disconnect()
    assert emg.last_command is not None
    assert x_min < -0.01                                   # moved left while the gesture streamed in
    p = ctl.snapshot().tcp_position.copy()
    ctl.run_for(0.5)
    assert ctl.snapshot().tcp_position == pytest.approx(p, abs=1e-4)   # held after REST


def test_robot_core_is_independent_of_emg():
    """The controller accepts any MotionCommand; swapping the EMG source needs no change in the core."""
    ctl = SimulatedRobotController()
    ctl.connect()
    for _ in range(60):
        ctl.submit(mc.MotionCommand(mc.CARTESIAN, "LEFT", speed=0.04, source="keyboard"))
        ctl.run_for(0.02)
    assert ctl.snapshot().tcp_position[0] < -0.02
