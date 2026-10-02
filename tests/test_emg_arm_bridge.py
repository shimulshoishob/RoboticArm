"""EMG device layer (single-file robotic_arm.py): prediction -> arm bridge, and an offscreen dashboard run."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import robotic_arm as ra


# ---------------------------------------------------------------- default mapping
@pytest.mark.parametrize("name,action", [
    ("Left", "Move Left  (-X)"), ("right", "Move Right (+X)"), ("Up", "Move Up    (+Z)"), ("down", "Move Down  (-Z)"),
    ("fist_close", "Close Gripper"), ("Open", "Open Gripper"), ("Rest", ra.IGNORE_ACTION), ("idle", ra.IGNORE_ACTION),
    ("wave", ra.IGNORE_ACTION),
])
def test_default_arm_action(name, action):
    assert ra.default_arm_action(name) == action


# ---------------------------------------------------------------- bridge
class Sink:
    def __init__(self):
        self.cmds = []

    def __call__(self, c):
        self.cmds.append(c)
        return True

    @property
    def types(self):
        return [(c.command_type, c.direction) for c in self.cmds]


def make_bridge(**eng):
    sink = Sink()
    b = ra.EMGArmBridge(sink, engine=ra.GestureDecisionEngine(**eng))
    b.set_mapping(["Rest", "Up", "Left", "fist_close"])
    b.enable(True)
    return b, sink


def feed(b, label, n, conf=0.95, margin=0.5, quality=True):
    return [b.on_prediction(label, conf, margin, quality) for _ in range(n)]


def test_gesture_needs_debounce_then_jogs_continuously():
    b, sink = make_bridge(consecutive_required=3)
    out = feed(b, "Up", 5)
    assert out[:2] == [ra.IGNORE_ACTION] * 2            # not yet sustained
    assert out[2:] == ["Move Up    (+Z)"] * 3
    jogs = [c for c in sink.cmds if c.command_type == ra.CARTESIAN]
    assert len(jogs) == 3 and all(c.direction == "UP" and c.is_continuous and c.source == "emg" for c in jogs)


def test_rest_after_motion_sends_one_hold_only():
    b, sink = make_bridge(consecutive_required=2)
    feed(b, "Left", 4)
    n = len(sink.cmds)
    feed(b, "Rest", 5)
    holds = [c for c in sink.cmds[n:] if c.command_type == ra.HOLD]
    assert len(holds) == 1 and len(sink.cmds) == n + 1  # does not spam HOLD (would cancel manual jogging)


def test_disabled_bridge_never_commands_the_arm():
    b, sink = make_bridge(consecutive_required=1)
    b.enable(False)
    feed(b, "Up", 6)
    assert sink.cmds == []


def test_low_confidence_small_margin_and_bad_signal_do_not_move():
    b, sink = make_bridge(consecutive_required=1, min_confidence=0.65, min_margin=0.10)
    feed(b, "Up", 4, conf=0.5)
    feed(b, "Up", 4, margin=0.02)
    feed(b, "Up", 4, quality=False)
    assert all(c.command_type != ra.CARTESIAN for c in sink.cmds)


def test_losing_quality_mid_motion_stops_the_arm():
    b, sink = make_bridge(consecutive_required=1)
    feed(b, "Up", 3)
    feed(b, "Up", 1, quality=False)
    assert sink.cmds[-1].command_type == ra.HOLD


def test_gripper_is_rate_limited_not_spammed():
    b, sink = make_bridge(consecutive_required=1, refractory_s=10.0)
    feed(b, "fist_close", 20)
    grips = [c for c in sink.cmds if c.command_type == ra.GRIPPER]
    assert len(grips) == 1 and grips[0].direction == "CLOSE"


def test_ignore_mapping_and_release():
    b, sink = make_bridge(consecutive_required=1)
    b.set_action("Up", ra.IGNORE_ACTION)
    feed(b, "Up", 4)
    assert sink.cmds == []
    b.set_action("Up", "Move Forward (+Y)")
    feed(b, "Up", 2)
    b.release("test")
    assert sink.cmds[-1].command_type == ra.HOLD
    with pytest.raises(ValueError):
        b.set_action("Up", "Fly")


# ---------------------------------------------------------------- vendored BioWave pipeline
def test_feature_count_matches_model_contract():
    win = np.random.default_rng(0).normal(0, 20, (100, 11)).astype(np.float32)
    feats = ra.extract_window_features(win, sample_rate=500)
    assert feats.shape == (ra.expected_feature_count(11),)


def test_ring_buffer_and_wireless_packet_roundtrip():
    ring = ra.SampleRingBuffer(11, 1000)
    ring.append(np.ones((150, 11), dtype=np.float32))
    assert ring.has_window(100) and ring.latest(100).shape == (100, 11)
    ring.append(np.ones((5, 11), dtype=np.float32), discontinuity=True)
    assert not ring.has_window(100)


# ---------------------------------------------------------------- offscreen dashboard, end to end
def synth(kind, n, rng):
    x = rng.normal(0, 4.0, (n, 11)).astype(np.float32) + 2000.0          # rest: baseline ~2000 ADC
    if kind == "up":
        x[:, :4] += rng.normal(0, 300.0, (n, 4)).astype(np.float32)
    elif kind == "all":                                                     # calibration FLEX: every EMG channel active
        x[:, :8] += rng.normal(0, 300.0, (n, 8)).astype(np.float32)
    elif kind == "fist":
        x[:, 4:8] += rng.normal(0, 300.0, (n, 4)).astype(np.float32)
    return x


@pytest.mark.skipif(not (ra.HAS_QT and ra.HAS_JOBLIB), reason="PyQt5 / joblib not installed")
def test_dashboard_calibrates_and_drives_arm(tmp_path, monkeypatch):
    for name in ("warning", "critical", "information"):          # a modal dialog would hang a headless test
        monkeypatch.setattr(ra.QMessageBox, name, staticmethod(lambda *a, **k: None))
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    rng = np.random.default_rng(1)
    X, y = [], []
    for kind, cls in (("rest", "Rest"), ("up", "Up"), ("fist", "fist_close")):
        for _ in range(120):
            win = synth(kind, 100, rng) - 2000.0                          # trainer sees baseline-centred data
            X.append(ra.extract_window_features(win, 500))
            y.append(cls)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(np.array(X), y)
    path = tmp_path / "m.joblib"
    joblib.dump({"model": rf, "class_names": sorted(set(y)), "window_samples": 100,
                 "stride_samples": 25, "input_channels": 11, "sample_rate": 500}, path)

    rt = ra.SimulationRuntime(headless=True, physics=False, save_log=False, objects=False, mode="full")
    app = ra.QApplication.instance() or ra.QApplication([])
    rt.start()
    win = ra.ArmDashboard(rt)
    try:
        assert win.load_model(str(path))
        assert win.bridge.action_map["Up"] == "Move Up    (+Z)"
        # pretend a wired 11-channel stream is connected (worker-less) and feed it in real time
        win.num_channels, win.emg_channel_count, win.is_connected = 11, 8, True
        win._reset_stream_state()
        win.check_ready_state()
        assert win.btn_calibrate.isEnabled()
        win.spin_rest_sec.setValue(3)
        win.spin_flex_sec.setValue(3)
        win.start_calibration_sequence()

        def pump(kind, seconds):
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                win.on_stream_batch(synth(kind, 25, rng))
                app.processEvents()
                time.sleep(0.05)

        while win.calibration_active and win.current_phase_key != "flex":
            pump("rest", 0.1)
        while win.calibration_active:
            pump("all", 0.1)
        assert win.is_calibrated, win.lbl_cal_status.text()
        assert win.btn_control.isEnabled()

        pump("rest", 1.0)
        z0 = rt.controller.snapshot().tcp_position[2]
        win.btn_control.setChecked(True)
        assert win.bridge.enabled
        pump("up", 2.0)
        z1 = rt.controller.snapshot().tcp_position[2]
        assert z1 > z0 + 0.01, f"arm did not rise: {z0:.3f} -> {z1:.3f} ({win.lbl_status.text()})"
        pump("rest", 1.0)
        z2 = rt.controller.snapshot().tcp_position[2]
        pump("rest", 0.5)
        assert abs(rt.controller.snapshot().tcp_position[2] - z2) < 0.005    # stopped on rest

        win.last_batch_received_monotonic = time.monotonic() - 10            # stream stalls -> control off
        win.refresh_ui()
        assert not win.bridge.enabled and not win.btn_control.isChecked()
    finally:
        win.close()
        rt.stop()
