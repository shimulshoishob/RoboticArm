"""EMG control end-to-end WITHOUT hardware: a noisy scripted classifier drives the simulated arm.

    python examples/example_emg_control.py

Shows the full chain  prediction -> confidence gate -> smoothing -> MotionCommand -> planner/IK -> servos,
and prints the measured latencies. To use your real classifier replace ``source`` (see README section
"Replacing the mock EMG").
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from control.controller import SimulatedRobotController
from emg.emg_controller import EMGController
from emg.mock_emg import NoisyEMGSource, ScriptedEMGSource, demo_script


def main() -> None:
    ctl = SimulatedRobotController()               # swap for HardwareRobotController later: nothing else changes
    ctl.connect()
    source = NoisyEMGSource(ScriptedEMGSource(demo_script()), flip_prob=0.10, low_conf_prob=0.10, seed=3)
    emg = EMGController(source, sink=ctl.submit, status_callback=ctl.set_emg_status, latency=ctl.latency)
    emg.connect()

    start = ctl.snapshot().tcp_position.copy()
    poll_steps = int(round(1 / (50 * ctl.dt)))     # 50 Hz classifier, 240 Hz control loop
    for i in range(int(14 / ctl.dt)):
        if i % poll_steps == 0:
            pred = emg.receive_gesture()
            if pred:
                emg.process_prediction(pred)
        ctl.step()
        if i % int(1 / ctl.dt) == 0:
            s = ctl.snapshot()
            print(f"t={s.sim_time:5.1f}s  gesture={s.gesture:<22} command={s.command:<12} "
                  f"TCP[mm]={np.round(s.tcp_position*1000, 1)}  gripper={s.gripper_opening*100:3.0f}%  safety={s.safety}")
    end = ctl.snapshot().tcp_position
    print("\nTCP moved by [mm]:", np.round((end - start) * 1000, 1))
    print("\nLatency:\n" + ctl.latency.summary())


if __name__ == "__main__":
    main()
