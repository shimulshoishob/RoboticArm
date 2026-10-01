"""RobotController interface + SimulatedRobotController.

EMG / keyboard / GUI code only ever sees ``RobotController``; swapping the simulator for
the real arm means swapping this one object (see control/hardware_controller.py).

Pipeline inside ``SimulatedRobotController.step``:
    queued MotionCommands -> MotionPlanner (+IK) -> safety/limit/collision vetting
    -> servo targets -> SimulatedServo dynamics -> physics backend -> safety evaluation -> CSV log
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config.robot_config import RobotConfig, default_config
from control import motion_command as mc
from control.motion_planner import MotionPlanner, PlanOutput
from control.safety import (COLLISION, CONTACT, ESTOP, JOINT_LIMIT, UNREACHABLE, SafetyMonitor, SafetyStatus,
                            Severity)
from robot.robot_arm import RobotArm
from simulation.collision import CollisionChecker
from simulation.physics import KinematicBackend, PhysicsBackend
from utils.logger import LatencyTracker, SessionLogger, get_logger

log = get_logger("controller")


class RobotController(ABC):
    """Everything an input layer (EMG, keyboard, GUI, ROS) may ask of a robot - simulated or real."""

    @abstractmethod
    def move_joint(self, joint_id: int, angle: float, wait: bool = False) -> None: ...

    @abstractmethod
    def move_joints(self, angles, wait: bool = False) -> None: ...

    @abstractmethod
    def move_cartesian(self, position, orientation=None, tool_pitch=None, speed=None, wait: bool = False) -> bool: ...

    @abstractmethod
    def open_gripper(self, wait: bool = False) -> None: ...

    @abstractmethod
    def close_gripper(self, wait: bool = False) -> None: ...

    @abstractmethod
    def set_gripper_position(self, value: float, wait: bool = False) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def home(self, wait: bool = False) -> None: ...

    @abstractmethod
    def submit(self, command: mc.MotionCommand) -> bool: ...

    @abstractmethod
    def snapshot(self): ...


@dataclass
class Telemetry:
    sim_time: float = 0.0
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(6))      # rad (servo state)
    joint_targets: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tcp_position: np.ndarray = field(default_factory=lambda: np.zeros(3))      # m
    tcp_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))           # rad
    gripper_opening: float = 1.0
    gesture: str = "-"
    confidence: float | None = None
    command: str = "-"
    collision: str = "NONE"
    safety: SafetyStatus = field(default_factory=SafetyStatus)
    estopped: bool = False
    connected: str = "SIMULATION"
    control_hz: float = 0.0
    torques: np.ndarray = field(default_factory=lambda: np.zeros(5))


class SimulatedRobotController(RobotController):
    def __init__(self, config: RobotConfig | None = None, backend: PhysicsBackend | None = None,
                 control_hz: float = 240.0, planner_hz: float = 100.0, command_timeout: float = 0.30,
                 session_logger: SessionLogger | None = None, payload_kg: float = 0.0):
        self.config = config or default_config()
        self.arm = RobotArm(self.config)
        self.kin = self.arm.kinematics
        self.dt = 1.0 / control_hz
        self.planner_period = 1.0 / planner_hz
        self.planner = MotionPlanner(
            self.kin, self.arm.ik_solver, self.config.workspace, self.config.home_angles,
            (self.arm.joints[5].limit.min_angle, self.arm.joints[5].limit.max_angle),
            command_timeout=command_timeout)
        self.safety = SafetyMonitor(self.config, payload_kg=payload_kg)
        self.collision_checker = CollisionChecker(self.kin, self.config)
        self.backend = backend or KinematicBackend()
        self.logger = session_logger
        self.latency = LatencyTracker()

        self._lock = threading.RLock()
        self._pending: deque = deque()
        self._planner_accum = 0.0
        self._step_count = 0
        self.sim_time = 0.0
        self.estopped = False
        self.rejected_commands = 0
        self.current_command: mc.MotionCommand | None = None
        self._awaiting_target: mc.MotionCommand | None = None
        self._gesture, self._confidence = "-", None
        self._collision_text, self._collision_t = "NONE", -1e9
        self._actual_q = self.arm.joint_angles[:5].copy()
        self.control_hz_measured = 0.0
        self.external_stepper = False          # True when a runtime thread calls step() for us

    # ------------------------------------------------------------------ lifecycle
    def connect(self, env=None) -> None:
        self.backend.connect(env)
        self._sync_backend()

    def shutdown(self) -> None:
        self.backend.disconnect()
        if self.logger:
            self.logger.close()

    # ------------------------------------------------------------------ helpers
    def _q_cmd(self) -> np.ndarray:
        return self.arm.joint_targets[:5]

    def _grip_cmd(self) -> float:
        return self.arm.gripper.target_opening

    def _sync_backend(self) -> None:
        self.backend.set_targets(self.arm.joint_angles[:5], self.arm.gripper.opening)

    @property
    def faulted(self) -> bool:
        return self.safety.faulted

    @property
    def blocked(self) -> bool:
        return self.estopped or self.faulted

    def set_emg_status(self, gesture: str, confidence: float | None) -> None:
        self._gesture, self._confidence = gesture, confidence

    # ------------------------------------------------------------------ command intake (thread-safe)
    def submit(self, command: mc.MotionCommand) -> bool:
        """Queue a command for the next control step. Returns False if rejected (e-stop/fault)."""
        if command.command_type in (mc.ESTOP,):
            self.emergency_stop()                       # never wait for the next tick
            return True
        if self.blocked and command.command_type != mc.RESET:
            self.rejected_commands += 1
            return False
        with self._lock:
            self._pending.append(command)
        return True

    def _process(self, cmd: mc.MotionCommand) -> None:
        if cmd.command_type == mc.RESET:
            self.reset()
            return
        if self.blocked:
            self.rejected_commands += 1
            return
        self.current_command = cmd
        self.latency.record("command_queue_ms", (time.perf_counter() - cmd.created_at) * 1000.0)
        self._awaiting_target = cmd
        out = self.planner.handle(cmd, self._q_cmd(), self._grip_cmd())
        self._apply(out)
        if not self.planner.active:                       # nothing more will be produced for this command
            self._awaiting_target = None

    # ------------------------------------------------------------------ applying planner output safely
    def _apply(self, out: PlanOutput) -> None:
        t = self.sim_time
        for code, msg, sev in out.events:
            self.safety.report(t, code, msg, sev)
        if out.stop_now:
            self.arm.stop()
        if out.gripper is not None:
            self.arm.gripper.set_position(out.gripper)
            self._record_target_latency()
        if out.joint_targets is not None:
            q = np.asarray(out.joint_targets, dtype=float)[:5]
            lo, hi = self.kin.lower, self.kin.upper
            for i in np.nonzero((q < lo - 1e-9) | (q > hi + 1e-9))[0]:
                self.safety.report(t, JOINT_LIMIT, f"J{i+1} joint limit reached", Severity.WARNING, key=f"J{i+1}")
            q = np.clip(q, lo, hi)
            rep = self.collision_checker.check(q, self.arm.gripper.target_opening)
            if rep.colliding and not self.collision_checker.check(self._q_cmd(), self.arm.gripper.target_opening).colliding:
                # Veto: never command a pose that collides. Keep the old target and drop the planner's path.
                self._set_collision(str(rep), t)
                self.safety.report(t, COLLISION, "Collision detected: target rejected (" + ", ".join(rep.pairs) + ")",
                                   Severity.WARNING)
                self.planner.cancel()
                return
            for i in range(5):
                self.arm.set_joint_angle(i + 1, float(q[i]))
            self.arm.drain_warnings()
            self._record_target_latency()

    def _record_target_latency(self) -> None:
        if self._awaiting_target is not None:
            self.latency.record("command_to_target_ms",
                                (time.perf_counter() - self._awaiting_target.created_at) * 1000.0)
            self._awaiting_target = None

    def _set_collision(self, text: str, t: float) -> None:
        self._collision_text, self._collision_t = text, t

    # ------------------------------------------------------------------ the control step
    def step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else dt
        with self._lock:
            while self._pending:
                self._process(self._pending.popleft())

            if not self.blocked and self.planner.active:
                self._planner_accum += dt
                if self._planner_accum >= self.planner_period - 1e-9:
                    out = self.planner.update(self._planner_accum, self._q_cmd(), self._grip_cmd())
                    self._planner_accum = 0.0
                    self._apply(out)
            else:
                self._planner_accum = 0.0

            self.arm.update(dt)
            self._sync_backend()
            self.backend.step()
            self._actual_q, _ = self.backend.read_state()
            self.sim_time += dt
            self._step_count += 1

            status = self.safety.evaluate(self.sim_time, dt, self.arm, self._actual_q if self.backend.supports_physics else None)
            if self.backend.supports_physics and self._step_count % 8 == 0:
                contacts = self.backend.get_contacts()
                if contacts:
                    self._set_collision("CONTACT: " + ", ".join(contacts), self.sim_time)
                    self.safety.report(self.sim_time, CONTACT, "Collision detected: " + ", ".join(contacts))
            if self.sim_time - self._collision_t > 1.0:
                self._collision_text = "NONE"
            if self.safety.faulted and not self.estopped:
                self._hold_after_fault()
            if self.logger:
                self._log_row()

    def _hold_after_fault(self) -> None:
        self.arm.stop()
        self.planner.cancel()
        self._pending.clear()
        log.error("Safety FAULT latched: %s - reset required", self.safety.status(self.sim_time))

    def _log_row(self) -> None:
        pose = self.arm.get_end_effector_pose()
        self.logger.log(self.sim_time, self._gesture if self._gesture != "-" else "", self._confidence,
                        self.current_command.describe() if self.current_command else "",
                        self.arm.joint_angles, pose.position, self._collision_text,
                        str(self.safety.status(self.sim_time)))

    # ------------------------------------------------------------------ waiting helpers
    def run_for(self, seconds: float) -> None:
        """Synchronously advance simulation time (deterministic; for scripts and tests)."""
        for _ in range(int(round(seconds / self.dt))):
            self.step()

    @property
    def settled(self) -> bool:
        return (not self._pending and not self.planner.active
                and all(j.servo.at_target for j in self.arm.joints))

    def wait_until_settled(self, timeout: float = 20.0) -> bool:
        if self.external_stepper:
            t0 = time.perf_counter()
            while not self.settled and time.perf_counter() - t0 < timeout:
                time.sleep(0.005)
            return self.settled
        for _ in range(int(timeout / self.dt)):
            if self.settled:
                return True
            self.step()
        return self.settled

    def pause(self, seconds: float) -> None:
        """Let time pass (steps the simulation itself unless a runtime thread is doing it)."""
        if self.external_stepper:
            time.sleep(seconds)
        else:
            self.run_for(seconds)

    # ------------------------------------------------------------------ RobotController API
    def move_joint(self, joint_id: int, angle: float, wait: bool = False) -> None:
        self.submit(mc.MotionCommand.joint_target(angle, joint=joint_id, source="api"))
        if wait:
            self.wait_until_settled()

    def move_joints(self, angles, wait: bool = False) -> None:
        self.submit(mc.MotionCommand.joint_target(angles, source="api"))
        if wait:
            self.wait_until_settled()

    def move_cartesian(self, position, orientation=None, tool_pitch=None, speed=None, wait: bool = False) -> bool:
        """Straight-line TCP move. Returns False (and raises a safety warning) if the target is unreachable."""
        with self._lock:
            res = self.arm.inverse_kinematics(position, orientation, seed=self._q_cmd(), tool_pitch=tool_pitch)
            if not res.success and orientation is None and tool_pitch is not None:
                res = self.arm.inverse_kinematics(position, seed=self._q_cmd())
            if not res.success:
                self.safety.report(self.sim_time, UNREACHABLE, res.message)
                log.warning(res.message)
                return False
        ok = self.submit(mc.MotionCommand.move_to(position, speed, orientation, tool_pitch, source="api"))
        if ok and wait:
            self.wait_until_settled()
        return ok

    def set_gripper_position(self, value: float, wait: bool = False) -> None:
        self.submit(mc.MotionCommand.gripper("SET", value, source="api"))
        if wait:
            self.wait_until_settled()

    def open_gripper(self, wait: bool = False) -> None:
        self.set_gripper_position(1.0, wait)

    def close_gripper(self, wait: bool = False) -> None:
        self.set_gripper_position(0.0, wait)

    def home(self, wait: bool = False) -> None:
        self.submit(mc.MotionCommand.home(source="api"))
        if wait:
            self.wait_until_settled()

    def stop(self) -> None:
        """Immediate stop of all motion (targets := current angles). Does not latch."""
        with self._lock:
            self._pending.clear()
            self.planner.cancel()
            self.arm.stop()

    def emergency_stop(self) -> None:
        """Latching stop: halts every joint at once; commands are refused until reset()."""
        with self._lock:
            self.estopped = True
            self._pending.clear()
            self.planner.cancel()
            self.arm.stop()
            self.safety.report(self.sim_time, ESTOP, "EMERGENCY STOP - reset required", Severity.FAULT)
        log.warning("EMERGENCY STOP")

    def reset(self, to_home: bool = False) -> None:
        """Explicitly clear e-stop and latched faults. Holds position (or snaps home in simulation)."""
        with self._lock:
            self.estopped = False
            self.safety.clear()
            self.planner.cancel()
            self._pending.clear()
            self._collision_text = "NONE"
            if to_home:
                self.arm.reset()
                if hasattr(self.backend, "reset_state"):
                    self.backend.reset_state(self.arm.joint_angles[:5], self.arm.gripper.opening)
            else:
                self.arm.stop()
            self._sync_backend()
        log.info("reset: controller re-armed")

    # ------------------------------------------------------------------ telemetry
    def snapshot(self) -> Telemetry:
        with self._lock:
            pose = self.arm.get_end_effector_pose()
            return Telemetry(
                sim_time=self.sim_time,
                joint_angles=self.arm.joint_angles.copy(),
                joint_targets=self.arm.joint_targets.copy(),
                tcp_position=pose.position.copy(), tcp_rpy=pose.rpy.copy(),
                gripper_opening=self.arm.gripper.opening,
                gesture=self._gesture, confidence=self._confidence,
                command=self.current_command.describe() if self.current_command else "-",
                collision=self._collision_text,
                safety=self.safety.status(self.sim_time),
                estopped=self.estopped, control_hz=self.control_hz_measured,
                torques=self.arm.gravity_torques(self.safety.payload_kg))
