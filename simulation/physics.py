"""Physics backends. The controller talks only to ``PhysicsBackend``.

* KinematicBackend - no dependencies; joints track their commands perfectly, no objects/contacts.
* PyBulletBackend  - gravity, collisions, joint motors with torque caps, friction grasping, GUI.

All PyBullet calls are serialised through ``backend.lock`` (control thread + UI thread share one client).
"""
from __future__ import annotations

import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from config.robot_config import RobotConfig
from simulation.environment import Environment
from simulation.objects import SimObject
from simulation.urdf import build_urdf

try:                                       # optional dependency
    import pybullet as _p
    import pybullet_data as _pd
except Exception:                          # pragma: no cover
    _p = _pd = None


def pybullet_available() -> bool:
    return _p is not None


class PhysicsBackend(ABC):
    supports_physics = False

    @abstractmethod
    def connect(self, env: Environment | None = None) -> None: ...

    @abstractmethod
    def set_targets(self, q5: np.ndarray, opening: float) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def read_state(self) -> tuple:
        """-> (q5 actual (rad), gripper opening actual 0..1)"""

    def get_contacts(self) -> list:
        return []

    def object_position(self, name: str):
        return None

    def disconnect(self) -> None:
        pass


class KinematicBackend(PhysicsBackend):
    """Perfect tracking, no physics. Useful for tests and machines without PyBullet."""

    def __init__(self):
        self._q = np.zeros(5)
        self._opening = 1.0
        self.env: Environment | None = None

    def connect(self, env=None):
        self.env = env

    def set_targets(self, q5, opening):
        self._q, self._opening = np.asarray(q5, float).copy(), float(opening)

    def step(self):
        pass

    def read_state(self):
        return self._q.copy(), self._opening

    def object_position(self, name):
        return None if self.env is None else self.env.get(name).position


class _World:
    """Handles of one PyBullet client."""

    def __init__(self, client: int):
        self.client = client
        self.robot = None
        self.table = None
        self.objects: dict[str, int] = {}
        self.joint_idx: dict[str, int] = {}
        self.link_names: dict[int, str] = {-1: "base"}

    @property
    def arm_idx(self) -> list:
        return [self.joint_idx[f"j{i}"] for i in range(1, 6)]

    @property
    def finger_idx(self) -> list:
        return [self.joint_idx["j6_finger_l"], self.joint_idx["j6_finger_r"]]


def _add_object(p, w: _World, obj: SimObject) -> int:
    c = w.client
    if obj.kind == "cube":
        h = obj.dims[0] / 2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h] * 3, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h] * 3, rgbaColor=list(obj.color), physicsClientId=c)
    elif obj.kind == "sphere":
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=obj.dims[0], physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=obj.dims[0], rgbaColor=list(obj.color), physicsClientId=c)
    else:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obj.dims[0], height=obj.dims[1], physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obj.dims[0], length=obj.dims[1],
                                  rgbaColor=list(obj.color), physicsClientId=c)
    bid = p.createMultiBody(obj.mass, col, vis, list(obj.position), physicsClientId=c)
    p.changeDynamics(bid, -1, lateralFriction=1.0, spinningFriction=0.005, rollingFriction=0.001, physicsClientId=c)
    w.objects[obj.name] = bid
    return bid


def make_world(client: int, cfg: RobotConfig, env: Environment, dt: float, gui: bool = False,
               show_frames: bool = True) -> _World:
    """Scene (table + floor), robot from the generated URDF, and objects, inside one PyBullet client.
    Used for the DIRECT physics client AND for the GUI viewer process (identical geometry)."""
    p, c = _p, client
    w = _World(client)
    p.setAdditionalSearchPath(_pd.getDataPath(), physicsClientId=c)
    p.setGravity(0, 0, -9.81, physicsClientId=c)
    p.setTimeStep(dt, physicsClientId=c)
    p.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=c)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=c)   # keys are ours
        for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            p.configureDebugVisualizer(flag, 0, physicsClientId=c)
        p.resetDebugVisualizerCamera(0.65, 35, -28, [0.0, 0.10, 0.10], physicsClientId=c)
    tx, ty = env.table_size
    th = env.table_thickness
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tx / 2, ty / 2, th / 2], physicsClientId=c)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[tx / 2, ty / 2, th / 2],
                              rgbaColor=[0.55, 0.42, 0.30, 1], physicsClientId=c)
    w.table = p.createMultiBody(0, col, vis, [0, ty / 2 - 0.2, -th / 2], physicsClientId=c)
    p.changeDynamics(w.table, -1, lateralFriction=0.9, physicsClientId=c)
    fcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 3, 0.05], physicsClientId=c)
    fvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 0.05], rgbaColor=[0.85, 0.85, 0.87, 1], physicsClientId=c)
    p.createMultiBody(0, fcol, fvis, [0, 0, -0.75], physicsClientId=c)
    tmp = Path(tempfile.mkdtemp(prefix="arm_urdf_")) / "arm.urdf"
    tmp.write_text(build_urdf(cfg, axes=gui and show_frames), encoding="utf-8")
    w.robot = p.loadURDF(str(tmp), [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE,
                         physicsClientId=c)
    for i in range(p.getNumJoints(w.robot, physicsClientId=c)):
        info = p.getJointInfo(w.robot, i, physicsClientId=c)
        w.joint_idx[info[1].decode()] = i
        w.link_names[i] = info[12].decode()
    for obj in env.objects:
        _add_object(p, w, obj)
    return w


class PyBulletBackend(PhysicsBackend):
    """Physics in a fast DIRECT client. Rendering is a separate concern: the GUI lives in its own process
    (ui/gui.py) and is fed ``view_state()``; any call into PyBullet's GUI client can block ~100+ ms."""

    supports_physics = True

    def __init__(self, config: RobotConfig, dt: float = 1.0 / 240.0, position_gain: float = 0.5,
                 velocity_gain: float = 1.0):
        if _p is None:
            raise RuntimeError("pybullet is not installed (pip install pybullet)")
        self.cfg = config
        self.dt = dt
        self.kp, self.kd = position_gain, velocity_gain
        self.p = _p
        self.lock = threading.RLock()
        self.phys: _World | None = None
        self.objects: dict[str, SimObject] = {}

    @property
    def client(self):
        return self.phys.client if self.phys else None

    def connect(self, env: Environment | None = None) -> None:
        p = self.p
        env = env or Environment()
        with self.lock:
            self.phys = make_world(p.connect(p.DIRECT), self.cfg, env, self.dt)
            for o in env.objects:
                self.objects[o.name] = o
                o.body_id = self.phys.objects[o.name]
            w, c = self.phys, self.phys.client
            self._torque = [self.cfg.joints[i].servo.torque_limit for i in range(5)]
            p.setCollisionFilterPair(w.robot, w.robot, *w.finger_idx, 0, physicsClientId=c)
            for fi in w.finger_idx:
                p.changeDynamics(w.robot, fi, lateralFriction=1.2, spinningFriction=0.01, physicsClientId=c)
            for ji in w.arm_idx:                         # pure position control: disable default velocity motor
                p.setJointMotorControl2(w.robot, ji, p.VELOCITY_CONTROL, force=0, physicsClientId=c)
            self.reset_state(np.array(self.cfg.home_angles[:5]), 1.0)

    # ------------------------------------------------------------------ objects
    def add_object(self, obj: SimObject) -> int:
        with self.lock:
            self.objects[obj.name] = obj
            obj.body_id = _add_object(self.p, self.phys, obj)
            return obj.body_id

    def object_position(self, name: str):
        with self.lock:
            pos, _ = self.p.getBasePositionAndOrientation(self.phys.objects[name], physicsClientId=self.client)
        return tuple(pos)

    def reset_object(self, name: str, position) -> None:
        with self.lock:
            bid = self.phys.objects[name]
            self.p.resetBasePositionAndOrientation(bid, list(position), [0, 0, 0, 1], physicsClientId=self.client)
            self.p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

    # ------------------------------------------------------------------ control / stepping
    def reset_state(self, q5, opening: float) -> None:
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            for ji, q in zip(w.arm_idx, q5):
                p.resetJointState(w.robot, ji, float(q), physicsClientId=c)
            s = opening * self.cfg.gripper.finger_travel
            for fi in w.finger_idx:
                p.resetJointState(w.robot, fi, s, physicsClientId=c)

    def set_targets(self, q5, opening: float) -> None:
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            p.setJointMotorControlArray(
                w.robot, w.arm_idx, p.POSITION_CONTROL, targetPositions=[float(v) for v in q5],
                forces=self._torque, positionGains=[self.kp] * 5, velocityGains=[self.kd] * 5, physicsClientId=c)
            s = float(np.clip(opening, 0, 1)) * self.cfg.gripper.finger_travel
            p.setJointMotorControlArray(
                w.robot, w.finger_idx, p.POSITION_CONTROL, targetPositions=[s, s],
                forces=[self.cfg.gripper.max_grip_force] * 2, physicsClientId=c)

    def step(self) -> None:
        with self.lock:
            self.p.stepSimulation(physicsClientId=self.client)

    def read_state(self):
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            st = p.getJointStates(w.robot, w.arm_idx, physicsClientId=c)
            fs = p.getJointStates(w.robot, w.finger_idx, physicsClientId=c)
        q = np.array([s[0] for s in st])
        travel = float(np.mean([s[0] for s in fs]))
        return q, travel / self.cfg.gripper.finger_travel

    def view_state(self) -> dict:
        """Everything a viewer needs to pose its copy of the scene (picklable)."""
        p, c, w = self.p, self.client, self.phys
        with self.lock:
            idx = w.arm_idx + w.finger_idx
            joints = [s[0] for s in p.getJointStates(w.robot, idx, physicsClientId=c)]
            objs = {n: p.getBasePositionAndOrientation(b, physicsClientId=c) for n, b in w.objects.items()}
        return {"joints": joints, "objects": objs}

    def get_contacts(self) -> list:
        """Unwanted contacts: robot vs table/objects, excluding base-table and finger-object (grasp)."""
        p, c, w = self.p, self.client, self.phys
        found = []
        with self.lock:
            pts = p.getContactPoints(bodyA=w.robot, physicsClientId=c)
        finger_links = set(w.finger_idx)
        obj_ids = {bid: n for n, bid in w.objects.items()}
        for cp in pts:
            link, other = cp[3], cp[2]
            if other == w.robot:
                continue
            if link == -1 and other == w.table:
                continue
            if link in finger_links and other in obj_ids:
                continue
            tag = "table" if other == w.table else obj_ids.get(other, f"body{other}")
            found.append(f"{w.link_names.get(link, link)}-{tag}")
        return sorted(set(found))

    def disconnect(self) -> None:
        with self.lock:
            if self.phys is not None and self.p.isConnected(self.phys.client):
                self.p.disconnect(self.phys.client)
            self.phys = None
