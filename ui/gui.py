"""PyBullet GUI viewer. Runs in its OWN PROCESS (see ui/viewer_link.py).

Why a separate process: any call into PyBullet's GUI client can block 100+ ms when the window is idle,
and on macOS the GUI must own a main thread. Keeping it away from the control process means the 240 Hz
control/physics loop can never be stalled by rendering or window events.

Parent -> viewer : ("state", {"joints", "objects", "telemetry"})  /  ("quit",)
Viewer -> parent : ("cmd", MotionCommand) / ("gesture", name | None) / ("closed",)

Window layout (PyBullet's own): sidebar = joint + Cartesian sliders and buttons, centre = 3D scene
(frames drawn on every link, TCP frame longer) with one status line; full telemetry is shown by the parent
in the terminal.
"""
from __future__ import annotations

import math
import time

from control import motion_command as mc
from simulation.physics import make_world
from ui.controls import KeyboardController


class _RemoteGestureKeys:
    """Stands in for KeyboardEMGSource inside the viewer: forwards mock-EMG key state to the parent."""

    def __init__(self, send):
        self.send = send
        self._cur = None

    def set_gesture(self, g):
        if g != self._cur:
            self._cur = g
            self.send(("gesture", g))

    def release(self):
        self.set_gesture(None)


class PyBulletGUI:
    def __init__(self, cfg, env, conn, mode: str = "manual", show_frames: bool = True, dt: float = 1 / 240):
        import pybullet as p
        self.p, self.cfg, self.conn, self.mode = p, cfg, conn, mode
        self.client = p.connect(p.GUI, options="--width=1400 --height=860")
        self.world = make_world(self.client, cfg, env, dt, gui=True, show_frames=show_frames)
        self.kb = KeyboardController(self._send_cmd, mode, _RemoteGestureKeys(self.conn.send) if mode == "emg" else None)
        self.fps = 0.0
        self.running = True
        self._slider_last, self._button_last = {}, {}
        self._text_id, self._last_status = None, None
        self.telemetry = None
        self._build_widgets()

    def _send_cmd(self, cmd: mc.MotionCommand) -> bool:
        self.conn.send(("cmd", cmd))
        return True

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self) -> None:
        p, cl, cfg = self.p, self.client, self.cfg
        self.joint_params = []
        for j in cfg.joints:
            lo, hi = math.degrees(j.limit.min_angle), math.degrees(j.limit.max_angle)
            pid = p.addUserDebugParameter(f"J{j.joint_id} {j.role} (deg)", lo, hi, math.degrees(j.home), physicsClientId=cl)
            self.joint_params.append(pid)
            self._slider_last[pid] = math.degrees(j.home)
        self.cart_params = {}
        for name, lo, hi, val in (("X (mm)", -300, 300, 0.0), ("Y (mm)", -300, 320, 136.0), ("Z (mm)", 0, 420, 64.0),
                                  ("Roll (deg)", -180, 180, 180.0), ("Pitch (deg)", -90, 90, 0.0),
                                  ("Yaw (deg)", -180, 180, 0.0)):
            self.cart_params[name] = p.addUserDebugParameter("Cartesian " + name, lo, hi, val, physicsClientId=cl)
        self.buttons = {}
        for name in ("GO to XYZ (tool down)", "GO to full pose (X,Y,Z,R,P,Y)", "HOME", "RESET", "STOP",
                     "OPEN GRIPPER", "CLOSE GRIPPER", "E-STOP"):
            pid = p.addUserDebugParameter(name, 1, 0, 0, physicsClientId=cl)      # rangeMin > rangeMax => button
            self.buttons[name] = pid
            self._button_last[pid] = 0

    def _poll_widgets(self) -> None:
        p, cl, send = self.p, self.client, self._send_cmd
        rd = lambda pid: p.readUserDebugParameter(pid, physicsClientId=cl)
        if self.mode == "manual":
            for i, pid in enumerate(self.joint_params):
                v = rd(pid)
                if abs(v - self._slider_last[pid]) > 1e-6:
                    self._slider_last[pid] = v
                    send(mc.MotionCommand.joint_target(math.radians(v), joint=i + 1, source="slider"))
        pressed = set()
        for n, pid in self.buttons.items():
            v = rd(pid)
            if v != self._button_last[pid]:
                self._button_last[pid] = v
                pressed.add(n)
        if "E-STOP" in pressed:
            send(mc.MotionCommand(mc.ESTOP, source="button"))
        if "RESET" in pressed:
            send(mc.MotionCommand(mc.RESET, source="button"))
        if "STOP" in pressed:
            send(mc.MotionCommand.stop(source="button"))
        if "HOME" in pressed:
            send(mc.MotionCommand.home(source="button"))
        if "OPEN GRIPPER" in pressed:
            send(mc.MotionCommand.gripper("OPEN", source="button"))
        if "CLOSE GRIPPER" in pressed:
            send(mc.MotionCommand.gripper("CLOSE", source="button"))
        if pressed & {"GO to XYZ (tool down)", "GO to full pose (X,Y,Z,R,P,Y)"}:
            c = {n: rd(pid) for n, pid in self.cart_params.items()}
            pos = (c["X (mm)"] / 1000, c["Y (mm)"] / 1000, c["Z (mm)"] / 1000)
            if "GO to XYZ (tool down)" in pressed:
                send(mc.MotionCommand.move_to(pos, tool_pitch=math.pi, source="slider"))
            else:
                rpy = tuple(math.radians(c[k]) for k in ("Roll (deg)", "Pitch (deg)", "Yaw (deg)"))
                send(mc.MotionCommand.move_to(pos, orientation=rpy, source="slider"))

    # ------------------------------------------------------------------ keyboard
    def _key_name(self, key: int):
        p = self.p
        special = {p.B3G_LEFT_ARROW: "left", p.B3G_RIGHT_ARROW: "right", p.B3G_UP_ARROW: "up",
                   p.B3G_DOWN_ARROW: "down", 27: "esc"}
        if key in special:
            return special[key]
        return chr(key).lower() if 32 <= key < 127 else None

    def _poll_keys(self) -> None:
        p = self.p
        held, trig = set(), set()
        for key, st in p.getKeyboardEvents(physicsClientId=self.client).items():
            name = self._key_name(key)
            if name is None:
                continue
            if st & p.KEY_IS_DOWN:
                held.add(name)
            if st & p.KEY_WAS_TRIGGERED:
                trig.add(name)
        self.kb.update(held, trig)

    # ------------------------------------------------------------------ state from the control process
    def _apply_state(self, st: dict) -> None:
        p, cl, w = self.p, self.client, self.world
        for ji, pos in zip(w.arm_idx + w.finger_idx, st["joints"]):
            p.resetJointState(w.robot, ji, pos, physicsClientId=cl)
        for name, (pos, orn) in st["objects"].items():
            p.resetBasePositionAndOrientation(w.objects[name], pos, orn, physicsClientId=cl)
        self.telemetry = st["telemetry"]

    def _draw_status(self) -> None:
        """ONE in-scene text line, rewritten only when it changes (debug text costs ~75 ms/update)."""
        s = self.telemetry
        if s is None:
            return
        conf = "-" if s.confidence is None else f"{s.confidence*100:.0f}%"
        status = f"{s.gesture} {conf} | {s.command} | " + ("ESTOP" if s.estopped else "SAFETY " + s.safety.level.name)
        if status == self._last_status:
            return
        color = (1, 0.3, 0.3) if (s.estopped or s.safety.level >= 2) else (1, 0.6, 0.1) if not s.safety.ok else (0.1, 0.5, 0.1)
        kw = dict(textColorRGB=list(color), textSize=1.3, physicsClientId=self.client)
        if self._text_id is not None:
            kw["replaceItemUniqueId"] = self._text_id
        self._text_id = self.p.addUserDebugText(status, [-0.05, 0.0, 0.50], **kw)
        self._last_status = status

    # ------------------------------------------------------------------ main loop (viewer process main thread)
    def run(self, target_fps: float = 60.0) -> None:
        period = 1.0 / target_fps
        last = time.perf_counter()
        try:
            while self.running and self.p.isConnected(self.client):
                latest = None
                while self.conn.poll():
                    msg = self.conn.recv()
                    if msg[0] == "quit":
                        return
                    if msg[0] == "state":
                        latest = msg[1]
                if latest is not None:
                    self._apply_state(latest)            # pose update first: it also wakes the GUI's render loop
                self._poll_keys()
                self._poll_widgets()
                self._draw_status()
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
                last = now
                time.sleep(max(0.0, period - 0.002 - (time.perf_counter() - now)))
        except self.p.error:                             # window closed
            pass
        finally:
            try:
                self.conn.send(("closed",))
            except Exception:
                pass


def viewer_main(conn, cfg, env, mode, show_frames, dt) -> None:
    """Entry point of the viewer process."""
    gui = PyBulletGUI(cfg, env, conn, mode, show_frames, dt)
    gui.run()
