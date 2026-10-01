"""Generate a URDF from RobotConfig, so geometry/mass/limits live in ONE place (the config).

Link frames follow robot/kinematics.py: link i's frame sits on joint i's axis and the link
extends along +Z by ``length`` to the next joint origin.
"""
from __future__ import annotations

from pathlib import Path

from config.robot_config import RobotConfig

ALU = "0.78 0.80 0.83 1"
DARK = "0.10 0.10 0.12 1"
FINGER = "0.62 0.65 0.70 1"


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


AX_COLORS = {"ax_x": "1 0 0 1", "ax_y": "0 0.8 0 1", "ax_z": "0 0 1 1"}
_MATERIALS = {"alu": ALU, "dark": DARK, "finger": FINGER}
_MATERIALS.update(AX_COLORS)


def _axis_links(parent: str, origin_z: float = 0.0, length: float = 0.03, r: float = 0.0008, tag: str = "") -> str:
    """Coordinate-frame arrows as three tiny fixed child links (the OpenGL viewer tints a whole link with ONE
    color, so each arrow needs its own link). Visual only: no collision."""
    out = ""
    for (name, size, xyz) in (("x", (length, 2 * r, 2 * r), (length / 2, 0, 0)),
                              ("y", (2 * r, length, 2 * r), (0, length / 2, 0)),
                              ("z", (2 * r, 2 * r, length), (0, 0, length / 2))):
        ln = f"ax_{parent}{tag}_{name}"
        out += (f'<link name="{ln}"><inertial><mass value="1e-4"/><inertia ixx="1e-9" iyy="1e-9" izz="1e-9" '
                f'ixy="0" ixz="0" iyz="0"/></inertial><visual><origin xyz="{_fmt(xyz)}"/><geometry>'
                f'<box size="{_fmt(size)}"/></geometry><material name="ax_{name}"/></visual></link>'
                f'<joint name="{ln}_j" type="fixed"><parent link="{parent}"/><child link="{ln}"/>'
                f'<origin xyz="0 0 {origin_z:.6g}"/></joint>')
    return out


def _box(size, xyz, color: str, name: str, collision: bool = True) -> str:
    mat = next(k for k, v in _MATERIALS.items() if v == color)
    vis = (f'<visual><origin xyz="{_fmt(xyz)}"/><geometry><box size="{_fmt(size)}"/></geometry>'
           f'<material name="{mat}"/></visual>')
    col = (f'<collision><origin xyz="{_fmt(xyz)}"/><geometry><box size="{_fmt(size)}"/></geometry></collision>'
           if collision else "")
    return vis + col


def _inertial(link) -> str:
    ixx, iyy, izz = link.inertia_diag()
    return (f'<inertial><origin xyz="{_fmt(link.com)}"/><mass value="{link.mass:.6g}"/>'
            f'<inertia ixx="{ixx:.6g}" iyy="{iyy:.6g}" izz="{izz:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>')


def build_urdf(cfg: RobotConfig, axes: bool = False) -> str:
    """``axes=True`` adds coordinate-frame arrows to every link + TCP (used by the GUI viewer only)."""
    L = cfg.links
    g = cfg.gripper
    parts = [f'<?xml version="1.0"?>\n<robot name="{cfg.name}">']
    parts += [f'<material name="{k}"><color rgba="{v}"/></material>' for k, v in _MATERIALS.items()]

    # ---- base: large metal plate
    w, d = cfg.base_size
    parts.append(f'<link name="base">{_inertial(L[0])}'
                 + _box((w, d, L[0].length), (0, 0, L[0].length / 2), ALU, "alu_base")
                 + "</link>")

    # ---- arm links 1..4 and end-effector body
    for i in range(1, 6):
        link = L[i]
        lw, ld = link.size
        # NOTE: the OpenGL viewer tints a whole link with its LAST visual's color, so the aluminum bracket
        # goes last and the dark servo housing first (it stays visible only in the collision/inertial model).
        body = ""
        if i == 5:
            body += _box((lw, ld, g.palm_length), (0, 0, g.palm_length / 2), ALU, f"alu_{i}")
        else:
            body += _box((lw * 1.15, ld * 1.5, 0.036), (0, 0, link.length - 0.018), DARK, f"servo_{i}", collision=False)
            body += _box((lw, ld, link.length), (0, 0, link.length / 2), ALU, f"alu_{i}")
        parts.append(f'<link name="{link.name}">{_inertial(link)}{body}</link>')

    # ---- revolute joints J1..J5
    for i in range(1, 6):
        j = cfg.joints[i - 1]
        a = j.axis
        parts.append(
            f'<joint name="j{i}" type="revolute"><parent link="{L[i-1].name}"/><child link="{L[i].name}"/>'
            f'<origin xyz="0 0 {L[i-1].length:.6g}"/><axis xyz="{_fmt(a)}"/>'
            f'<limit lower="{j.limit.min_angle:.6g}" upper="{j.limit.max_angle:.6g}" '
            f'effort="{j.servo.torque_limit:.6g}" velocity="{j.limit.max_velocity:.6g}"/>'
            f'<dynamics damping="0.02" friction="0.01"/></joint>')

    # ---- gripper fingers (prismatic, J6 drives both)
    fx = g.finger_thickness / 2
    for side, sgn in (("l", 1), ("r", -1)):
        name = f"finger_{side}"
        mass = g.finger_mass
        ix = mass / 12 * (g.finger_depth ** 2 + g.finger_length ** 2)
        iy = mass / 12 * (g.finger_thickness ** 2 + g.finger_length ** 2)
        iz = mass / 12 * (g.finger_thickness ** 2 + g.finger_depth ** 2)
        parts.append(
            f'<link name="{name}"><inertial><origin xyz="0 0 {g.finger_length/2:.6g}"/><mass value="{mass}"/>'
            f'<inertia ixx="{ix:.6g}" iyy="{iy:.6g}" izz="{iz:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>'
            + _box((g.finger_thickness, g.finger_depth, g.finger_length), (0, 0, g.finger_length / 2), FINGER,
                   f"finger_{side}") + "</link>")
        parts.append(
            f'<joint name="j6_{name}" type="prismatic"><parent link="ee"/><child link="{name}"/>'
            f'<origin xyz="{sgn*fx:.6g} 0 {g.palm_length:.6g}"/><axis xyz="{sgn} 0 0"/>'
            f'<limit lower="0" upper="{g.finger_travel:.6g}" effort="{g.max_grip_force}" velocity="0.2"/></joint>')

    # ---- TCP marker (fixed)
    parts.append('<link name="tcp"><inertial><mass value="0.001"/><inertia ixx="1e-9" iyy="1e-9" izz="1e-9" '
                 'ixy="0" ixz="0" iyz="0"/></inertial>' + '</link>')
    parts.append(f'<joint name="tcp_fixed" type="fixed"><parent link="ee"/><child link="tcp"/>'
                 f'<origin xyz="0 0 {cfg.tcp_offset:.6g}"/></joint>')
    if axes:                                   # appended LAST so the real joints keep their indices
        parts.append(_axis_links("base", L[0].length, 0.04, tag="_j1"))
        for i in range(1, 6):
            parts.append(_axis_links(L[i].name, 0.0, 0.03))
        parts.append(_axis_links("ee", g.palm_length, 0.02, tag="_j6"))
        parts.append(_axis_links("tcp", 0.0, 0.05, 0.001))
    parts.append("</robot>")
    return "\n".join(parts)


def export_urdf(cfg: RobotConfig, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(build_urdf(cfg), encoding="utf-8")
    return path
