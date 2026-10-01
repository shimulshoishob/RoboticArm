"""Analytic (NumPy-only) collision checks used to veto unsafe targets before they are executed.

Link capsules are tested against the table plane and against each other (non-adjacent pairs).
Contacts with simulated objects are reported separately by the physics backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.robot_config import RobotConfig
from robot.kinematics import Kinematics
from utils.math_utils import segment_distance


@dataclass
class CollisionReport:
    colliding: bool = False
    pairs: list = field(default_factory=list)

    def __str__(self) -> str:
        return "COLLISION: " + ", ".join(self.pairs) if self.colliding else "none"


class CollisionChecker:
    def __init__(self, kin: Kinematics, config: RobotConfig, margin: float = 0.002):
        self.kin = kin
        self.config = config
        self.margin = margin
        self.table_z = config.table_z
        self.radii = {i: 0.5 * max(l.size) for i, l in enumerate(config.links)}
        # Slimmer self-collision radii: brackets overlap by design at joints.
        self.self_scale = 0.6

    def _capsules(self, q, opening: float):
        """Segment (a, b, radius, name) per link 1..5 plus the turntable column."""
        fr = self.kin.frames(q)
        links = self.config.links
        caps = {}
        top = fr[1][:3, 3] + np.array([0, 0, links[1].length])
        caps["base"] = (np.zeros(3), top, self.radii[1])
        for k in range(2, 5):
            caps[f"link{k}"] = (fr[k][:3, 3], fr[k + 1][:3, 3], self.radii[k])
        tip = (fr[5] @ np.array([0, 0, links[5].length, 1.0]))[:3]
        caps["ee"] = (fr[5][:3, 3], tip, self.radii[5])
        # Fingertip points (outer corners) for the floor test.
        g = self.config.gripper
        half = opening * g.finger_travel + g.finger_thickness
        tips = [(fr[5] @ np.array([s * half, 0, links[5].length, 1.0]))[:3] for s in (-1, 1)]
        return caps, tips

    def check(self, q, opening: float = 1.0) -> CollisionReport:
        q = np.asarray(q, dtype=float)[:5]
        caps, tips = self._capsules(q, opening)
        floor = self.table_z + self.margin
        pairs = []
        for name, (a, b, r) in caps.items():
            if name == "base":
                continue
            if min(a[2], b[2]) - r * 0.5 < floor and name != "ee":
                pairs.append(f"{name}-table")
        if any(t[2] < floor for t in tips) or min(caps["ee"][0][2], caps["ee"][1][2]) < floor:
            pairs.append("gripper-table")
        s = self.self_scale
        for n1, n2 in (("base", "link4"), ("base", "ee"), ("link2", "link4"), ("link2", "ee"), ("link3", "ee")):
            a1, b1, r1 = caps[n1]
            a2, b2, r2 = caps[n2]
            if segment_distance(a1, b1, a2, b2) < (r1 + r2) * s:
                pairs.append(f"{n1}-{n2}")
        return CollisionReport(bool(pairs), pairs)
