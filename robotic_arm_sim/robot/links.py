"""Rigid link: geometry + inertial properties (all from LinkConfig, all configurable)."""
from __future__ import annotations

import numpy as np

from config.robot_config import LinkConfig


class Link:
    def __init__(self, index: int, cfg: LinkConfig):
        self.index = index            # 0 = base, 1..4 = arm links, 5 = end-effector
        self.cfg = cfg
        self.name = cfg.name
        self.length = cfg.length
        self.mass = cfg.mass
        self.com = np.asarray(cfg.com, dtype=float)
        self.size = cfg.size          # (width, depth) box used for visual + collision
        self.inertia = cfg.inertia_diag()

    @property
    def collision_radius(self) -> float:
        """Capsule radius used by the analytic collision checker."""
        return 0.5 * max(self.size)
