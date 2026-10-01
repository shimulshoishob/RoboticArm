"""Simple graspable objects (SI units)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimObject:
    name: str
    kind: str                      # "cube" | "sphere" | "cylinder"
    dims: tuple                    # cube: (edge,)  sphere: (radius,)  cylinder: (radius, height)
    mass: float
    position: tuple                # initial centre (x, y, z)
    color: tuple = (0.85, 0.25, 0.2, 1.0)
    body_id: Optional[int] = None  # filled by the physics backend

    @property
    def half_height(self) -> float:
        if self.kind == "cube":
            return self.dims[0] / 2
        return self.dims[0] if self.kind == "sphere" else self.dims[1] / 2

    @property
    def width(self) -> float:
        """Size across a gripper that grips it along X."""
        return self.dims[0] if self.kind == "cube" else 2 * self.dims[0]


def make_cube(name: str, x: float, y: float, edge: float = 0.030, mass: float = 0.025,
              color=(0.85, 0.25, 0.2, 1.0)) -> SimObject:
    return SimObject(name, "cube", (edge,), mass, (x, y, edge / 2), color)


def make_sphere(name: str, x: float, y: float, radius: float = 0.015, mass: float = 0.02,
                color=(0.2, 0.5, 0.85, 1.0)) -> SimObject:
    return SimObject(name, "sphere", (radius,), mass, (x, y, radius), color)


def make_cylinder(name: str, x: float, y: float, radius: float = 0.013, height: float = 0.045,
                  mass: float = 0.03, color=(0.25, 0.7, 0.35, 1.0)) -> SimObject:
    return SimObject(name, "cylinder", (radius, height), mass, (x, y, height / 2), color)
