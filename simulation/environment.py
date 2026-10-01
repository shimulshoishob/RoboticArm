"""Scene description: a table (top surface at z = 0) and a set of objects."""
from __future__ import annotations

from dataclasses import dataclass, field

from simulation.objects import SimObject, make_cube, make_cylinder, make_sphere


@dataclass
class Environment:
    table_size: tuple = (0.80, 0.60)           # x, y extent of the table top (m)
    table_thickness: float = 0.02
    objects: list = field(default_factory=list)

    def add(self, obj: SimObject) -> SimObject:
        self.objects.append(obj)
        return obj

    def get(self, name: str) -> SimObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    @classmethod
    def default_scene(cls) -> "Environment":
        env = cls()
        env.add(make_cube("cube", 0.0, 0.17))
        env.add(make_sphere("sphere", 0.10, 0.19))
        env.add(make_cylinder("cylinder", -0.10, 0.19))
        return env
