from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(slots=True)
class LinePiModel:
    id: int = field(default=-1, init=False)
    name: str 
    from_bus: str
    to_bus: str
    sbase_VA: float
    vbase_V: float
    fbase_Hz: float
    r_pu: float
    l_pu: float
    g_pu: float
    b_pu: float
    decomposed: bool = field(default=False)
    tags: ClassVar[list[str]] = ["line"]
    from_bus_id: int = None
    to_bus_id: int = None

    def assign_bus_id(self, buses: list):
        self.from_bus_id = next((n for n in buses if n.name == self.from_bus)).id
        self.to_bus_id = next((n for n in buses if n.name == self.to_bus)).id

@dataclass(slots=True)
class Line:
    id: int = field(default=-1, init=False)
    line: str
    bus_from: str
    bus_to: str
    r_pu: float
    x_pu: float
    g_pu: float
    b_pu: float
    rating_MVA: float
    transformer: bool
    angle_max_deg: float 
    angle_min_deg: float 
    bus_from_id: int = field(default=None, init=False)
    bus_to_id: int = field(default=None, init=False)

    def assign_indices(self, buses: list):
        self.bus_from_id = next((n for n in buses if n.bus == self.bus_from)).id
        self.bus_to_id = next((n for n in buses if n.bus == self.bus_to)).id
