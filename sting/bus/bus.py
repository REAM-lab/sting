# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo


# -------------
# Import sting code
# --------------
from sting.timescales.core import Timepoint, Scenario
from sting.utils.graph_matrices import build_admittance_matrix2

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Bus:
    idx: int = field(default=-1, init=False)
    bus: str
    bus_type: str = None
    sbase: float = None
    vbase: float = None
    fbase: float = None
    v_min: float = None
    v_max: float = None
    p_load: float = None
    q_load: float = None
    tags: ClassVar[list[str]] = []

    def __repr__(self):
        return f"Bus(idx={self.idx}, bus='{self.bus}')"

    def __hash__(self):
        return hash(self.idx)

  
@dataclass(slots=True)
class Load:
    idx: int = field(default=-1, init=False)
    bus: str
    scenario: str
    timepoint: str
    load_MW: float
    bus_idx: int = field(default=None, init=False)
    scenario_idx: int = field(default=None, init=False)
    timepoint_idx: int = field(default=None, init=False)

    def assign_indices(self, buses: list[Bus], scenarios: list[Scenario], timepoints: list[Timepoint]):
        self.bus_idx = next(filter(lambda b: b.bus == self.bus, buses)).idx
        self.scenario_idx = next(filter(lambda s: s.scenario == self.scenario, scenarios)).idx
        self.timepoint_idx = next(filter(lambda t: t.timepoint == self.timepoint, timepoints)).idx


def construct_capacity_expansion_model(system, model, model_settings):

    model.N = pyo.Set(initialize=system.bus)

    model.vTHETA = pyo.Var(model.N, model.S, model.T, within=pyo.Reals)

    slack_bus = next(n for n in system.bus if n.bus_type == 'slack')

    Y = build_admittance_matrix2(len(system.bus), system.line)
    B = Y.imag

    model.vTHETA[slack_bus.idx, :, :].fix(0.0)

    model.eFlowAtBus = pyo.Expression(model.N, model.S, model.T, rule=lambda m, n, s, t: 100 * sum(B[n.idx-1, k.idx-1] * (m.vTHETA[n, s, t] - m.vTHETA[k, s, t]) for k in model.N))


    