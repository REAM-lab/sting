# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pyomo.environ as pyo

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import DynamicalVariables


# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class Scenario:
    idx: int = field(default=-1, init=False)
    scenario: str
    probability: float

    def __hash__(self):
        return hash(self.idx)


@dataclass(slots=True)
class Timepoint:
    idx: int = field(default=-1, init=False)
    timepoint: str
    timeseries: str
    timeseries_idx: Optional[int] = field(init=False, default=None)
    weight: Optional[float] =field(init=False, default=None)
    duration_hr: Optional[float] = field(init=False, default=None)    
    prev_timepoint_idx: Optional[int] = field(default=None, init=False)

    def __hash__(self):
        return hash(self.idx)

@dataclass(slots=True)
class Timeseries:
    idx: int = field(default=-1, init=False)
    timeseries: str
    timepoint_duration_hr: float
    number_of_timepoints: int
    timeseries_scale_to_period: float
    timepoint_ids: Optional[list[int]] = field(default=None, init=False)



def timescale_calculations(tps : list[Timepoint], ts : list[Timeseries]):
    for t in tps:
        # Get timeseries idx
        t.timeseries_idx = next(filter(lambda p: p.timeseries == t.timeseries, ts)).idx

        # Get duration (hr) for the timepoint
        t.duration_hr = next(filter(lambda p: p.idx == t.timeseries_idx, ts)).timepoint_duration_hr

        # Weigth for each timepoint
        t.weight = t.duration_hr * next(filter(lambda p: p.idx == t.timeseries_idx, ts)).timeseries_scale_to_period

    for tt in ts:
        # Get timepoint ids
        tt.timepoint_ids = [t.idx for t in tps if t.timeseries_idx == tt.idx]

    for t in tps:
        # Get ids of the timepoint that are within the same timeseries
        tps_in_ts = [tp for tp in tps if tp.timeseries_idx == t.timeseries_idx]

        if t.idx == tps_in_ts[0].idx:
            t.prev_timepoint_idx = tps_in_ts[-1].idx
        else:
            t.prev_timepoint_idx = t.idx - 1

def construct_capacity_expansion_model(system, model, model_settings):
    
    model.S = pyo.Set(initialize=system.sc)
    model.T = pyo.Set(initialize=system.tp)