# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Generator:
    idx: int = field(default=-1, init=False)
    generator: str
    technology: str
    bus: str
    site: str
    cap_existing_power_MW: float
    cap_max_power_MW: float
    cost_fixed_power_USDperkW: float
    cost_variable_USDperMWh: float
    c0_USD: float
    c1_USDperMWh: float
    c2_USDperMWh2: float

@dataclass(slots=True)
class CapacityFactor:
    idx: int = field(default=-1, init=False)
    site: str
    technology: str
    scenario: str
    timepoint: str
    capacity_factor: float
