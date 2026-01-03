# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Storage:
    idx: int = field(default=-1, init=False)
    storage: str
    technology: str
    bus: str
    cap_existing_energy_MWh: float
    cap_existing_power_MW: float
    cap_max_power_MW: float
    cost_fixed_energy_USDperkWh: float
    cost_fixed_power_USDperkW: float
    cost_variable_USDperMWh: float
    duration_hr: float
    efficiency_charge: float
    efficiency_discharge: float
    c0_USD: float
    c1_USDperMWh: float
    c2_USDperMWh2: float
