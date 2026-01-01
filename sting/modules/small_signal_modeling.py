# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from typing import NamedTuple, Optional, ClassVar
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import block_diag
from more_itertools import transpose
import itertools
import pandas as pd


# ------------------
# Import sting code
# ------------------
from sting.system.core import System
import sting.system.selections as sl
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel, modal_analisis
from sting.utils.graph_matrices import get_ccm_matrices, build_ccm_permutation

# -----------
# Sub-classes
# -----------
class VariablesSSM(NamedTuple):
    """
    All variables in the system for small-signal modeling.
    """
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

class ComponentSSM(NamedTuple):
    """
    A component of the system that participates in small-signal modeling.

    #### Attributes:
    - type: `str`
            inf_src, se_rl, pa_rc, ... etc. 
    - idx: `int`
            Index of the component in its corresponding list in the system.
    """
    type: str
    idx: int

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SmallSignalModel:
    system: System 
    components: list[ComponentSSM] = field(init=False)
    ccm_matrices: list[np.ndarray] = field(init=False)
    model: StateSpaceModel = field(init=False)

    def __post_init__(self):
        self.system.clean_up()
        self.get_components()
        self.construct_components_ssm()
        self.get_ccm_matrices()

    def get_components(self):
        """
        Get components that qualified for building the system-scale small-signal model.
        Not all components in system, e.g., bus, line_pi, etc., participate in small-signal modeling.         
        """

        components = []
        for component in self.system:
            if (    hasattr(component, "_load_power_flow_solution") 
                and hasattr(component, "_calculate_emt_initial_conditions") 
                and hasattr(component, "_build_small_signal_model")
                ):
                components.append(ComponentSSM(type = component.type, idx = component.idx))
        
        self.components = components


    def apply(self, method: str, *args):
        """
        Apply a method to the components for small-signal modeling.
        """
        for c in self.components:
               component = getattr(self.system, c.type)[c.idx-1]
               getattr(component, method)(*args)

    def get_ccm_matrices(self):
        """
        Get the CCM matrices in dq frame for the small-signal modeling.
        """
        
        F, G, H, L = get_ccm_matrices(self.system, attribute="ssm", dimI=2)

        T = build_ccm_permutation(self.system)
        T = block_diag(T, np.eye(F.shape[0] - T.shape[0]))

        F = T @ F
        G = T @ G

        self.ccm_matrices = [F, G, H, L]

    def construct_components_ssm(self):
        """
        Create each small-signal model of each component
        """
        self.apply("_calculate_emt_initial_conditions")
        self.apply("_build_small_signal_model")

    def construct_system_ssm(self):
        """
        Return a state-space model of all interconnected components
        """

        # Get components in order of generators, then shunts, then branches
        generators, = self.system.generators.select("ssm")
        shunts, = self.system.shunts.select("ssm")
        branches, = self.system.branches.select("ssm")

        models = itertools.chain(generators, shunts, branches)
     
        # Input of system are device inputs according to defined G matrix
        u = lambda stacked_u: stacked_u[stacked_u.type == "device"]

        # Output of system are all outputs according to defined H matrix
        y = lambda stacked_y: stacked_y
                
        # Then interconnect models
        self.model = StateSpaceModel.from_interconnected(models, self.ccm_matrices, u, y)

        # Print modal analysis
        modal_analisis(self.model.A, show=True)

        # Export small-signal model to CSV files
        self.model.to_csv(os.path.join(self.system.case_directory, "outputs", "small_signal_model"))
