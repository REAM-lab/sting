# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass
import copy
import numpy as np
from scipy.linalg import solve

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.line.pi_model import LinePiModel
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.utils.data_tools import mat2cell



# -----------
# Main class
# -----------
@dataclass
class KronReduction():
    # g - d = Y * z
    #   g: generation
    #   d: demand
    #   z: bus angle (relative to slack)
    system: System
    remove_buses: set = None
    
    def __post_init__(self):
        self.system = copy.deepcopy(self.system)

    def reduce(self):
        
        first, last = [], []
        for b in self.system.bus:
            if b.name in self.remove_buses:
                last.append(b)
            else:
                first.append(b)
        
        
        # Sort all system buses placing buses to remove last
        self.system.bus = []
        for b in (first + last):
            self.system.add(b)
        
        # Update all line and generator indices
        self.system.apply("assign_indices", self.system)

        # Number of total, unused, and real buses
        n_bus = len(self.system.bus)
        q = len(self.remove_buses)
        p = n_bus - q

        # Build & partition admittance matrix
        Y = build_admittance_matrix_from_lines(n_bus, self.system.line_pi)
        (Y_pp, Y_pq), (Y_qp, Y_qq) = mat2cell(Y, [p,q], [p,q])
        # Back substitute to get reduced matrix
        invY_qq = solve(Y_qq, np.eye(q))
        Y_red = Y_pp - Y_pq @ (invY_qq) @ Y_qp

        # Build the new reduced lines
        self.system.line_pi = []
        self.system.bus = self.system.bus[:p]

        for i, j in zip(*np.triu_indices(p)):
            # Skip self loops and unconnected nodes
            if (i == j) or (Y_red[i, j] == 0):
                continue
            
            y = -Y_red[i, j]
            z = 1/y

            line = LinePiModel(
                name=f"Y_kron_{i}{j}",
                # Line connectivity
                from_bus=self.system.bus[i].name, from_bus_id=i,
                to_bus=self.system.bus[j].name, to_bus_id=j,
                # Line parameters
                r_pu=z.real, x_pu=z.imag, # Z = R + jX
                g_pu=y.real, b_pu=y.imag
            )
            self.system.add(line)


    def line_cap():
        # Create graph object

        # for each bus to remove:
        #.  1. look up buses nearest neighbors
        #.  3. Create new edges between *all* neighbors with a weight given by min of both edges
        #.  4. Delete the bus.
        pass