import numpy as np
import pandas as pd
import os

from scipy.linalg import block_diag 
from typing import NamedTuple, Optional
from sting.utils import graph_matrices

def matrix_to_csv(filepath, matrix, index, columns):
        df = pd.DataFrame(matrix, index=index , columns=columns)
        df.to_csv(filepath)


class CompositeModel():
    
    def __init__(self, interconnections, models, names=None):
        self.N = interconnections
        
        self.name = names    # ordering of components, maybe some way to access systematically (e.g, return all ssm's of gens)
        self.models = models

    @classmethod
    def from_system(cls, system):
        pass


    def permute():
        pass

    def interconnect():
        pass

    def stack():
        pass

    def apply():
        pass

class StateSpaceModel(NamedTuple):
    A: np.ndarray 
    B: np.ndarray 
    C: np.ndarray 
    D: np.ndarray 
    #device_side_inputs: Optional[list] = None
    #grid_side_inputs:  Optional[list] = None
    states:  Optional[list] = None
    outputs:  Optional[list] = None
    inputs:  Optional[list] = None
    input_type:  Optional[list] = None

    def __post_init__(self):
        # Check that sizes match for A,B,C,D and inputs/outputs
        # If inputs and outputs are not given add a blank list
        x, x_other = self.A.shape
        assert x == x_other, "A is not square."
        B_x, u = self.B.shape
        assert x == B_x, "Incorrect dimensions for A and B."
        y, C_x = self.C.shape
        assert x == C_x, "Incorrect dimensions for A and C."
        D_y, D_u = self.D.shape
        assert D_y == y, "Incorrect dimensions for C and D."
        assert D_u == u, "Incorrect dimensions for B and D."
        
        self.states = self.states if self.states else [f'x{i}' for i in range(x)]
        self.inputs = self.inputs if self.inputs else [f'u{i}' for i in range(u)]
        self.outputs = self.outputs if self.outputs else [f'y{i}' for i in range(y)]
        self.inputs_type = self.inputs_type if self.inputs_type else ['grid']*u

    @classmethod
    def from_stacked(cls, models):

        stack = dict(zip(StateSpaceModel._fields, zip(*models)))
        A = block_diag(*stack['A'])
        B = block_diag(*stack['B'])
        C = block_diag(*stack['C'])
        D = block_diag(*stack['D'])

        states = sum(stack['states'], [])
        inputs = sum(stack['inputs'], [])
        outputs = sum(stack['outputs'], [])  
        input_type = sum(stack['input_type'], [])

        return cls(A=A, B=B, C=C, D=D, states=states, inputs=inputs, outputs=outputs, input_type=input_type)


    @classmethod
    def from_interconnected(cls, models, F, G, H, L):

        sys = cls.from_stacked(models)
        I_y = np.eye(F.shape[1])
        I_u = np.eye(F.shape[0])

        A = sys.A + sys.B @ F @ np.linalg.inv(I_y - sys.D @ F) @ sys.C
        B = sys.B @ np.linalg.inv(I_u - F @ sys.D) @ G
        C = H @ np.linalg.inv(I_y - sys.D @ F ) @ sys.C
        D = H @ np.linalg.inv(I_y - sys.D @ F ) @ sys.D @ G + L
        sys.A, sys.B, sys.C, sys.D = A, B, C, D

        return sys
    
    @classmethod
    def from_csv(cls, filepath):
        pass


    def coordinate_transform():
        pass

    def modal_analysis():
        pass

    def to_csv(self, filepath, silent=True):
        os.makedirs(filepath, exist_ok=True)
        matrix_to_csv(filepath=os.path.join(filepath, "A.csv"), index=self.states, columns=self.states)
        matrix_to_csv(filepath=os.path.join(filepath, "B.csv"), index=self.states, columns=self.inputs)
        matrix_to_csv(filepath=os.path.join(filepath, "C.csv"), index=self.outputs, columns=self.states)
        matrix_to_csv(filepath=os.path.join(filepath, "D.csv"), index=self.outputs, columns=self.inputs)
        if not silent:
            print(f'  - StateSpaceModel saved to {filepath} ')

    def __str__():
        pass




def get_interconnections(system, generators, branches, shunts):
    T1 = []
    T4 = []
    T5 = []

    for typ in system.generator_types_list:
        list_of_gens = getattr(system, typ)

        if not list_of_gens:
            continue

        n = len(list_of_gens)
        u_type = np.array(list_of_gens[0].ssm.input_type)

        n_d, n_g = sum(u_type == 'device'), sum(u_type == 'grid')

        T2 = np.kron(np.eye(n), 
                    np.hstack( ( np.eye(n_d), np.zeros((n_d, n_g)) ) ) )
        T3 = np.kron(np.eye(n), 
                    np.hstack( ( np.zeros((n_g, n_d)), np.eye(n_g) ) ) )

        T1.append(np.vstack((T2, T3)))
        T4.append(np.hstack(( np.eye(n*n_d), np.zeros((n*n_d, n*n_g)) )) )
        T5.append(np.hstack(( np.zeros((n*n_g, n*n_d)), np.eye(n*n_g) )) )

    T1 = block_diag(*T1)
    T4 = block_diag(*T4)
    T5 = block_diag(*T5)
    T6 = np.vstack((T4, T5))

    ##

    num_buses = len(system.bus)

    gen_bus_connections = []

    list_of_gentypes = system.generator_types_list
    for typ in list_of_gentypes:
        list_of_gens = getattr(system, typ)
        gen_bus_connections.extend([g.bus_idx for g in list_of_gens])

    branch_frombus_tobus = []
    list_of_branchtypes = system.branch_types_list
    for typ in list_of_branchtypes:
        list_of_branches = getattr(system, typ)
        branch_frombus_tobus.extend([(b.from_bus, b.to_bus) for b in list_of_branches])

    gen_cx = graph_matrices.build_generation_connection_matrix(num_buses, gen_bus_connections)
    or_inc = graph_matrices.build_oriented_incidence_matrix(num_buses, branch_frombus_tobus)

    un_inc = abs(or_inc)

    u_type = np.array(generators.input_type)

    d_gen = sum(u_type == 'device')
    g_gen = sum(u_type == 'grid')
    y_gen = len(generators.outputs)

    g_br = len(branches.inputs)
    y_br = len(branches.outputs)

    g_sh = len(shunts.inputs)
    y_sh = len(shunts.outputs)

    y = y_gen + y_sh + y_br

    F11 = np.zeros( (d_gen, y_gen) )
    F12 = np.zeros( (d_gen, y_sh) )
    F13 = np.zeros( (d_gen, y_br) )

    F21 = np.zeros( (g_gen, y_gen) )
    F22 = np.kron( gen_cx, np.eye(2) )
    F23 = np.zeros( (g_gen, y_br) )

    F31 = np.kron( np.transpose(gen_cx), np.eye(2))
    F32 = np.zeros( (g_sh, y_sh) )
    F33 = np.kron( or_inc, np.eye(2) )

    F41 = np.zeros( (g_br, y_gen) )
    F42 = np.kron( 0.5*(np.kron( np.transpose(un_inc) , np.array([[1], [1]]) )
                        + np.kron( np.transpose(or_inc) , np.array([[-1], [1]]) ) ) 
                            , np.eye(2) )
    F43 = np.zeros( (g_br, y_br) )

    F = np.block( [   [F11, F12, F13],
                        [F21, F22, F23],
                        [F31, F32, F33],
                        [F41, F42, F43] ])
    
    G11 = np.eye(d_gen)
    G21 = np.zeros((g_gen, d_gen))
    G31 = np.zeros((g_sh, d_gen))
    G41 = np.zeros((g_br, d_gen))

    G = np.block( [[G11], 
                    [G21],
                    [G31],
                    [G41]])
    
    H = np.eye(y)
    L = np.zeros((y, d_gen))

    F = np.linalg.inv(T6) @ np.linalg.inv(T1) @ F
    G = np.linalg.inv(T6) @ np.linalg.inv(T1) @ G

    return F, G, H, L







