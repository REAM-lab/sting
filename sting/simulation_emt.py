# Import python packages
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import itertools
from more_itertools import transpose

# Import sting code
from sting.system.core import System
import sting.system.selections as sl
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.graph_matrices import get_ccm_matrices

@dataclass(slots=True)
class SimulationEMT:
    system: System
    components: list = field(init=False)
    x: DynamicalVariables = field(init=False)
    u: DynamicalVariables = field(init=False)
    y: DynamicalVariables = field(init=False)
    ccm_abc_matrices: list[np.ndarray] = field(init=False)

    
    def define_variables(self):
        """
        Define EMT variables for all components in the system
        """
        self.system.apply("_define_variables_emt")

        generators, = self.system.generators.select("var_emt")
        shunts, = self.system.shunts.select("var_emt")
        branches, = self.system.branches.select("var_emt")

        variables_emt = itertools.chain(generators, shunts, branches)

        fields = ["x", "u", "y"]
        selection = [[getattr(c, f) for f in fields] for c in variables_emt]
        stack = dict(zip(fields, transpose(selection)))

        x = sum(stack["x"], DynamicalVariables(name=[]))
        y = sum(stack["y"], DynamicalVariables(name=[]))

        u = sum(stack["u"], DynamicalVariables(name=[]))
        ud = u[u.type == "device"]
        ug = u[u.type == "grid"]
        u = ud + ug

        components = np.unique(x.component).tolist()
        components = [[c, c.rpartition('_')[0], int(c.rpartition('_')[2])] for c in components]
        for c  in components:
            c.append([x.component == c[0]])
            c.append((u.component == c[0]) & (u.type == "grid"))
        # [['inf_src', '1', [...]], ['inf_src', '2', [...]], ['pa_rc', '1', [...]], ['pa_rc', '2', [...]], ['se_rl', '1', [...]]]

        self.components = components
        self.x = x
        self.u = u
        self.y = y

        self.ccm_abc_matrices = get_ccm_matrices(self.system, "var_emt", 3)


    def update_x_emt_value(self, system_state):

        for component, component_type, idx, x_idx, ug_idx in self.components:
            getattr(getattr(self.system, component_type)[idx-1], "var_emt").x.value_t = system_state[x_idx]



    def sim(self, t_max, inputs):
        """
        Simulate the EMT dynamics of the system using scipy.integrate.solve_ivp
        """
        
        F, G, H, L = self.ccm_abc_matrices
        
        def system_ode(t, x, ud):

            angle_sys = x[-1]  # last state is system angle

            y_stack = []

            for _, component_type, idx, x_idx, ug_idx in self.components:
                component = getattr(self.system, component_type)[idx-1]
                getattr(component, "var_emt").x.value = x[x_idx]
                y = getattr(component, "_get_output_emt")(t, ud)
                y_stack.extend(y)

            y_stack = np.array(y_stack).flatten()

            ustack = F @ y_stack 

            dx = []
        
            for _, component_type, idx, x_idx, ug_idx in self.components:
                component = getattr(self.system, component_type)[idx-1]
                ud_vals = ud.get(f"{component_type}_{component_idx}", 0)
                ug_vals = ustack[ug_idx]
                dx_comp = getattr(component, "_get_derivative_state_emt")(t, ud_vals, ug_vals, angle_sys)
                dx.extend(dx_comp)

            d_angle_sys = 2 * np.pi * 60 # rad/s
            dx.append(d_angle_sys)

            dx = np.array(dx).flatten()

            return dx
        
        x_init = self.x_emt.init
        x_init = np.append(x_init, [0.0])  # initial system angle

        solution = solve_ivp(system_ode, 
                        [0, t_max], # timeperiod 
                        x_init, # initial conditions
                        dense_output=True,  
                        args=(inputs, ),
                        method='Radau', 
                        max_step=0.001)
        
        return solution