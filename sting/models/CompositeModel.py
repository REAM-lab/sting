import numpy as np
import os
from sting.models import StateSpaceModel, Variables

from dataclasses import dataclass
from sting.utils.graph_matrices import build_ccm_matrices, build_ccm_permutations

class CompositeModel():
    """
    Wrapper class for a collection of state-space models
    """
    def __init__(self, components, connections, attr=None):
        # Make these values frozen
        self._components = components
        self._connections = connections
        # Component attributes: name, type, zone, subtype, id 
        self._attr = attr 
        
    @classmethod
    def from_system(cls, system):

        def collect(component_list):
            """
            Helper function for collecting components of common type.
            """
            components = []
            
            for typ in component_list:
                # Get all components of a specific subtype (e.g., GFM_c)
                subtype_components = getattr(system, typ)
                # If they exist in the system, add all them to the main list
                if subtype_components:
                    components.extend(subtype_components)

            return components
        
        # TODO: Do we even need collect? Run in debug mode...

        generators = collect(system.generator_types_list)
        shunts = collect(system.shunt_types_list)
        branches = collect(system.branch_types_list)

        # Construct CCM matrices 
        connections = build_ccm_matrices(
            system=system, generators=generators, shunts=shunts,  branches=branches)
        # Define components by concatenating all state-space models
        components = generators + shunts + branches

        invT1, invT2 = build_ccm_permutations(system=system)

        return cls(components=components, connections=connections)

        

    def permute(self, index):
        assert len(set(index)) == len(self.components)
        pass

        # Get the number of elements in each subsystem ccm vector
        #y_stack = cell_size(subsystems.y_stack, 1) [c.n_y for c in self.components]
        #y_grid = cell_size(subsystems.y_grid, 1).  
        #u_stack = cell_size(subsystems.u_stack, 1)
        #u_grid =  cell_size(subsystems.u_grid, 1)

        # Permute each component connection matrix
        #F = block_permute(self.connections.F, u_stack, y_stack, index)
        #G = block_permute(self.connections.G, u_stack, u_grid,  index)
        #H = block_permute(self.connections.H, y_grid,  y_stack, index)
        #L = block_permute(self.connections.L, y_grid,  u_grid,  index)

        #components = [self.components[i] for i in index]
        # Re-order the rows in the attributes table
        #attr = self.attr.loc[index, :]

        #return CompositeModel(components, (F,G,H,L), attr=attr)
    
    def sort_components(self, by):
        index = self._attr.sort_values(by=by)['id']
        sys = self.permute(index=index)
        return sys
        

    def groupby(self, group):
        # Permute so all groups are together
        # Stack all components within groups
        sys = self.sort_components(group)

        components = []
        for value in self._attr[group].unique():
            models = sys.sel(key=group, value=value)
            components.append(StateSpaceModel.from_stacked(components=models))

        # TODO: sys._attr.groupby(group)

        return CompositeModel(components=components, connections=sys._connections)

    def interconnect(self):
        return StateSpaceModel.from_interconnected(self._components, self._connections)

    def stack(self):
        return StateSpaceModel.from_stacked(self._components)

    def apply(self, func, **kwargs):
        return [func(c, **kwargs) for c in self._components]

    def sel(self, key, value):
        """Return all component state-space models with attribute 'key' equal to 'value'."""
        idx = self._attr.loc[self._attr[key] == value, 'id']
        return [self._components[i] for i in idx]
    
    def to_matlab_session(self, session_name = None):
        pass


@dataclass(slots=True)
class ComponentConnections:
    F: np.ndarray 
    G: np.ndarray 
    H: np.ndarray 
    L: np.ndarray 
    u_stack: Variables = None
    y_stack: Variables = None
    u_grid: Variables = None
    y_grid: Variables = None

    def __post_init__(self):
        pass # TODO: Check that the sizes are valid

    @property
    def data(self):
        return self.F, self.G, self.H, self.L
