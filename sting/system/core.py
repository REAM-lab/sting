import pandas as pd
import importlib
import os
import itertools
from typing import get_type_hints
from dataclasses import fields
from sting import __logo__
from sting import data_files
from sting.line.core import decompose_lines
from sting.utils import data_tools

# from sting.shunt.core import combine_shunts
from sting.utils.graph_matrices import get_ccm_matrices
from sting.utils.dynamical_systems import StateSpaceModel
import sting.system.selections as sl


class System:
    """
    A power system object comprised of multiple components. 

    A list components of all possible are components within the system
    can be supplied during initialization. If no such list is supplied, 
    the power system will be initialized to accommodate all components 
    in ./data_files/components_metadata.csv.
    """
    # ------------------------------------------------------------
    # Construction + Read/Write
    # ------------------------------------------------------------
    def __init__(self, components=None):

        print(__logo__)
        print("> System initialization", end=" ")

        data_dir = os.path.dirname(data_files.__file__)
        filepath = os.path.join(data_dir, "components_metadata.csv")
        self.components = pd.read_csv(filepath)

        # If components are given, only use the relevant meta-data
        if components:
            active_components = self.components["type"].isin(components)
            self.components = self.components[active_components]

        # Mapping of a components class to its string representation
        self.class_to_str = dict(zip(self.components["class"], self.components["type"]))

        # Create a new attribute (an empty list) for each component type 
        for component_name in self.components["type"]:
            setattr(self, component_name, [])

        print("... ok.")

    @classmethod
    def from_csv(cls, inputs_dir=None, components=None):

        # If no input directory is given, try using the working directory
        if not inputs_dir:
            inputs_dir = os.getcwd()

        inputs_dir = os.path.join(inputs_dir, "inputs")
        self = cls(components=components)

        print("> Load components via CSV files from:")

        for _, c_name, c_class, c_module, filename in self.components.itertuples(
            name=None
        ):
            # Expected file with components 
            filepath = os.path.join(inputs_dir, filename)
            # If no such file exits, continue
            if not os.path.exists(filepath):
                continue

            # Import module, class, and expected data types
            class_module = importlib.import_module(c_module)
            component_class = getattr(class_module, c_class)
            parameters_dtypes = get_type_hints(component_class)
            parameters_dtypes = {
                key: value
                for key, value in parameters_dtypes.items()
                if value.__module__ == "builtins"
            }

            # Read components csv
            print(f"\t- '{filepath}'", end=" ")
            df = pd.read_csv(filepath, dtype=parameters_dtypes)

            # Create a component for each row (i.e., component) in the csv
            for row in df.itertuples(index=False):
                component = component_class(**row._asdict())
                # Add the component to the system
                self.add(component)

            print("... ok.")

        return self

    def to_csv(self, output_dir=None):
        # TODO: This is untested
        for name in self.components["type"]:
          lst = getattr(self, name)
          if lst:
              # Assumes each component is a dataclass with fields
              cols = fields(lst[0])
              df = self.query(name).to_table(cols)
              df.to_csv(os.path.join(output_dir, self.components["input_csv"]))

    def to_matlab(self, session_name=None):
        # TODO: Not sure if this has been tested
        import matlab.engine

        current_matlab_sessions = matlab.engine.find_matlab()

        if not session_name in current_matlab_sessions:
            print("> Initiate Matlab session, as a session was not founded or entered.")
            eng = matlab.engine.start_matlab()
        else:
            eng = matlab.engine.connect_matlab(session_name)
            print(f"> Connect to Matlab session: {session_name} ... ok.")

        components_types = self.component_types
        for typ in components_types:
            components = getattr(self, typ)

            components_dict = [
                data_tools.convert_class_instance_to_dictionary(i) for i in components
            ]

            eng.workspace[typ] = components_dict

        eng.quit()

    # ------------------------------------------------------------
    # Component Management + Searching
    # ------------------------------------------------------------
    def add(self, component):
        """Add a new component to the system."""
        # Get the component string representation (InfiniteSource -> inf_src)
        component_attr = self.class_to_str[type(component).__name__]
        component_list = getattr(self, component_attr)
        # Assign the component a 1-based index value
        component.idx = len(component_list) + 1
        # Add the component to the list
        component_list.append(component)
        
    def _generator(self, names):
        # Collect all lists of components in the component_types
        all_components = [getattr(self, name) for name in names]
         # Yield all components following the order in component_types
        return itertools.chain(*all_components)

    def query(self, *args):
        """
        Return a Stream over a set of component types. Analogous to FROM in 
        SQL, specifying which tables to access data from. For example, 
        "FROM gfmi_a, inf_src SELECT idx" would be written as:
        >>> power_sys.query("gfmi_a", "inf_src").select("idx")
        
        If no tables are provided runs a Stream over all components.
        """
        if not args:
            return sl.Stream(self, index_map=self.class_to_str)
        # Unpack all args calling on self if they are a function
        names = [arg(self) if callable(arg) else [arg] for arg in args]
        # Flatten the list of component types to query from
        names = itertools.chain(*names)

        return sl.Stream(self._generator(names), index_map=self.class_to_str)

    def __iter__(self):
        return self._generator(self.components["type"])
    
    def apply(self, method, *args):
        """Call a given method on all components with the method."""
        for component in self:
            if hasattr(component, method):
                getattr(component, method)(*args)

    @property
    def generators(self):
        return self.query(sl.generators())

    @property
    def shunts(self):
        return self.query(sl.shunts())

    @property
    def branches(self):
        return self.query(sl.branches())

    # ------------------------------------------------------------
    # Small-Signal Modeling
    # ------------------------------------------------------------
    def clean_up(self):
        """
        Apply any component clean up needed prior to methods like power flow.
        """
        decompose_lines(self)
        #TODO: I think combine_shunts(self) is untested

    def construct_ssm(self, pf_instance):
        """
        Create each components SSM given a power flow solution
        """
        # Build each components SSM
        self.apply("_load_power_flow_solution", pf_instance)
        self.apply("_calculate_emt_initial_conditions")
        self.apply("_build_small_signal_model")

        # Construct the component connection matrices for the system model
        self.connections = get_ccm_matrices(self)

    def interconnect(self):
        """
        Return a state-space model of all interconnected components
        """
        # Get components in order of generators, then shunts, then branches
        generators, = self.generators.select("ssm")
        shunts, = self.shunts.select("ssm")
        branches, = self.branches.select("ssm")

        models = itertools.chain(generators, shunts, branches)

        # Then interconnect models
        return StateSpaceModel.from_interconnected(models, self.connections)

    # ------------------------------------------------------------
    # Model Reduction (TBD?)
    # ------------------------------------------------------------
    def create_zone(self, c_names):
        pass

    def permute(self, index):
        pass
