"""
Testcase1 simulates a two infinite sources connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

Second (optional), we transfer the system data and initial conditions 
to Simulink. STING has a functionality to transfer this data. 
In EMT simulation is in the file sim_emt.slx. 

Comparison between the small-signal simulation and EMT simultion
shows a proximity of these domain responses. Dec 7, 2025.
"""

# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = os.path.join(os.getcwd(), "examples", "testcase1")

# Construct system and small-signal model
main.run_emt(case_dir)

print('ok')