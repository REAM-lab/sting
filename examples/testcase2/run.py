import os
from sting import main
from sting.utils.linear_systems_tools import modal_analisis
import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Specify path of the case study directory
case_dir = os.path.join(os.getcwd(), "examples", "testcase2")

# Construct system and small-signal model
sys, ssm = main.run_ssm(case_dir)

# Send system information to MATLAB session (optional)
# First, in MATLAB, execute: matlab.engine.shareEngine('s1')
matlab_session = 's1' 
sys.to_matlab(session_name = matlab_session, excluded_attributes=['ssm'])

eng = matlab.engine.connect_matlab(matlab_session)
eng.workspace['ts'] = 1E-5
eng.quit()

tps = np.arange(0, 3.001, 0.001)
def u_func(t):
    v_dc_ref = 0.1 if t>= 0.3 else 0
    i_q_ref = -0.1 if t>= 2 else 0
    i_dc_src = 0
    v_int_d = 0 
    v_int_q = 0
    
    return np.array([v_dc_ref, i_q_ref, i_dc_src, v_int_d, v_int_q])

dx = ssm.sim(tps, u_func)
x_ssm = dx + np.atleast_2d(ssm.x.init).T
np.savetxt(os.path.join(case_dir, 'outputs', 'sim_ssm.csv'), x_ssm.T, delimiter=',', fmt='%f')


# In MATLAB, execute: writematrix(out.simout.Data, 'sps_output.csv');
filepath = os.path.join(case_dir, "emt", "sps_output.csv")

# Initialize an empty dictionary to store the column lists
x_emt = np.loadtxt(filepath, delimiter=',').T

states = ssm.x.name
dir = os.path.join(case_dir, "outputs", "small_signal_model")
for (i, x) in enumerate(states):
    fig, ax = plt.subplots(dpi=1200, figsize=(4,3.0))
    ax.plot(tps, x_ssm[i], color='red', linestyle='solid')
    ax.plot(tps, x_emt[i], color='blue', linestyle='dashed')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'{x}')
    plt.savefig(os.path.join(dir, f"{x}.pdf"))


print('ok')