import os
from sting import main
from sting.utils.linear_systems_tools import modal_analisis
import matlab.engine
import numpy as np
import pandas as pd

case_dir = os.path.join(os.getcwd(), "examples", "testcase1")
sys, ssm = main.run_ssm(case_dir)

matlab_session = 's1'
sys.to_matlab(session_name = matlab_session, excluded_attributes=['ssm'])

tps = np.arange(0, 1.001, 0.001)
def u_func(t):
    return np.zeros((ssm.u.n_device,))

dx = ssm.sim(tps, u_func)
x = dx + np.atleast_2d(ssm.x.init).T
np.savetxt(os.path.join(case_dir, 'sim_ssm.csv'), x.T, delimiter=',', fmt='%f')

filepath = os.path.join(case_dir, "emt", "sps_output.csv")

# Initialize an empty dictionary to store the column lists
sps_output = np.loadtxt(filepath, delimiter=',').T
print(np.allclose(x, sps_output,  rtol=1e-05, atol=1e-04))

eng = matlab.engine.connect_matlab(matlab_session)
eng.workspace['ts'] = 1E-5
eng.quit()


print('ok')