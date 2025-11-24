import unittest
import os
from sting.main import run_ssm
from sting.utils.dynamical_systems import StateSpaceModel

class TestExampleSystems(unittest.TestCase):

    def testcase1(self):
        # Read the correct state-space matrices
        outputs_dir = os.path.join(os.getcwd(), "testcase1_ssm")
        true_ssm = StateSpaceModel.from_csv(outputs_dir)
        # Compute SSM with current version of STING
        input_dir = os.path.join(os.getcwd(), os.pardir, os.pardir,"examples", "testcase1")
        sys, ssm = run_ssm(input_dir, write_outputs=False)
        # Compare
        tol = 1e-12
        self.assertTrue((ssm.A - true_ssm.A).sum() < tol)
        self.assertTrue((ssm.B - true_ssm.B).sum() < tol)
        self.assertTrue((ssm.C - true_ssm.C).sum() < tol)
        self.assertTrue((ssm.D - true_ssm.D).sum() < tol)
        

if __name__ == '__main__':
    unittest.main()