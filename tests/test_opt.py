import os
import unittest
from ase.io import read
# from xtb.ase.calculator import XTB
from ase.calculators.lj import LennardJones



class BFGS_RTest(unittest.TestCase):
    def setUp(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        xyz_file = os.path.join(cwd, "./test_files/CH2OH.xyz")
        self.mol = read(xyz_file)

    def test_bfgs(self):
        from ase.optimize import BFGS

        dyn = BFGS(self.mol)
        self.mol.calc = LennardJones()
        dyn.run()


    def test_bfgs_internal(self):
        from molopt import BFGS_R
        dyn = BFGS_R(self.mol)
        self.mol.calc = LennardJones()
        dyn.run()