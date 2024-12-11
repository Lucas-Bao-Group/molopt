import os
import time
import unittest
import logging
from math import sin

import numpy as np
import rmsd
from ase.io import read
# from xtb.ase.calculator import XTB
from ase.calculators.lj import LennardJones
from molopt.internal.converter import Cartesian2Internal, Cartesian2DelocalizedInternal, clean_pi
from molopt.internal.z_mat_type import Stretching, Bending, Torsion


class Cartesian2InternalTest(unittest.TestCase):
    def setUp(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        xyz_file = os.path.join(cwd, "./test_files/CH2OH.xyz")
        self.mol = read(xyz_file)
        self.internal = Cartesian2Internal(self.mol)
        self.delocalizedinternal = Cartesian2DelocalizedInternal(self.mol)
        self.geom = self.mol.get_positions()
        self.mol1 = self.mol.copy()
        self.mol1.rattle(0.2)
        self.geom1 = self.mol1.get_positions()
        print(f'DOF cart {self.geom.shape[0] * 3} DOF internal: {len(self.internal.internal_coords)}')

    def test_convert2internal(self):
        time1 = time.time()
        for i in self.internal.internal_coords:
            try:
                if isinstance(i, Stretching):
                    ase_value = self.mol.get_distance(i.A, i.B)
                if isinstance(i, Bending):
                    ase_value = self.mol.get_angle(i.A, i.B, i.C) * 3.1415926 / 180
                if isinstance(i, Torsion):
                    ase_value = self.mol.get_dihedral(i.A, i.B, i.C, i.D) * 3.1415926 / 180
            except ZeroDivisionError:
                ase_value = 'ERROR'
            print(f'{i}, Test Value:{i.q(self.geom)} ase_ref:{ase_value}')
            if ase_value != 'ERROR':
                assert sin(i.q(self.geom) - ase_value) < 1e-6
        print("Cartesian2Internal TIme: ", time.time() - time1)

    def test_convert2delocalized(self):
        s = self.delocalizedinternal.get_q_values(self.geom)
        s1 = self.delocalizedinternal.get_q_values(self.geom1)
        print(s)
        print(s1)

    def test_convert2delocalizedforce(self):
        f = self.mol.get_forces().reshape(-1)
        fs = self.delocalizedinternal.f_xyz_to_q(f, self.geom)

        print(fs)

    # def test_ad_get_b_matrix(self):
    #     time1 = time.time()
    #     b1 = self.internal.get_b_matrix_ad(self.geom)
    #     print("B Matrix AD time:", time.time() - time1)
    #     b2 = self.internal.get_b_matrix_ad(self.geom1)
    #     print("B Matrix AD time *2 :", time.time() - time1)

    def test_get_b_matrix(self):
        time1 = time.time()
        b1 = self.internal.get_b_matrix_analytic(self.geom)

        print("B Matrix time:", time.time() - time1)

        for i, int_coords in enumerate(self.internal.internal_coords):
            dqdx = int_coords.DqDx(self.geom1)
        print("B Matrix time *2 :", time.time() - time1)

    # def test_b_matrix_ad_analytic_eq(self):
    #
    #     time1 = time.time()
    #     b1 = self.internal.get_b_matrix_ad(self.geom1)
    #     b2 = self.internal.get_b_matrix_analytic(self.geom1)
    #     print('b1',b1)
    #     print('b2',b2)
    #     assert (b1 - b2).sum() < 1e-7

    def test_convert_force_to_internal(self):
        f = self.mol.get_forces().reshape(-1)
        f_intern = self.internal.f_xyz_to_q(f, self.geom)
        print(f_intern)

    def test_internal2cartesian(self):
        time_ = time.time()
        print('xyz_diff\n', self.geom - self.geom1)
        int_ = Cartesian2Internal(self.mol)
        int1 = Cartesian2Internal(self.mol1)
        # int_.fixBendAxes(self.geom)
        int_.updateDihedralOrientations(self.geom)
        int1.updateDihedralOrientations(self.geom1)
        # int1.fixBendAxes(self.geom1)
        np.random.seed(2)
        geom_with_noise = self.geom + np.random.normal(0, 0.1, self.geom.shape)
        geom_with_noise_1 = self.geom1 + np.random.normal(0, 0.1, self.geom.shape)
        q_ = int_.get_q_values(geom_with_noise)
        q1 = int1.get_q_values(geom_with_noise_1)
        print('Internal coords:\n', list(zip(self.internal.internal_coords, q_)))
        geom1_new_from_internal1 = int1.q_to_xyz(self.geom1, q1)
        geom_new_from_internal = int_.q_to_xyz(self.geom, q_)
        rmsd1 = rmsd.kabsch_rmsd(geom1_new_from_internal1, geom_with_noise_1, translate=True)
        rmsd_ = rmsd.kabsch_rmsd(geom_new_from_internal, geom_with_noise, translate=True)

        print('rmsd:', rmsd1, rmsd_)
        assert rmsd1 < 1e-5
        assert rmsd_ < 1e-5
        print(f'test_internal2cartesian TIME {time.time() - time_}')

    def test_internal2cartesian_by_zmat(self):
        time_ = time.time()
        print('xyz_diff\n', self.geom - self.geom1)
        internal = np.array([i.q(self.geom) for i in self.internal.internal_coords])
        print('Internal coords:\n', list(zip(self.internal.internal_coords, internal)))
        internal1 = np.array([i.q(self.geom1) for i in self.internal.internal_coords])
        geom1_new_from_internal1 = self.internal.q_to_xyz_by_zmat(internal1)
        geom_new_from_internal = self.internal.q_to_xyz_by_zmat(internal)
        rmsd1 = rmsd.kabsch_rmsd(geom1_new_from_internal1, self.geom1, translate=True)
        rmsd_ = rmsd.kabsch_rmsd(geom_new_from_internal, self.geom, translate=True)

        print('rmsd:', rmsd1, rmsd_)
        assert rmsd1 < 1e-5
        assert rmsd_ < 1e-5
        print(f'test_internal2cartesian_by_zmat TIME {time.time() - time_}')

    def do_not_test_dlc2ic(self):
        """
        this is example we should not test.
        DLC>IC lose some information and is not 1to1 mapping.
        It will not match! Only test DLC>XYZ!
        """
        s1 = self.delocalizedinternal.get_q_values(self.geom1)
        q1 = self.delocalizedinternal.redundant_internal.get_q_values(self.geom1)
        q_ = self.delocalizedinternal.s_to_q(self.geom, s1)
        diff = clean_pi(self.delocalizedinternal.redundant_internal.type_array, q1 - q_)
        print('test_dlc2ic Max diff:', max(abs(diff)))
        assert max((q1 - q_) ** 2) < 1e-8

    def test_delocalized2cartesian(self):
        time_ = time.time()
        print('xyz_diff\n', self.geom - self.geom1)
        int_ = Cartesian2DelocalizedInternal(self.mol)
        int1 = Cartesian2DelocalizedInternal(self.mol1)
        int_.fixBendAxes(self.geom)
        int_.updateDihedralOrientations(self.geom)
        int1.updateDihedralOrientations(self.geom1)
        int1.fixBendAxes(self.geom1)
        np.random.seed(2)
        geom_with_noise = self.geom + np.random.normal(0, 0.1, self.geom.shape)
        geom_with_noise_1 = self.geom1 + np.random.normal(0, 0.1, self.geom.shape)
        q_ = int_.get_q_values(geom_with_noise)
        q1 = int1.get_q_values(geom_with_noise_1)
        print('Internal coords:\n', list(zip(self.internal.internal_coords, q_)))
        geom1_new_from_internal1 = int1.q_to_xyz_safe(self.geom1, q1)
        geom_new_from_internal = int_.q_to_xyz_safe(self.geom, q_)
        rmsd1 = rmsd.kabsch_rmsd(geom1_new_from_internal1, geom_with_noise_1, translate=True)
        rmsd_ = rmsd.kabsch_rmsd(geom_new_from_internal, geom_with_noise, translate=True)

        print('rmsd:', rmsd1, rmsd_)
        assert rmsd1 < 1e-5
        assert rmsd_ < 1e-5
        print(f'delocalized2cartesian TIME {time.time() - time_}')



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()
