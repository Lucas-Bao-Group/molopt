
import logging
from abc import ABC, abstractmethod
from itertools import combinations, permutations

import numpy as np
import numpy as jnp  # Switch to jax.numpy enable jax support.[debug purpose]
import scipy.optimize

from ase import Atoms
from typing import List, Union

from scipy.optimize import Bounds

from molopt.internal.utils import get_conn, make_virtual_connectivity, make_hydrogen_bond
from molopt.internal.z_mat_type import Stretching, Bending, Torsion, InternalBase, OOFP

logger = logging.getLogger(__name__)
def clean_pi(type_array, internal_coords: np.ndarray):
    """
    clean torsion internal_coords by periodic condition. if not -pi <q_torsion<= pi
    This is also valid for delta q_torsion.
    we assume no q_torsion > 2pi or < -2pi
    # Note bending angle should be constrained to [0~pi]. so we do not clean here
    Args:
        internal_coords: in shape [r] as [q_1,q_2...q_r]

    Returns: cleand internal_coords: in shape [r] as [q_1,q_2...q_r]

    """
    # Maybe store it
    diff_array = np.zeros(len(internal_coords))
    # diff_array += ((type_array == 1) & (internal_coords > jnp.pi)) * (- np.pi)
    # diff_array += ((type_array == 1) & (internal_coords < 0)) * np.pi
    diff_array += ((type_array == 2) & (internal_coords <= -jnp.pi)) * 2 * np.pi
    diff_array += ((type_array == 2) & (internal_coords > jnp.pi)) * 2 * (- np.pi)
    cleaned = internal_coords + diff_array
    return cleaned


class NonCartesianSystem(ABC):
    """
        class for Non-Cartesian coord.system.
        xyz_3N>Q_M
        M should >= 3N-6

    """

    @abstractmethod
    def f_xyz_to_q(self, f: np.ndarray, geom: np.ndarray) -> np.ndarray:
        """

        Args:
            f: force/gradient  in Cartesian 3N
            geom: xyz geometry in (N,3) shape

        Returns: force/gradient in Q coord. ndarray in 3M shape.

        """

    @abstractmethod
    def q_to_xyz(self, geom: Union[np.ndarray, None],
                 new_internal_coords: np.ndarray,
                 tol: float = 1e-15,
                 ) -> np.ndarray:
        """

        Args:
            geom:  reference xyz geometry in (N,3) shape. Some conversion method require an initial guess.
                    pass None will use last cached geometry.
            new_internal_coords: geometry in Q coord. ndarray in 3q shape.
            tol: tolerance for max(delta_Q)

        Returns: xyz geometry in (N,3) shape

        """

    @abstractmethod
    def get_q_values(self, geom: np.ndarray) -> np.ndarray:
        """
        Get Q values from given xyz geometry.
        Args:
            geom: xyz geometry in (N,3) shape

        Returns:
            Q values {Q_1,Q_2,...,Q_M}
        """

    @abstractmethod
    def f_q_to_xyz(self, f: np.ndarray, geom: np.ndarray) -> np.ndarray:
        """

        Args:
            f: force/gradient  in ICs
            geom: xyz geometry in (N,3) shape

        Returns: force/gradient in xyz coords.. ndarray in M shape.

        """

    @property
    @abstractmethod
    def bounds(self) -> scipy.optimize.Bounds:
        """
        Up and low bounds for scipy optimizer.
        Returns:
        scipy.optimize.Bounds
        """
        ...

    @property
    @abstractmethod
    def dof(self) -> int:
        ...

    @abstractmethod
    def get_g(self,geom: np.ndarray, rcond: float = 1e-6)->np.ndarray:
        ...

    @abstractmethod
    def get_g_inv(self, geom: np.ndarray, rcond: float = 1e-6) ->np.ndarray:
        pass

class Cartesian2Internal(NonCartesianSystem):

    def __init__(self, atoms: Atoms, cutoff: float = 1.3, length_coeff=1):
        """
        Cartesian to redundant internal coord. system.
        Args:
            atoms: ase Atoms object for build the conversion. Note: position from the atoms will be used.
            cutoff: cutoff radius for build connectivity.
        """
        self.bohr_coff: Union[None, jnp.ndarray] = None
        self.length_coeff = length_coeff
        self._bounds: Union[None, Bounds] = None
        self.mol = atoms.copy()
        self.n_atom = len(self.mol)
        self.connectivity = get_conn(self.mol, cutoff=cutoff)
        self.virtual_connectivity = self.make_virtual_bond()
        self.internal_coords = self.get_internal_coords()
        self.b_matrix_func = get_b_matrix_jitfunction(self.internal_coords)
        self.init_geom = self.mol.get_positions()
        self.last_geom = self.init_geom
        # cached matrix
        self._geom: Union[None, np.ndarray] = None
        self.b_matrix: Union[None, np.ndarray] = None
        self.b_matrix_geom: Union[None, np.ndarray] = None
        self.g_inv: Union[None, np.ndarray] = None
        self.g_inv_geom: Union[None, np.ndarray] = None
        self.type_array: Union[None, jnp.ndarray] = None
        self.geom = self.mol.get_positions()
        self.create_type_array()
        if not self.test_complete():
            logger.info('redundant internal coordinates is not complete set, adding torsion ')
            self.make_any_torsion()
            self.b_matrix: Union[None, np.ndarray] = None
            self.b_matrix_geom: Union[None, np.ndarray] = None
            self.g_inv: Union[None, np.ndarray] = None
            self.g_inv_geom: Union[None, np.ndarray] = None
            self.create_type_array()
        try:
            self.conn_ls, self.zmat_mapping = self.make_zmat_mapping()
        except CannotBuildZmat:
            self.conn_ls, self.zmat_mapping = None, None

    @property
    def dof(self):
        return len(self.internal_coords)


    def make_zmat_mapping(self):
        """
        J. Chem. Phys. 110, 4986 (1999); https://doi.org/10.1063/1.478397
        For non-iterative back-transformation from DLC.
        make Z matrix from ICs. idk how it works but it works.
        This should only be called once. Maybe don't bother to optimize it.
        Returns: atoms_ls , mapping array

        """
        torsion_ls = []

        for inter in self.internal_coords:
            if type(inter) is Torsion:
                torsion_ls.append(inter)
        for i in range(len(torsion_ls)):
            atoms_ls = []
            zmat_ls = []
            conn_ls = []
            torsion_l = torsion_ls.copy()
            zmat_ls.append(Stretching(*torsion_ls[i].atoms[:2]))  # Atom 2 R12
            zmat_ls.append(Stretching(*torsion_ls[i].atoms[1:3]))  # Atom 3 R23
            zmat_ls.append(Bending(*torsion_ls[i].atoms[:3]))  # Atom 3 B123
            atoms_ls.extend(torsion_ls[i].atoms[:3])
            conn_ls.append(torsion_ls[i].atoms)
            while i < len(torsion_l):
                if set(torsion_l[i].atoms[:3]).issubset(set(atoms_ls)) and torsion_l[i].atoms[-1] not in set(atoms_ls):
                    zmat_ls.append(Stretching(*torsion_l[i].atoms[2:4]))  # Atom n R
                    zmat_ls.append(Bending(*torsion_l[i].atoms[1:4]))  # Atom n B
                    zmat_ls.append(torsion_l[i])  # Atom n D
                    atoms_ls.append(torsion_l[i].atoms[-1])
                    conn_ls.append(torsion_l[i].atoms)
                    torsion_l.remove(torsion_l[i])
                    i = 0
                    continue
                elif set(torsion_l[i].atoms[1:]).issubset(set(atoms_ls)) and torsion_l[i].atoms[0] not in set(atoms_ls):
                    zmat_ls.append(Stretching(*torsion_l[i].atoms[0:2]))  # Atom n R
                    zmat_ls.append(Bending(*torsion_l[i].atoms[0:3]))  # Atom n B
                    zmat_ls.append(torsion_l[i])  # Atom n D
                    atoms_ls.append(torsion_l[i].atoms[0])
                    conn_ls.append(torsion_l[i].atoms[::-1])
                    torsion_l.remove(torsion_l[i])
                    i = 0
                    continue
                i += 1

            if len(zmat_ls) == len(self.mol) * 3 - 6:
                valid = True
                mapping = np.zeros((len(zmat_ls), len(self.internal_coords)))
                for row, zmat in enumerate(zmat_ls):
                    if zmat not in self.internal_coords:
                        valid = False
                    else:
                        mapping[row, self.internal_coords.index(zmat)] = 1
                if valid:
                    return conn_ls, mapping
        raise CannotBuildZmat

    def get_zmat_values(self, q: np.ndarray) -> np.ndarray:
        """
        J. Chem. Phys. 110, 4986 (1999); https://doi.org/10.1063/1.478397
        For non-iterative back-transformation from DLC.
        Args:
            q: ICs values

        Returns: Z matrix vaules

        """
        return self.zmat_mapping @ q

    def q_to_xyz_by_zmat(self, q: np.ndarray) -> np.ndarray:
        """
        J. Chem. Phys. 110, 4986 (1999); https://doi.org/10.1063/1.478397
        For non-iterative back-transformation from DLC.

        Args:
            q: ICs values

        Returns: Cartesian np.ndarray

        """
        z_mat_values = self.get_zmat_values(q)
        pos = np.zeros((len(self.mol), 3))
        # 0 in 0,0,0
        # 1 in [r01, 0, 0]
        pos[self.conn_ls[0][1]] = np.array([z_mat_values[0], 0, 0]) / self.length_coeff
        # third atom in the xy-plane
        r = z_mat_values[1] / self.length_coeff
        theta = z_mat_values[2]
        x = -1 * r * np.cos(theta)
        y = r * np.sin(theta)

        pos[self.conn_ls[0][2]] = pos[self.conn_ls[0][1]] + np.array([x, y, 0])

        for i, z_mat_value in enumerate(z_mat_values[3:].reshape(-1, 3)):
            r, theta, phi = z_mat_value
            r /= self.length_coeff
            sintheta = np.sin(theta)
            costheta = np.cos(theta)
            sinphi = np.sin(phi)
            cosphi = np.cos(phi)
            dx = r * costheta
            dy = r * cosphi * sintheta
            dz = r * sinphi * sintheta

            a = pos[self.conn_ls[i + 1][0]]
            b = pos[self.conn_ls[i + 1][1]]
            c = pos[self.conn_ls[i + 1][2]]
            ab = b - a
            bc = c - b
            bc = bc / np.linalg.norm(bc)
            nv = np.cross(ab, bc)
            nv = nv / np.linalg.norm(nv)
            ncbc = np.cross(nv, bc)

            new_x = c[0] - bc[0] * dx + ncbc[0] * dy + nv[0] * dz
            new_y = c[1] - bc[1] * dx + ncbc[1] * dy + nv[1] * dz
            new_z = c[2] - bc[2] * dx + ncbc[2] * dy + nv[2] * dz
            pos[self.conn_ls[i + 1][3]] = np.array([new_x, new_y, new_z])
        return pos

    def test_complete(self):
        """
        test if missing torsion. e.g., Baker_30 5 will need this.
        Returns:

        """
        if len(self.mol) >= 4 and not (np.any(self.type_array == 2) or np.any(self.type_array == 4) ):
            return False
        else:
            return True

    def make_any_torsion(self) -> None:
        """
        Make torsion of i-j k-l . j k does not have to conn.
        Make torsion of i j-k-l . i j does not have to conn.
        This is for some molecule missing torsion e.g., Baker_30 05
        and molecule without fully connected

        Returns: None

        """
        for i, j in permutations(range(self.n_atom), 2):
            if self.virtual_connectivity[i, j]:
                for k in range(self.n_atom):
                    if not self.connectivity[k, i] and (k not in [i, j]):
                        for l in range(self.n_atom):
                            if self.virtual_connectivity[k, l] and (l not in [i, j, k]):
                                for i_, j_, k_ in permutations([i, j, k, l], 3):
                                    if self.mol.get_angle(i_, j_, k_) <= 5 or self.mol.get_angle(i_, j_, k_) >= 175:
                                        break
                                else:
                                    t = Torsion(i, j, k, l)
                                    for t_1, t_2, t_3, t_4 in permutations([i, j, k, l], 4):
                                        if Torsion(t_1, t_2, t_3, t_4) in self.internal_coords:
                                            break
                                    else:
                                        self.internal_coords.append(t)

        # for i, j in permutations(range(self.n_atom), 2):
        #     if True:
        #         for k in range(self.n_atom):
        #             if self.virtual_connectivity[j, k] and (k not in [i, j]):
        #                 for l in range(self.n_atom):
        #                     if self.virtual_connectivity[k, l] and (l not in [i, j, k]):
        #                         for i_, j_, k_ in permutations([i, j, k, l], 3):
        #                             if self.mol.get_angle(i_, j_, k_) <= 5 or self.mol.get_angle(i_, j_, k_) >= 175:
        #                                 break
        #                         else:
        #                             t = Torsion(i, j, k, l)
        #                             for t_1, t_2, t_3, t_4 in permutations([i, j, k, l], 4):
        #                                 if Torsion(t_1, t_2, t_3, t_4) in self.internal_coords:
        #                                     break
        #                             else:
        #                                 self.internal_coords.append(t)

    def make_virtual_bond(self) -> np.ndarray:
        """
        detect fragments and make virtual bond
        Args:
        Returns: connectivity matrix. -1 as virtual bond

        """
        conn = np.array(self.connectivity.copy().todense())
        # may use dist-radius for distance_matrix
        distance_matrix = self.mol.get_all_distances()
        conn = make_virtual_connectivity(conn, distance_matrix)
        conn = make_hydrogen_bond(conn,distance_matrix,self.mol)
        return conn

    @property
    def geom(self) -> np.ndarray:
        """
        cache geometry
        Returns: Cached Cartesian geometry

        """
        return self._geom

    @geom.setter
    def geom(self, geom: np.ndarray) -> None:
        """

        Args:
            geom: geometry in shape [3,n]

        Returns: None

        """
        self._geom = geom

    def get_internal_coords(self) -> List[InternalBase]:
        """

        Returns: list of internal coord. instances [Stretching1,Bending1,Torsion1...]

        """
        internal_coords = []
        linear_bendings = []
        # Maybe there is more elegant way?
        # Stretching
        for i, j in combinations(range(self.n_atom), 2):
            if self.virtual_connectivity[i, j]:
                s = Stretching(i, j)
                if s not in internal_coords:
                    internal_coords.append(s)
        # Bending
        for i, j in permutations(range(self.n_atom), 2):
            if self.virtual_connectivity[i, j] == 1:
                for k in range(i + 1, self.n_atom):
                    if self.virtual_connectivity[j, k] == 1:
                        if self.mol.get_angle(i, j, k) > 175 or self.mol.get_angle(i, j, k) <= 5:
                            b = Bending(i, j, k)
                            b_complement = Bending(i, j, k, bendType='COMPLEMENT')
                            if self.mol.get_angle(i, j, k) <= 5 or self.mol.get_angle(i, j, k) >= 175:
                                b = Bending(i, j, k, bendType='LINEAR')
                                linear_bendings.append((i, j, k))
                                linear_bendings.append((k, j, i))
                            if b_complement not in internal_coords:
                                internal_coords.append(b_complement)
                        else:
                            b = Bending(i, j, k)
                        if b not in internal_coords:
                            internal_coords.append(b)

        # Torsion
        torsion_added = False
        for i, j in permutations(range(self.n_atom), 2):
            if self.virtual_connectivity[i, j] == 1:
                for k in range(self.n_atom):
                    if self.virtual_connectivity[j, k] == 1 and not self.virtual_connectivity[k, i] == 1 and (
                            k not in [i, j]):
                        for l in range(self.n_atom):
                            if self.virtual_connectivity[k, l] == 1 and (l not in [i, j, k]):
                                for linear_bending in linear_bendings:
                                    if linear_bending == (i, j, k) or linear_bending == (j, k, l): break
                                else:
                                    t = Torsion(i, j, k, l)
                                    if t not in internal_coords:
                                        torsion_added = True
                                        internal_coords.append(t)

        # Out of Plane Torsion
        # if not torsion_added:
        #
        #     for i, j in permutations(range(self.n_atom), 2):
        #         if self.virtual_connectivity[i, j] == 1:
        #             for k in range(self.n_atom):
        #                 if self.virtual_connectivity[j, k] == 1 and not self.virtual_connectivity[k, i] == 1 and (
        #                         k not in [i, j]):
        #                     for l in range(self.n_atom):
        #                         if self.virtual_connectivity[j, l] and (l not in [i, j, k]):
        #                             for linear_bending in linear_bendings:
        #                                 if linear_bending == (l, j, k): break
        #                             else:
        #                                 t = OOFP(i, j, k, l)
        #                                 if t not in internal_coords:
        #                                     internal_coords.append(t)
        return internal_coords

    def get_q_values(self, geom: np.ndarray) -> np.ndarray:
        """

        Args:
            geom: geometry in shape [3,n]

        Returns: internal coordinates [q1,q2...q_r]
        """
        geom = geom.reshape(-1, 3)
        q = jnp.array([i.q(geom) for i in self.internal_coords]) * self.bohr_coff
        return q

    def get_dq(self, geom: np.ndarray, internal_coords: np.ndarray) -> np.ndarray:
        """

        Args:
            geom: geometry in shape [3,n]
            internal_coords: in shape [r] as [q_1,q_2...q_r]

        Returns: delta q array [dq_1,dq_2...dq_r]

        """
        diff = internal_coords - self.get_q_values(geom)
        diff = clean_pi(self.type_array, diff)
        # flip if over pi
        return diff

    def get_b_matrix_ad(self, geom):
        """
        Developing purpose only. use to verify analytical differentiation
        Switch to jax.numpy first
        Args:
            geom:

        Returns:

        """
        return self.b_matrix_func(geom)[0].reshape(-1, self.n_atom * 3) * self.bohr_coff # why this is tuple

    def get_b_matrix_analytic(self, geom):
        """

        Args:
            geom: geometry in shape [3,n]

        Returns: analytical B matrix in shape [r,3n]
                                      [[dq_11,dq_12...dq_1_3n]
                                      [dq_21,dq_22...dq_2_3n]
                                        ...............
                                      [dq_r1,dq_r2....dq_r_3n]]

        """
        if not np.array_equal(self.b_matrix_geom, geom) or self.b_matrix is None:
            b = []
            for int_coord in self.internal_coords:
                b.append(int_coord.DqDx(geom))

            self.b_matrix = np.array(b)
            self.b_matrix_geom = geom
        return (self.b_matrix.T*self.bohr_coff).T

    def f_xyz_to_q(self, f: np.ndarray, geom: np.ndarray) -> np.ndarray:
        """

        Args:
            f: force array [f_1...f_3n]
            geom: geometry in shape [3,n]

        Returns: f_q force in internal system. shape [r]  [q1...qr]

        """
        B = self.get_b_matrix_analytic(geom)
        temp_arr = np.dot(B, f.T)
        g_inv = self.get_g_inv(geom)
        g_intern = np.dot(g_inv, temp_arr)
        return g_intern

    def f_q_to_xyz(self, f: np.ndarray, geom: np.ndarray) -> np.ndarray:
        b = self.get_b_matrix_analytic(geom)
        return b.T @ f

    def get_g_inv(self, geom: np.ndarray, rcond: float = 1e-6):
        """
        calc G inversion if not buffered and return it.
        Args:
            geom: geometry in shape [3,n]
            rcond: ignored if eigvalue less than rcond

        Returns: G inversion matrix

        """
        if not np.array_equal(self.g_inv_geom, geom) or self.g_inv is None:
            b = self.get_b_matrix_analytic(geom)
            g = np.dot(b, b.T)
            try:
                self.g_inv = np.linalg.pinv(g, hermitian=True, rcond=rcond)
            except:
                raise  # debug flag
            self.g_inv_geom = geom
        return self.g_inv


    def get_g(self,geom: np.ndarray, rcond: float = 1e-6):
        b = self.get_b_matrix_analytic(geom)
        g = np.dot(b, b.T)
        return g

    def q_to_xyz(self, geom: Union[np.ndarray, None],
                 new_internal_coords: np.ndarray,
                 tol: float = 1e-10,
                 ) -> np.ndarray:
        """
        iterative conversion of internal to Cartesian
        Args:
            tol: end conversion if max(dq) < tol
            geom: old geometry in shape [3,n]
                    None will use cached geometry
            new_internal_coords: in shape [r] as [q1...qr]

        Returns: new geometry in shape [3,n]

        """
        logger.debug(f'q_to_xyz:{new_internal_coords}')
        _iter = 0
        internal_coords = new_internal_coords.copy()
        # internal_coords = clean_pi(self.type_array, internal_coords)
        max_dq = 10
        xyz = geom.copy() if geom is not None else self.last_geom.copy()
        best_dq = 0
        last_dq = 0
        best_xyz = xyz.copy()
        # self.fixBendAxes(xyz)
        # self.updateDihedralOrientations(xyz)
        # Three case we stop. 1. converged 2. out of iter cycle 3. slow converge
        while max_dq > tol and _iter < 50:
            _iter += 1
            dq = self.get_dq(xyz, internal_coords)
            max_dq = max(abs(dq))
            if _iter == 1 or max_dq < best_dq:
                best_dq = max_dq
                best_xyz = xyz.copy()
                t = 1
            else:
                t = 0.5
                xyz = best_xyz
            b = self.get_b_matrix_analytic(xyz)
            g_inv = self.get_g_inv(xyz)
            dx = np.dot(b.T, np.dot(g_inv, dq))
            xyz += dx.reshape(xyz.shape) * t
            if abs(max_dq-last_dq) < 1e-16:
                break
            last_dq = max_dq
            logger.debug(f'get_cart_from_internal RMS:{max_dq}')
        if max_dq >= tol:
            logger.info(f'get_cart_from_internal did not converge. max_dq :{best_dq}')
            return best_xyz
        self.last_geom = xyz
        return xyz

    def create_type_array(self) -> None:
        """
        create type array for internal coord. For the purpose of treating different coords. differently.
        Create up and low bounds as well.
        self.type_array: [type_1...type_r] as:
            Bending:1
            Torsion:2
            Stretching:0
        Returns: None

        """
        type_array = np.zeros(len(self.internal_coords))
        for i, internal in enumerate(self.internal_coords):
            if isinstance(internal, Bending):
                if internal.bendType == 'REGULAR':
                    type_array[i] = 1
                else:
                    type_array[i] = 3
            if isinstance(internal, Torsion):
                type_array[i] = 2
            if isinstance(internal, OOFP):
                type_array[i] = 4
        self.type_array = type_array
        self.bohr_coff = np.array([self.length_coeff if i == 0 else 1 for i in type_array])
        lb = np.array([0.0 if u<=1 else -np.inf for u in type_array])
        lb += np.array([0.6*self.length_coeff if u == 0 else 0 for u in type_array])
        up = np.array([np.pi if u == 1 else np.inf for u in type_array])
        self._bounds = Bounds(lb, up)

    @property
    def bounds(self):
        return self._bounds

    def updateDihedralOrientations(self, geom):
        for intco in self.internal_coords:
            if isinstance(intco, Torsion) or isinstance(intco, OOFP):
                intco.updateOrientation(geom)
        return

    def fixBendAxes(self, geom):
        for intco in self.internal_coords:
            if isinstance(intco, Bending):
                intco.fixBendAxes(geom)
        return

    def unfixBendAxes(self):
        for intco in self.internal_coords:
            if isinstance(intco, Bending):
                intco.unfixBendAxes()
        return


class Cartesian2DelocalizedInternal(NonCartesianSystem):
    """
    J. Chem. Phys. 105, 192 (1996)
    Delocalized internal system. [s_1,s_2...,s_M]
    M = 3N-6

    """

    def __init__(self, atoms: Atoms, length_coeff=1):
        self.redundant_internal = Cartesian2Internal(atoms, length_coeff=length_coeff)
        self.geom = self.init_geom = atoms.get_positions()
        self.u = self.get_u(self.geom)
        self.s = self.get_q_values(self.geom)
        self.last_geom = self.geom
        if not self.z_mat_method_enabled:
            logger.info("Cannot use Z-MAT quick back transformation. Switch to iterative method ")

    @property
    def dof(self):
        return len(self.s)

    def get_g(self,geom: np.ndarray, rcond: float = 1e-6) ->np.ndarray:
        raise NotImplementedError

    def get_g_inv(self, geom: np.ndarray, rcond: float = 1e-6) ->np.ndarray:
        raise NotImplementedError

    @property
    def z_mat_method_enabled(self):
        """
        if z_mat quick back transformation enabled
        Returns:

        """
        return self.redundant_internal.zmat_mapping is not None

    def get_u(self, geom) -> np.ndarray:
        """
        Calc U.T matrix. Use 1e-14 to filter numerical noise
        Note for convenience we return the transposed U.T instead of U
        Args:
            geom: geometry

        Returns: U matrix

        """
        b = self.redundant_internal.get_b_matrix_analytic(geom)
        g = np.dot(b, b.T)
        eig, vec = np.linalg.eigh(g)
        vec = vec.T
        u = vec[abs(eig) > 1e-6]
        return u

    def get_dq(self, geom, internal_coords: np.ndarray) -> np.ndarray:
        """
        get delta_s.
        Args:
            geom:
            internal_coords: reference internal coords. s0

        Returns:
        """
        diff = internal_coords - self.get_q_values(geom)
        return diff

    def get_q_values(self, geom, q=None) -> np.ndarray:
        """
        calc s vales with reference geometry or reference s values
        Note:
            This is to avoid changes over pi
        Args:
            geom: if q is not given, use geometry to calculated reference s values
            q: referenced s values s0

        Returns: delta_s = s - s0

        """
        if q is None:
            q = self.redundant_internal.get_q_values(geom)
        q_ref = self.redundant_internal.get_q_values(self.init_geom)
        diff = q - q_ref
        diff = clean_pi(self.redundant_internal.type_array, diff)
        q = q_ref + diff

        return self.u @ q

    def s_to_q(self, geom: np.ndarray, new_internal_coords: np.ndarray,
               tol: float = 1e-8, ) -> np.ndarray:
        """
        Quick conversion of s>q with Z-mat. J. Chem. Phys. 110, 4986 (1999)
        Args:
            geom:
            new_internal_coords:
            tol: tolerance of max(delta_s)

        Returns: redundant coord. values q
        """
        _iter = 0
        internal_coords = new_internal_coords.copy()
        q0 = self.redundant_internal.get_q_values(geom)
        s0 = self.get_q_values(geom, q0)
        t = 1
        best_q0 = q0
        best_max_ds = 99
        xyz_last_step = geom.copy()
        d_xyz = np.array([1])
        while True:
            _iter += 1
            ds = (internal_coords - s0)
            max_ds = max(abs(ds))
            dq = self.u.T @ ds.T

            if max_ds < best_max_ds and _iter!=1:
                best_q0 = q0
                t = 1

            else:
                t = t * 0.5
            best_max_ds = max_ds
            if max_ds < tol or d_xyz.max()<tol or _iter > 30: # TODO handle slow converge
                if max_ds > tol:
                    logger.info("use Zmat for s>xyz did not converge : max ds: " + str(best_max_ds))
                break
            q0 += dq * t
            xyz = self.redundant_internal.q_to_xyz_by_zmat(q0)
            d_xyz = xyz - xyz_last_step
            xyz_last_step = xyz.copy()
            s0 = self.get_q_values(xyz)
            # s0 = self.get_q_values(None,q0) # Note direct conversion from q0>s0 will not work, we need update r
            logger.debug(f'use Zmat for s>xyz max_ds : {max_ds}  max(d_xyz): {d_xyz.max()} step :{_iter}')
        return best_q0

    def q_to_xyz_safe(self, geom: np.ndarray, new_internal_coords: np.ndarray,
                      tol: float = 1e-8, ) -> np.ndarray:
        """
        Safe conversion of q>xyz. JPC 1999 method will not always work. e.g., Baker_30 id=2
        Args:
            geom:
            new_internal_coords:
            tol: tolerance of max(delta_s)

        Returns: xyz geometry in [N,3] ndarray

        """
        # can be improved
        _iter = 1
        internal_coords = new_internal_coords.copy()
        max_dq = 1
        xyz = geom.copy()
        dq_ls = [1000, 1000]
        t = 1
        xyz_ls = [xyz, xyz]
        dx = [1]
        while max_dq > tol and max(dx) > tol and _iter <40: # TODO handle slow converge
            _iter += 1
            ds = self.get_dq(xyz, internal_coords)
            b = self.redundant_internal.get_b_matrix_analytic(xyz)
            b_star = self.u @ b
            b_star_inv = np.linalg.pinv(b_star @ b_star.T, rcond=1e-6) @ b_star
            dx = b_star_inv.T @ ds
            if dq_ls[-1] >= dq_ls[-2]:
                t = t * 0.5
                dt = t
            else:
                t= 1
                dt = t
            dx = dx * dt
            xyz += dx.reshape(xyz.shape)
            max_dq = max(abs(ds))
            dq_ls.append(max_dq)
            xyz_ls.append(xyz.copy())
            logger.debug((f'get_cart_from_DLC RMS  max_dq :{max_dq}  max_dx {max(dx)}'))
        if max_dq >= tol:
            logger.info(f'x>s did not converge. max_ds :{min(dq_ls)}')
        return xyz_ls[dq_ls.index(min(dq_ls))]

    def q_to_xyz(self, geom: np.ndarray, new_internal_coords: np.ndarray,
                 tol: float = 1e-9, ) -> np.ndarray:
        """
        Conversion of s>xyz
        Args:
            geom:
            new_internal_coords:
            tol: tolerance of max(delta_s)

        Returns: xyz geometry in [N,3] ndarray

        """
        geom = geom if geom is not None else self.last_geom.copy()
        if self.z_mat_method_enabled:
            xyz = self.redundant_internal.q_to_xyz_by_zmat(self.s_to_q(geom, new_internal_coords, tol=tol))
        else:
            xyz = self.q_to_xyz_safe(geom, new_internal_coords, tol=tol)
        self.last_geom = xyz
        return xyz

    def f_xyz_to_q(self, f: np.ndarray, geom: np.ndarray) -> np.ndarray:
        """
        convert Cartesian force/gradient to DLC system.
        Args:
            f: force array [f_1...f_3N]
            geom: geometry for the conversion. DLC is not linear conversion so xyz geometry will be needed as well.

        Returns: f_s force in DLC system. shape [M]  [f_s1...f_sM]

        """
        # This is fater than calc (B.T@B)^-1@B@f in which B=U@B_prime
        try:
            f_q = self.redundant_internal.f_xyz_to_q(f, geom)
            fs = self.u @ f_q
        except:
            raise  # DEBUG raise
        return fs

    @property
    def bounds(self):
        """
        No bounds for DLC
        Returns:

        """
        # Any constrain seems not help at all.
        # q = self.get_q_values(self.init_geom)
        # lb = q-0.5
        # ub = q+0.5
        # bounds = Bounds(lb, ub)

        return None  # bounds

    def f_q_to_xyz(self, f: np.ndarray, geom: np.ndarray) -> np.ndarray:
        b = self.redundant_internal.get_b_matrix_analytic(geom)
        u = self.u
        return (u @ b).T @ f

    def fixBendAxes(self, geom):
        self.redundant_internal.fixBendAxes(geom)

    def updateDihedralOrientations(self, geom):
        self.redundant_internal.updateDihedralOrientations(geom)

    @property
    def internal_coords(self):
        return str(self.u) + "\n" + str(self.redundant_internal.internal_coords)

class Cartesian2DelocalizedInternalNoZmat(Cartesian2DelocalizedInternal):
    @property
    def z_mat_method_enabled(self):
        """
        if z_mat quick back transformation enabled
        Returns:

        """
        return False
def get_b_matrix_jitfunction(internal_coords):
    """
    debug function for AD
    Args:
        internal_coords:

    Returns:

    """
    from jax import jacfwd
    def get_allq(geom: np.ndarray):
        return jnp.array([intco.q(geom) for intco in internal_coords])

    return jacfwd(get_allq, argnums=[0])


class CannotBuildZmat(Exception):
    ...
