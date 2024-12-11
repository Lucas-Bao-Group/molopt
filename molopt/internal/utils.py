import ase
import scipy
from ase import neighborlist, Atoms
from ase.data import vdw_radii
import numpy as np
from math import sqrt
import numpy as jnp  # Switch to jax.numpy enable jax support.[AD debug purpose]
import time

TORS_ANGLE_LIM = 0.017
TORS_COS_TOL = 1e-10
DOT_PARALLEL_LIMIT = 1.e-10


def zeta(a, m, n):
    if a == m:
        return 1
    elif a == n:
        return -1
    else:
        return 0


def get_conn(mol: Atoms, cutoff=1, sparse=True):
    """
    get connectivity from molecule
    Args:
        sparse: return a sparse matrix or full matrix
        mol:
        cutoff:

    Returns:

    """
    cutOff = neighborlist.natural_cutoffs(mol, mult=cutoff)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True, skin=0.0)
    neighborList.update(mol)
    return neighborList.get_connectivity_matrix(sparse=sparse)


def angle(A, B, C, tol=1.0e-14):
    eBA = eAB(B, A)
    eBC = eAB(B, C)
    dotprod = jnp.dot(eBA, eBC)
    if dotprod > 1.0 - tol:
        phi = 0.0
    elif dotprod < -1.0 + tol:
        phi = jnp.arccos(-1.0)
    else:
        phi = jnp.arccos(dotprod)

    return phi


def eAB(p1, p2):
    eAB = p2 - p1
    eAB /= jnp.linalg.norm(eAB)
    return eAB


def Bmat(intcos, mol, masses=None):
    Nint = len(intcos)
    Ncart = len(mol) * 3

    B = np.zeros((Nint, Ncart), float)
    for i, intco in enumerate(intcos):
        intco.DqDx(mol, B[i])

    if masses is not None:
        print("mass weighting B matrix\n")
        for i in range(len(intcos)):
            for a in range(len(mol)):
                for xyz in range(3):
                    B[i, 3 * a + xyz] /= sqrt(masses[a])

    return B


def tors(A, B, C, D):
    phi_lim = TORS_ANGLE_LIM
    tors_cos_tol = TORS_COS_TOL

    # Form e vectors
    EAB = eAB(A, B)
    EBC = eAB(B, C)
    ECD = eAB(C, D)

    # Compute bond angles
    phi_123 = angle(A, B, C)
    phi_234 = angle(B, C, D)

    tmp = jnp.cross(EAB, EBC)
    tmp2 = jnp.cross(EBC, ECD)
    tval = jnp.dot(tmp, tmp2) / (jnp.sin(phi_123) * jnp.sin(phi_234))

    if tval >= 1.0 - tors_cos_tol:  # accounts for numerical leaking out of range
        tau = 0.0
    elif tval <= -1.0 + tors_cos_tol:
        tau = jnp.arccos(-1)
    else:
        tau = jnp.arccos(tval)

    # determine sign of torsion ; this convention matches Wilson, Decius and Cross
    if tau != jnp.arccos(-1):  # no torsion will get value of -pi; Range is (-pi,pi].
        tmp = jnp.cross(EBC, ECD)
        tval = jnp.dot(EAB, tmp)
        if tval < 0:
            tau *= -1

    return tau


def oofp(A, B, C, D):
    eBA = eAB(B, A)
    eBC = eAB(B, C)
    eBD = eAB(B, D)

    phi_CBD = angle(C, B, D)

    # This shouldn't happen unless angle B-C-D -> 0,
    if jnp.sin(phi_CBD) < TORS_COS_TOL:  # reusing parameter
        return False, 0.0

    dotprod = jnp.dot(jnp.cross(eBC, eBD), eBA) / jnp.sin(phi_CBD)

    if dotprod > 1.0:
        tau = jnp.acos(-1)
    elif dotprod < -1.0:
        tau = -1 * jnp.acos(-1)
    else:
        tau = jnp.arcsin(dotprod)
    return True, tau


# Are two vectors parallel?
def are_parallel(u, v):
    if abs(jnp.dot(u, v) - 1.0e0) < DOT_PARALLEL_LIMIT:
        return True
    else:
        return False


# Are two vectors parallel?
def are_antiparallel(u, v):
    if abs(jnp.dot(u, v) + 1.0e0) < DOT_PARALLEL_LIMIT:
        return True
    else:
        return False


def are_parallel_or_antiparallel(u, v):
    return are_parallel(u, v) or are_antiparallel(u, v)


def make_virtual_connectivity(conn, distance_matrix):
    # can be improved
    n = conn.shape[0]
    n_part, arr = scipy.sparse.csgraph.connected_components(conn)
    while n_part > 1:
        mask = np.full((n, n), True)
        a = np.where(arr == 0)
        mask[a, :] = False
        mask[:, a] = True
        dist = np.ma.array(distance_matrix, mask=mask)
        index_ = np.argmin(dist)
        i, j = index_ // n, index_ % n
        index_2 = dist < dist[i, j] + 2
        conn[index_2 == True] = -1
        conn.T[index_2 == True] = -1
        conn[i, j] = 1
        conn[j, i] = 1
        n_part, arr = scipy.sparse.csgraph.connected_components(conn)
    return conn


def make_hydrogen_bond(conn, distance_matrix, mol: ase.Atoms):
    conn_temp = conn.copy()
    n = len(mol)
    Y_ls = [7, 8, 9, 15, 16, 17]
    atomic_number_array = np.array(mol.get_atomic_numbers())
    for i in range(n):
        if atomic_number_array[i] in Y_ls:
            for j in range(n):
                if atomic_number_array[j] == 1 and conn[i, j] == 1:
                    for k in range(n):
                        if k == i:
                            continue
                        elif if_less_than_vdw_radii(distance_matrix[j, k], atomic_number_array[j],
                                                    atomic_number_array[k]):
                            try:
                                angle = mol.get_angle(i, j, k)
                            except ZeroDivisionError:
                                angle = 0
                            if angle > 90:
                                conn_temp[j, k] = 1
                                conn_temp[k, j] = 1
    return conn_temp


def if_less_than_vdw_radii(distance, atom1, atom2, factor=0.9):
    r = True if distance < (vdw_radii[atom1] + vdw_radii[atom2]) * factor else False
    return r


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r  %2.4f sec' % \
              (f.__name__, te - ts))
        return result

    return timed
