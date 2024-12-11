from abc import ABC, abstractmethod
from math import sqrt

import numpy as jnp  # Switch to jax.numpy enable jax support.[debug purpose]
import numpy as np

from molopt.internal.utils import eAB, angle, tors, zeta, are_parallel_or_antiparallel, oofp

"""
This is file contains 3 internal coordinates type classes: Stretch,Bending,Torsion
"""

FIX_VAL_NEAR_PI = 1.57


class InternalBase(ABC):

    def __init__(self, atoms, frozen=False, fixedEqVal=None):
        # these lines use the property's and setters below
        self.atoms = atoms  # atom indices for internal definition
        self.frozen = frozen  # bool - is internal coordinate frozen? NOT IMPLEMENTED YET
        self.fixedEqVal = fixedEqVal  # target value if artificial forces are to be added NOT IMPLEMENTED YET

    def __repr__(self):
        return str(self)

    @abstractmethod
    def DqDx(self, geom):
        """
        first order derivative DqDx
        Args:
            geom: np.array. [3N] shape in CARTESIAN

        Returns:

        """
        ...

    @abstractmethod
    def q(self, geom: jnp.array):
        """
        get the value of this internal coordinate
        Args:
            geom: jnp.array. [3N] shape in CARTESIAN

        Returns:

        """
        raise NotImplementedError

    @property
    def A(self):
        try:
            return self.atoms[0]
        except:
            raise RuntimeError("A() called but atoms[0] does not exist")

    @property
    def B(self):
        try:
            return self.atoms[1]
        except:
            raise RuntimeError("B() called but atoms[1] does not exist")

    @property
    def C(self):
        try:
            return self.atoms[2]
        except:
            raise RuntimeError("C() called but atoms[2] does not exist")

    @property
    def D(self):
        try:
            return self.atoms[3]
        except:
            raise RuntimeError("D() called but atoms[3] does not exist")

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, type(self)):
            return False
        else:
            return True


class Stretching(InternalBase):
    """
    Stretching
    """

    def __init__(self, a, b, frozen=False, fixedEqVal=None, inverse=False):

        self.inverse = inverse  # bool - is really 1/R coordinate?

        if a < b:
            atoms = (a, b)
        else:
            atoms = (b, a)

        InternalBase.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen:
            s = '*'
        else:
            s = ' '

        if self.inverse:
            s += '1/R'
        else:
            s += 'R'

        s += "(%d,%d)" % (self.A + 1, self.B + 1)
        if self.fixedEqVal:
            s += "[%.4f]" % (self.fixedEqVal * self.qShowFactor)
        return s

    def q(self, geom):
        return jnp.linalg.norm(geom[self.A] - geom[self.B])

    def DqDx(self, geom):

        EAB = eAB(geom[self.A], geom[self.B])  # A->B

        dqdx = np.zeros(len(geom.reshape(-1)))
        dqdx[self.A * 3:self.A * 3 + 3] = -1 * EAB[0:3]
        dqdx[self.B * 3:self.B * 3 + 3] = EAB[0:3]

        #
        # if self.inverse:
        #     val = self.q(mol)
        #     dqdx[startA:startA + 3] *= -1.0 * val * val  # -(1/R)^2 * (dR/da)
        #     dqdx[startB:startB + 3] *= -1.0 * val * val

        return dqdx


class Bending(InternalBase):
    """
    Bending
    """

    def __init__(self, a, b, c, frozen=False, fixedEqVal=None, bendType="REGULAR"):

        if a < c:
            atoms = (a, b, c)
        else:
            atoms = (c, b, a)

        self.bendType = bendType
        self._axes_fixed = False
        self._x = jnp.zeros(3, float)
        self._w = jnp.zeros(3, float)

        InternalBase.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen:
            s = '*'
        else:
            s = ' '

        if self.bendType == "REGULAR":
            s += "B"
        elif self.bendType == "LINEAR":
            s += "L"
        elif self.bendType == "COMPLEMENT":
            s += "l"

        s += "(%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1)

        return s

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, Bending):
            return False
        elif self.bendType != other.bendType:
            return False
        else:
            return True

    @property
    def bendType(self):
        return self._bendType

    @bendType.setter
    def bendType(self, intype):
        if intype in "REGULAR" "LINEAR" "COMPLEMENT":
            self._bendType = intype
        else:
            raise RuntimeError("BEND.bendType must be REGULAR, LINEAR, or COMPLEMENT")

    def q(self, geom):

        if not self._axes_fixed:
            self.compute_axes(geom)

        u = eAB(geom[self.B], geom[self.A])  # B->A
        v = eAB(geom[self.B], geom[self.C])  # B->C

        # linear bend is sum of 2 angles, u.x + v.x
        origin = jnp.zeros(3)
        phi = angle(u, origin, self._x)
        phi2 = angle(self._x, origin, v)
        phi += phi2
        # phi = jnp.clip(phi,-np.pi,np.pi)
        return phi

    def compute_axes(self, geom):
        u = eAB(geom[self.B], geom[self.A])  # B->A
        v = eAB(geom[self.B], geom[self.C])  # B->C

        if self.bendType == "REGULAR":  # not a linear-bend type
            self._w = jnp.cross(u, v)  # orthogonal vector
            self._w /= jnp.linalg.norm(self._w)
            self._x = u + v  # angle bisector
            self._x /= jnp.linalg.norm(self._x)
            return
        tv1 = np.array([1, 0, 0], float)  # hope not to create 2 bends that both break
        tv2 = np.array([0, 1, 1], float)  # a symmetry plane, so 2nd is off-axis
        tv2 /= jnp.linalg.norm(tv2)

        # handle both types of linear bends
        if not are_parallel_or_antiparallel(u, v):
            self._w = jnp.cross(u, v)  # orthogonal vector
            self._w /= jnp.linalg.norm(self._w)
            self._x = u + v  # angle bisector
            self._x /= jnp.linalg.norm(self._x)

        # u || v but not || to tv1.
        elif not are_parallel_or_antiparallel(u, tv1) \
                and not are_parallel_or_antiparallel(v, tv1):
            self._w = jnp.cross(u, tv1)
            self._w /= jnp.linalg.norm(self._w)
            self._x = jnp.cross(self._w, u)
            self._x /= jnp.linalg.norm(self._x)

        # u || v but not || to tv2.
        elif not are_parallel_or_antiparallel(u, tv2) \
                and not are_parallel_or_antiparallel(v, tv2):
            self._w = jnp.cross(u, tv2)
            self._w /= jnp.linalg.norm(self._w)
            self._x = jnp.cross(self._w, u)
            self._x /= jnp.linalg.norm(self._x)

        if self._bendType == "COMPLEMENT":
            w2 = np.copy(self._w)  # x_normal -> w_complement
            self._w = -1.0 * self._x  # -w_normal -> x_complement
            self._x = w2
        return

    def DqDx(self, geom):
        if not self._axes_fixed:
            self.compute_axes(geom)
        u = geom[self.A] - geom[self.B]  # B->A
        v = geom[self.C] - geom[self.B]  # B->C
        Lu = np.linalg.norm(u)  # RBA
        Lv = np.linalg.norm(v)  # RBC
        u *= 1.0 / Lu  # u = eBA
        v *= 1.0 / Lv  # v = eBC

        uXw = np.cross(u, self._w)
        wXv = np.cross(self._w, v)
        dqdx = np.zeros(len(geom.reshape(-1)))
        # B = overall index of atom; a = 0,1,2 relative index for delta's
        # TODO performance
        for a, B in enumerate(self.atoms):
            dqdx[3 * B: 3 * B + 3] = zeta(a, 0, 1) * uXw[0:3] / Lu + \
                                     zeta(a, 2, 1) * wXv[0:3] / Lv
        return dqdx

    def fixBendAxes(self, geom):
        if self.bendType == 'LINEAR' or self.bendType == 'COMPLEMENT':
            self.compute_axes(geom)
            self._axes_fixed = True

    def unfixBendAxes(self):
        self._axes_fixed = False


class Torsion(InternalBase):
    """
    Torsion
    """

    def __init__(self, a, b, c, d, frozen=False, fixedEqVal=None):

        if a < d:
            atoms = (a, b, c, d)
        else:
            atoms = (d, c, b, a)
        self._near180 = 0

        InternalBase.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen:
            s = '*'
        else:
            s = ' '

        s += "D"

        s += "(%d,%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1, self.D + 1)
        return s

    @property
    def near180(self):
        return self._near180

    # keeps track of orientation
    def updateOrientation(self, geom):
        tval = self.q(geom)
        if tval > FIX_VAL_NEAR_PI:
            self._near180 = +1
        elif tval < -1 * FIX_VAL_NEAR_PI:
            self._near180 = -1
        else:
            self._near180 = 0
        return

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, Torsion):
            return False
        else:
            return True

    def q(self, geom):
        tau = tors(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        # Extend values domain of torsion angles beyond pi or -pi, so that
        # delta(values) can be calculated
        if self._near180 == -1 and tau > FIX_VAL_NEAR_PI:
            return tau - 2.0 * np.pi
        elif self._near180 == +1 and tau < -1 * FIX_VAL_NEAR_PI:
            return tau + 2.0 * np.pi
        else:
            return tau

    def DqDx(self, geom):
        # There are several errors in JCP, 22, 9164, (2002) in Dq2Dx2
        # I identified incorrect signs by making the equations invariant to reversing the atom indices
        # (0,1,2,3) -> (3,2,1,0) and checking terms against finite differences.  Also, the last terms
        # with sin^2 in the denominator are incorrectly given as only sin^1 in the paper.
        # Torsion is m-o-p-n.  -RAK 2010

        dqdx = np.zeros(len(geom.reshape(-1)))
        u = geom[self.A] - geom[self.B]  # u=m-o eBA
        v = geom[self.D] - geom[self.C]  # v=n-p eCD
        w = geom[self.C] - geom[self.B]  # w=p-o eBC
        Lu = np.linalg.norm(u)  # RBA
        Lv = np.linalg.norm(v)  # RCD
        Lw = np.linalg.norm(w)  # RBC
        u *= 1.0 / Lu  # eBA
        v *= 1.0 / Lv  # eCD
        w *= 1.0 / Lw  # eBC

        cos_u = np.dot(u, w)
        cos_v = -np.dot(v, w)

        # abort and leave zero if 0 or 180 angle
        if 1.0 - cos_u * cos_u <= 1.0e-12 or 1.0 - cos_v * cos_v <= 1.0e-12:
            return dqdx

        sin_u = sqrt(1.0 - cos_u * cos_u)
        sin_v = sqrt(1.0 - cos_v * cos_v)
        uXw = np.cross(u, w)
        vXw = np.cross(v, w)

        # a = relative index; B = full index of atom
        for a, B in enumerate(self.atoms):
            for i in range(3):  # i=a_xyz
                tval = 0.0

                if a == 0 or a == 1:
                    tval += zeta(a, 0, 1) * uXw[i] / (Lu * sin_u * sin_u)

                if a == 2 or a == 3:
                    tval += zeta(a, 2, 3) * vXw[i] / (Lv * sin_v * sin_v)

                if a == 1 or a == 2:
                    tval += zeta(a, 1, 2) * uXw[i] * cos_u / (Lw * sin_u * sin_u)

                # "+" sign for zeta(a,2,1)) differs from JCP, 117, 9164 (2002)
                if a == 1 or a == 2:
                    tval += -zeta(a, 2, 1) * vXw[i] * cos_v / (Lw * sin_v * sin_v)

                dqdx[3 * B + i] = tval

        return dqdx


class OOFP(InternalBase):
    def __init__(self, a, b, c, d, frozen=False, fixedEqVal=None):

        if c < d:
            atoms = (a, b, c, d)
        else:
            atoms = (a, b, d, c)
        self._near180 = 0
        InternalBase.__init__(self, atoms, frozen, fixedEqVal)

    def __str__(self):
        if self.frozen:
            s = '*'
        else:
            s = ' '

        s += "O"

        s += "(%d,%d,%d,%d)" % (self.A + 1, self.B + 1, self.C + 1, self.D + 1)
        if self.fixedEqVal:
            s += "[%.4f]" % self.fixedEqVal
        return s

    def __eq__(self, other):
        if self.atoms != other.atoms:
            return False
        elif not isinstance(other, OOFP):
            return False
        else:
            return True

    @property
    def near180(self):
        return self._near180

    def updateOrientation(self, geom):
        tval = self.q(geom)
        if tval > FIX_VAL_NEAR_PI:
            self._near180 = +1
        elif tval < -1 * FIX_VAL_NEAR_PI:
            self._near180 = -1
        else:
            self._near180 = 0
        return

    @property
    def qShowFactor(self):
        return 180.0 / np.pi

    def qShow(self, geom):  # return in degrees
        return self.q(geom) * self.qShowFactor

    # compute angle and return value in radians
    def q(self, geom):
        check, tau = oofp(geom[self.A], geom[self.B], geom[self.C], geom[self.D])
        if not check:
            raise RuntimeError("OOFP::compute.q: unable to compute out-of-plane value")

        # Extend domain of out-of-plane angles to beyond pi
        if self._near180 == -1 and tau > FIX_VAL_NEAR_PI:
            return tau - 2.0 * np.pi
        elif self._near180 == +1 and tau < -1 * FIX_VAL_NEAR_PI:
            return tau + 2.0 * np.pi
        else:
            return tau

    # out-of-plane is m-o-p-n
    # Assume angle phi_CBD is OK, or we couldn't calculate the value anyway.
    def DqDx(self, geom):
        dqdx = np.zeros(len(geom.reshape(-1)))
        eBA = geom[self.A] - geom[self.B]
        eBC = geom[self.C] - geom[self.B]
        eBD = geom[self.D] - geom[self.B]
        rBA = np.linalg.norm(eBA)
        rBC = np.linalg.norm(eBC)
        rBD = np.linalg.norm(eBD)
        eBA *= 1.0 / rBA
        eBC *= 1.0 / rBC
        eBD *= 1.0 / rBD
        # compute out-of-plane value, C-B-D angle
        val = self.q(geom)
        phi_CBD = angle(geom[self.C], geom[self.B], geom[self.D])

        # S vector for A
        tmp = jnp.cross(eBC, eBD)
        tmp /= jnp.cos(val) * jnp.sin(phi_CBD)
        tmp2 = jnp.tan(val) * eBA
        dqdx[3 * self.A:3 * self.A + 3] = (tmp - tmp2) / rBA

        # S vector for C
        tmp = jnp.cross(eBD, eBA)
        tmp = tmp / (jnp.cos(val) * jnp.sin(phi_CBD))
        tmp2 = jnp.cos(phi_CBD) * eBD
        tmp3 = -1.0 * tmp2 + eBC
        tmp3 *= jnp.tan(val) / (jnp.sin(phi_CBD) * jnp.sin(phi_CBD))
        dqdx[3 * self.C:3 * self.C + 3] = (tmp - tmp3) / rBC

        # S vector for D
        tmp = jnp.cross(eBA, eBC)
        tmp /= jnp.cos(val) * jnp.sin(phi_CBD)
        tmp2 = jnp.cos(phi_CBD) * eBC
        tmp3 = -1.0 * tmp2 + eBD
        tmp3 *= jnp.tan(val) / (jnp.sin(phi_CBD) * jnp.sin(phi_CBD))
        dqdx[3 * self.D:3 * self.D + 3] = (tmp - tmp3) / rBD

        # S vector for B
        dqdx[3 * self.B:3 * self.B + 3] = -1.0 * dqdx[3 * self.A:3 * self.A + 3] \
                                          - dqdx[3 * self.C:3 * self.C + 3] - dqdx[3 * self.D:3 * self.D + 3]
        return dqdx
