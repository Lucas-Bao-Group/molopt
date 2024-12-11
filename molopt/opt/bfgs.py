from abc import ABC
from typing import Type

import numpy as np
from ase.optimize import BFGS
from numpy.linalg import eigh
from molopt.internal.converter import NonCartesianSystem, Cartesian2Internal


class BFGSInternal(BFGS,ABC):
    internal_adapter_cls: Type[NonCartesianSystem]
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None):
        self.internal_adapter = self.internal_adapter_cls(atoms)
        super().__init__(atoms=atoms, restart=restart, logfile=logfile,
                         trajectory=trajectory,maxstep=maxstep, master=master,alpha=alpha)


    def initialize(self):
        # initial hessian
        self.H0 = np.eye(self.internal_adapter.dof) * self.alpha
        # r = self.atoms.get_positions()
        # p = self.internal_adapter.get_g(r) @ self.internal_adapter.get_g_inv(r)
        # self.H0 = p @ self.H0 @ p + 1000 * (np.eye(self.internal_adapter.dof) - p)
        self.H = None
        self.r0 = None
        self.f0 = None
        self.q0 = None
        self.fq0  = None

    def update(self, r, f, r0, f0):
        if self.H is None:
            self.H = self.H0
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b


    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        q = self.internal_adapter.get_q_values(r)
        f = f.reshape(-1)
        f_q = self.internal_adapter.f_xyz_to_q(f, r)
        # p = self.internal_adapter.get_g(r) @ self.internal_adapter.get_g_inv(r)
        # f_q_hat = p@f_q
        self.update(q, f_q, self.q0, self.fq0)
        # H_hat = p@self.H@p
        # self.H = p@self.H@p + 1000 * (np.eye(self.internal_adapter.dof)-p)
        omega, V = eigh(self.H)
        dq = np.dot(V, np.dot(f_q, V) / np.fabs(omega))
        steplengths = (dq**2).sum()**0.5
        dq = self.determine_step(dq, steplengths)
        q0 = q+dq
        new_r = self.internal_adapter.q_to_xyz(self.r0,q0)
        atoms.set_positions(new_r)
        self.r0 = r.copy()
        self.q0 = q.copy()
        self.f0 = f.copy()
        self.fq0 = f_q.copy()

        self.dump((self.H, self.r0, self.f0, self.maxstep))


class BFGS_R(BFGSInternal):
    internal_adapter_cls = Cartesian2Internal