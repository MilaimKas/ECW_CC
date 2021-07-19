#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################################################
#
# ECW-CCS v1
# -----------
# Experimentaly constrained wave function coupled cluster single
# ---------------------------------------------------------------
#
# File containing all CC related functions
# - L1 and T1 canonical and factorized equations
# - L1 and T1 intermediates 
# - EXC-CCS Hessian
#
#
###################################################################
import copy
import sys

import numpy as np
# from . import utilities
# import pyscf.cc.eom_gccsd
import utilities

############################
# reduced density matrix
############################

def gamma_unsym_CCS(ts, ls):
    '''
    Unsymmetrized one-particle reduced density matrix CCS
    - Stanton 1993 with ria = 0 and  r0=1
    - Stasis: same results except l0 term

    :param ts: t1 amplitudes
    :param ls: l1 amplitudes
    :return:
    '''

    nocc, nvir = ts.shape

    doo = -np.einsum('ie,je->ij', ts, ls)
    dvv = np.einsum('mb,ma->ab', ts, ls)
    dov = ls.copy()
    dvo = -np.einsum('ie,ma,me->ai', ts, ts, ls) + ts.transpose()

    dm1 = np.empty((nocc + nvir, nocc + nvir))
    dm1[:nocc, :nocc] = doo
    dm1[:nocc, nocc:] = dov
    dm1[nocc:, :nocc] = dvo
    dm1[nocc:, nocc:] = dvv

    dm1[np.diag_indices(nocc)] += 1

    return dm1


def gamma_es_CCS(ts, ln, rn, r0n, l0n):
    """
    Unsymmetrized CCS one-particle reduced density matrix for a excited states n
    Psi_n must be normalized: sum(ln*rn)+(l0*r0) = 1

    :param ts: t1 amplitudes
    :param ln: l1 amplitudes
    :param rn: r1 amplitudes
    :param r0n: r0 amplitude
    :return:
    """

    nocc, nvir = ts.shape

    # GS case:
    if rn is None:
        rn = np.zeros_like(ts)
        r0n = 1.
    if ln is None:
        ln = np.zeros_like(ts)
        l0n = 1.

    # gamma_ij
    doo = -r0n * np.einsum('ie,je->ij', ts, ln)
    doo -= np.einsum('ie,je->ij', rn, ln)

    # gamma_ia
    dov = r0n * ln

    # gamma_ab
    dvv = r0n * np.einsum('mb,ma->ab', ts, ln)
    dvv += np.einsum('mb,ma->ab', rn, ln)

    # gamma_ai
    dvo = r0n * np.einsum('ie,ma,me->ai', ts, ts, ln)
    dvo -= np.einsum('ma,ie,me->ai', ts, rn, ln)
    dvo -= np.einsum('ie,ma,me->ai', ts, rn, ln)
    dvo += np.einsum('ia->ai', ts)
    dvo += l0n * rn.transpose()

    dm1 = np.empty((nocc + nvir, nocc + nvir))
    dm1[:nocc, :nocc] = doo
    dm1[:nocc, nocc:] = dov
    dm1[nocc:, :nocc] = dvo
    dm1[nocc:, nocc:] = dvv

    # G format
    dm1[np.diag_indices(nocc)] += 1

    return dm1


def gamma_tr_CCS(ts, ln, rk, r0k, l0n):
    """
    CCS one-particle reduced transition density matrix between state n and k

    <Psi_n|apaq|Psi_k>
    if Psi_k = Psi_GS then r0=1 and rk=0
    if Psi_n = Psi_GS then l0=1 and lk=lam_k
    ln,l0 and rk,r0k must be orthogonal: sum(ln*rk)+(r0*l0) = 0

    :param ts: t1 amplitude
    :param ln: l1 amplitudes for state n
    :param rk: r1 amplitudes for state k
    :param r0k: r0 amplitude for state k

    :return: tr_rdm1 in G format
    """

    nocc, nvir = ts.shape

    # if Psi_k = GS
    if isinstance(rk, float) or isinstance(rk, int) or rk is None:
        if rk != 0 and rk is not None:
            print('Warning: wrong input for the r vector, assuming r=0')
        rk = np.zeros_like(ln)
        r0k = 1.

    # if Psi_n = GS
    if l0n == 0 or l0n is None:
        l0n = 1.

    doo = -r0k * np.einsum('ie,je->ij', ts, ln)
    doo -= np.einsum('ie,je->ij', rk, ln)

    dov = r0k * ln

    dvv = r0k * np.einsum('mb,ma->ab', ts, ln)
    dvv += np.einsum('mb,ma->ab', rk, ln)

    dvo = -r0k * np.einsum('ie,ma,me->ai', ts, ts, ln)
    dvo -= np.einsum('ma,ie,me->ai', ts, rk, ln)
    dvo -= np.einsum('ie,ma,me->ai', ts, rk, ln)
    dvo += l0n * r0k * ts.transpose()
    dvo += np.einsum('jb,ia,jb->ai', ln, ts, rk)
    dvo += l0n * rk.transpose()  # not present in Stanton because l0n = 0

    dm = np.empty((nocc + nvir, nocc + nvir))
    dm[:nocc, :nocc] = doo
    dm[:nocc, nocc:] = dov
    dm[nocc:, :nocc] = dvo
    dm[nocc:, nocc:] = dvv

    return dm

def gamma_CCS(ts, ls):
    '''
    Symmetrized one-particle reduced density matrix CCS

    :param ts: t1 amplitudes
    :param ls: l1 amplitudes from the GS
    :return: symmetrized rdm1
    '''

    # relevant gccsd_rdm1 PySCF functions
    # _gamma1_intermediates(mycc, t1, t2, l1, l2)
    # make_rdm1(mycc, t1, t2, l1, l2, ao_repr=False)
    # _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False)
    # Checked

    # calculate occ-occ, vir-vir and occ-vir blocs
    nocc, nvir = ts.shape
    doo = -np.einsum('ja,ia->ij', ts, ls)
    dvv = np.einsum('ia,ib->ab', ts, ls)
    xtv = np.einsum('ie,me->im', ts, ls)
    dvo = ts.T - np.einsum('im,ma->ai', xtv, ts)
    dov = ls

    # build density matrix from the blocs
    dm1 = np.empty((nocc + nvir, nocc + nvir), dtype=doo.dtype)
    dm1[:nocc, :nocc] = doo + doo.conj().T
    dm1[:nocc, nocc:] = dov + dvo.conj().T
    dm1[nocc:, :nocc] = dm1[:nocc, nocc:].conj().T
    dm1[nocc:, nocc:] = dvv + dvv.conj().T
    dm1 *= .5

    dm1[np.diag_indices(nocc)] += 1

    return dm1

############################
# CLASS: Generalized CCS
############################

class Gccs:
    def __init__(self, eris, fock=None, M_tot=None):
        '''
        All equations are given in spin-orbital basis

        :param eris: two electron integrals in Physics notation (<pq||rs>= <pq|rs> - <pq|sr>)
        :param fock: fock matrix in MOs basis
        :param M_tot: number of measurements
        '''
        if M_tot is None:
            self.M_tot = 1
        else:
            self.M_tot = M_tot

        if fock is None:
            self.fock = np.asarray(eris.fock)
        else:
            self.fock = fock

        self.eris = eris

        self.nocc = eris.nocc  
        self.nvir = self.fock.shape[0]-self.nocc 

    # -----------------------------------------
    # Energy
    # -----------------------------------------

    def energy_ccs(self, ts, fsp, rsn=None, r0n=None, vn=None):
        """
        E'_{0}
        :param rsn: list of rs amplitude for excited states n
        :param vn: list of exp potential V{n0}
        """

        if fsp is None:
            fsp = self.fock.copy()

        nocc, nvir = ts.shape
        e = np.einsum('ia,ia', fsp[:nocc, nocc:], ts)
        e += 0.5 * np.einsum('ia,jb,ijab', ts, ts, self.eris.oovv)

        # add contribution to excited states
        if rsn is not None:
            for rs, v, r0 in zip(rsn, vn, r0n):
                if v.any():
                    v_ov = v[:nocc, nocc:]
                    e += np.einsum('ia,ia', v_ov, rs)
                    e += r0*np.einsum('ia,ia', v_ov, ts)
                    e += r0*np.einsum('jj', v[:nocc, :nocc])

        return e

    # -------------------------------------------------------------------
    # RDM1
    # -------------------------------------------------------------------

    def gamma(self, ts, ls):
        return gamma_CCS(ts, ls)

    def gamma_unsym(self, ts, ls):
        return gamma_unsym_CCS(ts, ls)
    
    def gamma_es(self, ts, ln, rn, r0n, l0n):
        return gamma_es_CCS(ts, ln, rn, r0n, l0n)

    def gamma_tr(self, ts, ln, rk, r0k, l0n):
        return gamma_tr_CCS(ts, ln, rk, r0k, l0n)

    # -----------------------------------------------------------------------------
    # T1 equation
    # -----------------------------------------------------------------------------

    def T1eq(self, ts, fsp):
        """
        T1 equations using intermediates

        :param ts: t1 amplitudes
        :param fsp: fock matrix
        :return: T1 nocc x nvir matrix
        """
        
        Fae, Fmi, Fai = self.T1inter(ts, fsp)

        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ie,ae->ia', ts, Fae)
        T1 -= np.einsum('ma,mi->ia', ts, Fmi)

        return T1

    def tsupdate(self, ts, T1inter, rsn=None, r0n=None, vn=None):
        """
        SCF update of the t1 amplitudes with additional Vexp coupling terms

        :param ts: t1 amplitudes
        :param T1inter: T1 intermediates
        :param rsn: ria and r0 amplitudes for the states n, list of ((ria_1,r0_1),(ria_2,r0_2), ...)
                    where ria is a occ x nocc matrix and r0 a float
        :param vn: "right" Vexp[0,n] exp potentials
        :return: updated ts amplitudes
        """

        Fae, Fmi, Fai = T1inter
        nocc, nvir = ts.shape

        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # remove diagonal of the fock matrix
        Fae[np.diag_indices(nvir)] -= diag_vv
        Fmi[np.diag_indices(nocc)] -= diag_oo

        # update ts
        tsnew = np.einsum('ai->ia', Fai)
        tsnew += np.einsum('ie,ae->ia', ts, Fae)
        tsnew -= np.einsum('ma,mi->ia', ts, Fmi)
        
        # add coupling terms with excited states
        # assuming that the additional terms are small
        if rsn is not None:
            if r0n is None:
                raise ValueError('if Vexp are to be calculated, list of r0 amp must be given')
            if len(vn) != len(rsn):
                raise ValueError('Number of experimental potentials must be equal to number of r amplitudes')
            for r, v, r0 in zip(rsn, vn, r0n):
                if v.any():

                    v = np.asarray(v)
                    v_oo = v[:nocc, :nocc]
                    v_vv = v[nocc:, nocc:]
                    v_ov = v[:nocc, nocc:]

                    # Z intermediates
                    Z = np.trace(v_oo)
                    Z += np.einsum('jb,jb', v_ov, ts)

                    Z0 = v_ov.copy()
                    Z0 += np.einsum('ib,ab->ia', ts, v_vv)
                    Z0 -= np.einsum('ja,ji->ia', ts, v_oo)
                    tmp = np.einsum('ja,jb->ab', ts, v_ov)
                    Z0 -= np.einsum('ab,ib->ia', tmp, ts)

                    Zab = v_vv.copy()
                    Zab -= np.einsum('ja,jb->ab', ts, v_ov)

                    Zji = -v_oo.copy()
                    Zji -= np.einsum('ib,jb->ji', ts, v_ov)

                    tsnew += r*Z
                    tsnew += r0*Z0
                    tsnew += np.einsum('ab,ib->ia', Zab, r)
                    tsnew += np.einsum('ji,ja->ia', Zji, r)

        tsnew /= (diag_oo[:, None] - diag_vv)

        return tsnew

    def tsupdate_L1(self, ts, T1inter, alpha):
        """
        SCF+L1 (regularization) update of the t1 amplitudes

        :param ts: t1 amplitudes
        :param T1inter: T1 intermediates
        :param alpha: L1 regularization parameter
        :return: updated ts
        """

        Fae, Fmi, Fai = T1inter
        nocc, nvir = ts.shape

        # remove diagonal of the fock matrix
        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # update ts
        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ie,ae->ia', ts, Fae)
        T1 -= np.einsum('ma,mi->ia', ts, Fmi)

        # subdifferential
        dW = utilities.subdiff(T1, ts, alpha)

        # remove diagonal elements
        eia = diag_oo[:, None] - diag_vv
        dW += ts*eia

        return dW/eia

    def T1inter(self, ts, fsp):
        """
        T1 intermediates from T1 equation Stasis

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix in spin-orbital MO basis
        :return:
        """

        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fvo = self.fock[nocc:, :nocc].copy()
            fvv = self.fock[nocc:, nocc:].copy()
            fov = self.fock[:nocc, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fvo = fsp[nocc:, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        Fai = fvo.copy()
        Fai += np.einsum('kc,kaci->ai', ts, self.eris.ovvo)  # Crawford
        # Fai += np.einsum('me,amie->ai', ts, self.eris.ovvo)  # Stanton

        Fab = fvv.copy()
        Fab -= np.einsum('kb,ka->ab', fov, ts)
        Fab += np.einsum('kc,kacb->ab', ts, self.eris.ovvv)

        Fji = foo.copy()
        Fji += np.einsum('kc,kjci->ji', ts, self.eris.oovo)
        tmp = np.einsum('kc,kjcd->jd', ts, self.eris.oovv)
        Fji += np.einsum('id,jd->ji', ts, tmp)

        return Fab, Fji, Fai

    def T1inter_PySCF(self, ts, fsp):
        """
        T1 intermediates in spin orbital MO basis
        see also PySCF cc/gintermediates

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix in spin orbital MO basis
        :return:
        """

        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fvv = self.fock[nocc:, nocc:].copy()
            fov = self.fock[:nocc, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        # make tau
        fac = 0.5
        tsts = np.einsum('ia,jb->ijab', fac * 0.5 * ts, ts)
        tsts = tsts - tsts.transpose(1, 0, 2, 3)
        tau = tsts - tsts.transpose(0, 1, 3, 2)

        # Fvv
        Fae = fvv.copy()
        Fae -= 0.5 * np.einsum('me,ma->ae', fov, ts)
        Fae += np.einsum('mf,amef->ae', ts, self.eris.vovv)
        Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tau, self.eris.oovv)

        # Foo
        Fmi = foo.copy()
        Fmi += 0.5 * np.einsum('me,ie->mi', fov, ts)
        Fmi += np.einsum('ne,mnie->mi', ts, self.eris.ooov)
        Fmi += 0.5 * np.einsum('inef,mnef->mi', tau, self.eris.oovv)

        # Fvo - Fov
        Fai = fov.copy()
        Fai -= np.einsum('nf,naif->ia', ts, self.eris.ovov)
        Fai = np.einsum('ia->ai', Fai)

        return Fae, Fmi, Fai

    def T1inter_Stanton(self, ts, fsp):
        """
        Row implementation of T1 intermediates in Stanton paper

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix
        :return:
        """

        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fvo = self.fock[nocc:, :nocc].copy()
            fov = self.fock[:nocc, nocc:].copy()
            fvv = self.fock[nocc:, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()
            fvo = fsp[nocc:, :nocc].copy()

        tau = np.zeros((nocc, nocc, nvir, nvir))
        for i in range(0, nocc):
            for j in range(0, nocc):
                for a in range(0, nvir):
                    for b in range(0, nvir):
                        tau[i, j, a, b] = 0.25 * (
                                    ts[i, a] * ts[j, b] - ts[j, a] * ts[i, b] - ts[i, b] * ts[j, a] + ts[j, b] * ts[
                                i, a])

        Fae = fvv.copy()
        Fae -= 0.5 * np.einsum('me,ma->ae', fov, ts)
        # positif in Stanton ; negatif in Crawford
        Fae += np.einsum('mf,amef->ae', ts, self.eris.vovv)
        Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tau, self.eris.oovv)

        Fmi = foo.copy()
        Fmi += 0.5 * np.einsum('ie,me->mi', ts, fov)
        # mnie eris.ooov in Stanton ; nmei eris.oovo in Crawford
        Fmi += np.einsum('ne,mnie->mi', ts, self.eris.ooov)
        Fmi += 0.5 * np.einsum('inef,mnef->mi', tau, self.eris.oovv)

        Fai = fvo.copy()
        Fai += np.einsum('me,amie->ai', ts, self.eris.voov)

        return Fae, Fmi, Fai

    # ------------------------------------------------------------------------------------------
    # Lambda 1 equation
    # -------------------------------------------------------------------------------------------

    def L1eq(self, ts, ls, fsp):
        """
        Value of the Lambda 1 equations using intermediates

        :param ts: t1 amplitudes
        :param ls: l1 amplitudes
        :param fsp: fock matrix
        :return: Lambda1 value
        """
        
        Fia, Fea, Fim, Wieam, E = self.L1inter(ts, fsp)

        L1 = Fia.copy()
        L1 += np.einsum('ie,ea->ia', ls, Fea)
        L1 -= np.einsum('ma,im->ia', ls, Fim)
        L1 += np.einsum('me,ieam->ia', ls, Wieam)
        L1 += ls*E

        return L1

    def lsupdate(self, ts, ls, L1inter, rsn=None, lsn=None, r0n=None, l0n=None, vn=None):
        """
        SCF update of the lambda singles amplitudes ls

        see PySCF module:
        from gccsd import update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        where imds are the intermediates taken from make_intermediates(mycc, t1, t2, eris)

        :param rsn: list with r amplitudes associated to excited state n
        :param lsn: list with l amplitudes associated to excited state n
        :param vn: exp potential Vexp[n,0]
        :return lsnew: updated lambda 1 values
        """

        Fia, Fba, Fij, Wjiba, E = L1inter

        nocc, nvir = ls.shape

        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # remove diagonal of the fock matrix
        Fba[np.diag_indices(nvir)] -= diag_vv
        Fij[np.diag_indices(nocc)] -= diag_oo

        lsnew = Fia.copy()
        lsnew += np.einsum('ib,ba->ia', ls, Fba)
        lsnew -= np.einsum('ja,ij->ia', ls, Fij)
        lsnew += np.einsum('jb,jiba->ia', ls, Wjiba)
        lsnew += ls*E

        # add terms from coupling to excited states
        if rsn is not None:

            # check length
            if len(lsn) != len(rsn) or len(vn) != len(rsn):
                raise ValueError('v0n, l and r list must be of same length')
            if r0n is None or l0n is None:
                raise ValueError('r0 and l0 values must be given')

            for r, l, v, r0, l0 in zip(rsn, lsn, vn, r0n, l0n):

                if v.any():
                    v = np.asarray(v)
                    v_oo = v[:nocc, :nocc]
                    v_vv = v[nocc:, nocc:]
                    v_ov = v[:nocc, nocc:]

                    # P_lam intermediate
                    Pl = np.einsum('jb,jb', r, v_ov)
                    Pl += r0*np.einsum('jb,jb', ts, v_ov)
                    Pl += r0*np.trace(v_oo)

                    # P_0 intermediate => v_ov

                    # P intermediate
                    P = v_oo.copy()
                    P += np.einsum('jb,jb', ts, v_ov)

                    # Pba intermediate
                    Pba = v_vv.copy()
                    Pba -= np.einsum('jb,ja', ts, v_ov)

                    # Pij intermediate
                    Pij = -v_oo.copy()
                    Pij -= np.einsum('jb,ib', ts, v_ov)

                    # add Vexp terms
                    lsnew += ls*Pl
                    lsnew += l0*v_ov
                    lsnew += l*P
                    lsnew += np.einsum('ib,ba', l, Pba)
                    lsnew += np.einsum('ja,ij', l, Pij)

        lsnew /= (diag_oo[:, None] - diag_vv)

        return lsnew

    def lsupdate_L1(self, ls, L1inter, alpha):
        """
        SCF+L1 regularization for the updated lambda single amplitudes

        :param ls: lambda 1 amplitudes
        :param L1inter: Lambda 1 intermediates
        :param alpha: L1 reg. parameter
        :return: updated ls
        """

        Fia, Fba, Fij, Wjiba, E = L1inter

        nocc, nvir = ls.shape

        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # remove diagonal of the fock matrix
        Fba[np.diag_indices(nvir)] -= diag_vv
        Fij[np.diag_indices(nocc)] -= diag_oo

        lsnew = Fia.copy()
        lsnew += np.einsum('ib,ba->ia', ls, Fba)
        lsnew -= np.einsum('ja,ij->ia', ls, Fij)
        lsnew += np.einsum('jb,jiba->ia', ls, Wjiba)
        lsnew += ls*E

        # subdifferential
        dW = utilities.subdiff(lsnew, ls, alpha)

        # remove diagonal elements
        eia = diag_oo[:, None] - diag_vv
        dW += ls*eia
        dW /= eia

        return dW

    def lsupdate_PySCF(self, ts, ls, fsp):
        """
        PySCF module for lambda update with l2=0

        :return:
        """

        from pyscf.cc import gccsd_lambda

        class tmp:
            def __init__(self):
                self.stout = sys.stdout
                self.verbose = 0
                self.level_shift = 0

        nocc, nvir = ts.shape
        t2 = np.zeros((nocc, nocc, nvir, nvir))
        l2 = np.zeros_like(t2)

        tmp_eris = copy.deepcopy(self.eris)
        tmp_eris.fock = fsp

        imds = gccsd_lambda.make_intermediates(tmp, t1, t2, tmp_eris)
        lsnew = gccsd_lambda.update_lambda(tmp, ts, t2, l1, l2, tmp_eris, imds)[0]

        return lsnew

    def L1inter(self, ts, fsp):
        """
        Lambda 1 intermediates from equation ...

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix in spin-orbital MO basis
        :return:
        """

        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fov = self.fock[:nocc, nocc:].copy()
            fvv = self.fock[nocc:, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        Fba = fvv.copy()
        Fba -= np.einsum('ja,jb->ba', fov, ts)
        Fba += np.einsum('jbca,jc->ba', self.eris.ovvv, ts)
        tmp = np.einsum('jkca,jc->kca', self.eris.oovv, ts)
        Fba -= np.einsum('kca,kb->ba', tmp, ts)

        Fij = foo.copy()
        Fij += np.einsum('ib,jb->ij', fov, ts)
        Fij += np.einsum('kibj,kb->ij', self.eris.oovo, ts)
        tmp = np.einsum('kibc,jc->ibj', self.eris.oovv, ts)
        Fij += np.einsum('ibj,kb->ij', tmp, ts)

        Wbija = self.eris.voov.copy()
        Wbija -= np.einsum('kija,kb->bija', self.eris.ovvv, ts)
        tmp = np.einsum('kica,kb->icab', self.eris.oovv, ts)
        Wbija -= np.einsum('icab,jc->bija', tmp, ts)
        Wbija -= np.einsum('bica,jc->bija', self.eris.vovv, ts)

        Fia = fov.copy()
        Fia += np.einsum('jiba,jb->ia', self.eris.oovv, ts)

        # energy term
        E = -np.einsum('jb,jb', ts, fov)
        E -= 0.5*np.einsum('jb,kc,jkbc', ts, ts, self.eris.oovv)

        return Fia, Fba, Fij, Wbija, E

    def L1inter_Stanton(self, ts, fsp):
        """
        Intermediates for the Lambda1 equation
        Stanton 95 with t2=0

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix
        :return:
        """

        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fov = self.fock[:nocc, nocc:].copy()
            fvv = self.fock[nocc:, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        #tau = np.zeros((nocc, nocc, nvir, nvir))
        #for i in range(0, nocc):
        #    for j in range(0, nocc):
        #        for a in range(0, nvir):
        #            for b in range(0, nvir):
        #                tau[i, j, a, b] = 0.25 * (
        #                            ts[i, a] * ts[j, b] - ts[j, a] * ts[i, b] - ts[i, b] * ts[j, a] + ts[j, b] * ts[
        #                        i, a])
        fac = 0.5
        tsts = np.einsum('ia,jb->ijab', fac * 0.5 * ts, ts)
        tsts = tsts - tsts.transpose(1, 0, 2, 3)
        tau = tsts - tsts.transpose(0, 1, 3, 2)

        TFea = fvv.copy()
        TFea -= 0.5 * np.einsum('ma,me->ea', fov, ts)
        TFea += np.einsum('mf,emaf->ea', ts, self.eris.vovv)
        TFea -= 0.5 * np.einsum('mnef,mnaf->ea', tau, self.eris.oovv)

        TFie = fov.copy()
        TFie += np.einsum('nf,inef->ie', ts, self.eris.oovv)

        TFim = foo.copy()
        TFim += 0.5 * np.einsum('me,ie->im', ts, fov)
        TFim += np.einsum('ne,inme->im', ts, self.eris.ooov)
        TFim += 0.5 * np.einsum('mnef,inef->im', tau, self.eris.oovv)

        TFma = TFie.copy()

        Fea = TFea.copy()
        Fea -= 0.5 * np.einsum('me,ma->ea', ts, TFma)

        Fim = TFim.copy()
        Fim += 0.5 * np.einsum('me,ie->im', ts, TFie)

        Weima = self.eris.ovvo.copy()
        Weima += np.einsum('mf,ieaf->ieam', ts, self.eris.ovvv)
        # inam oovo becomes nima ooov in PySCF but same result
        Weima -= np.einsum('ne,inam->ieam', ts, self.eris.oovo)
        # inaf becomes nifa in PySCF but same result
        Weima -= np.einsum('mf,ne,inaf->ieam', ts, ts, self.eris.oovv)
        Weima = Weima.transpose(1, 0, 3, 2)  # ieam to eima

        Fia = TFie.copy()

        # energy term not present due to the use of commutator
        E = 0.

        return Fia, Fea, Fim, Weima, E

    # ------------------------------------------------------------------------------------------
    # R1 equations
    # ------------------------------------------------------------------------------------------

    def R1inter(self, ts, fsp, vm):
        """
        Calculates the R1 intermediates for state m: equations (14) -> (21) in ES-ECW-CCS file

        :param ts: t1 amplitudes
        :param fsp: Effective Fock matrix of state m (containing the Vmm exp potential)
        :param vm: m0Vexp potential
        :return: set of one and two electron intermediates
        """

        nocc,nvir = ts.shape

        if fsp is None:
            fsp = self.fock.copy()

        # Fock matrix
        foo = fsp[:nocc, :nocc].copy()
        fvo = fsp[nocc:, :nocc].copy()
        fvv = fsp[nocc:, nocc:].copy()
        fov = fsp[:nocc, nocc:].copy()

        # r intermediates
        # (commented lines are the additional terms if f=kin)
        # ---------------------------------------------------
        
        # Fab: equation (14)
        Fab = fvv.copy()
        # Fab += np.einsum('akbk->ab', self.eris.vovo)
        Fab -= np.einsum('ja,jb->ab', ts, fov)
        Fab += np.einsum('jc,jacb->ab', ts, self.eris.ovvv)
        # Fab -= np.einsum('ja,jkbk->ab', ts, self.eris.oovo)
        Fab -= np.einsum('jc,ka,jkcb->ab', ts, ts, self.eris.oovv)

        # Fji: equation (15)
        Fji = foo.copy()
        # Fji += np.einsum('jkik->ji', self.eris.oooo)  ##
        Fji += np.einsum('ib,jb->ji', ts, fov)
        # Fji += np.einsum('ib,jkbk->ji', ts, self.eris.oovo)  ##
        Fji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        Fji += np.einsum('kb,ic,kjbc->ji', ts, ts, self.eris.oovv)
        
        # Wakic: equation (16)
        Wakic  = self.eris.voov.copy()
        Wakic += np.einsum('ib,akbc->akic', ts, self.eris.vovv)
        Wakic -= np.einsum('ib,ja,jkbc->akic', ts, ts, self.eris.oovv)
        Wakic -= np.einsum('ja,jkic->akic', ts, self.eris.ooov)

        # ria intermediates (energy term)
        # ----------------------------------

        # Fjb: equation (17)
        Fjb = fov.copy()
        # Fjb += np.einsum('jkbk->jb', self.eris.oovo)
        Fjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        Er = np.einsum('jb,jb', ts, Fjb)
        del Fjb

        # r0 intermediates (Zai/Fai and T eq terms)
        # --------------------------------------------
        
        # Zab: equation (18)
        Zab = fvv.copy()
        Zab -= np.einsum('ja,jb->ab', ts, fov)
        # Zab += np.einsum('akbk->ab', self.eris.vovo)  ##
        # Zab -= np.einsum('ka,kjbj->ab', ts, self.eris.oovo)  ##
        
        # Zji: equation (19)
        Zji = foo.copy()
        # Zji += np.einsum('jkik->ji', self.eris.oooo)  ##
        Zji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        tmp = np.einsum('ic,jkbc->ijkb', ts, self.eris.oovv)
        Zji += np.einsum('kb,ijkb->ji', ts, tmp)  # recheck sign ?
        del tmp

        # Zai: equation (20)
        Zai = fvo.copy()
        # Zai += np.einsum('akik->ai', self.eris.vooo)  ##
        Zai += np.einsum('jb,jabi->ai', ts, self.eris.ovvo)
        Zai += np.einsum('jb,ic,jabc->ai', ts, ts, self.eris.ovvv)

        # Tia: equation (21)
        Tia = np.einsum('ai->ia', Zai)
        Tia += np.einsum('ib,ab->ia', ts, Zab)
        Tia -= np.einsum('ja,ji->ia', ts, Zji)
        del Zab, Zji, Zai

        # Vexp intermediate P: equation (22)
        v_vo = vm[nocc:, :nocc]
        v_vv = vm[nocc:, nocc:]
        v_oo = vm[:nocc, :nocc]
        Pia = v_vo.copy()
        Pia += np.einsum('ab,ib->ai', v_vv, ts)
        Pia -= np.einsum('ii,ja,ib->ai', v_oo, ts, ts)
        Pia = np.einsum('ai->ia', Pia)

        return Fab, Fji, Wakic, Er, Tia, Pia
    
    def R0inter(self, ts, fsp, vm):
        """
        one and two particles intermediates for state m as well as the Vexp intermediate
        for the R0 equations

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix for the state m
        :param vm: m0V potential
        :return: Fjb, Zjb, P intermediates
        """
        
        nocc,nvir = ts.shape

        if fsp is None:
            fsp = self.fock.copy()

        fov = fsp[:nocc, nocc:].copy()
        
        # r intermediates
        # ------------------
        
        # Fjb: equation (23)
        Fjb = fov.copy()
        # Fjb += np.einsum('jkbk->jb',self.eris.oovo)
        # tmp = Fjb.copy()
        Fjb += np.einsum('kc,kjcb->jb', ts, self.eris.oovv)
           
        # r0 intermediates
        # ------------------
        
        # Zjb: equation (25) --> Same as Fjb in R1inter
        # Zjb and ts are contracted here
        Zjb = fov.copy()
        Zjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        Z = np.einsum('jb,jb', ts, Zjb)
        del Zjb
        #del tmp

        # Vexp inter
        # -------------------
        vm_oo = vm[:nocc, :nocc]
        vm_ov = vm[:nocc, nocc:]
        P = np.einsum('jj', vm_oo)
        P += np.einsum('jb,jb', ts, vm_ov)
        
        return Fjb, Z, P

    def Extract_Em_r(self, rs, r0, Rinter, ov=None):
        """
        Extract Em from the largest r1 element

        :param rs: r1 amplitude of state m
        :param r0: r0 amplitude of state m
        :param Rinter: R1 intermediates
        :return: Em and index of largest r1 element
        """

        Fab, Fji, W, F, Zia, Pia = Rinter

        # largest r1 if indices not given
        if ov is None:
            o, v = np.unravel_index(np.argmax(abs(rs), axis=None), rs.shape)
        else:
            o, v = ov

        # Ria = ria*En' matrix
        Ria = np.einsum('ab,ib->ia', Fab, rs)
        Ria -= np.einsum('ji,ja->ia', Fji, rs)
        Ria += np.einsum('akic,kc->ia', W, rs)
        Rov = Ria[o, v]

        del Ria
        Rov += rs[o, v] * F
        Rov += r0 * Zia[o, v]
        Rov += Pia[o, v]

        Em = Rov/rs[o, v]

        return Em, o, v

    def Extract_r0(self, r1, ts, fsp, vm):
        """
        Use R1 and R0 equations to calculate r0 from given r1

        :param r1: r1 amplitude vector
        :return: r0
        """

        if fsp is None:
            f = self.fock
        else:
            f = fsp.copy()

        Fab, Fji, W, F, Zia, Pia = self.R1inter(ts, f, vm)
        Fjb, Z, P = self.R0inter(ts, f, vm)

        R1 = np.einsum('ab, ib->ia', Fab, r1)
        R1 -= np.einsum('ji, ja->ia', Fji, r1)
        R1 += np.einsum('kc, akic->ia', r1, W)
        R1 += r1*F
        R1 += Pia
        
        c = -np.einsum('jb, jb', r1, Fjb)
        c -= P

        if c == 0.:
            return 0
        else:
            # largest r1
            i, j = np.unravel_index(np.argmax(abs(r1), axis=None), r1.shape)
            a = Zia[i, j] / r1[i, j]
            b = R1[i, j] / r1[i, j]
            b -= Z

            # solve quadratic equation using delta
            r0_1 = (-b + np.sqrt((b**2)-(4*a*c))) / c
            r0_2 = (-b - np.sqrt((b**2)-(4*a*c))) / c

            if r0_1 > 0:
                return r0_1
            elif r0_2 > 0:
                return r0_2
            else:
                raise ValueError('Both solution for r0 are negative')

    def rsupdate(self, rs, r0, Rinter, Em, idx=None):
        """
        Update r1 amplitudes using Ria equations and given E for iteration k

        :param rs: matrix of r1 amplitudes for state m at k-1 iteration
        :param r0: r0 amplitude for state m at iteration k-1
        :param Rinter: r1 intermediates for state m
        :param Em: Energy of state m
        :return: updated list of r1 amplitudes and index of the largest ria

        """

        Fab, Fji, W, F, Zia, Pia = Rinter
        nocc, nvir = rs.shape

        # remove diagonal of the fock matrix
        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])
        Fab[np.diag_indices(nvir)] -= diag_vv
        Fji[np.diag_indices(nocc)] -= diag_oo

        # Ria = ria*En' matrix

        rsnew  = np.einsum('ab,ib->ia', Fab, rs)
        rsnew -= np.einsum('ji,ja->ia', Fji, rs)
        rsnew += np.einsum('akic,kc->ia', W, rs)
        rsnew += rs*F
        rsnew += r0*Zia
        rsnew += Pia
        #rsnew -= rs*Em  # uncomment if E stays in the right hand side

        if idx is not None:
            o, v = idx
            rov = rsnew[o-1, v-1]-rs[o-1, v-1]*Em
            rov /= (diag_oo[o-1]-diag_vv[v-1])
            rsnew /= (Em+diag_oo[:, None]-diag_vv)  # comment if E stays in the right hand side
            rsnew[o-1, v-1] = rov
            return rsnew

        # divide by diag
        rsnew /= (Em+diag_oo[:, None]-diag_vv)  # comment if E stays in the right hand side
        #rsnew /= (diag_oo[:, None] - diag_vv)  # uncomment if E stays in the right hand side

        return rsnew

    def R1eq(self, rs, r0, Rinter):
        """
        Return the Ria values

        :param rs: r1 amplitudes
        :param r0: r0 amplitude
        :param Rinter: R1 intermediates
        :return: Ria
        """

        Fab, Fji, W, F, Zia, Pia = Rinter

        # Ria = ria*En' matrix
        Ria = np.einsum('ab,ib->ia', Fab, rs)
        Ria -= np.einsum('ji,ja->ia', Fji, rs)
        Ria += np.einsum('akic,kc->ia', W, rs)
        Ria += rs * F
        Ria += r0 * Zia
        Ria += Pia

        return Ria

    def r0update(self, rs, r0, Em, R0inter):
        """

        :param rs: r1 amplitude
        :param Em: energy of state m
        :param R0inter: intermediates for the R0 equation
        :return: updated r0 for state m
        """

        Fjb, Z, P = R0inter
        F = np.einsum('jb,jb', rs, Fjb)
        r0new = F+P+(r0*Z)
        r0new /= Em

        return r0new

    def get_ov(self, ls, l0, rs, r0, ind):
        """
        Extract missing ria/lia value from normality relation

        :param ls: l1 amplitudes for state m
        :param l0: l0 amplitude for state m
        :param rs: r1 amplitudes for state m
        :param r0: r0 amplitude for state m
        :param ind: index of missing rov amplitude
        :return: updated rov
        """

        o, v = ind
        r = rs.copy()
        r[o, v] = 0
        rov = 1 - r0 * l0 - np.einsum('ia,ia', r, ls)
        rov /= ls[o, v]

        return rov

    def R0eq(self, En, t1, r1, vm0, fsp=None):
        """
        Returns the r0 value for a CCS state from the R0 equation

        :param En: correlation energy of the state
        :param t1: t1 amp
        :param r1: r1 amp
        :param vm0: constraint potential Vm0
        :param fsp: fock matrix
        :return: r0
        """

        if fsp is None:
            fsp = self.fock.copy()

        nocc, nvir = r1.shape

        vov = vm0[:nocc, nocc:].copy()
        voo = vm0[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        
        d = 0.
        d += En
        d -= np.einsum('jb,jb', t1, fov)
        # d -= np.einsum('jb,jkbk', t1, self.eris.oovo)
        d -= 0.5 * np.einsum('jb,kc,jkbc', t1, t1, self.eris.oovv)

        r0 = 0.
        r0 += np.einsum('jb,jb', r1, fov)
        # r0 += np.einsum('jb,jkbk', t1, self.eris.oovo)
        r0 += np.einsum('kc,jb,jkbc', r1, t1, self.eris.oovv)
        r0 += np.einsum('jb, jb', t1, vov)
        r0 += np.trace(voo)
        r0 /= d

        return r0

    # -----------------------------------------------------------------------------------------------
    # L1 equations
    # -----------------------------------------------------------------------------------------------

    def es_L1inter(self, ts, fsp, vm):
        """
        Returns the intermediates for the L1 equations of state m

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix containing the potential Vmm
        :param vm: coupling potential V0m
        :return:
        """

        nocc,nvir = ts.shape

        # Fock matrix
        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fov = self.fock[:nocc, nocc:].copy()
            fvv = self.fock[nocc:, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        # l intermediates
        # (commented lines are additional term when f=kin)
        # -----------------

        # Fba: equation (30)
        Fba = fvv.copy()
        # Fba += np.einsum('bkak->ba', self.eris.vovo)  ##
        Fba -= np.einsum('jb,ja->ba', ts, fov)
        Fba += np.einsum('jc,jbca->ba', ts, self.eris.ovvv)
        # Fba -= np.einsum('jb,jkak->ba', ts, self.eris.oovo)  ##
        Fba -= np.einsum('jc,kb,jkca->ba', ts, ts, self.eris.oovv)

        # Fij: equation (31)
        Fij = foo.copy()
        #Fij += np.einsum('ikjk->ij', self.eris.oooo)  ##
        Fij += np.einsum('jb,ib->ij', ts, fov)
        #Fij += np.einsum('kb,kibj->ij', ts, self.eris.oovo)  ##
        Fij += np.einsum('jb,jibk->ij', ts, self.eris.oovo)
        Fij += np.einsum('kb,jc,kibc->ij', ts, ts, self.eris.oovv)

        # Wbija: equation (32)
        W = self.eris.voov.copy()
        W -= np.einsum('kb,kija->bija', ts, self.eris.ooov)
        W += np.einsum('jc,bica->bija', ts, self.eris.vovv)
        W -= np.einsum('jc,kb,kica->bija', ts, ts, self.eris.oovv)

        # El: equation (33) --> same as for R1inter (energy term)
        Fjb = fov.copy()
        #Fjb += np.einsum('jkbk->jb',self.eris.oovo)  ##
        Fjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        El = np.einsum('jb,jb', ts, Fjb)
        del Fjb

        # l0 intermediate
        # ------------------

        Zia  = fov.copy()
        #Zia += np.einsum('ikak->ia', self.eris.oovo)  ##
        Zia += np.einsum('jb,jiba->ia', ts, self.eris.oovv)

        # Vexp intermediate
        # ---------------------

        P = vm[:nocc, nocc:].copy()

        return Fba, Fij, W, El, Zia, P

    def L0inter(self, ts, fsp, vm):
        '''
        L0 intermediates for the L0 equation of state m

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix containing the Vmm potential
        :param vm: V0m coupling potential
        :return: L0 intermediates
        '''
       
        nocc,nvir = ts.shape

        if fsp is None:
            fsp = self.fock.copy()

        # Effective fock sub-matrices
        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()
        fvo = fsp[nocc:, :nocc].copy()

        # Fbj: eq (36)
        Fbj  = fvo.copy()
        Fbj -= np.einsum('kb,kj->bj', ts, foo)
        Fbj += np.einsum('ja,ba->bj', ts, fvv)
        Fbj -= np.einsum('jc,kb,kc->bj', ts, ts, fov)

        # Wjb: eq (37)
        tmp  = self.eris.ovvo.copy()
        tmp += np.einsum('lb,jd,lkcd->kbcj', ts, ts, self.eris.oovv)
        tmp -= np.einsum('lb,klcj->kbcj', ts, self.eris.oovo)
        tmp += np.einsum('jd,kbcd->kbcj', ts, self.eris.ovvv)
        Wjb = np.einsum('kc,kbcj->jb', ts, tmp)
        del tmp
        #Wjb -= np.einsum('kb,kljl->jb', ts, self.eris.oooo)
        #Wjb -= np.einsum('jv,kb,klcl->jb', ts, ts, self.eris.oovo)
        #Wjb += np.einsum('jc,bkck->jb', ts, self.eris.vovo)
        #Wjb += np.einsum('bkjk->jb', self.eris.vooo)

        # Z: eq (38) --> same as Z in R0 equation --> R1inter
        Zjb = fov.copy()
        #Zjb += np.einsum('jkbk->jb', self.eris.oovo)
        Zjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        Z = np.einsum('jb,jb', ts, Zjb)
        del Zjb
        
        # P: eq (39)
        P = np.einsum('ia,ia', ts, vm[:nocc, nocc:])
        P += np.sum(np.diagonal(vm[:nocc, :nocc]))

        return Fbj, Wjb, Z, P

    def Extract_Em_l(self, ls, l0, L1inter, ov=None):
        """
        Extract Em from the largest l1 element

        :param ls: l1 amplitude of state m
        :param l0: l0 amplitude of state m
        :param L1inter: L1 intermediates
        :return: Em and index of largest l1 element
        """

        Fba, Fij, W, F, Zia, P = L1inter
        
        # largest r1
        if ov is None:
            o, v = np.unravel_index(np.argmax(abs(ls), axis=None), ls.shape)
        else:
            o, v = ov

        # Lia = lia*En' matrix
        Lia  = np.einsum('ib,ba->ia', ls, Fba)
        Lia -= np.einsum('ja,ij->ia', ls, Fij)
        Lia += np.einsum('jb,bija->ia', ls, W)
        Lov = Lia[o,v]
        del Lia

        Lov += ls[o,v] * F
        Lov += l0 * Zia[o,v]
        Lov += P[o,v]

        Em = Lov/ls[o,v]

        return Em, o,v

    def Extract_l0(self, l1, ts, fsp, vm):
        '''
        Use L1 and L0 equations to calculate l0 from given r1

        :param l1: l1 amplitude vector
        :return: r0
        '''

        if fsp is None:
            f = self.fock
        else:
            f = fsp.copy()

        Fba, Fij, W, F, Zia, P = self.es_L1inter(ts, f, vm)
        Fbj, Wjb, Z, P = self.L0inter(ts, f, vm)

        L1 = np.einsum('ba, ib->ia', Fba, l1)
        L1 -= np.einsum('ij, ja->ia', Fij, l1)
        L1 += np.einsum('jb, bija->ia', l1, W)
        L1 += l1 * F
        L1 += P

        c = -np.einsum('jb, bj', l1, Fbj)
        c -= P

        if c == 0.:
            return 0
        else:
            # largest l1
            i, j = np.unravel_index(np.argmax(abs(l1), axis=None), l1.shape)
            a = Zia[i, j] / l1[i, j]
            b = L1[i, j] / l1[i, j]
            b -= Z

            # solve quadratic equation using delta
            l0_1 = (-b + np.sqrt((b ** 2) - (4 * a * c))) / 2*c
            l0_2 = (-b - np.sqrt((b ** 2) - (4 * a * c))) / 2*c

            if l0_1 > 0:
                return l0_1
            elif l0_2 > 0:
                return l0_2
            else:
                raise ValueError('Both solution for l0 are negative')

    def es_lsupdate(self, ls, l0, Em, L1inter, idx=None):
        '''
        Update the l1 amplitudes for state m

        :param ls: list of l amplitudes for the m excited state
        :param l0: l0 amplitude for state m
        :param Em: Energy of the state m
        :param L1inter: intermediates for the L1 equation of state m
        :return: updated matrix of ls amplitudes for state m
        '''

        Fba, Fij, W, F, Zia, P = L1inter
        nocc, nvir = ls.shape

        # remove diagonal of the fock matrix
        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])
        Fba[np.diag_indices(nvir)] -= diag_vv
        Fij[np.diag_indices(nocc)] -= diag_oo

        # get lia
        lsnew  = np.einsum('ib,ba->ia', ls, Fba)
        lsnew -= np.einsum('ja,ij->ia', ls, Fij)
        lsnew += np.einsum('jb,bija->ia', ls, W)
        lsnew += ls*F
        lsnew += l0*Zia
        lsnew += P
        #lsnew -= ls*Em
        if idx is not None:
            o, v = idx
            lov = lsnew[o-1, v-1]-ls[o-1, v-1]*Em
            lov /= (diag_oo[o-1]-diag_vv[v-1])
            lsnew /= (Em+diag_oo[:, None]-diag_vv)
            lsnew[o-1, v-1] = lov
            return lsnew

        # divide by diag
        lsnew /= (Em+diag_oo[:,None]-diag_vv)
        #lsnew /= (diag_oo[:, None] - diag_vv)

        return lsnew

    def es_L1eq(self, ls, l0, L1inter):
        """
        Update the l1 amplitudes for state m

        :param ls: list of l amplitudes for the m excited state
        :param l0: l0 amplitude for state m
        :param L1inter: intermediates for the L1 equation of state m
        :return: updated matrix of ls amplitudes for state m
        """

        Fba, Fij, W, F, Zia, P = L1inter

        # get lia
        Lia  = np.einsum('ib,ba->ia', ls, Fba)
        Lia -= np.einsum('ja,ij->ia', ls, Fij)
        Lia += np.einsum('jb,bija->ia', ls, W)
        Lia += ls*F
        Lia += l0*Zia
        Lia += P

        return Lia

    def l0update(self, ls, l0, Em, L0inter):
        '''
        Update the l0 amplitude

        :param ls: l1 amplitudes for state m
        :param Em: energy for state m
        :param L0inter: L1 intermediates for state m
        :return:
        '''

        Fjb, Wjb, Z, P = L0inter
        F = np.einsum('jb,bj', ls, Fjb)
        W = np.einsum('jb,jb', ls, Wjb)
        l0new = F+W+P+(l0*Z)
        l0new /= Em

        return l0new

    def L0eq(self, En, t1, l1, v0m, fsp=None):
        '''
        Returns l0 term for a CCS state
        See equation 23 in Derivation_ES file

        :param En: correlation energy (En+EHF=En_tot) of the state n
        :param t1: t1 amp
        :param l1: l1 amp of state n
        :param v0m: contraint potential matrix V0m
        :param fsp: fock matrix
        :return: l0
        '''

        nocc, nvir = t1.shape

        if fsp is None:
            fsp = self.fock.copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()
        foo = fsp[:nocc, :nocc].copy()

        vov = v0m[:nocc, nocc:]
        voo = v0m[:nocc, :nocc]

        d = En
        #d -= np.einsum('jb,jkbk', t1, self.eris.oovo)
        d -= 0.5 * np.einsum('jb,kc,jkbc', t1, t1, self.eris.oovv)

        l0 = np.einsum('jb,jb', l1, fov)
        l0 += np.einsum('jb,ab,ja', t1, fvv, l1)
        l0 -= np.einsum('jb,kb,kj', l1, t1, foo)
        l0 -= np.einsum('jc,kb,kc,jb', t1, t1, fov, l1)
        #l0 += np.einsum('jb,bkjk', l1, self.eris.vooo)
        l0 += np.einsum('jb,kc,kbcj', l1, t1, self.eris.ovvo)
        #l0 -= np.einsum('jb,kb,kljl', l1, t1, self.eris.oooo)
        #l0 += np.einsum('jb,jc,bkck', l1, t1, self.eris.vovo)
        tmp = np.einsum('jb,jd->bd', l1, t1)
        l0 += np.einsum('bd,kb,lc,klcd', tmp, t1, t1, self.eris.oovv)
        del tmp
        tmp = np.einsum('jb,lb->jl', l1, t1)
        l0 -= np.einsum('jl,kc,klcj', tmp, t1, self.eris.oovo)
        del tmp
        #tmp = np.einsum('jb,jl->bl', l1, t1)
        #l0 += np.einsum('bl,kc,kbcl', tmp, t1, self.eris.ovvo)
        tmp = np.einsum('jb,jd->bd', l1, t1)
        l0 += np.einsum('bd,kc,kbcd', tmp, t1, self.eris.ovvv)
        #tmp = np.einsum('jb,jc->bc', l1, t1)
        #l0 -= np.einsum('bc,kb,klcl', tmp, t1, self.eris.oovo)
        del tmp
        l0 += np.einsum('ia, ia', t1, vov)
        l0 += np.trace(voo)

        l0 /= d

        return l0

#################################
#   ECW-GCCS gradient equations #
#################################

class ccs_gradient:
    def __init__(self, eris, M_tot=1, sum_sig=1):
        '''
        Gradient of the ECW-CCS equations and Newton's method

        :param eris: two electron integrals
        :param M_tot: scale of the Vexp potential (number of measurements)
        :param sum_sig: sum of all sig_i, means for each sets of measurements
        '''

        self.M_tot = M_tot
        self.sum_sig = sum_sig

        self.fock = eris.fock
        self.eris = eris

        self.nocc = eris.nocc
        self.nvir = self.fock.shape[0]-self.nocc 

    ################
    # T1 equations
    ################

    def T1eq(self, ts, fsp):
        """
        T1 equations using intermediates

        :param ts: t1 amplitudes
        :param fsp: fock matrix
        :return: T1 nocc x nvir matrix
        """
        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fvo = self.fock[nocc:, :nocc].copy()
            fvv = self.fock[nocc:, nocc:].copy()
            fov = self.fock[:nocc, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fvo = fsp[nocc:, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        Fai = fvo.copy()
        Fai += np.einsum('kc,kaci->ai', ts, self.eris.ovvo)

        Fab = fvv.copy()
        Fab -= np.einsum('kb,ka->ab', fov, ts)
        Fab += np.einsum('kc,kacb->ab', ts, self.eris.ovvv)

        Fji = foo.copy()
        Fji += np.einsum('kc,kjci->ji', ts, self.eris.oovo)
        tmp = np.einsum('kc,kjcd->jd', ts, self.eris.oovv)
        Fji += np.einsum('id,jd->ji', ts, tmp)

        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ie,ae->ia', ts, Fab)
        T1 -= np.einsum('ma,mi->ia', ts, Fji)

        return T1

    #################
    # L1 equations
    #################

    def L1eq(self, ts, ls, fsp):
        """
        Value of the Lambda 1 equations using intermediates

        :param ts: t1 amplitudes
        :param ls: l1 amplitudes
        :param fsp: fock matrix
        :return: Lambda1 value
        """

        nocc, nvir = ts.shape

        if fsp is None:
            foo = self.fock[:nocc, :nocc].copy()
            fov = self.fock[:nocc, nocc:].copy()
            fvv = self.fock[nocc:, nocc:].copy()
        else:
            foo = fsp[:nocc, :nocc].copy()
            fov = fsp[:nocc, nocc:].copy()
            fvv = fsp[nocc:, nocc:].copy()

        Fba = fvv.copy()
        Fba += np.einsum('ja,jb->ba', fov, ts)
        Fba += np.einsum('jbca,jc->ba', self.eris.ovvv, ts)
        tmp = np.einsum('kjca,jc->kca', self.eris.oovv, ts)
        Fba += np.einsum('kca,kb->ba', tmp, ts)

        Fij = foo.copy()
        Fij += np.einsum('ib,jb->ij', fov, ts)
        Fij += np.einsum('kibj,kb->ij', self.eris.oovo, ts)
        tmp = np.einsum('kibc,kc->ibc', self.eris.oovv, ts)
        Fij -= np.einsum('ibc,jb->ij', tmp, ts)

        Wibaj = self.eris.ovvo.copy()
        Wibaj -= np.einsum('ibca,jc->ibaj', self.eris.ovvv, ts)
        tmp = np.einsum('kica,kb->icab', self.eris.oovv, ts)
        Wibaj -= np.einsum('icab,jc->ibaj', tmp, ts)
        Wibaj -= np.einsum('ikaj,kb->ibaj', self.eris.oovo, ts)

        Fia = fov.copy()
        Fia += np.einsum('jiba,jb->ia', self.eris.oovv, ts)

        L1 = Fia.copy()
        L1 += np.einsum('ib,ba->ia', ls, Fba)
        L1 -= np.einsum('ja,ij->ia', ls, Fij)
        L1 += np.einsum('jb,ibaj->ia', ls, Wibaj)

        return L1

    ############
    # dT/dt
    ############

    def dTdt(self, ts, ls, fsp, L):

        nocc, nvir = ts.shape

        dT = np.zeros((nocc * nvir, nocc * nvir))
        C = L * (1 / self.M_tot) * self.sum_sig

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

        # contractions
        doidga = C
        tmp = np.einsum('klcd,kc->ld', self.eris.oovv, ts)
        doidga += -np.einsum('ld,la->a', tmp, ts)

        doi1 = -np.einsum('ka,kg->ag', ts, fov)
        doi2 = -np.einsum('kc,kacg->ag', ts, self.eris.ovvv)
        doi3 = -C * np.einsum('ka,kg->ag', ts, ls)
        dga1 = -np.einsum('ie,oe->io', ts, fov)

        tmp = np.einsum('kc,kocd->o', ts, self.eris.oovv)
        dga2 = -np.einsum('ia,o->aio', ts, tmp)

        dga3 = -np.einsum('kc,koci->oi', ts, self.eris.oovo)

        tmp = np.einsum('ic,oa->ioa', ts, ts)
        int1 = np.einsum('ioa,mg,mc->ioag', tmp, ls, ts)

        tmp = np.einsum('ig,ka->iga', ts, ts)
        int2 = np.einsum('iga,oe,ke->igao', tmp, ls, ts)

        int3 = -np.einsum('id,oagd->oagi', ts, self.eris.ovvv)

        tmp = np.einsum('olgd,la->oga', self.eris.oovv, ts)
        int4 = -np.einsum('oga,ia->ogia', tmp, ts)

        int5 = -np.einsum('la,olgi->ogia', ts, self.eris.oovo)

        for p in range(0, nvir * nocc):
            for q in range(0, nocc * nvir):
                o, g = np.unravel_index(p, (nocc, nvir))
                i, a = np.unravel_index(q, (nocc, nvir))
                if o == i:
                    if g == a:
                        dT[p, q] += doidga[a]
                    dT[p, q] += fvv[a, g]
                    dT[p, q] += doi1[a, g]
                    dT[p, q] += doi2[a, g]
                    dT[p, q] += doi3[a, g]
                if g == a:
                    # dT[p,q] += C*foo[o,i]
                    dT[p, q] += -foo[o, i]
                    dT[p, q] += dga1[i, o]
                    dT[p, q] += dga2[a, i, o]
                    dT[p, q] += dga3[o, i]
                dT[p, q] += C * (ts[i, g] * ls[o, a] - ts[o, a] * ls[i, g])
                dT[p, q] += self.eris.ovvo[o, a, g, i]
                dT[p, q] += -C * ts[i, g] * ts[o, a]
                dT[p, q] += C * (int1[i, o, a, g] + int2[i, g, a, o])
                dT[p, q] += int3[o, a, g, i]
                dT[p, q] += int4[o, g, i, a]
                dT[p, q] += int5[o, g, i, a]

        return dT

    ###########
    # dT/dl
    ###########

    def dTdl(self, ts, L):

        nocc, nvir = ts.shape

        dT = np.zeros((nocc * nvir, nocc * nvir))
        C = L * (1 / self.M_tot) * self.sum_sig

        # contractions
        int1 = np.einsum('ic,oc->io', ts, ts)
        int2 = -np.einsum('ka,kg->ag', ts, ts)

        tmp = np.einsum('ic,ka->ia', ts, ts)
        int3 = -np.einsum('ia,oc,kg->iaog', tmp, ts, ts)

        for p in range(0, nvir * nocc):
            for q in range(0, nocc * nvir):
                o, g = np.unravel_index(p, (nocc, nvir))
                i, a = np.unravel_index(q, (nocc, nvir))
                if g == a:
                    dT[p, q] += int1[i, o]
                    if o == i:
                        dT[p, q] += 1
                if o == i:
                    dT[p, q] += int2[a, g]
                dT[p, q] += int3[i, a, o, g]

        dT = dT * C

        return dT

        # dL/dt

    def dLdt(self, ts, ls, fsp, L):

        nocc, nvir = ts.shape
        fov = fsp[:nocc, nocc:].copy()
        dL = np.zeros((nocc * nvir, nocc * nvir))
        C = L * (1 / self.M_tot) * self.sum_sig

        # contractions involving doi and dag
        doi1 = -np.einsum('ma,mg->ag', ts, ls)
        doi2 = -np.einsum('ja,jg->ag', ls, ls)
        doi3 = -np.einsum('ja,jg->ag', ls, ts)

        tmp = np.einsum('ja,jb->a', ls, ts)
        doi4 = np.einsum('a,mb,mg->ag', tmp, ts, ls)

        dag1 = -np.einsum('ie,oe->io', ts, ls)
        dag2 = np.einsum('ib,ob->io', ls, ls)
        dag3 = -np.einsum('ib,ob->io', ls, ts)

        tmp = np.einsum('ib,jb->i', ls, ts)
        dag4 = np.einsum('i,je,oe->io', tmp, ts, ls)

        # ts and ls contractions
        tmp = np.einsum('ib,ob->oi', ls, ts)
        int1 = C * np.einsum('oi,ma,mg->ogia', tmp, ts, ls)

        tmp = np.einsum('ja,jg->ga', ls, ts)
        int2 = C * np.einsum('ga,ie,oe->ogia', tmp, ts, ls)

        # eris, ls, ts contractions
        int3 = -np.einsum('icga,oc->ogia', self.eris.ovvv, ls)
        int4 = np.einsum('ocga,ic->ogia', self.eris.ovvv, ls)

        int5 = np.einsum('okga,ic,kc->ogia', self.eris.oovv, ls, ts)
        int6 = np.einsum('ojab,ig,jb->ogia', self.eris.oovv, ls, ts)
        int7 = np.einsum('oigb,ja,jb->ogia', self.eris.oovv, ls, ts)
        int8 = np.einsum('ikgc,oa,kc->ogia', self.eris.oovv, ls, ts)
        int9 = -np.einsum('oiac,kg,kc->ogia', self.eris.oovv, ls, ts)
        int10 = -np.einsum('ijag,ob,jb->ogia', self.eris.oovv, ls, ts)
        int11 = -np.einsum('oigk,ka->ogia', self.eris.oovo, ls)
        int12 = -np.einsum('oiaj,jg->ogia', self.eris.oovo, ls)

        for p in range(0, nvir * nocc):
            for q in range(0, nocc * nvir):
                o, g = np.unravel_index(p, (nocc, nvir))
                i, a = np.unravel_index(q, (nocc, nvir))
                if g == a:
                    dL[p, q] += C * (dag1[i, o] + dag2[i, o] + dag3[i, o] + dag4[i, o])
                    if o == i:
                        dL[p, q] += C
                if o == i:
                    dL[p, q] += C * (doi1[a, g] + doi2[a, g] + doi3[a, g] + doi4[a, g])
                dL[p, q] += int1[o, g, i, a] + int2[o, g, i, a] + int3[o, g, i, a] \
                            + int4[o, g, i, a] + int5[o, g, i, a]
                dL[p, q] += int6[o, g, i, a] + int7[o, g, i, a] + int8[o, g, i, a] \
                            + int9[o, g, i, a] + int10[o, g, i, a]
                dL[p, q] += int11[o, g, i, a] + int12[o, g, i, a]
                dL[p, q] += C * (2 * ls[o, a] * ls[i, g])
                dL[p, q] += -(fov[o, a] * ls[i, g]) - (fov[i, g] * ls[o, a])
                dL[p, q] += self.eris.oovv[o, i, g, a]

        return dL

        # dL/dl

    def dLdl(self, ts, ls, fsp, L):

        nocc, nvir = ts.shape
        fov = fsp[:nocc, nocc:].copy()
        foo = fsp[:nocc, :nocc].copy()
        fvv = fsp[nocc:, nocc:].copy()

        dL = np.zeros((nocc * nvir, nocc * nvir))
        C = L * (1 / self.M_tot) * self.sum_sig

        # contraction involving dag and doi
        # dag1 = 2*C*np.einsum('ib,ob->oi',ls,ts)
        dag2 = -np.einsum('ib,ob->oi', fov, ts)

        tmp = np.einsum('kibc,kc->ib', self.eris.oovv, ts)
        dag3 = np.einsum('ib,ob->oi', tmp, ts)

        dag4 = -np.einsum('jiob,jb->oi', self.eris.ooov, ts)
        doi1 = -C * np.einsum('ja,jg->ag', ls, ts)
        doi2 = np.einsum('ja,jg->ag', fov, ts)
        doi3 = -C * np.einsum('ja,jg->ag', ls, ts)
        doi4 = np.einsum('jgba,jb->ag', self.eris.ovvv, ts)

        tmp = np.einsum('kjab,jb->ak', self.eris.oovv, ts)
        doi5 = np.einsum('ak,kg->ag', tmp, ts)

        # other contraction
        tmp = np.einsum('ib,jb->ij', ls, ts)
        int1 = C * np.einsum('ij,oa,jg->iaog', tmp, ts, ts)

        tmp = np.einsum('ja,jb->a', ls, ts)
        int2 = C * np.einsum('a,ob,ig->iaog', tmp, ts, ts)
        int3 = -np.einsum('igba,ob->iaog', self.eris.ovvv, ts)

        tmp = np.einsum('jiac,oc->iaoj', self.eris.oovv, ts)
        int4 = -np.einsum('iaoj,jg->iaog', tmp, ts)
        int5 = -np.einsum('ikoa,kg->iaog', self.eris.ooov, ts)

        for p in range(0, nvir * nocc):
            for q in range(0, nocc * nvir):
                o, g = np.unravel_index(p, (nocc, nvir))
                i, a = np.unravel_index(q, (nocc, nvir))
                if g == a:
                    dL[p, q] += dag2[o, i] + dag3[o, i] + dag4[o, i]
                    dL[p, q] += -foo[o, i]
                    if o == i:
                        dL[p, q] += C
                if o == i:
                    dL[p, q] += fvv[a, g]
                    dL[p, q] += doi1[a, g] + doi2[a, g] + doi3[a, g] \
                                + doi4[a, g] + doi5[a, g]
                dL[p, q] += int1[i, a, o, g] + int2[i, a, o, g] + int3[i, a, o, g] \
                            + int4[i, a, o, g] + int5[i, a, o, g]
                dL[p, q] += C * (-ts[o, a] * ts[i, g] + ls[i, g] * ts[o, a] - ls[o, a] * ts[i, g])
                dL[p, q] += self.eris.ovov[i, g, o, a]

        return dL

    def Jacobian(self, ts, ls, fsp, L):

        # build Jacobian
        J00 = self.dTdt(ts, ls, fsp, L)
        J01 = self.dTdl(ts, L)
        J10 = self.dLdt(ts, ls, fsp, L)
        J11 = self.dLdl(ts, ls, fsp, L)
        J = np.block([[J00, J01], [J10, J11]])

        return J

    ###################
    # Newton's method
    ###################

    def Newton(self, ts, ls, fsp, L):

        nocc, nvir = ts.shape

        # make T1 and L1 eq. vectors and build X
        T1 = self.T1eq(ts, fsp).flatten()
        L1 = self.L1eq(ts, ls, fsp).flatten()
        X = np.concatenate((T1, L1))

        # build Jacobian
        J = self.Jacobian(ts, ls, fsp, L)

        # Solve J.Dx=-X
        Dx = np.linalg.solve(J, -X)
        # split Dx into Dt and Dl arrays
        Dt, Dl = np.split(Dx, 2)

        # build new t and l amplitudes
        tsnew = ts + Dt.reshape(nocc, nvir)
        lsnew = ls + Dl.reshape(nocc, nvir)

        return tsnew, lsnew

    def Gradient_Descent(self, beta, ts, ls, fsp, L):

        nocc, nvir = ts.shape

        # make T1 and L1 eq. vectors and build X
        T1 = self.T1eq(ts, fsp).flatten()
        L1 = self.L1eq(ts, ls, fsp).flatten()
        X = np.concatenate((T1, L1))

        # build Jacobian
        J = self.Jacobian(ts, ls, fsp, L)

        # make ts and ls vectors
        ts = ts.flatten()
        ls = ls.flatten()
        tls = np.concatenate((ts, ls))

        # build new t and l amplitudes
        tlsnew = tls - beta * np.dot(J.transpose(), X)
        tsnew, lsnew = np.split(tlsnew, 2)

        tsnew = tsnew.reshape(nocc, nvir)
        lsnew = lsnew.reshape(nocc, nvir)

        return tsnew, lsnew

if __name__ == "__main__":
    # execute only if run as a script

    np.random.seed(2)

    from pyscf import gto, scf, cc
    import Eris, CC_raw_equations

    mol = gto.Mole()
    #mol.atom = [
    #    [8 , (0. , 0.     , 0.)],
    #    [1 , (0. , -0.757 , 0.587)],
    #    [1 , (0. , 0.757  , 0.587)]]
    mol.atom = """
    H 0 0 0
    H 0 0 1
    """

    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()

    # generalize HF and CC
    mgf = scf.RHF(mol)
    mgf.kernel()
    mgf = scf.addons.convert_to_ghf(mgf)
    mo_occ = mgf.mo_occ
    mocc = mgf.mo_coeff[:, mo_occ > 0]
    mvir = mgf.mo_coeff[:, mo_occ == 0]
    gnocc = mocc.shape[1]
    gnvir = mvir.shape[1]
    gdim = gnocc + gnvir

    mygcc = cc.GCCSD(mgf)
    geris = Eris.geris(mygcc) #mygcc.ao2mo(mgf.mo_coeff)
    fock = geris.fock.copy()

    mccsg = Gccs(geris)

    # random ts and ls amplitudes
    gts = np.random.rand(gnocc//2, gnvir//2)
    gts = utilities.convert_r_to_g_amp(gts)
    gls = np.random.rand(gnocc//2, gnvir//2)
    gls = utilities.convert_r_to_g_amp(gls)
    gfs = np.random.rand(gdim//2, gdim//2)
    gfs = utilities.convert_r_to_g_rdm1(gfs)

    print()
    print('####################################################')
    print(' Test T and L equations using random t1, l1 and f   ')
    print('####################################################')

    T1eq_1 = mccsg.T1eq(gts, gfs)
    T1eq_2 = CC_raw_equations.T1eq(gts, geris)

    La1eq_1 = mccsg.L1eq(gts, gls, gfs)
    La1eq_2 = CC_raw_equations.La1eq(gts, gls, geris)

    print()
    print("---------------------------------")
    print("Difference between intermediates ")
    print("---------------------------------")
    print()

    print('T1 intermediates')
    print('----------------')
    R_St = mccsg.T1inter_Stanton(gts, gfs)
    R_PySCF = mccsg.T1inter_PySCF(gts, gfs)
    R = mccsg.T1inter(gts, gfs)
    for r_st, r_py, r in zip(R_St, R_PySCF, R):
        print('Inter')
        print("With Stanton ")
        print(np.max(np.subtract(r_st, r)))
        print()
        print("With PySCF ")
        print(np.max(np.subtract(r_py, r)))
        print()

    print('Lambda 1 intermediates')
    print('----------------')
    L_St = mccsg.L1inter_Stanton(gls, gfs)
    L = mccsg.L1inter(gls, gfs)
    for l_st, l in zip(L_St, L):
        print('Inter')
        print("With Stanton ")
        print(np.max(np.subtract(l_st, l)))
        print()

    print()
    print("-------------------------------------------------")
    print("Difference between raw eq and with intermediates ")
    print("-------------------------------------------------")
    print()
    print("T1 ")
    print(np.max(np.subtract(T1eq_1, T1eq_2)))
    print()
    print("L1 ")
    print(np.max(np.subtract(La1eq_1, La1eq_2)))
    print()

    print("--------------------------------")
    print(" ts_update with L1 reg          ")
    print("--------------------------------")

    print('ts updated with alpha = 0.')
    inter = mccsg.T1inter(gts, gfs)
    ts_L1 = mccsg.tsupdate_L1(gts, inter, 0.)
    ts_up = mccsg.tsupdate(gts, inter)
    print(np.max(np.subtract(ts_up, ts_L1)))
    print()
    print('ls updated with alpha = 0.')
    inter = mccsg.L1inter(gts, gfs)
    ls_L1 = mccsg.lsupdate_L1(gls, inter, 0.)
    ls_up = mccsg.lsupdate(gts, gls, inter)
    print(np.max(np.subtract(ls_up, ls_L1)))

    print()
    print('####################')
    print(' TEST JACOBIAN      ')
    print('####################')
    print()

    mgrad = ccs_gradient(geris)

    ts_G = gts.copy()*0.1
    ls_G = gls.copy()*0.1
    conv_thre = 1.
    conv = True
    ite = 1
    norm = 0.
    while conv_thre > 10**-5:
        norm_old = norm
        tsnew, lsnew = mgrad.Newton(ts_G, ls_G, gfs, 0)
        #tsnew, lsnew = mgrad.Gradient_Descent(0.01, ts_G, ls_G, gfs, 0)
        ts_G = tsnew
        ls_G = lsnew
        norm = np.concatenate((ts_G.flatten(), ls_G.flatten()))
        conv = np.linalg.norm(norm-norm_old)
        ite += 1
        if ite > 50:
            conv = False
            print("t and l amplitudes NOT converged after {} iteration".format(ite))
            break
    if conv:
        print("Newton's method with Lamdba = 0 from random initial guess")
        print("t and l amplitudes converged after {} iteration".format(ite))
        print()

    ts_G = np.zeros_like(gts)
    ls_G = gls.copy()
    conv_thre = 1.
    ite = 1
    norm = 0.
    while conv_thre > 10**-5:
        norm_old = norm
        tsnew, lsnew = mgrad.Newton(ts_G, ls_G, gfs, 0.5)
        #tsnew, lsnew = mgrad.Gradient_Descent(0.01, ts_G, ls_G, gfs, 0)
        ts_G = tsnew
        ls_G = lsnew
        norm = np.concatenate((ts_G.flatten(), ls_G.flatten()))
        conv = np.linalg.norm(norm-norm_old)
        ite += 1
        if ite > 50:
            conv = False
            print("t and l amplitudes NOT converged after {} iteration".format(ite))
            break
    if conv:
        print("Newton's method with Lamdba = 0.1")
        print("t and l amplitudes converged after {} iteration".format(ite))

    print()
    print("#############################")
    print(" Test different GCCS rdm1    ")
    print("#############################")
    print()

    print('gamma for GS')
    g1 = mccsg.gamma(gts, gls)  # symetrized rdm1
    g2 = mccsg.gamma_unsym(gts, gls)  # unsymetrized rdm1
    print(g2)
    print('Difference in Ek between symetrized and unsymetrized rdm1:')
    print('DEk= ', np.subtract(utilities.Ekin(mol, g1, aobasis=False, mo_coeff=mgf.mo_coeff),
                               utilities.Ekin(mol, g2, aobasis=False, mo_coeff=mgf.mo_coeff)))
    print()
    print(" Difference between GS gamma and ES gamma with r1=0, r0=0")
    g1 = mccsg.gamma_es(gts, gls, np.zeros_like(gts), 0., 1.)
    print(np.max(np.subtract(g1, g2)))

    print()
    print('trace of transition rdm1 ')
    # building random amplitudes vectors
    t1 = np.random.random((gnocc, gnvir))*0.1
    r1 = np.random.random((gnocc, gnvir))*0.1
    l1 = np.random.random((gnocc, gnvir))*0.1
    # orthogonalize r and l amp
    Matvec = np.zeros((gnocc*gnvir, 2))
    Matvec[:, 0] = np.ravel(r1)
    Matvec[:, 1] = np.ravel(l1)
    Matvec = utilities.ortho_QR(Matvec)
    rk = Matvec[:, 0].reshape(gnocc, gnvir)
    ln = Matvec[:, 1].reshape(gnocc, gnvir)
    # get r0 and l0 with E = 0.1
    r0k = mccsg.R0eq(0.1, t1, rk, np.zeros_like(gfs))
    l0n = mccsg.L0eq(0.1, t1, ln, np.zeros_like(gfs))
    tr_rdm1 = mccsg.gamma_tr(t1, ln, rk, r0k, l0n)
    print(tr_rdm1.trace())
    print()
    
    print('trace of rdm1 for excited state - nelec')
    # normalize r and l
    c = utilities.get_norm(r1, l1, r0k, l0n)
    l1 /= c
    # get r0 and l0
    r0 = mccsg.R0eq(0.1, t1, r1, np.zeros_like(gfs))
    l0 = mccsg.L0eq(0.1, t1, l1, np.zeros_like(gfs))
    rdm1 = mccsg.gamma_es(t1, l1, r1, r0, l0)
    print(rdm1.trace()-np.sum(mol.nelec))

    print()
    print("######################################")
    print(" Test R and L intermediates           ")
    print("######################################")
    print()
    
    import CC_raw_equations
    
    vn = np.zeros_like(gfs)
    ls = gls.copy()
    rs = gts.copy()
    r0 = 0.1
    l0 = 0.1

    print('Difference between R1 and L1 inter for t=0 (should be zero)')
    print('-------------------------------------------------------------')
    ts = np.zeros((gnocc, gnvir))
    Rinter = mccsg.R1inter(ts, gfs, vn)    #Fab, Fji, W, F, Zia, Pia
    Linter = mccsg.es_L1inter(ts, gfs, vn) #Fba, Fij, W, F, Zia, P
    for R, L in zip(Rinter, Linter):
        print('Inter')
        print(np.max(np.subtract(R, L)))

    print()
    print('Difference between R1 and L1 equations for t=0 and ls=rs (should be zero)')
    print('-------------------------------------------------------------------------')
    print('with intermediates')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), mccsg.es_L1eq(ls, l0, Linter))))
    print('raw equations')
    print(np.max(np.subtract(CC_raw_equations.R1eq(ts, rs, r0, geris), CC_raw_equations.es_L1eq(ts, ls, l0, geris))))

    print()
    print('Difference between R0 and L0 equations for t=0 and l0=r0 (should be zero)')
    print('-------------------------------------------------------------------------')
    tmp = np.zeros_like(t1)
    En = 0.5
    print('raw equations')
    print(np.max(np.subtract(mccsg.R0eq(En, tmp, rs, gfs), mccsg.L0eq(En, tmp, ls, gfs))))
    print('r0 and l0 update')
    v = np.zeros_like(gfs)
    R0inter = mccsg.R0inter(tmp, gfs, v)
    L0inter = mccsg.L0inter(tmp, gfs, v)
    print(np.max(np.subtract(mccsg.r0update(rs, r0, En, R0inter), mccsg.l0update(ls, l0, En, L0inter))))

    print()
    print('Difference between inter and raw equations for t=0 (should be zero)')
    print('-------------------------------------------------------------------')

    print('R1 difference')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), CC_raw_equations.R1eq(ts, rs, r0, geris))))
    print('L1 difference')
    print(np.max(np.subtract(mccsg.es_L1eq(ls, l0, Linter), CC_raw_equations.es_L1eq(ts, ls, l0, geris))))

    print()
    print('Difference between inter and raw equations for t random (should be zero)')
    print('------------------------------------------------------------------------')

    ts = np.random.random((gnocc//2, gnvir//2))*0.1
    ts = utilities.convert_r_to_g_amp(ts)

    Rinter = mccsg.R1inter(ts, gfs, np.zeros_like(gfs))
    Linter = mccsg.es_L1inter(ts, gfs, np.zeros_like(gfs))

    print('R1 difference')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), CC_raw_equations.R1eq(ts, rs, r0, geris, fsp=gfs))))
    print('L1 difference')
    print(np.max(np.subtract(mccsg.es_L1eq(ls, l0, Linter), CC_raw_equations.es_L1eq(ts, ls, l0, geris, fsp=gfs))))

    print()
    print('Energy from R1 and R0 equations')
    print('-------------------------------------------------------')
    print()

    r1, DE = utilities.koopman_init_guess(mgf.mo_energy, mgf.mo_occ, (2, 0))
    ts = np.zeros_like(r1[0])
    vm = np.zeros_like(gfs)
    print('State 1')
    r0 = mccsg.Extract_r0(r1[0], ts, None, vm)
    print('r0= ', r0)
    Rinter = mccsg.R1inter(ts, gfs, vm)
    print('Koopman DE= ', DE[0])
    print('E= ', mccsg.Extract_Em_r(r1[0], r0, Rinter)[0])
    print()
    print('State 2')
    r0 = mccsg.Extract_r0(r1[1], ts, None, vm)
    print('r0= ', r0)
    print('Koopman DE= ', DE[1])
    print('E= ', mccsg.Extract_Em_r(r1[1], r0, Rinter)[0])
    print()

    # Note: R and L intermediates are the same for HF basis (f off diag = 0) --> lambda = 0
    #       except the Zia intermediates, which contracts with r0 and l0
    #       note anymore when lambda > 0
    # Fab, Fji, W, F, Zia, Pia
