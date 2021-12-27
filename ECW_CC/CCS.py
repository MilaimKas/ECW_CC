#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
File containing all CC related functions
 - Lambda1, T1, L1, R1, L0 and R0 factorized equations
 - Corresponding intermediates
 - one-particle reduced density matrix
 - EXC-CCS Jacobian and gradient methods
"""

import copy
import sys

import numpy as np
import utilities

############################
# reduced density matrix
############################


def gamma_unsym_CCS(ts, ls):
    """
    Unsymmetrized one-particle reduced density matrix CCS
    Same as gamma_es with r=0 and r0=1 (GS case)

    :param ts: t1 amplitudes
    :param ls: l1 amplitudes
    :return:
    """

    nocc, nvir = ts.shape

    doo = -np.einsum('ie,je->ij', ts, ls)
    dvv = np.einsum('ib,ia->ab', ts, ls)
    dvo = ls.transpose()
    dov = -np.einsum('ja,ib,jb->ia', ts, ts, ls) + ts

    dm1 = np.empty((nocc + nvir, nocc + nvir))
    dm1[:nocc, :nocc] = doo
    dm1[:nocc, nocc:] = dov
    dm1[nocc:, :nocc] = dvo
    dm1[nocc:, nocc:] = dvv

    dm1[np.diag_indices(nocc)] += 1

    return dm1


def gamma_es_CCS(ts, ln, rk, r0k, l0n):
    """
    Unsymmetrized CCS one-particle reduced density matrix

    GS case: ln = lambda, rn = l0n = 0 and r0 = 1
    ES case: rk = rn and r0k = r0n
    transition rdm: <Psi_n|ap.aq|Psi_k>

    :param ts: t1 amplitudes
    :param ln: l1 amplitudes or lambda 1 amplitudes
    :param rk: r1 amplitudes
    :param r0k: r0 amplitude
    :return:
    """

    nocc, nvir = ts.shape

    # GS case:
    if rk is None or isinstance(rk, float) or isinstance(rk, int):
        rk = np.zeros_like(ts)
        r0k = 1.
        l0n = 0.

    # gamma_ij
    doo = -r0k * np.einsum('ie,je->ij', ts, ln)
    doo -= np.einsum('ie,je->ij', rk, ln)

    # gamma_ai
    dvo = r0k * ln.transpose()

    # gamma_ab
    dvv = r0k * np.einsum('mb,ma->ab', ts, ln)
    dvv += np.einsum('mb,ma->ab', rk, ln)

    # gamma_ia
    tmp = np.einsum('ja,jb->ab', ts, ln)
    dov = -r0k * np.einsum('ib,ab->ia', ts, tmp)
    dov -= np.einsum('ma,ie,me->ia', ts, rk, ln)
    dov -= np.einsum('ie,ma,me->ia', ts, rk, ln)
    dov += ts
    dov += l0n * rk

    dm1 = np.empty((nocc + nvir, nocc + nvir))
    dm1[:nocc, :nocc] = doo
    dm1[:nocc, nocc:] = dov
    dm1[nocc:, :nocc] = dvo
    dm1[nocc:, nocc:] = dvv

    # G format: Hartree Fock term
    dm1[np.diag_indices(nocc)] += 1

    return dm1


def gamma_tr_CCS(ts, ln, rk, r0k, l0n):
    """
    Unsymmetrized CCS transition rdm: <Psi_n|ap.aq|Psi_k>
    same as gamma_es without HF term (1 in diagonal oo bloc)

    GS cases:
    <Psi_0|ap.aq|Psi_k>: ln = lambda and l0n = 0
    <Psi_n|ap.aq|Psi_0>: rk = 0 and r0k = 1

    :param ts: t1 amplitudes
    :param ln: l1 amplitudes or lambda1 amplitudes
    :param rk: r1 amplitudes
    :param r0k: r0 amplitude or 0 or None (for the GS case)
    :param l0n: l0 amplitude or 1 (GS case)
    :return:
    """

    nocc, nvir = ts.shape

    # GS case:
    if rk is None or isinstance(rk, float) or isinstance(rk, int) or r0k is None:
        rk = np.zeros_like(ts)
        r0k = 1.

    # gamma_ij
    doo = -r0k * np.einsum('ie,je->ij', ts, ln)
    doo -= np.einsum('ie,je->ij', rk, ln)

    # gamma_ai
    dvo = r0k * ln.transpose()

    # gamma_ab
    dvv = r0k * np.einsum('mb,ma->ab', ts, ln)
    dvv += np.einsum('mb,ma->ab', rk, ln)

    # gamma_ia
    tmp = np.einsum('ja,jb->ab', ts, ln)
    dov = -r0k * np.einsum('ib,ab->ia', ts, tmp)
    dov -= np.einsum('ma,ie,me->ia', ts, rk, ln)
    dov -= np.einsum('ie,ma,me->ia', ts, rk, ln)
    dov += ts
    dov += l0n * rk

    dm1 = np.empty((nocc + nvir, nocc + nvir))
    dm1[:nocc, :nocc] = doo
    dm1[:nocc, nocc:] = dov
    dm1[nocc:, :nocc] = dvo
    dm1[nocc:, nocc:] = dvv

    return dm1


def gamma_CCS(ts, ls):
    """
    Symmetrized one-particle reduced density matrix CCS from PySCF with t2=l2=0

    :param ts: t1 amplitudes
    :param ls: l1 amplitudes from the GS
    :return: symmetrized rdm1
    """

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
        """
        All equations are given in spin-orbital basis

        :param eris: two electron integrals in Physics notation (<pq||rs>= <pq|rs> - <pq|sr>)
        :param fock: fock matrix in MOs basis
        :param M_tot: number of measurements
        """

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
                if v is not None:
                    v_ov = -v[:nocc, nocc:]
                    e += np.einsum('ia,ia', v_ov, rs)
                    e += r0*np.einsum('ia,ia', v_ov, ts)
                    e += r0*np.einsum('jj', -v[:nocc, :nocc])

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
        
        Fab, Fji, Fai = self.T1inter(ts, fsp)

        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ib,ab->ia', ts, Fab)
        T1 -= np.einsum('ja,ji->ia', ts, Fji)

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

        Fab, Fji, Fai = T1inter
        nocc, nvir = ts.shape

        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # remove diagonal of the fock matrix
        Fab[np.diag_indices(nvir)] -= diag_vv
        Fji[np.diag_indices(nocc)] -= diag_oo

        # update ts
        tsnew = np.einsum('ai->ia', Fai)
        tsnew += np.einsum('ib,ab->ia', ts, Fab)
        tsnew -= np.einsum('ja,ji->ia', ts, Fji)
        
        # add coupling terms with excited states
        if rsn is not None:
            if r0n is None:
                raise ValueError('if Vexp are to be calculated, list of r0 amp must be given')
            if len(vn) != len(rsn):
                raise ValueError('Number of experimental potentials must be equal to number of r amplitudes')
            for r, v, r0 in zip(rsn, vn, r0n):
                if v is not None:

                    v_oo = -v[:nocc, :nocc]
                    v_vv = -v[nocc:, nocc:]
                    v_ov = -v[:nocc, nocc:]

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
        # todo: does not give the same result as tsupdate for alpha = 0
        """
        SCF+L1 (regularization) update of the t1 amplitudes

        :param ts: t1 amplitudes
        :param T1inter: T1 intermediates
        :param alpha: L1 regularization parameter
        :return: updated ts
        """

        Fab, Fji, Fai = T1inter
        nocc, nvir = ts.shape

        # store diagonals
        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # T1 eq
        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ib,ab->ia', ts, Fab)
        T1 -= np.einsum('ja,ji->ia', ts, Fji)

        # subdifferential
        dW = utilities.subdiff(T1, ts, alpha)

        # remove diagonal elements
        eia = diag_oo[:, None] - diag_vv
        dW += ts*eia
        dW /= eia

        return dW

    def tsupdate_PySCF(self, cc, ts, fsp):
        """
        PySCF module for ts update
        For comparison purpose

        :return:
        """

        from pyscf.cc import gccsd

        nocc, nvir = ts.shape
        t2 = np.zeros((nocc, nocc, nvir, nvir))

        eris = gccsd._make_eris_outcore(cc)
        eris.fock = fsp

        tsnew = gccsd.update_amps(cc, ts, t2, eris)[0]

        return tsnew

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
        Fai += np.einsum('jb,jabi->ai', ts, self.eris.ovvo)

        Fab = fvv.copy()
        Fab -= np.einsum('jb,ja->ab', fov, ts)
        Fab += np.einsum('jc,jacb->ab', ts, self.eris.ovvv)

        Fji = foo.copy()
        Fji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        tmp = np.einsum('kc,jkcb->jb', ts, self.eris.oovv)
        Fji -= np.einsum('ib,jb->ji', ts, tmp)

        return Fab, Fji, Fai

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

    def L1eq(self, ts, ls, fsp, E_term=True):
        """
        Value of the Lambda 1 equations using intermediates

        :param ts: t1 amplitudes
        :param ls: l1 amplitudes
        :param fsp: fock matrix
        :param E_term: False if energy term in Lambda eq is zero
        :return: Lambda1 value
        """
        
        Fia, Fba, Fij, Wbija, E = self.L1inter(ts, fsp, E_term=E_term)

        L1 = Fia.copy()
        L1 += np.einsum('ib,ba->ia', ls, Fba)
        L1 -= np.einsum('ja,ij->ia', ls, Fij)
        L1 += np.einsum('jb,bija->ia', ls, Wbija)
        L1 += ls*E

        return L1

    def lsupdate(self, ts, ls, L1inter, rsn=None, lsn=None, r0n=None, l0n=None, vn=None):
        """
        SCF update of the lambda singles amplitudes ls

        :param rsn: list with r amplitudes associated to excited state n
        :param lsn: list with l amplitudes associated to excited state n
        :param vn: exp potential Vexp[n,0]
        :return lsnew: updated lambda 1 values
        """

        Fia, Fba, Fij, Wbija, E = L1inter

        nocc, nvir = ls.shape

        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        # remove diagonal of the fock matrix
        Fba[np.diag_indices(nvir)] -= diag_vv
        Fij[np.diag_indices(nocc)] -= diag_oo

        lsnew = Fia.copy()
        lsnew += np.einsum('ib,ba->ia', ls, Fba)
        lsnew -= np.einsum('ja,ij->ia', ls, Fij)
        lsnew += np.einsum('jb,bija->ia', ls, Wbija)
        lsnew += ls*E

        # add terms from coupling to excited states
        if rsn is not None:

            # check length
            if len(lsn) != len(rsn) or len(vn) != len(rsn):
                raise ValueError('v0n, l and r list must be of same length')
            if r0n is None or l0n is None:
                raise ValueError('r0 and l0 values must be given')

            for r, l, v, r0, l0 in zip(rsn, lsn, vn, r0n, l0n):

                if v is not None:

                    v_oo = -v[:nocc, :nocc]
                    v_vv = -v[nocc:, nocc:]
                    v_ov = -v[:nocc, nocc:]

                    # P_lam intermediate
                    Pl = np.einsum('jb,jb', r, v_ov)
                    Pl += r0*np.einsum('jb,jb', ts, v_ov)
                    Pl += r0*np.trace(v_oo)

                    # P_0 intermediate => v_ov

                    # P intermediate
                    P = np.sum(np.diag(v_oo))
                    P += np.einsum('jb,jb', ts, v_ov)

                    # Pba intermediate
                    Pba = v_vv.copy()
                    Pba -= np.einsum('jb,ja->ba', ts, v_ov)

                    # Pij intermediate
                    Pij = -v_oo.copy()
                    Pij -= np.einsum('jb,ib->ij', ts, v_ov)

                    # add Vexp terms
                    lsnew += ls*Pl
                    lsnew += l0*v_ov
                    lsnew += l*P
                    lsnew += np.einsum('ib,ba->ia', l, Pba)
                    lsnew += np.einsum('ja,ij->ia', l, Pij)

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

        Fia, Fba, Fij, Wbija, E = L1inter

        nocc, nvir = ls.shape

        # store diagonals
        diag_vv = np.diagonal(self.fock[nocc:, nocc:])
        diag_oo = np.diagonal(self.fock[:nocc, :nocc])

        L1 = Fia.copy()
        L1 += np.einsum('ib,ba->ia', ls, Fba)
        L1 -= np.einsum('ja,ij->ia', ls, Fij)
        L1 += np.einsum('jb,bija->ia', ls, Wbija)
        L1 += ls * E

        # subdifferential
        dW = utilities.subdiff(L1, ls, alpha)

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
                self.stdout = sys.stdout
                self.verbose = 0
                self.level_shift = 0

        nocc, nvir = ts.shape
        t2 = np.zeros((nocc, nocc, nvir, nvir))
        l2 = np.zeros_like(t2)

        cc = tmp()
        tmp_eris = copy.deepcopy(self.eris)
        tmp_eris.fock = fsp
        tmp_eris.mo_energy = fsp.diagonal()

        imds = gccsd_lambda.make_intermediates(cc, ts, t2, tmp_eris)
        lsnew = gccsd_lambda.update_lambda(cc, ts, t2, ls, l2, tmp_eris, imds)[0]

        return lsnew

    # @profile
    def L1inter(self, ts, fsp, E_term=True):
        """
        Lambda 1 intermediates

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix in spin-orbital MO basis
        :param E_term: False is energy term is 0.
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
        tmp = np.einsum('jkca,jc->ka', self.eris.oovv, ts)
        Fba -= np.einsum('ka,kb->ba', tmp, ts)

        Fij = foo.copy()
        Fij += np.einsum('ib,jb->ij', fov, ts)
        Fij += np.einsum('kibj,kb->ij', self.eris.oovo, ts)
        tmp = np.einsum('kibc,kb->ic', self.eris.oovv, ts)
        Fij += np.einsum('ic,jc->ij', tmp, ts)

        Wbija = self.eris.voov.copy()
        Wbija -= np.einsum('kija,kb->bija', self.eris.ooov, ts)
        tmp = np.einsum('kica,kb->icab', self.eris.oovv, ts)
        Wbija -= np.einsum('icab,jc->bija', tmp, ts)
        Wbija += np.einsum('bica,jc->bija', self.eris.vovv, ts)

        Fia = fov.copy()
        Fia += np.einsum('jiba,jb->ia', self.eris.oovv, ts)

        # energy term
        if E_term:
            E = -np.einsum('jb,jb', ts, fov)
            E -= 0.5*np.einsum('jb,kc,jkbc', ts, ts, self.eris.oovv)
        else:
            E = 0.

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
        Zji -= np.einsum('kb,ijkb->ji', ts, tmp)
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
        if vm is None:
            Pia = np.zeros_like(Tia)
        else:
            v_vo = -vm[nocc:, :nocc]
            v_vv = -vm[nocc:, nocc:]
            v_oo = -vm[:nocc, :nocc]
            Pia = v_vo.copy()
            Pia += np.einsum('ab,ib->ai', v_vv, ts)
            Pia -= np.einsum('ii,ja,ib->ai', v_oo, ts, ts)
            Pia = np.einsum('ai->ia', Pia)

        return Fab, Fji, Wakic, Er, Tia, Pia

    def Extract_Em_r(self, rs, r0, Rinter, ov=None):
        """
        Extract Em from the largest r1 element

        :param ov: index for initial Koopman excitation. If not given, takes the largest element
        :param rs: r1 amplitude of state m
        :param r0: r0 amplitude of state m
        :param Rinter: R1 intermediates
        :return: Em and index of largest r1 element
        """

        Fab, Fji, W, F, Zia, Pia = Rinter

        # Ria = ria*En' matrix
        Ria = np.einsum('ab,ib->ia', Fab, rs)
        Ria -= np.einsum('ji,ja->ia', Fji, rs)
        Ria += np.einsum('akic,kc->ia', W, rs)

        # largest r1 if indices not given
        if ov is None:
            o, v = np.unravel_index(np.argmax(abs(rs), axis=None), rs.shape)
        else:
            o, v = ov

        Rov = Ria[o, v]
        del Ria

        Rov += rs[o, v] * F
        Rov += r0 * Zia[o, v]
        Rov += Pia[o, v]
        Em = Rov/rs[o, v]

        return Em, o, v

    def rsupdate(self, rs, r0, Rinter, Em, force_alpha=True):
        """
        Update r1 amplitudes using Ria equations and given E for iteration k

        :param force_alpha: make all beta excitation 0
        :param rs: matrix of r1 amplitudes for state m at k-1 iteration
        :param r0: r0 amplitude for state m at iteration k-1
        :param Rinter: r1 intermediates for state m
        :param Em: Energy of state m
        :return: updated list of r1 amplitudes

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
        rsnew /= (Em + diag_oo[:, None] - diag_vv)

        if force_alpha:
            rsnew[0::2, :] = 0.  # force alpha transition

        return rsnew

    def get_ov(self, ls, l0, rs, r0, ind):
        """
        Extract missing ria value from normality condition

        :param ls: l1 amplitudes for state m
        :param l0: l0 amplitude for state m
        :param rs: r1 amplitudes for state m
        :param r0: r0 amplitude for state m
        :param ind: index of missing rov amplitude
        :return: updated rov
        """

        o, v = ind
        r = rs.copy()
        r[o, v] = 0.
        rov = 1. - r0 * l0 - np.einsum('ia,ia', r, ls)
        rov /= ls[o, v]

        return rov

    def R1eq(self, rs, r0, Rinter):
        """
        Return the Ria values

        :param rs: r1 amplitudes
        :param r0: r0 amplitude
        :param Rinter: R1 intermediates
        :return: Ria
        """

        Fab, Fji, W, F, Tia, Pia = Rinter

        # Ria = ria*En' matrix
        Ria = np.einsum('ab,ib->ia', Fab, rs)
        Ria -= np.einsum('ji,ja->ia', Fji, rs)
        Ria += np.einsum('akic,kc->ia', W, rs)
        Ria += rs * F
        Ria += r0 * Tia
        Ria += Pia

        return Ria

    def R0inter(self, ts, fsp, vm):
        """
        one and two particles intermediates for state m as well as the Vexp intermediate
        for the R0 equations

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix for the state m
        :param vm: m0V potential
        :return: Fjb, Zjb, P intermediates
        """

        nocc, nvir = ts.shape

        if fsp is None:
            fsp = self.fock.copy()

        fov = fsp[:nocc, nocc:].copy()

        # (commented lines are additional term when f=kin)

        # r intermediates
        # ------------------

        # Fjb: equation (23)
        Fjb = fov.copy()
        # Fjb += np.einsum('jkbk->jb',self.eris.oovo)
        # tmp = Fjb.copy()
        Fjb += np.einsum('kc,kjcb->jb', ts, self.eris.oovv)

        # r0 intermediates (energy term)
        # -------------------------------

        # Zjb: equation (25) --> Same as Fjb in R1inter
        # Zjb and ts are contracted here to form E
        Zjb = fov.copy()
        Zjb += 0.5 * np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        E = np.einsum('jb,jb', ts, Zjb)
        del Zjb

        # Vexp intermediate P
        # ---------------------

        vm_oo = vm[:nocc, :nocc]
        vm_ov = vm[:nocc, nocc:]
        P = np.einsum('jj', vm_oo)
        P += np.einsum('jb,jb', ts, vm_ov)

        return Fjb, E, P

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
        R1 += r1 * F
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
            r0_1 = (-b + np.sqrt((b ** 2) - (4 * a * c))) / c
            r0_2 = (-b - np.sqrt((b ** 2) - (4 * a * c))) / c

            if r0_1 > 0:
                return r0_1
            elif r0_2 > 0:
                return r0_2
            else:
                raise ValueError('Both solution for r0 are negative')

    def r0update(self, rs, r0, Em, R0inter):
        """
        SCF update of the r0 amplitude

        :param rs: r1 amplitude from the previous iteration
        :param Em: energy of state m
        :param R0inter: intermediates for the R0 equation
        :return: updated r0 for state m
        """

        Fjb, E, P = R0inter
        F = np.einsum('jb,jb', rs, Fjb)
        r0new = F+P+(r0*E)
        r0new /= Em

        return r0new

    def R0eq(self, rs, r0, R0inter):
        """
        Returns the E*r0 from R0 equation using intermediates

        :param rs: r1 amplitude for state m
        :param r0: r0 coefficient for state m
        :param R0inter: R0 intermediates
        :return:
        """

        Fjb, E, P = R0inter

        R0 = np.einsum('jb,jb', rs, Fjb)
        R0 += r0*E
        R0 += P

        return R0

    def r0_fromE(self, En, t1, r1, vm0, fsp=None):
        """
        Returns the r0 value for a CCS state with given energy from the R0 equation

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

        if vm0 is not None:
            vov = -vm0[:nocc, nocc:].copy()
            voo = -vm0[:nocc, :nocc].copy()
        else:
            vov = np.zeros((nocc, nvir))
            voo = np.zeros((nocc, nocc))

        fov = fsp[:nocc, nocc:].copy()

        # (commented lines are additional term when f=kin)
        
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
        # Fij += np.einsum('ikjk->ij', self.eris.oooo)  ##
        Fij += np.einsum('jb,ib->ij', ts, fov)
        Fij += np.einsum('kb,kibj->ij', ts, self.eris.oovo)
        # Fij += np.einsum('jb,ikbk->ij', ts, self.eris.oovo) ##
        Fij += np.einsum('kb,jc,kibc->ij', ts, ts, self.eris.oovv)

        # Wbija: equation (32)
        W = self.eris.voov.copy()
        W -= np.einsum('kb,kija->bija', ts, self.eris.ooov)
        W += np.einsum('jc,bica->bija', ts, self.eris.vovv)
        W -= np.einsum('jc,kb,kica->bija', ts, ts, self.eris.oovv)

        # El: equation (33) --> same as for R1inter (energy term)
        Fjb = fov.copy()
        # Fjb += np.einsum('jkbk->jb',self.eris.oovo)  ##
        Fjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        El = np.einsum('jb,jb', ts, Fjb)
        del Fjb

        # l0 intermediate
        # ------------------

        Zia  = fov.copy()
        # Zia += np.einsum('ikak->ia', self.eris.oovo)  ##
        Zia += np.einsum('jb,jiba->ia', ts, self.eris.oovv)

        # Vexp intermediate P
        # ---------------------

        if vm is None:
            P = np.zeros((nocc, nvir))
        else:
            P = -vm[:nocc, nocc:].copy()

        return Fba, Fij, W, El, Zia, P

    def L0inter(self, ts, fsp, vm):
        '''
        L0 intermediates for the L0 equation of state m

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix containing the Vmm potential
        :param vm: V0m coupling potential
        :return: L0 intermediates
        '''
       
        nocc, nvir = ts.shape

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
        #Zjb -= np.einsum('jkbk->jb', self.eris.oovo)
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
        """
        Use L1 and L0 equations to calculate l0 from given r1

        :param l1: l1 amplitude vector
        :return: r0
        """

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

    def es_lsupdate(self, ls, l0, Em, L1inter, force_alpha=True):
        """
        Update the l1 amplitudes for state m

        :param force_alpha: force beta transition to be 0
        :param ls: list of l amplitudes for the m excited state
        :param l0: l0 amplitude for state m
        :param Em: Energy of the state m
        :param L1inter: intermediates for the L1 equation of state m
        :return: updated matrix of ls amplitudes for state m
        """

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
        lsnew /= (Em + diag_oo[:, None] - diag_vv)

        if force_alpha:
            lsnew[0::2, :] = 0.  # force alpha transition

        return lsnew

    def es_L1eq(self, ls, l0, es_L1inter):
        """
        Return the value of Lia (lia*E)

        :param ls: l1 amplitudes
        :param l0: l0 amplitude
        :param es_L1inter: intermediates for the L1 equation of state m
        :return:
        """

        Fba, Fij, W, El, Zia, P = es_L1inter

        # get lia
        Lia  = np.einsum('ib,ba->ia', ls, Fba)
        Lia -= np.einsum('ja,ij->ia', ls, Fij)
        Lia += np.einsum('jb,bija->ia', ls, W)
        Lia += ls*El
        Lia += l0*Zia
        Lia += P

        return Lia

    def l0update(self, ls, l0, Em, L0inter):
        """
        Update the l0 amplitude

        :param ls: l1 amplitudes for state m
        :param Em: energy for state m
        :param L0inter: L1 intermediates for state m
        :return:
        """

        Fjb, Wjb, Z, P = L0inter
        F = np.einsum('jb,bj', ls, Fjb)
        W = np.einsum('jb,jb', ls, Wjb)
        l0new = F+W+P+(l0*Z)
        l0new /= Em

        return l0new

    def L0eq(self, ls, l0, L0inter):
        """
        Returns Lia (lia*E) values

        :param ls: l1 amplitudes for state m
        :param L0inter: L1 intermediates for state m
        :return: Lia
        """

        Fbj, Wjb, El, P = L0inter

        Lia = np.einsum('jb,bj', ls, Fbj)
        Lia += np.einsum('jb,jb', ls, Wjb)
        Lia += l0*El
        Lia += P

        return Lia

    def l0_fromE(self, En, t1, l1, v0m, fsp=None):
        """
        Returns l0 term for a CCS state
        See equation 23 in Derivation_ES file

        :param En: correlation energy (En+EHF=En_tot) of the state n
        :param t1: t1 amp
        :param l1: l1 amp of state n
        :param v0m: contraint potential matrix V0m
        :param fsp: fock matrix
        :return: l0
        """

        nocc, nvir = t1.shape

        if fsp is None:
            fsp = self.fock.copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()
        foo = fsp[:nocc, :nocc].copy()

        if v0m is not None:
            vov = v0m[:nocc, nocc:]
            voo = v0m[:nocc, :nocc]
        else:
            vov = np.zeros((nocc, nvir))
            voo = np.zeros((nocc, nocc))


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

#####################################
#   ECW-GCCS old gradient equations #
#####################################

class ccs_gradient:
    def __init__(self, eris, Vexp_model=1, exp_pot=None):
        """
        Gradient of the ECW-CCS equations and Newton's method

        :param eris: two electron integrals
        :param exp_pot: exp_pot class containing the exp_data and needed MO matrices
        :param Vexp_model: form of the Vexp model
        """

        self.fock = eris.fock
        self.eris = eris

        self.nocc = eris.nocc
        self.nvir = self.fock.shape[0]-self.nocc

        # form of Vexp
        if Vexp_model == 1:
            self.DV = self.DV1
        elif Vexp_model == 2:
            self.DV = self.DV2
            if exp_pot is None:
                raise ValueError('exp_pot class is needed')
            self.exp_data = exp_pot.exp_data[0, 0]  # ['Aj':Aj]
            self.A = exp_pot.dic_int  # {'Aj':<p|Aj|q>, ...}
        elif Vexp_model == 3:
            self.DV = self.DV3
            if exp_pot is None:
                raise ValueError('exp_pot class is needed')
            self.exp_data = exp_pot.exp_data[0, 0]  # ['Aj':Aj]
            self.A = exp_pot.dic_int  # {'Aj':<p|Aj|q>, ...}
        else:
            raise ValueError('Vexp model is 1,2 or 3')

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
        Fai += np.einsum('jb,jabi->ai', ts, self.eris.ovvo)

        Fab = fvv.copy()
        Fab -= np.einsum('jb,ja->ab', fov, ts)
        Fab += np.einsum('jc,jacb->ab', ts, self.eris.ovvv)

        Fji = foo.copy()
        Fji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        tmp = np.einsum('kc,jkcb->jb', ts, self.eris.oovv)
        Fji -= np.einsum('ib,jb->ji', ts, tmp)

        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ib,ab->ia', ts, Fab)
        T1 -= np.einsum('ja,ji->ia', ts, Fji)

        return T1

    #################
    # L1 equations
    #################

    def L1eq(self, ts, ls, fsp, E_term=False):
        """
        Value of the Lambda 1 equations using intermediates

        :param ts: t1 amplitudes
        :param ls: l1 amplitudes
        :param fsp: fock matrix
        :param E_term: False if energy term in Lambda eq is zero
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
        Fba -= np.einsum('ja,jb->ba', fov, ts)
        Fba += np.einsum('jbca,jc->ba', self.eris.ovvv, ts)
        tmp = np.einsum('jkca,jc->ka', self.eris.oovv, ts)
        Fba -= np.einsum('ka,kb->ba', tmp, ts)

        Fij = foo.copy()
        Fij += np.einsum('ib,jb->ij', fov, ts)
        Fij += np.einsum('kibj,kb->ij', self.eris.oovo, ts)
        tmp = np.einsum('kibc,kb->ic', self.eris.oovv, ts)
        Fij += np.einsum('ic,jc->ij', tmp, ts)

        Wbija = self.eris.voov.copy()
        Wbija -= np.einsum('kija,kb->bija', self.eris.ooov, ts)
        tmp = np.einsum('kica,kb->icab', self.eris.oovv, ts)
        Wbija -= np.einsum('icab,jc->bija', tmp, ts)
        Wbija += np.einsum('bica,jc->bija', self.eris.vovv, ts)

        Fia = fov.copy()
        Fia += np.einsum('jiba,jb->ia', self.eris.oovv, ts)

        # energy term
        if E_term:
            E = -np.einsum('jb,jb', ts, fov)
            E -= 0.5*np.einsum('jb,kc,jkbc', ts, ts, self.eris.oovv)
        else:
            E = 0.

        L1 = Fia.copy()
        L1 += np.einsum('ib,ba->ia', ls, Fba)
        L1 -= np.einsum('ja,ij->ia', ls, Fij)
        L1 += np.einsum('jb,bija->ia', ls, Wbija)
        L1 += ls * E

        return L1

    ###############################
    # T1 and Lambda 1 derivatives
    ###############################

    def dT(self, ts, ls, fsp, L):
        """
        dTai/dtog = dTdt_pq and dTai/dlog = dTdl_pq

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix
        :return:
        """
        nocc, nvir = ts.shape

        # iaog -> xy
        dTdt_pq = np.zeros((nocc * nvir, nocc * nvir))
        dTdl_pq = np.zeros((nocc * nvir, nocc * nvir))

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

        dVt, dVl = self.DV(ts, ls)
        dVoo_t = dVt[:nocc, :nocc, :, :].copy()
        dVvv_t = dVt[nocc:, nocc:, :, :].copy()
        dVov_t = dVt[:nocc, nocc:, :, :].copy()
        dVoo_l = dVl[:nocc, :nocc, :, :].copy()
        dVvv_l = dVl[nocc:, nocc:, :, :].copy()
        dVov_l = dVl[:nocc, nocc:, :, :].copy()

        # dio terms
        dio = fvv.copy()
        dio -= np.einsum('jg,ja->ag', fov, ts)
        dio += np.einsum('jc,jacg->ag', ts, self.eris.ovvv)
        tmp = np.einsum('kc,jkcg->jg', ts, self.eris.oovv)
        dio += np.einsum('ja,jg->ag', ts, tmp)

        # dag terms
        dag = -foo.copy()
        dag -= np.einsum('ob,ib->io', fov, ts)
        dag -= np.einsum('kb,kobi->io', ts, self.eris.oovo)
        tmp = np.einsum('kc,okcb->ob', ts, self.eris.oovv)
        dag += np.einsum('ib,ob->io', ts, tmp)

        # Vexp terms
        # dVdt
        Viaog_t = dVov_t.copy()
        Viaog_t -= np.einsum('abog,ib->iaog', dVvv_t, ts)
        Viaog_t += np.einsum('jiog,ja->iaog', dVoo_t, ts)
        tmp = np.einsum('jbog,ja->boga', dVov_t, ts)
        Viaog_t += np.einsum('boga,ib->iaog', tmp, ts)
        # dVdl
        Viaog_l = dVov_l.copy()
        Viaog_l -= np.einsum('abog,ib->iaog', dVvv_l, ts)
        Viaog_l += np.einsum('jiog,ja->iaog', dVoo_l, ts)
        tmp = np.einsum('jbog,ja->boga', dVov_l, ts)
        Viaog_l += np.einsum('boga,ib->iaog', tmp, ts)

        # oagi terms
        oagi = self.eris.ovvo.copy()
        oagi -= np.einsum('ja,ojgi->oagi', ts, self.eris.oovo)
        oagi += np.einsum('ib,oagb->oagi', ts, self.eris.ovvv)
        tmp = np.einsum('ja,jogb->ogba', ts, self.eris.oovv)
        oagi += np.einsum('ib,ogba->oagi', ts, tmp)
        del tmp

        # todo: vectorize
        for p in range(0, nvir * nocc):
            for q in range(0, nocc * nvir):
                o, g = np.unravel_index(q, (nocc, nvir))
                i, a = np.unravel_index(p, (nocc, nvir))

                if o == i:
                    dTdt_pq[p, q] += dio[a, g]
                if a == g:
                    dTdt_pq[p, q] += dag[i, o]

                dTdt_pq[q, p] += L*Viaog_t[i, a, o, g]
                dTdt_pq[q, p] += L*oagi[o, a, g, i]

                dTdl_pq[q, p] = L*Viaog_l[i, a, o, g]

        return dTdt_pq, dTdl_pq

    def dL(self, ts, ls, fsp, L):
        """
        dLai/dtog and dLai/dlog

        :param ts: t1 amplitudes
        :param ls: lambda 1 amplitudes
        :param fsp: effective fock matrix
        :return:
        """
        nocc, nvir = ts.shape

        # aiog -> pq
        dLdt_pq = np.zeros((nocc * nvir, nocc * nvir))
        dLdl_pq = np.zeros((nocc * nvir, nocc * nvir))

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

        dVt, dVl = self.DV(ts, ls)
        dVoo_t = dVt[:nocc, :nocc, :, :].copy()
        dVvv_t = dVt[nocc:, nocc:, :, :].copy()
        dVov_t = dVt[:nocc, nocc:, :, :].copy()
        dVoo_l = dVl[:nocc, :nocc, :, :].copy()
        dVvv_l = dVl[nocc:, nocc:, :, :].copy()
        dVov_l = dVl[:nocc, nocc:, :, :].copy()

        # dio terms
        dio = fvv.copy()
        dio -= np.einsum('jg,ja->ag', ts, fov)
        dio -= np.einsum('jgba,jb->ag', self.eris.ovvv, ts)
        tmp = np.einsum('kjba,jb->ka', self.eris.oovv, ts)
        dio += np.einsum('ka,kg->ag', tmp, ts)

        # dag terms
        dag = -foo.copy()
        dag -= np.einsum('ob,ib->io', ts, fov)
        dag -= np.einsum('jibo,jb->io', self.eris.oovo, ts)
        tmp = np.einsum('kibc,kc->ib', self.eris.oovv, ts)
        dag += np.einsum('ib,ob->io', tmp, ts)

        # diodag terms (energy terms)
        diodag = - np.einsum('jb, jb', ts, fov)
        tmp = np.einsum('jkbc, kc->jb', self.eris.oovv, ts)
        diodag -= 0.5*np.einsum('jb, jb', ts, tmp)

        # Vexp terms
        # dVdl
        Viaog_l = -dVov_l.copy()
        Viaog_l -= np.einsum('baog,ib->iaog', dVvv_l, ls)
        Viaog_l += np.einsum('ijog,ja->iaog', dVoo_l, ls)
        tmp = np.einsum('jaog,jb->aogb', dVov_l, ts)
        Viaog_l += np.einsum('aogb,ib->iaog', tmp, ls)
        tmp = np.einsum('ja,jb->ab', ls, ts)
        Viaog_l += np.einsum('ab,ibog->iaog', tmp, dVov_l)
        Viaog_l += np.einsum('ia, jbog, jb->iaog', ls, dVov_l, ts)  # energy term
        # dVdt
        Viaog_t = -dVov_t.copy()
        Viaog_t -= np.einsum('baog,ib->iaog', dVvv_t, ls)
        Viaog_t += np.einsum('ijog,ja->iaog', dVoo_t, ls)
        tmp = np.einsum('jaog,jb->aogb', dVov_t, ts)
        Viaog_t += np.einsum('aogb,ib->iaog', tmp, ls)
        tmp = np.einsum('ja,jb->ab', ls, ts)
        Viaog_t += np.einsum('ab,ibog->iaog', tmp, dVov_t)
        Viaog_t += np.einsum('ia, jbog, jb->iaog', ls, dVov_t, ts)  # energy term

        # gioa terms for dLdl
        gioa_l = self.eris.voov.copy()
        gioa_l += np.einsum('giba,ob->gioa', self.eris.vovv, ts)
        tmp = np.einsum('jica,oc->jiao', self.eris.oovv, ts)
        gioa_l -= np.einsum('jiao,jg->gioa', tmp, ts)
        gioa_l -= np.einsum('kioa,kg->gioa', self.eris.ooov, ts)

        # oiga terms for dLdt
        oiga_t = self.eris.oovv.copy()
        oiga_t -= np.einsum('oa,ig->oiga', fov, ls)
        oiga_t -= np.einsum('ig,oa->oiga', fov, ls)
        oiga_t += np.einsum('ciga,oc->oiga', self.eris.vovv, ls)
        oiga_t += np.einsum('ocga,ic->oiga', self.eris.ovvv, ls)
        tmp = np.einsum('ic,kc->ik', ls, ts)
        oiga_t += np.einsum('koga,ik->oiga', self.eris.oovv, tmp)
        tmp = np.einsum('ojba,jb->oa', self.eris.oovv, ts)
        oiga_t += np.einsum('oa,ig->oiga', tmp, ls)
        tmp = np.einsum('oibg,jb->oijg', self.eris.oovv, ts)
        oiga_t += np.einsum('oijg,ja->oiga', tmp, ls)
        tmp = np.einsum('kigc,kc->ig', self.eris.oovv, ts)
        oiga_t += np.einsum('ig,oa->oiga', tmp, ls)
        tmp = np.einsum('jiga,jb->igab', self.eris.oovv, ts)
        oiga_t -= np.einsum('igab,ob->oiga', tmp, ls)
        tmp = np.einsum('oica,kc->oika', self.eris.oovv, ts)
        oiga_t += np.einsum('oika,kg->oiga', tmp, ls)
        oiga_t -= np.einsum('oigk,ka->oiga', self.eris.oovo, ls)
        oiga_t -= np.einsum('oija,jg->oiga', self.eris.ooov, ls)
        oiga_t -= np.einsum('ia,og->oiga', ls, fov)  # energy term
        oiga_t -= 0.5*np.einsum('ia,kc,okgc->oiga', ls, ts, self.eris.oovv)  # energy term
        oiga_t -= 0.5 * np.einsum('ia,jb,jobg->oiga', ls, ts, self.eris.oovv)  # energy term
        del tmp

        for p in range(0, nvir * nocc):
            for q in range(0, nocc * nvir):
                o, g = np.unravel_index(q, (nocc, nvir))
                i, a = np.unravel_index(p, (nocc, nvir))

                if o == i:
                    dLdl_pq[p, q] += dio[a, g]
                if a == g:
                    dLdl_pq[p, q] += dag[i, o]
                if a == g and o ==i:
                    dLdl_pq[p, q] += diodag

                dLdl_pq[p, q] += L*Viaog_l[i, a, o, g]
                dLdl_pq[p, q] += gioa_l[g, i, o, a]

                dLdt_pq[p, q] = L*Viaog_t[i, a, o, g]
                dLdt_pq[p, q] = oiga_t[o, i, g, a]

        return dLdt_pq, dLdl_pq

    #############################
    # Vexp derivatives
    #############################

    def DV1(self, ts, ls):
        """
        CCS gradient of Vexp with Vexp = K*gamma_esp-gamma_calc

        :param ts: t1 amplitudes
        :param ls: lambda 1 amplitudes
        :return: dV[rs]dt[og] and dV[rs]dl[og] (l=lambda) tensors
        """

        nocc, nvir = ts.shape
        dim = nocc+nvir

        dVdt = np.zeros((dim, dim, nocc, nvir))
        dVdl = np.zeros((dim, dim, nocc, nvir))

        # explicit loop (slow)
        # for r in range(dim):
        #    for s in range(dim):
        #        for o in range(nocc):
        #            for g in range(nvir):
        #                if r < nocc:
        #                    # oo bloc
        #                    if s < nocc and o == r:
        #                        dVdt[r, s, o, g] -= ls[s, g]
        #                    if nocc > s == o:
        #                        dVdl[r, s, o, g] -= ts[r, g]
        #
        #                    # ov bloc
        #                    if s >= nocc:
        #                        dVdl[r, s, o, g] += ts[o, s - nocc] * ts[r, g]
        #                        if o == r and s == g + nocc:
        #                            dVdt[r, s, o, g] += 1.
        #                        if s == g + nocc:
        #                            dVdt[r, s, o, g] += - np.sum(ts[r, :] * ls[o, :])
        #                        if o == r:
        #                            dVdt[r, s, o, g] += - np.sum(ts[:, s - nocc] * ls[:, g])
        #                # vv bloc
        #                if r >= nocc and s >= nocc:
        #                    if s == g + nocc:
        #                        dVdt[r, s, o, g] += ls[o, r - nocc]
        #                    if r == g + nocc:
        #                        dVdl[r, s, o, g] += ts[o, s - nocc]
        #
        #                # vo bloc
        #                if r >= nocc > s and r == g and s == o:
        #                    dVdl[r, s, o, g] += 1

        # vectorization (fast)

        # oo
        doo_t = np.einsum('isig->isg', dVdt[:nocc, :nocc, :, :])
        doo_t[:, :, :] = -np.broadcast_to(ls, (nocc, nocc, nvir))
        doo_l = np.einsum('riig->irg', dVdl[:nocc, :nocc, :, :])
        doo_l[:, :, :] = -np.broadcast_to(ts, (nocc, nocc, nvir))

        # vv
        dvv_t = np.einsum('raoa->aor', dVdt[nocc:, nocc:, :, :])
        dvv_t[:, :, :] = np.broadcast_to(ls, (nvir, nocc, nvir))
        dvv_l = np.einsum('asoa->aos', dVdl[nocc:, nocc:, :, :])
        dvv_l[:, :, :] = np.broadcast_to(ts, (nvir, nocc, nvir))

        # ov
        np.einsum('iaia->ia', dVdt[:nocc, nocc:, :, :])[:, :] += 1
        tmp = np.einsum('rb,ob->ro', ts, ls)
        np.einsum('raoa->aro', dVdt[:nocc, nocc:, :, :])[:, :, :] -= np.broadcast_to(tmp, (nvir, nocc, nocc))
        tmp = np.einsum('js,jg->sg', ts, ls)
        np.einsum('isig->isg', dVdt[:nocc, nocc:, :, :])[:, :, :] -= np.broadcast_to(tmp, (nocc, nvir, nvir))
        dVdl[:nocc, nocc:, :, :] += np.einsum('os,rg->rsog', ts, ts)

        # vo
        np.einsum('aiia->ia', dVdl[nocc:, :nocc, :, :])[:, :] += 1.

        return dVdt, dVdl

    def DV2(self, ts, ls):
        """
        CCS gradient of Vexp with
        Vexp,rs = K*sum_j A_rs,j*(A_exp,j-sum(gamma_calc*A_calc,j)/sig_j

        :return:
        """

        Nr = self.Nr  # total number of exp data (prop)
        sig = self.exp_pot.sig  # array of sigma values for each prop
        A = self.exp_pot  # needed <p|hat(A)|q> matrices

        nocc, nvir = ts.shape
        dim = nocc+nvir

        dVdt = np.zeros((dim, dim, nocc, nvir))
        dVdl = np.zeros((dim, dim, nocc, nvir))

        for j in range(Nr):

            Aoo = A[j][:nocc, :nocc].copy()
            Aov = A[j][:nocc, nocc:].copy()
            Avv = A[j][nocc:, nocc:].copy()
            Avo = A[j][nocc:, :nocc].copy()

            tmp = -np.einsum('lo,lg->og', Aoo, ls)
            tmp += np.einsum('ga,oa->og', Avv, ls)
            tmp += Avo
            tmp -= np.einsum('gl,lc,oc->og', Avo, ts, ls)
            tmp -= np.einsum('bo,mb,mg->og', Avo, ts, ls)
            dVdt += np.einsum('rs,og->rsog', A[j], tmp) / sig[j]

            tmp = -np.einsum('oj,jg->og', Aoo, ts)
            tmp += np.einsum('ag,oa->og', Avv, ts)
            tmp += Aov
            contr = np.einsum('aj,oj->ao', Avo, ts)
            tmp += np.einsum('ao,ag->og', contr, ts)
            dVdl += np.einsum('rs,og->rsog', A[j], tmp) / sig[j]

        return (2./Nr)*dVdt, (2./Nr)*dVdl

    def DV3(self, ts, ls):
        """
        CCS gradient of Vexp with
        K_j = (2/Nr*sig_j)*A_rs,j*(sum_pq A_pq,j gamma_qp)
        Vexp_rs = sum_j K_j*(A_exp,j-sum_pqtu (A_pq,j*A_tu.conj,j*gamma_pq*gamma_ut)

        where |Aj|^2, the squared norm of the expectation values are being compared
        :return:
        """

        Nr = self.exp_pot.Nr  # total number of exp data (prop)
        sig = self.exp_pot.sig  # array of sigma values for each prop
        A = self.exp_pot  # list of needed <p|hat(A)|q> matrices
        Aexp = self.exp_pot.Aexp  # list of experimental data
        gamma_calc = self.exp_data.gamma_calc

        nocc, nvir = ts.shape
        dim = nocc+nvir

        dVdt = np.zeros((dim, dim, nocc, nvir))
        dVdl = np.zeros((dim, dim, nocc, nvir))

        for j in range(Nr):

            Aoo = A[j][:nocc, :nocc].copy()
            Aov = A[j][:nocc, nocc:].copy()
            Avv = A[j][nocc:, nocc:].copy()
            Avo = A[j][nocc:, :nocc].copy()

            Kp = np.einsum('qp, pq', A[j].conj, gamma_calc)
            K = np.einsum('pq, pq', A[j], gamma_calc)
            Kexp = Aexp[j] - Kp*K

            # dVdl

            dVdl -= np.einsum('po,pg->rsog', Aoo, ts)
            dVdl += np.einsum('gq,oq->rsog', Avv, ts)
            tmp = np.einsum('pq,oq->po', Aov, ts)
            dVdl += np.einsum('po,pg->rsog', tmp, ts)
            dVdl += np.einsum('go->rsog', Avo)
            dVdl *= Kp

            dV_tmp = np.zeros_like(dVdl)
            dV_tmp -= np.einsum('oq,qg->rsog', Aoo.conj, ts)
            dV_tmp += np.einsum('pg,op->rsog', Avv.conj, ts)
            dV_tmp += np.einsum('og->rsog', Aov)
            tmp = np.einsum('pq,op->qo', Avo, ts)
            dV_tmp += np.einsum('qo,qg->rsog', tmp, ts)

            dVdl += K*dV_tmp
            dVdl *= Kp
            dVdl += Kexp*dV_tmp
            dVdl *= np.einsum('rs->rsog', A[j])/sig[j]

            # dVdt

            dVdt -= np.einsum('oq,qg->rsog', Aoo, ls)
            dVdt += np.einsum('pg,op->rsog', Avv, ls)
            dVdt += np.einsum('go->rsog', Aov)
            tmp = np.einsum('pb,ob->po', ts, ls)
            dVdt -= np.einsum('po,pg->rsog', tmp, Aov)
            tmp = np.einsum('jq,jg->qg', ts, ls)
            dVdt -= np.einsum('qg,oq->rsog', tmp, Aov)

            dVdt *= Kp

            dV_tmp = np.zeros_like(dVdl)
            dV_tmp -= np.einsum('po,pg->rsog', Aoo.conj, ls)
            dV_tmp += np.einsum('gq,oq->rsog', Avv.conj, ls)
            dV_tmp += np.einsum('go->rsog', Avo.conj)
            tmp = np.einsum('qb,ob->po', ts, ls)
            dV_tmp -= np.einsum('po,gq->rsog', tmp, Avo.conj)
            tmp = np.einsum('jp,jg->pg', ts, ls)
            dV_tmp -= np.einsum('pg,po->rsog', tmp, Avo.conj)

            dVdt += K*dV_tmp
            dVdt *= Kp
            dVdt += Kexp*dV_tmp
            dVdt *= np.einsum('rs->rsog', A[j])/sig[j]

        del dV_tmp, tmp

        return (2./Nr)*dVdt, (2./Nr)*dVdl

    ################################
    # Jacobian and Gradient methods
    ################################

    def Jacobian(self, ts, ls, fsp, L):
        """
        Build Jacobian matrix

        :param ts: t1 amplitudes
        :param ls: l1 aamplitudes
        :param fsp: effective fock matrix
        :param L: weight of the experimental data lambda
        :return:
        """

        dTdt, dTdl = self.dT(ts, ls, fsp, L)
        dLdt, dLdl = self.dL(ts, ls, fsp, L)

        J00 = dTdt.copy()
        J01 = dTdl.copy()
        J10 = dLdt.copy()
        J11 = dLdl.copy()

        return np.block([[J00, J01], [J10, J11]])

    def Newton(self, ts, ls, fsp, L):
        """
        Apply Newton's method to the ECW-CCS model

        :param ts: t1 amplitudes
        :param ls: l1 aamplitudes
        :param fsp: effective fock matrix
        :param L: weight of the experimental data lambda
        :return:
        """

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
        """
        Apply Graident descend method to the ECW-CCS model

        :param beta: gradient step
        :param ts: t1 amplitudes
        :param ls: l1 aamplitudes
        :param fsp: effective fock matrix
        :param L: weight of the experimental data lambda
        :return:
        """

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

#####################################
#   ECW-GCCS old gradient equations #
#####################################

class ccs_gradient_old:
    def __init__(self, eris, M_tot=1, sum_sig=1):
        """
        Gradient of the ECW-CCS equations and Newton's method

        :param eris: two electron integrals
        :param M_tot: scale of the Vexp potential (number of measurements)
        :param sum_sig: sum of all sig_i, means for each sets of measurements
        """

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
        Fai += np.einsum('jb,jabi->ai', ts, self.eris.ovvo)

        Fab = fvv.copy()
        Fab -= np.einsum('jb,ja->ab', fov, ts)
        Fab += np.einsum('jc,jacb->ab', ts, self.eris.ovvv)

        Fji = foo.copy()
        Fji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        tmp = np.einsum('kc,jkcb->jb', ts, self.eris.oovv)
        Fji -= np.einsum('ib,jb->ji', ts, tmp)

        T1 = np.einsum('ai->ia', Fai)
        T1 += np.einsum('ib,ab->ia', ts, Fab)
        T1 -= np.einsum('ja,ji->ia', ts, Fji)

        return T1

    #################
    # L1 equations
    #################

    def L1eq(self, ts, ls, fsp, E_term=False):
        """
        Value of the Lambda 1 equations using intermediates

        :param ts: t1 amplitudes
        :param ls: l1 amplitudes
        :param fsp: fock matrix
        :param E_term: False if energy term in Lambda eq is zero
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
        Fba -= np.einsum('ja,jb->ba', fov, ts)
        Fba += np.einsum('jbca,jc->ba', self.eris.ovvv, ts)
        tmp = np.einsum('jkca,jc->ka', self.eris.oovv, ts)
        Fba -= np.einsum('ka,kb->ba', tmp, ts)

        Fij = foo.copy()
        Fij += np.einsum('ib,jb->ij', fov, ts)
        Fij += np.einsum('kibj,kb->ij', self.eris.oovo, ts)
        tmp = np.einsum('kibc,kb->ic', self.eris.oovv, ts)
        Fij += np.einsum('ic,jc->ij', tmp, ts)

        Wbija = self.eris.voov.copy()
        Wbija -= np.einsum('kija,kb->bija', self.eris.ooov, ts)
        tmp = np.einsum('kica,kb->icab', self.eris.oovv, ts)
        Wbija -= np.einsum('icab,jc->bija', tmp, ts)
        Wbija += np.einsum('bica,jc->bija', self.eris.vovv, ts)

        Fia = fov.copy()
        Fia += np.einsum('jiba,jb->ia', self.eris.oovv, ts)

        # energy term
        if E_term:
            E = -np.einsum('jb,jb', ts, fov)
            E -= 0.5*np.einsum('jb,kc,jkbc', ts, ts, self.eris.oovv)
        else:
            E = 0.

        L1 = Fia.copy()
        L1 += np.einsum('ib,ba->ia', ls, Fba)
        L1 -= np.einsum('ja,ij->ia', ls, Fij)
        L1 += np.einsum('jb,bija->ia', ls, Wbija)
        L1 += ls * E

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
        """
        Build Jacobian matrix

        :param ts:
        :param ls:
        :param fsp:
        :param L:
        :return:
        """

        J00 = self.dTdt(ts, ls, fsp, L)
        J01 = self.dTdl(ts, L)
        J10 = self.dLdt(ts, ls, fsp, L)
        J11 = self.dLdl(ts, ls, fsp, L)

        return np.block([[J00, J01], [J10, J11]])

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
    # execute only if file is run as a script

    np.random.seed(2)

    from pyscf import gto, scf, cc
    import Eris
    import CC_raw_equations

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
    gfs_sym = gfs + gfs.T
    gfs = utilities.convert_r_to_g_rdm1(gfs)  # non-symmetric fock matrix
    gfs_sym = utilities.convert_r_to_g_rdm1(gfs_sym)  # symmetric fock matrix

    print()
    print('####################################################')
    print(' Test T and L equations using random t1, l1 and f   ')
    print('####################################################')

    T1eq = mccsg.T1eq(gts, gfs)
    T1eq_raw = CC_raw_equations.T1eq(gts, geris, fsp=gfs)

    La1eq = mccsg.L1eq(gts, gls, gfs, E_term=False)
    La1eq_raw = CC_raw_equations.La1eq(gts, gls, geris, fsp=gfs)

    print()
    print(" Difference between raw eq and with intermediates ")
    print("--------------------------------------------------")
    print()
    print("T1 ")
    print(np.max(np.subtract(T1eq, T1eq_raw)))
    print()
    print("L1 ")
    print(np.max(np.subtract(La1eq, La1eq_raw)))
    print()

    print('Difference between T1 and L1 inter for t=0 (should be zero)')
    print('-----------------------------------------------------------')
    ts = np.zeros((gnocc, gnvir))
    Tinter = mccsg.T1inter(ts, gfs)  # Fab, Fji, Fai
    Linter = mccsg.L1inter(ts, gfs)  # Fia, Fba, Fij, Wbija, E
    print('Fba: ', np.max(np.subtract(Tinter[0], Linter[1])))
    print('Fij: ', np.max(np.subtract(Tinter[1], Linter[2])))
    print()
    del ts, Tinter, Linter

    print(" Difference between updated amp with different intermediates    ")
    print("----------------------------------------------------------------")
    print()

    print('l1 update')
    print()

    print('symmetric fock matrix')
    pyscf = mccsg.lsupdate_PySCF(gts, gls, gfs_sym)
    Linter = mccsg.L1inter_Stanton(gts, gfs_sym)
    stanton = mccsg.lsupdate(gts, gls, Linter)
    Linter = mccsg.L1inter(gts, gfs_sym, E_term=False)
    stasis = mccsg.lsupdate(gts, gls, Linter)
    print('PySCF-Stanton: ', np.max(np.subtract(pyscf, stanton)))
    print('PySCF-Stasis: ', np.max(np.subtract(pyscf, stasis)))
    print('Stanton-Stasis: ', np.max(np.subtract(stanton, stasis)))
    print()

    print('non-symmetric fock matrix')
    pyscf = mccsg.lsupdate_PySCF(gts, gls, gfs)
    Linter = mccsg.L1inter_Stanton(gts, gfs)
    stanton = mccsg.lsupdate(gts, gls, Linter)
    Linter = mccsg.L1inter(gts, gfs, E_term=False)
    stasis = mccsg.lsupdate(gts, gls, Linter)
    print('PySCF-Stanton: ', np.max(np.subtract(pyscf, stanton)))
    print('PySCF-Stasis: ', np.max(np.subtract(pyscf, stasis)))
    print('Stanton-Stasis: ', np.max(np.subtract(stanton, stasis)))

    print()
    print('t1 update')
    print()

    print('Symmetric fock matrix')
    pyscf = mccsg.tsupdate_PySCF(mygcc, gts, np.diag(gfs.diagonal()))
    Tinter = mccsg.T1inter_Stanton(gts, np.diag(gfs.diagonal()))
    stanton = mccsg.tsupdate(gts, Tinter)
    Tinter = mccsg.T1inter(gts, np.diag(gfs.diagonal()))
    stasis = mccsg.tsupdate(gts, Tinter)
    print('PySCF-Stanton: ', np.max(np.subtract(pyscf, stanton)))
    print('PySCF-Stasis: ', np.max(np.subtract(pyscf, stasis)))
    print('Stanton-Stasis: ', np.max(np.subtract(stanton, stasis)))
    print()

    print('Non symmetric fock matrix')
    pyscf = mccsg.tsupdate_PySCF(mygcc, gts, gfs)
    Tinter = mccsg.T1inter_Stanton(gts, gfs)
    stanton = mccsg.tsupdate(gts, Tinter)
    Tinter = mccsg.T1inter(gts, gfs)
    stasis = mccsg.tsupdate(gts, Tinter)
    print('PySCF-Stanton: ', np.max(np.subtract(pyscf, stanton)))
    print('PySCF-Stasis: ', np.max(np.subtract(pyscf, stasis)))
    print('Stanton-Stasis: ', np.max(np.subtract(stanton, stasis)))
    print()

    print("--------------------------------")
    print(" ts_update with L1 reg          ")
    print("--------------------------------")

    print('ts updated with alpha = 0')
    Tinter = mccsg.T1inter(gts, gfs)
    ts_L1 = mccsg.tsupdate_L1(gts, Tinter, 0.)
    ts_up = mccsg.tsupdate(gts, Tinter)
    print(np.max(np.subtract(ts_up, ts_L1)))

    print()
    print('ls updated with alpha = 0')
    Linter = mccsg.L1inter(gts, gfs)
    ls_L1 = mccsg.lsupdate_L1(gls, Linter, 0.)
    ls_up = mccsg.lsupdate(gts, gls, Linter)
    print(np.max(np.subtract(ls_up, ls_L1)))

    print()
    print('####################')
    print(' TEST JACOBIAN      ')
    print('####################')
    print()

    mgrad = ccs_gradient(geris)
    # print(np.diag(mgrad.Jacobian(gts, gls, gfs, 0.)))
    print(np.diag(mgrad.dT(np.zeros_like(gts), np.zeros_like(gls), gfs, 0.)[0]))
    print(np.diag(gfs[gnocc:])[:, None] - np.diag(gfs[:gnocc]))

    ts_G = gts.copy()*0.1
    ls_G = gls.copy()*0.1
    conv_thre = 1.
    conv = True
    ite = 1
    norm = 0.
    L = 0.
    while conv_thre > 10**-5:
        norm_old = norm
        tsnew, lsnew = mgrad.Newton(ts_G, ls_G, gfs, L)
        #tsnew, lsnew = mgrad.Gradient_Descent(0.01, ts_G, ls_G, gfs, 0)
        ts_G = tsnew
        ls_G = lsnew
        norm = np.concatenate((ts_G.flatten(), ls_G.flatten()))
        conv_thre = np.linalg.norm(norm-norm_old)
        ite += 1
        if ite > 20:
            conv = False
            print("t and l amplitudes NOT converged after {} iteration".format(ite))
            break
    if conv:
        print("Newton's method with Lambda = 0 from random initial guess")
        print("t and l amplitudes converged after {} iteration".format(ite))
        print()

    ts_G = np.zeros_like(gts)
    ls_G = gls.copy()
    conv_thre = 1.
    ite = 1
    norm = 0.
    L = 0.1
    while conv_thre > 10**-5:
        norm_old = norm
        tsnew, lsnew = mgrad.Newton(ts_G, ls_G, gfs, L)
        #tsnew, lsnew = mgrad.Gradient_Descent(0.01, ts_G, ls_G, gfs, L)
        ts_G = tsnew
        ls_G = lsnew
        norm = np.concatenate((ts_G.flatten(), ls_G.flatten()))
        conv_thre = np.linalg.norm(norm-norm_old)
        ite += 1
        if ite > 20:
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

    print('Difference in Ek between symetrized and unsymetrized rdm1:')
    print('DEk= ', np.subtract(utilities.Ekin(mol, g1, aobasis=False, mo_coeff=mgf.mo_coeff),
                               utilities.Ekin(mol, g2, aobasis=False, mo_coeff=mgf.mo_coeff)))

    print()
    print(" Difference between GS gamma and ES gamma with r1=0, r0=0")
    g1 = mccsg.gamma_es(gts, gls, None, 1, 0)
    print(np.max(np.subtract(g1, g2)))

    print()
    print('trace of transition rdm1 ')
    # building random amplitudes vectors
    t1 = np.random.random((gnocc, gnvir))*0.1
    r1 = np.random.random((gnocc, gnvir))*0.1
    l1 = np.random.random((gnocc, gnvir))*0.1
    r0 = mccsg.r0_fromE(0.1, t1, r1, np.zeros_like(gfs))
    l0 = mccsg.l0_fromE(0.1, t1, l1, np.zeros_like(gfs))
    tr_rdm1 = mccsg.gamma_tr(t1, l1, r1, r0, l0)
    print(tr_rdm1.trace())
    print()
    
    print('trace of rdm1 for excited state - nelec')
    # normalize r and l
    c = utilities.get_norm(r1, l1, r0, l0)
    l1 /= c
    # get r0 and l0
    r0 = mccsg.r0_fromE(0.1, t1, r1, np.zeros_like(gfs))
    l0 = mccsg.l0_fromE(0.1, t1, l1, np.zeros_like(gfs))
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

    print('Difference between R1 and L1 inter for t=0 ')
    print('-------------------------------------------')
    ts = np.zeros((gnocc, gnvir))
    Rinter = mccsg.R1inter(ts, gfs, vn)[:3]    # Fab, Fji, W, E, Tia, Pia
    Linter = mccsg.es_L1inter(ts, gfs, vn) # Fba, Fij, W, E, Tia, P
    inter = ['Fba', 'Fij', 'W', 'E']
    i = 0
    for R, L in zip(Rinter, Linter):
        print('Inter ', inter[i])
        print(np.max(np.subtract(R, L)))
        i += 1

    print()
    print('Difference between R1 and L1 equations for t=0 and ls=rs')
    print('--------------------------------------------------------')
    print()

    print('Symmetric f matrix (should be zero)')
    Rinter = mccsg.R1inter(np.zeros_like(gts), gfs_sym, vn)
    Linter = mccsg.es_L1inter(np.zeros_like(gts), gfs_sym, vn)
    print('with intermediates')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), mccsg.es_L1eq(rs, r0, Linter))))
    print('raw equations')
    print(np.max(np.subtract(CC_raw_equations.R1eq(ts, rs, r0, geris),
                             CC_raw_equations.es_L1eq(ts, rs, r0, geris))))

    print()
    print('Random f matrix')
    Rinter = mccsg.R1inter(np.zeros_like(gts), gfs, vn)
    Linter = mccsg.es_L1inter(np.zeros_like(gts), gfs, vn)
    print('with intermediates')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), mccsg.es_L1eq(rs, r0, Linter))))
    print('raw equations')
    print(np.max(np.subtract(CC_raw_equations.R1eq(np.zeros_like(gts), rs, r0, geris, fsp=gfs),
                             CC_raw_equations.es_L1eq(np.zeros_like(gts), rs, r0, geris, fsp=gfs))))

    print()
    print('Difference between inter and raw equations for t=0 (should be zero)')
    print('-------------------------------------------------------------------')

    Rinter = mccsg.R1inter(np.zeros_like(gts), gfs, vn)
    Linter = mccsg.es_L1inter(np.zeros_like(gts), gfs, vn)
    print('R1 difference')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), CC_raw_equations.R1eq(ts, rs, r0, geris, fsp=gfs))))
    print('L1 difference')
    print(np.max(np.subtract(mccsg.es_L1eq(ls, l0, Linter), CC_raw_equations.es_L1eq(ts, ls, l0, geris, fsp=gfs))))

    print()
    print('Difference between inter and raw equations for t random (should be zero)')
    print('------------------------------------------------------------------------')

    Rinter = mccsg.R1inter(gts, gfs, vn)
    Linter = mccsg.es_L1inter(gts, gfs, vn)
    print('R1 difference')
    print(np.max(np.subtract(mccsg.R1eq(rs, r0, Rinter), CC_raw_equations.R1eq(gts, rs, r0, geris, fsp=gfs))))
    print('L1 difference')
    print(np.max(np.subtract(mccsg.es_L1eq(ls, l0, Linter), CC_raw_equations.es_L1eq(gts, ls, l0, geris, fsp=gfs))))

    R0inter = mccsg.R0inter(gts, gfs, vn)
    L0inter = mccsg.L0inter(gts, gfs, vn)
    print('R0 difference')
    print(np.max(np.subtract(mccsg.R0eq(rs, r0, R0inter), CC_raw_equations.R10eq(gts, rs, r0, geris, fsp=gfs))))
    print('L0 difference')
    print(np.max(np.subtract(mccsg.L0eq(ls, l0, L0inter), CC_raw_equations.es_L10eq(gts, ls, l0, geris, fsp=gfs))))

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
