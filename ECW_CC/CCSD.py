#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################################################
#
# ECW-CCS v1
# -----------
# Experimentaly constrained wave function coupled cluster single
# ---------------------------------------------------------------
#
# File containing all PySCF RCCSD and GCCSD relevant functions
# - L1/T1 and T2/L2 intermediates
# - t and l updates
#
# todo: add energy term to the Lambda equation and intermediates
# todo: recast Lambda intermediates in a class (like for T)
# todo: L1 regularization term only for doubles
# todo: add EOM(1)-CCSD equations
#
###################################################################

import numpy as np
from pyscf import lib
import utilities

einsum = lib.einsum

#############
# FUNCTIONS
#############

##################
# Transition rdm1
##################

def tr_rdm1_inter(t1, t2, l1, l2, r1, r2, r0):
    """
    Intermediates for the transition density matrix <Psi_{n}(t,l)|ap^{dag}aq|Psi_{k}(t,r)>
    All amplitudes must be given in amp format (nocc x nvir)

    raw equation from Stanton 95

    :param: single and double amplitudes
    :return: intermediates
    """

    # oo inter
    Yijem = np.einsum('if,jmfe->ijem', t1, l2)

    # vv inter
    Yabn = np.einsum('me,mnea->abn', r1, l2)

    # vo inter

    # Yim
    Yim = -np.einsum('ie,me->im', t1, l1)
    Yim += -0.5 * np.einsum('inef,mnef->im', t2, l2)
    Yim *= r0
    Yim += -np.einsum('ie,me->im', r1, l1)
    Yim += -0.5 * np.einsum('inef,mnef->im', r2, l2)
    Yim += -np.einsum('ie,nf,mnef->im', t1, r1, l2)

    # Yea
    Yea = -0.5 * r0 * np.einsum('mnaf,mnef->ea', t2, l2)
    Yea += -np.einsum('ma,me->ea', r1, l1)
    Yea += -0.5 * np.einsum('mnaf,mnef->ea', r2, l2)

    # Yea_p
    Yea_p = -0.5 * np.einsum('mnaf,mnef->ea', t2, l2)
    # Yanef
    Yanef = -0.5 * np.einsum('ma,mnef->anef', r1, l2)
    # Yainf
    Yainf = np.einsum('imae,mnef->ainf', t2, l2)

    return Yijem, Yabn, Yim, Yea, Yea_p, Yanef, Yainf


def tr_rdm1(t1, t2, l1, l2, r1, r2, r0, inter=None):
    """
    Calculates the transition reduced density matrix between two states m and n
    <Psi_m|a^{dag}_p a_q|Psi_n>
    Psi_m -> t,l
    Psi_n -> t,r
    For ground state (n=0): r=0 and r0=1
                     (m=0): l=lambda and l0=1

    raw equation from Stanton 95

    :param t1: t1 amplitudes
    :param l1: l1 amplitudes for state m
    :param r1: r1 amplitudes for state n
    :param t2: t2 amplitudes
    :param l2: l2 amplitudes for state m
    :param r2: r2 amplitudes for state n
    :param inter: intermediates
    :return:
    """

    if inter is None:
       Yijem, Yabn, Yim, Yea, Yea_p, Yanef, Yainf = tr_rdm1_inter(t1, t2, l1, l2, r1, r2, r0)
    else:
        Yijem, Yabn, Yim, Yea, Yea_p, Yanef, Yainf = inter

    # oo
    rdm1oo = np.einsum('ie,je->ij', t1, l1)
    rdm1oo += 0.5 * np.einsum('imfe,jmfe->ij', t2, l2)
    rdm1oo *= -r0
    rdm1oo += -np.einsum('ie,je->ij', r1, l1)
    rdm1oo += -0.5 * np.einsum('imfe,jmfe->ij', r2, l2)
    rdm1oo += np.einsum('me,ijem->ij', r1, Yijem)

    # vv
    rdm1vv = np.einsum('mb,am->ab', t1, l1)
    rdm1vv += 0.5 * np.einsum('mneb,mnea->ab', t2, l2)
    rdm1vv *= r0
    rdm1vv += np.einsum('mb,ma->ab', r1, l1)
    rdm1vv += 0.5 * np.einsum('mne,mnea->ab', r2, l2)
    rdm1vv += np.einsum('nb,abn->ab', t1, Yabn)

    # ov
    rdm1ov = r0 * l1 + np.einsum('imae,me->ia', l2, r1)

    # vo
    rdm1vo = r0 * np.einsum('imae,me->ai', t2, l1)
    rdm1vo += np.einsum('ia->ai', t1)
    rdm1vo += np.einsum('imae,me->ai', r2, l1)
    rdm1vo += np.einsum('ie,ea->ai', r1, Yea_p)
    rdm1vo += np.einsum('inef,anef->ai', t2, Yanef)
    rdm1vo += np.einsum('nf,ainf->ai', r1, Yainf)
    rdm1vo += np.einsum('ma,im->ai', t1, Yim)
    rdm1vo += np.einsum('ea,ie->ai', Yea, t1)

    # construct total rdm1
    rdm1 = np.block([[rdm1oo, rdm1ov], [rdm1vo, rdm1vv]])

    return rdm1


def gamma_CCSD(t1, t2, l1, l2):
    """
    Construct symetrized CCSD reduced density matrix for the ground state
    Taken from PySCF GCCSD_rdm1 file

    (Give the same result has CCS.gamma_ccs with t2 and l2 = 0)

    :param t1: t1 amplitudes in G format (nocc, nvir)
    :param t2: t2 amplitudes in G format (nocc, nocc, nvir, nvir)
    :param l1: lambda 1 amplitudes
    :param l2: lambda 2 amplitudes
    :return:
    """

    doo, dov, dvo, dvv = gamma_inter(t1, t2, l1, l2)
    nocc, nvir = dov.shape
    nmo = nocc + nvir

    dm1 = np.empty((nmo, nmo), dtype=doo.dtype)
    dm1[:nocc, :nocc] = doo + doo.conj().T
    dm1[:nocc, nocc:] = dov + dvo.conj().T
    dm1[nocc:, :nocc] = dm1[:nocc, nocc:].conj().T
    dm1[nocc:, nocc:] = dvv + dvv.conj().T
    dm1 *= .5
    dm1[np.diag_indices(nocc)] += 1

    return dm1

def gamma_inter(t1, t2, l1, l2):

    doo = -np.einsum('ie,je->ij', l1, t1)
    doo -= np.einsum('imef,jmef->ij', l2, t2) * .5

    dvv = np.einsum('ma,mb->ab', t1, l1)
    dvv += np.einsum('mnea,mneb->ab', t2, l2) * .5

    xt1 = np.einsum('mnef,inef->mi', l2, t2) * .5
    xt2 = np.einsum('mnfa,mnfe->ae', t2, l2) * .5
    xt2 += np.einsum('ma,me->ae', t1, l1)
    dvo = np.einsum('imae,me->ai', t2, l1)
    dvo -= np.einsum('mi,ma->ai', xt1, t1)
    dvo -= np.einsum('ie,ae->ai', t1, xt2)
    dvo += t1.T

    dov = l1

    return doo, dov, dvo, dvv


class GCC:
    def __init__(self,eris, fock=None):
        self.eris = eris
        self.nocc = eris.nocc
        if fock is None:
            self.fock = eris.fock
        self.nvir = self.fock.shape[0]-self.nocc

#######
# rdm1
#######

    def gamma(self, t1, t2, l1, l2):
        return gamma_CCSD(t1, t2, l1, l2)

    def gamma_inter(self, t1, t2, l1, l2):
        return gamma_inter(t1,t2,l1,l2)

##################
# Transition rdm1
##################

    def tr_rdm1_inter(self,t1, t2, l1, l2, r1, r2, r0):
        return tr_rdm1_inter(t1, t2, l1, l2, r1, r2, r0)

    def tr_rdm1(self,t1, t2, l1, l2, r1, r2, r0, inter):
        return tr_rdm1(t1,t2,l1,l2,r1,r2,r0,inter)

##########
# energy
##########

    def energy(self, t1, t2, fsp):
        """
        CCSD energy equation

        :param t1:
        :param t2:
        :param fsp: one-electron operator
        :return:
        """

        nocc, nvir = t1.shape

        e = np.einsum('ia,ia', fsp[:nocc, nocc:], t1)

        eris_oovv = np.array(self.eris.oovv)
        e += 0.25 * np.einsum('ijab,ijab', t2, eris_oovv)
        e += 0.5 * np.einsum('ia,jb,ijab', t1, t1, eris_oovv)

        return e.real

###########
# t update
###########

    def tupdate(self, t1, t2, fsp=None, alpha=None, equation=False):
        """
        SCF update of the t1 and t2 amplitudes
        See PySCF.cc.gccsd

        :param t1: t1 amplitudes in spin-orbital basis
        :param t2: t2 amplitudes in spin-orbital basis
        :param fsp: effective fock matrix in MO basis
        :param alpha: L1 reg coefficient
        :param equation: True if T1 is to be calculate
        :return:
        """

        eris = self.eris
        fock = self.fock.copy()
        
        nocc, nvir = t1.shape

        if fsp is None:
            fsp = fock.copy()

        fov = fsp[:nocc, nocc:].copy()
        diag_vv = np.diagonal(fock[nocc:, nocc:])
        diag_oo = np.diagonal(fock[:nocc, :nocc])

        tau = self.make_tau(t2, t1, t1)

        Fvv = self.cc_Fvv(t1, t2, fsp)
        Foo = self.cc_Foo(t1, t2, fsp)
        Fov = self.cc_Fov(t1, t2, fsp)
        Woooo = self.cc_Woooo(t1, t2)
        Wvvvv = self.cc_Wvvvv(t1, t2)
        Wovvo = self.cc_Wovvo(t1, t2)

        # Move diagonal terms to the other side
        if not equation and alpha is None:
            Fvv[np.diag_indices(nvir)] -= diag_vv
            Foo[np.diag_indices(nocc)] -= diag_oo

        # T1 equation
        t1new = einsum('ie,ae->ia', t1, Fvv)
        t1new += -einsum('ma,mi->ia', t1, Foo)
        t1new += einsum('imae,me->ia', t2, Fov)
        t1new += -einsum('nf,naif->ia', t1, eris.ovov)
        t1new += -0.5 * einsum('imef,maef->ia', t2, eris.ovvv)
        t1new += -0.5 * einsum('mnae,mnie->ia', t2, eris.ooov)
        t1new += fov.conj()

        # T2 equation
        Ftmp = Fvv - 0.5 * einsum('mb,me->be', t1, Fov)
        tmp = einsum('ijae,be->ijab', t2, Ftmp)
        t2new = tmp - tmp.transpose(0, 1, 3, 2)
        Ftmp = Foo + 0.5 * einsum('je,me->mj', t1, Fov)
        tmp = einsum('imab,mj->ijab', t2, Ftmp)
        t2new -= tmp - tmp.transpose(1, 0, 2, 3)
        t2new += np.asarray(eris.oovv).conj()
        t2new += 0.5 * einsum('mnab,mnij->ijab', tau, Woooo)
        t2new += 0.5 * einsum('ijef,abef->ijab', tau, Wvvvv)
        tmp = einsum('imae,mbej->ijab', t2, Wovvo)
        tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
        tmp = tmp - tmp.transpose(1, 0, 2, 3)
        tmp = tmp - tmp.transpose(0, 1, 3, 2)
        t2new += tmp
        tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
        t2new += (tmp - tmp.transpose(1, 0, 2, 3))
        tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
        t2new -= (tmp - tmp.transpose(0, 1, 3, 2))
        
        if alpha is not None:

            dW1 = utilities.subdiff(t1new, t1, alpha)
            dW2 = utilities.subdiff(t2new, t2, alpha)
            if equation:
                return dW1, dW2
            eia = diag_oo[:, None] - diag_vv
            eijab = lib.direct_sum('ia,jb->ijab', eia, eia)

            dW1 += t1*eia
            dW2 += t2*eijab
            dW1 /= eia
            dW2 /= eijab
            return dW1, dW2

        elif not equation:
            eia = diag_oo[:, None] - diag_vv
            eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
            t1new /=eia
            t2new /=eijab

        return t1new, t2new

#################
# T intermediate
#################

    # See PySCF GCCSD file

    def make_tau(self, t2, t1a, t1b, fac=1.):

       t1t1 = np.einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
       t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
       tau1 = t1t1 - t1t1.transpose(0,1,3,2)
       tau1 += t2

       return tau1

    def cc_Fvv(self, t1, t2, fsp):

       eris = self.eris

       nocc, nvir = t1.shape
       fov = fsp[:nocc,nocc:].copy()
       fvv = fsp[nocc:,nocc:].copy()

       tau_tilde = self.make_tau(t2, t1, t1,fac=0.5)
       Fae = fvv - 0.5*np.einsum('me,ma->ae',fov, t1)
       Fae += np.einsum('mf,amef->ae', t1, eris.vovv)
       Fae -= 0.5*np.einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)

       return Fae

    def cc_Foo(self,t1, t2, fsp):

       eris = self.eris

       nocc, nvir = t1.shape
       fov = fsp[:nocc,nocc:]
       foo = fsp[:nocc,:nocc]
       tau_tilde = self.make_tau(t2, t1, t1,fac=0.5)
       Fmi = ( foo + 0.5*np.einsum('me,ie->mi',fov, t1)
              + np.einsum('ne,mnie->mi', t1, eris.ooov)
              + 0.5*np.einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
       return Fmi

    def cc_Fov(self,t1, t2, fsp):
       nocc, nvir = t1.shape
       fov = fsp[:nocc,nocc:]
       Fme = fov + np.einsum('nf,mnef->me', t1, self.eris.oovv)
       return Fme

    def cc_Woooo(self, t1, t2):
        tau = self.make_tau(t2, t1, t1)
        tmp = np.einsum('je,mnie->mnij', t1, self.eris.ooov)
        Wmnij = self.eris.oooo + tmp - tmp.transpose(0, 1, 3, 2)
        Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tau, self.eris.oovv)
        return Wmnij

    def cc_Wvvvv(self, t1, t2):
        tau = self.make_tau(t2, t1, t1)
        eris_ovvv = np.asarray(self.eris.ovvv)
        tmp = np.einsum('mb,mafe->bafe', t1, eris_ovvv)
        Wabef = np.asarray(self.eris.vvvv) - tmp + tmp.transpose(1, 0, 2, 3)
        Wabef += np.einsum('mnab,mnef->abef', tau, 0.25 * np.asarray(self.eris.oovv))
        return Wabef

    def cc_Wovvo(self, t1, t2):
        eris=self.eris
        eris_ovvo = -np.asarray(eris.ovov).transpose(0, 1, 3, 2)
        eris_oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)
        Wmbej = np.einsum('jf,mbef->mbej', t1, eris.ovvv)
        Wmbej -= np.einsum('nb,mnej->mbej', t1, eris_oovo)
        Wmbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, eris.oovv)
        Wmbej -= np.einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
        Wmbej += eris_ovvo
        return Wmbej

###########
# l update
###########

    def lupdate(self, t1, t2, l1, l2, fsp=None, alpha=None, equation=False):
        """
        SCF update of the Lambda amplitudes
        see pyscf.cc.gccsd_lambda file

        :param t1: t1 amplitudes in spin-orbital basis
        :param t2: t2 amplitudes in spin-orbital basis
        :param l1: lambda1 amplitudes in spin-orbital basis
        :param l2: lambda2 amplitudes in spin-orbital basis
        :param fsp: effective Fock matrix in molecular spin-orbital basis
        :param alpha: L1 reg coefficient
        :param equation: True if the Lambda equations are to be returned
        :return: updates l1 and l2 amplitudes or L1 and L2 equations
        """

        eris = self.eris
        fock = self.fock.copy()
        nocc, nvir = t1.shape

        if fsp is None:
            fsp = fock

        imds = self.Linter(t1, t2, fsp=fsp)
        fov = fsp[:nocc, nocc:].copy()

        diag_vv = np.diagonal(fock[nocc:, nocc:])
        diag_oo = np.diagonal(fock[:nocc, :nocc])

        if equation is False and alpha is None:
            v1 = imds.v1 - np.diag(diag_vv)
            v2 = imds.v2 - np.diag(diag_oo)
        else:
            v1 = imds.v1.copy()
            v2 = imds.v2.copy()

        l1new = np.zeros_like(l1)
        l2new = np.zeros_like(l2)

        mba = einsum('klca,klcb->ba', l2, t2) * .5
        mij = einsum('kicd,kjcd->ij', l2, t2) * .5
        m3 = einsum('klab,ijkl->ijab', l2, np.asarray(imds.woooo))
        tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
        tmp = einsum('ijcd,klcd->ijkl', l2, tau)
        oovv = np.asarray(eris.oovv)
        m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
        tmp = einsum('ijcd,kd->ijck', l2, t1)
        m3 -= einsum('kcba,ijck->ijab', eris.ovvv, tmp)
        m3 += einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5

        l2new += oovv
        l2new += m3
        fov1 = fov + einsum('kjcb,kc->jb', oovv, t1)
        tmp = einsum('ia,jb->ijab', l1, fov1)
        tmp += einsum('kica,jcbk->ijab', l2, np.asarray(imds.wovvo))
        tmp = tmp - tmp.transpose(1, 0, 2, 3)
        l2new += tmp - tmp.transpose(0, 1, 3, 2)
        tmp = einsum('ka,ijkb->ijab', l1, eris.ooov)
        tmp += einsum('ijca,cb->ijab', l2, v1)
        tmp1vv = mba + einsum('ka,kb->ba', l1, t1)
        tmp += einsum('ca,ijcb->ijab', tmp1vv, oovv)
        l2new -= tmp - tmp.transpose(0, 1, 3, 2)
        tmp = einsum('ic,jcba->jiba', l1, eris.ovvv)
        tmp += einsum('kiab,jk->ijab', l2, v2)
        tmp1oo = mij + einsum('ic,kc->ik', l1, t1)
        tmp -= einsum('ik,kjab->ijab', tmp1oo, oovv)
        l2new += tmp - tmp.transpose(1, 0, 2, 3)

        l1new += fov
        l1new += einsum('jb,ibaj->ia', l1, eris.ovvo)
        l1new += einsum('ib,ba->ia', l1, v1)
        l1new -= einsum('ja,ij->ia', l1, v2)
        l1new -= einsum('kjca,icjk->ia', l2, imds.wovoo)
        l1new -= einsum('ikbc,bcak->ia', l2, imds.wvvvo)
        l1new += einsum('ijab,jb->ia', m3, t1)
        l1new += einsum('jiba,bj->ia', l2, imds.w3)
        tmp = (t1 + einsum('kc,kjcb->jb', l1, t2)
               - einsum('bd,jd->jb', tmp1vv, t1)
               - einsum('lj,lb->jb', mij, t1))
        l1new += np.einsum('jiba,jb->ia', oovv, tmp)
        l1new += np.einsum('icab,bc->ia', eris.ovvv, tmp1vv)
        l1new -= np.einsum('jika,kj->ia', eris.ooov, tmp1oo)
        tmp = fov - einsum('kjba,jb->ka', oovv, t1)
        l1new -= np.einsum('ik,ka->ia', mij, tmp)
        l1new -= np.einsum('ca,ic->ia', mba, tmp)

        if alpha is not None:

            dW1 = utilities.subdiff(l1new, l1, alpha)
            dW2 = utilities.subdiff(l2new, l2, alpha)
            if equation:
                return dW1, dW2
            eia = diag_oo[:, None] - diag_vv
            eijab = lib.direct_sum('ia,jb->ijab', eia, eia)

            dW1 += l1*eia
            dW2 += l2*eijab
            dW1 /= eia
            dW2 /= eijab
            return dW1, dW2

        elif not equation:
            eia = diag_oo[:, None] - diag_vv
            eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
            l1new /= eia
            l2new /= eijab

        return l1new, l2new

###################
# L intermediates
###################

    # see PySCF gccsd_lambda file

    def Linter(self, t1, t2, fsp=None):
        '''
        Generalized Lambda CCSD intermediates
        see cc.gccsd_lambda PySCF file

        :param t1: t1 amplitudes in spin-orbital basis
        :param t2: t2 amplitudes in spin-orbital basis
        :param fsp: effective fock matrix in MO spin basis
        :return:
        '''

        eris = self.eris
        nocc, nvir = t1.shape

        if fsp is None:
            fsp = self.fock

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvo = fsp[nocc:, :nocc].copy()
        fvv = fsp[nocc:, nocc:].copy()

        tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2

        v1 = fvv - einsum('ja,jb->ba', fov, t1)
        v1 -= np.einsum('jbac,jc->ba', eris.ovvv, t1)
        v1 += einsum('jkca,jkbc->ba', eris.oovv, tau) * .5

        v2 = foo + einsum('ib,jb->ij', fov, t1)
        v2 -= np.einsum('kijb,kb->ij', eris.ooov, t1)
        v2 += einsum('ikbc,jkbc->ij', eris.oovv, tau) * .5

        v3 = einsum('ijcd,klcd->ijkl', eris.oovv, tau)
        v4 = einsum('ljdb,klcd->jcbk', eris.oovv, t2)
        v4 += np.asarray(eris.ovvo)

        v5 = fvo + np.einsum('kc,jkbc->bj', fov, t2)
        tmp = fov - np.einsum('kldc,ld->kc', eris.oovv, t1)
        v5 += np.einsum('kc,kb,jc->bj', tmp, t1, t1)
        v5 -= einsum('kljc,klbc->bj', eris.ooov, t2) * .5
        v5 += einsum('kbdc,jkcd->bj', eris.ovvv, t2) * .5

        w3 = v5 + np.einsum('jcbk,jb->ck', v4, t1)
        w3 += np.einsum('cb,jb->cj', v1, t1)
        w3 -= np.einsum('jk,jb->bk', v2, t1)

        woooo = np.asarray(eris.oooo) * .5
        woooo += v3 * .25
        woooo += einsum('jilc,kc->jilk', eris.ooov, t1)

        wovvo = v4 - np.einsum('ljdb,lc,kd->jcbk', eris.oovv, t1, t1)
        wovvo -= einsum('ljkb,lc->jcbk', eris.ooov, t1)
        wovvo += einsum('jcbd,kd->jcbk', eris.ovvv, t1)

        wovoo = einsum('icdb,jkdb->icjk', eris.ovvv, tau) * .25
        wovoo += np.einsum('jkic->icjk', np.asarray(eris.ooov).conj()) * .5
        wovoo += einsum('icbk,jb->icjk', v4, t1)
        wovoo -= einsum('lijb,klcb->icjk', eris.ooov, t2)

        wvvvo = einsum('jcak,jb->bcak', v4, t1)
        wvvvo += einsum('jlka,jlbc->bcak', eris.ooov, tau) * .25
        wvvvo -= np.einsum('jacb->bcaj', np.asarray(eris.ovvv).conj()) * .5
        wvvvo += einsum('kbad,jkcd->bcaj', eris.ovvv, t2)

        class _IMDS: pass
        imds = _IMDS()
        imds.woooo = woooo
        imds.wovvo = wovvo
        imds.wovoo = wovoo
        imds.wvvvo = wvvvo
        imds.v1 = v1
        imds.v2 = v2
        imds.w3 = w3

        return imds

if __name__ == "__main__":
    # execute only if run as a script
    # test on water

    from pyscf import gto, scf, cc
    from pyscf.cc import gccsd, gccsd_lambda
    import Eris, utilities, CC_raw_equations

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    #mol.atom = """
    #H 0 0 0
    #H 0 0 1
    #"""

    mol.basis = 'sto3g'
    mol.spin = 0
    mol.build()

    # GHF
    mgf = scf.GHF(mol)
    mgf.kernel()
    mo_occ = mgf.mo_occ
    mocc = mgf.mo_coeff[:, mo_occ > 0]
    mvir = mgf.mo_coeff[:, mo_occ == 0]
    gnocc = mocc.shape[1]
    gnvir = mvir.shape[1]
    gdim = gnocc + gnvir

    print()
    print("nocc x nvir")
    print(gnocc,gnvir)
    print()

    mycc = cc.GCCSD(mgf)
    eris = Eris.geris(mycc)
    fsp = eris.fock
    myccsd = GCC(eris)

    #from pyscf
    mycc.kernel()
    t1 = mycc.t1*0.1
    t2 = mycc.t2*0.1
    l1 = t1*0.05
    l2 = t2*0.05

    T1,T2 = myccsd.tupdate(t1, t2, fsp, equation=True)
    L1,L2 = myccsd.lupdate(t1, t2, l1, l2, fsp, equation=True)

    T1_eq, T2_eq = CC_raw_equations.T1T2eq(t1, t2, eris)
    L1_eq, L2_eq = CC_raw_equations.La1La2eq(t1, t2, l1, l2, eris)
    
    print() 
    print('###############################################')
    print('Difference between CCSD class and raw equations')
    print('###############################################')
    print()
    print("T1 - T1_eq")
    print(np.sum(np.subtract(T1_eq,T1)))

    print()
    print("T2 - T2_eq")
    print(np.sum(np.subtract(T2_eq, T2)))

    print()
    print("L1 - L1_eq")
    print(np.sum(np.subtract(L1_eq,L1)))

    print()
    print("L2 - L2_eq")
    print(np.sum(np.subtract(L2_eq, L2)))

    print()
    print("Gamma shape")
    print(gdim)
    print(myccsd.gamma(t1,t2,l1,l2).shape)
    print()

    # pyscf
    pyscf_eris = cc.GCCSD(mgf).ao2mo()
    pyscf_tupdate = gccsd.update_amps(cc.GCCSD(mgf),t1,t2,pyscf_eris)
    imds = gccsd_lambda.make_intermediates(cc.GCCSD(mgf),t1,t2,pyscf_eris)
    pyscf_lupdate = gccsd_lambda.update_lambda(cc.GCCSD(mgf),t1,t2,l1,l2,pyscf_eris,imds)

    # ccsd
    t_update = myccsd.tupdate(t1,t2)
    l_update = myccsd.lupdate(t1,t2,l1,l2)

    # t1 and t2 OK
    # l1 and l2 OK
    print('t and l update comparison with PySCF')
    print('------------------------------------')
    print('l1')
    print(np.sum(np.subtract(pyscf_lupdate[0],l_update[0])))
    print('l2')
    print(np.sum(np.subtract(pyscf_tupdate[1],t_update[1])))

    print()
    print('###############################################')
    print(' L1 regularization in CCSD                     ')
    print('###############################################')
    print()

    print('Difference between t1new alpha=0 and alpha=None -> should be zero')
    print()
    T1,T2 = myccsd.tupdate(t1,t2,alpha=None)
    W1,W2 = myccsd.tupdate(t1,t2,alpha=0)
    print('t1')
    print(np.sum(np.subtract(W1,T1)))
    print('t2')
    print(np.sum(np.subtract(W2,T2)))
    print()
    L1, L2 = myccsd.lupdate(t1,t2,l1,l2,None)
    W1, W2 = myccsd.lupdate(t1,t2,l1,l2,alpha=0)
    print('l1')
    print(np.sum(np.subtract(W1,L1)))
    print('l2')
    print(np.sum(np.subtract(W2,L2)))
    print()
