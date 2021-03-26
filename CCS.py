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
# TODO: factorize rdm1_es equations
# TODO: factorize gradient terms
#
###################################################################

import numpy as np
import utilities

np.random.seed(2)


############################
# CLASS: Generalized CCS
############################

class Gccs:
    def __init__(self, eris, fock=None, M_tot=None):
        '''

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

    # ----------------------------------------------------------------------------
    # Energy
    # -----------------------------------------------------------------------------

    def energy_ccs(self, ts, fsp, rs=None, vnn=None):

        '''
        E'_{0}
        :param rs: list of rs amplitude for excited states n
        :param vn: list of exp potential V{nn}
        '''

        # from gccsd import energy(cc, t1, t2, eris)
        nocc, nvir = ts.shape
        e = np.einsum('ia,ia', fsp[:nocc, nocc:], ts)
        e += 0.5 * np.einsum('ia,jb,ijab', ts, ts, self.eris.oovv)

        # add contribution to excited states
        if rs is not None:
            for r,v in zip(rs,vnn):
                if v.any():
                    e += np.einsum('ia,ia',v,r)

        return e

    # -------------------------------------------------------------------
    # RDM1
    # -------------------------------------------------------------------

    def gamma(self, ts, ls):
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

    def gamma_unsym(self, ts, ls):
        '''
        Unsymmetrized one-particle reduced density matrix CCS
        - Stanton 1993 with ria = 0 and  r0=1
        - Stasis: same results except l0 term

        :param ts: t1 amplitudes
        :param ls: l1 amplitudes
        :return:
        '''

        nocc,nvir = ts.shape

        doo = -np.einsum('ie,je->ij',ts,ls)
        dvv = np.einsum('mb,ma->ab',ts,ls)
        dov = ls
        dvo = -np.einsum('ie,ma,me->ai',ts,ts,ls)+ts.transpose()

        dm1 = np.empty((nocc + nvir, nocc + nvir))
        dm1[:nocc, :nocc] = doo
        dm1[:nocc, nocc:] = dov
        dm1[nocc:, :nocc] = dvo
        dm1[nocc:, nocc:] = dvv

        dm1[np.diag_indices(nocc)] += 1

        return dm1
    
    def gamma_es(self, ts, ln, rn, r0n, l0n):
        '''
        Unsymmetrized CCS one-particle reduced density matrix for a excited states n
        Psi_n must be normalized: sum(ln*rn)+(l0*r0)=1

        :param ts: t1 amplitudes
        :param ln: l1 amplitudes
        :param rn: r1 amplitudes
        :param r0n: r0 amplitude
        :return:
        '''

        nocc, nvir = ts.shape

        # GS case:
        if isinstance(rn,float) or isinstance(rn,int):
            if rn == 0:
               rn = np.zeros((nocc,nvir))
               r0n = 1
            else:
                raise ValueError('r1 amplitudes must be either 0 or an matrix')

        doo = -r0n * np.einsum('ie,je->ij', ts, ln)
        doo -= np.einsum('ie,je->ij', rn, ln)
        dov = r0n * ln
        dvv = r0n * np.einsum('mb,ma->ab', ts, ln)
        dvv += np.einsum('mb,ma->ab', rn, ln)
        dvo = r0n * np.einsum('ie,ma,me->ai', ts, ts, ln)
        dvo -= np.einsum('ma,ie,me->ai', ts, rn, ln)
        dvo -= np.einsum('ie,ma,me->ai', ts, rn, ln)
        dvo += np.einsum('ia->ai', ts)
        dvo += l0n*rn.transpose()

        dm1 = np.empty((nocc + nvir, nocc + nvir))
        dm1[:nocc, :nocc] = doo
        dm1[:nocc, nocc:] = dov
        dm1[nocc:, :nocc] = dvo
        dm1[nocc:, nocc:] = dvv

        # G format
        dm1[np.diag_indices(nocc)] += 1

        return dm1

    def gamma_tr(self, ts, ln, rk, r0k, l0n):
        # todo: what about l0*r1 term ?

        '''
        CCS one-particle reduced transition density matrix between state n and k
        <Psi_n|apaq|Psi_k>
        if Psi_k = Psi_GS then r0=1 and rk=0
        if Psi_n = Psi_GS then l0=1 and lk=0
        ln,l0 and rk,r0k must be orthogonal: sum(ln*rk)+(r0*l0) = 0
        

        :param ts: t1 amplitude
        :param ln: l1 amplitudes for state n
        :param rk: r1 amplitudes for state k
        :param r0k: r0 amplitude for state k

        :return: tr_rdm1 in G format
        '''

        nocc,nvir = ts.shape

        # if Psi_k or Psi_l = GS
        if isinstance(rk,float) or isinstance(rk,int):
            if rk == 0:
               rk = np.zeros_like(ln)
               r0k = 1
            else:
                raise ValueError('r1 amplitudes must be either 0 or a matrix')

        if isinstance(ln,float) or isinstance(ln,int):
            if ln == 0:
               ln = np.zeros_like(rk)
               l0n = 1
            else:
                raise ValueError('l1 amplitudes must be either 0 or a matrix')

        doo = -r0k * np.einsum('ie,je->ij', ts, ln)
        doo -= np.einsum('ie,je->ij', rk, ln)
        dov = r0k * ln
        dvv = r0k * np.einsum('mb,ma->ab', ts, ln)
        dvv += np.einsum('mb,ma->ab', rk, ln)
        dvo = r0k * np.einsum('ie,ma,me->ai', ts, ts, ln)
        dvo -= np.einsum('ma,ie,me->ai', ts, rk, ln)
        dvo -= np.einsum('ie,ma,me->ai', ts, rk, ln)
        # dvo += ts.transpose() # this terms drops if k != n
        dvo += l0n*rk.transpose()

        dm = np.empty((nocc + nvir, nocc + nvir))
        dm[:nocc, :nocc] = doo
        dm[:nocc, nocc:] = dov
        dm[nocc:, :nocc] = dvo
        dm[nocc:, nocc:] = dvv

        return dm

    # -----------------------------------------------------------------------------
    # T1 equation
    # -----------------------------------------------------------------------------

    # equations
    def T1eq(self, ts, fsp):
        '''
        T1 equations using intermediates

        :param ts: t1 amplitudes
        :param fsp: fock matrix
        :return: T1 nocc x nvir matrix
        '''
        
        Fae, Fmi, fov = self.T1inter(ts,fsp)

        T1 = 0
        T1 += fov
        T1 += np.einsum('ie,ae->ia', ts, Fae)
        T1 -= np.einsum('ma,mi->ia', ts, Fmi)
        T1 -= np.einsum('nf,naif->ia', ts, self.eris.ovov)

        return T1

    # ts update
    def tsupdate(self, ts, T1inter, rsn=None, r0n=None, vn=None):
        '''
        SCF update of the t1 amplitudes with additional coupling terms

        :param ts: t1 amplitudes
        :param T1inter: T1 intermediates
        :param fsp: effective Fock matrix elements
        :param rsn: ria and r0 amplitudes for the states n, list of ((ria_1,r0_1),(ria_2,r0_2), ...)
                    where ria is a occ x nocc matrix and r0 a number
        :param vn: Vexp[0,n] exp potentials
        :return: updated ts amplitudes
        '''

        Fae,Fmi,fov = T1inter
        fock = self.fock
        nocc, nvir = ts.shape

        # remove diagonal of the fock matrix
        diag_vv = np.diagonal(fock[nocc:, nocc:])
        diag_oo = np.diagonal(fock[:nocc, :nocc])
        
        Fae[np.diag_indices(nvir)] -= diag_vv
        Fmi[np.diag_indices(nocc)] -= diag_oo

        # update ts
        tsnew = fov
        tsnew += np.einsum('ie,ae->ia', ts, Fae)
        tsnew -= np.einsum('ma,mi->ia', ts, Fmi)
        tsnew -= np.einsum('nf,naif->ia', ts, self.eris.ovov)
        
        # add coupling terms with excited states
        # assuming that the additional terms are small
        if rsn is not None:
            if r0n is None:
                raise ValueError('if Vexp are to be calculated, list of r0 amp must be given')
            if len(vn) != len(rsn):
                raise ValueError('Number of experimental potentials must be equal to number of r amplitudes')
            for r,v,r0 in zip(rsn,vn,r0n):
                if v.any():
                    v = np.asarray(v)
                    v_oo = v[:nocc, :nocc]
                    v_vv = v[nocc:, nocc:]
                    v_ov = v[:nocc, nocc:]

                    tsnew += r*np.trace(v_oo)
                    tsnew += r0*v_ov
                    tsnew += np.einsum('ib,ab->ia', r, v_vv)
                    tsnew -= np.einsum('ja,ji->ia', r, v_oo)
                    tsnew += r*np.einsum('jb,jb', v_ov, ts)
                    tsnew -= np.einsum('ja,ib,jb->ia', r, ts, v_ov)
                    tsnew -= np.einsum('ib,ja,jb->ia', r, ts, v_ov)
                    tsnew += r0*np.einsum('ib,ab->ia', ts, v_vv)
                    tsnew -= r0*np.einsum('ja,ji->ia',r, v_oo)
                    tsnew -= r0*np.einsum('ib,ja,jb->ia', ts, ts, v_ov)


        tsnew /= (diag_oo[:, None] - diag_vv)

        return tsnew
    
    # ts update with L1 reg
    def tsupdate_L1(self,ts,T1inter,alpha):
        '''

        SCF+L1 (regularization) update of the t1 amplitudes

        :param ts: t1 amplitudes
        :param T1inter: T1 intermediates
        :param fsp: effective fock
        :param alpha: L1 regularization parameter
        :return: updated ts
        '''

        Fae,Fmi,fov = T1inter
        fock = self.fock
        nocc, nvir = ts.shape

        diag_vv = np.diagonal(fock[nocc:, nocc:])
        diag_oo = np.diagonal(fock[:nocc, :nocc])

        # T1 equations
        T1 = fov
        T1 += np.einsum('ie,ae->ia', ts, Fae)
        T1 -= np.einsum('ma,mi->ia', ts, Fmi)
        T1 -= np.einsum('nf,naif->ia', ts, self.eris.ovov)

        # subdifferential
        W = utilities.subdiff(T1,ts,alpha)

        # remove diagonal elements
        # with outer
        tmp = np.subtract.outer(diag_vv,diag_oo).transpose()
        W -= ts*tmp
        # with loop
        #for i in range(nocc):
        #    for a in range(nvir):
        #       W[i,a] -= ts[i,a]*(diag_vv[a]-diag_oo[i])

        tsnew = W/(diag_oo[:, None] - diag_vv)

        return tsnew

    # T1 intermediates
    def T1inter(self, ts, fsp):
        # script from PySCF
        # cc/gintermediates.py

        nocc, nvir = ts.shape

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

        # make tau
        fac = 0.5
        tsts = np.einsum('ia,jb->ijab', fac * 0.5 * ts, ts)
        tsts = tsts - tsts.transpose(1, 0, 2, 3)
        tau = tsts - tsts.transpose(0, 1, 3, 2)

        # make ts intermediates
        Fae = fvv.copy()
        Fae -= 0.5 * np.einsum('me,ma->ae', fov, ts)
        Fae += np.einsum('mf,amef->ae', ts, self.eris.vovv)
        Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tau, self.eris.oovv)

        Fmi = foo.copy()
        Fmi += 0.5 * np.einsum('me,ie->mi', fov, ts)
        Fmi += np.einsum('ne,mnie->mi', ts, self.eris.ooov)
        Fmi += 0.5 * np.einsum('inef,mnef->mi', tau, self.eris.oovv)

        return Fae, Fmi, fov

    def T1inter_Stanton(self, ts, fsp):
        # Row implementation of T1 inetrmedaite in Staton paper

        nocc, nvir = ts.shape

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

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

        return Fae, Fmi, fov

    # ------------------------------------------------------------------------------------------
    # Lambda 1 equation
    # -------------------------------------------------------------------------------------------
    
    # equation
    def L1eq(self, ts, ls, fsp):
        '''
        Value of the Lambda 1 equations using intermediates
        
        :param ts: t1 amplitudes 
        :param ls: l1 amplitudes
        :param fsp: fock matrix
        :return: Lambda1 value
        '''
        
        Fia,Fea,Fim,Wieam = self.L1inter(ts,fsp)

        L1 = Fia.copy()
        L1 += np.einsum('ie,ea->ia', ls, Fea)
        L1 -= np.einsum('ma,im->ia', ls, Fim)
        L1 += np.einsum('me,ieam->ia', ls, Wieam)

        return L1

    # lambda update
    def lsupdate(self, ts, ls, L1inter, rsn=None, lsn=None, r0n=None, l0n=None, vn=None):
        # from gccsd import update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        # where imds are the intermediates taken from make_intermediates(mycc, t1, t2, eris)
        '''

        :param rsn: list with r amplitudes associated to excited state n
        :param lsn: list with l amplitudes associated to excited state n
        :param vn: exp potential Vexp[n,0]
        :return lsnew: updated lambda 1 values
        '''

        Fia,Fea,Fim,Wieam = L1inter

        nocc, nvir = ts.shape
        fock = self.fock

        # remove diagonal of the fock matrix
        diag_vv = np.diagonal(fock[nocc:, nocc:])
        diag_oo = np.diagonal(fock[:nocc, :nocc])

        Fea[np.diag_indices(nvir)] -= diag_vv
        Fim[np.diag_indices(nocc)] -= diag_oo

        lsnew = Fia.copy()
        lsnew += np.einsum('ie,ea->ia', ls, Fea)
        lsnew -= np.einsum('ma,im->ia', ls, Fim)
        lsnew += np.einsum('me,ieam->ia', ls, Wieam)

        # add l_R terms from coupling to excited states
        # assuming that l_r are small
        if rsn is not None:

            # check length
            if len(lsn) != len(rsn) or len(vn) != len(rsn):
                raise ValueError('v0n, l and r list must be of same length')
            if r0n is None or l0n is None:
                raise ValueError('r0 and l0 values must be given')

            for r,l,v,r0,l0 in zip(rsn,lsn,vn,r0n,l0n):

                if v.any():
                    v = np.asarray(v)
                    v_oo = v[:nocc, :nocc]
                    v_vv = v[nocc:, nocc:]
                    v_ov = v[:nocc, nocc:]

                    # term with l (lambda) and r
                    lsnew += np.einsum('jb,jb', r, v_ov)
                    lsnew += r0*np.einsum('jb,jb', ts, v_ov)
                    lsnew += r0*np.trace(v_oo)
                    lsnew *= ls
                    # term with lsn
                    lsnew += l*np.trace(v_oo)
                    lsnew += np.einsum('ib,ba->ia', l, v_vv)
                    lsnew -= np.einsum('ij,ja->ia', v_oo, l)
                    lsnew += l*np.einsum('jb,jb', ts, v_ov)
                    tmp    = np.einsum('ja,jb->ab',l, ts)
                    lsnew -= np.einsum('ab,ib->ia', tmp, v_ov)
                    tmp    = np.einsum('ib,jb->ij', l, ts)
                    lsnew -= np.einsum('ij,ja->ia', tmp, v_ov)
                    lsnew += l0*v_ov

        lsnew /= (diag_oo[:, None] - diag_vv)

        return lsnew
    
    # lambda update with L1 reg
    def lsupdate_L1(self,ls,ts,L1inter,alpha):

        '''

        SCF+L1 regularization for the updated lambda amplitudes

        :param ts: t1 amplitudes
        :param T1inter: T1 intermediates
        :param fsp: effective fock
        :param alpha: L1 parameter
        :return: updated ts
        '''

        Fia,Fea,Fim,Wieam = L1inter

        nocc, nvir = ts.shape
        fock = self.fock

        diag_vv = np.diagonal(fock[nocc:, nocc:])
        diag_oo = np.diagonal(fock[:nocc, :nocc])

        # Lambda 1 equations
        L1 = Fia.copy()
        L1 += np.einsum('ie,ea->ia', ls, Fea)
        L1 -= np.einsum('ma,im->ia', ls, Fim)
        L1 += np.einsum('me,ieam->ia', ls, Wieam)

        # subdifferential
        W = utilities.subdiff(L1,ls,alpha)

        # remove diagonal elements
        # with outer
        tmp = np.subtract.outer(diag_vv,diag_oo).transpose()
        W -= ls*tmp
        # with loop
        #for i in range(nocc):
        #    for a in range(nvir):
        #       W[i,a] -= ls[i,a]*(diag_vv[a]-diag_oo[i])

        lsnew = W/(diag_oo[:, None] - diag_vv)

        return lsnew

    # Lambda intermediates
    def L1inter(self, ts, fsp):

        nocc, nvir = ts.shape

        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

        tau = np.zeros((nocc, nocc, nvir, nvir))
        for i in range(0, nocc):
            for j in range(0, nocc):
                for a in range(0, nvir):
                    for b in range(0, nvir):
                        tau[i, j, a, b] = 0.25 * (
                                    ts[i, a] * ts[j, b] - ts[j, a] * ts[i, b] - ts[i, b] * ts[j, a] + ts[j, b] * ts[
                                i, a])

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

        Wieam = self.eris.ovvo.copy()
        Wieam += np.einsum('mf,ieaf->ieam', ts, self.eris.ovvv)
        # inam oovo becomes nima ooov in PySCF but same resuls
        Wieam -= np.einsum('ne,inam->ieam', ts, self.eris.oovo)
        # inaf becomes nifa in PySCF but same result
        Wieam -= np.einsum('mf,ne,inaf->ieam', ts, ts, self.eris.oovv)

        Fia = TFie.copy()

        return Fia, Fea, Fim, Wieam

    # ------------------------------------------------------------------------------------------
    # R1 equations
    # ------------------------------------------------------------------------------------------

    def R1inter(self, ts, fsp, vm):
        # todo: calculates all Rn inter simultaneously
        '''
        Calculates the R1 intermediates for state m: equations (14) -> (21) in ES-ECW-CCS file

        :param ts: t1 amplitudes
        :param fsp_m: Effective Fock matrix of state m (containing the Vmm exp potential)
        :param vm: m0Vexp potential
        :return: set of one and two electron intermediates
        '''

        nocc,nvir = ts.shape

        # Fock matrix
        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()
        fvo = fsp[nocc:,:nocc].copy()

        # r intermediates
        # -----------------
        
        # Fab: equation (14)
        Fab = fvv.copy()
        Fab += np.einsum('akbk->ab', self.eris.vovo)
        Fab -= np.einsum('ja,jb->ab', ts, fov)
        Fab += np.einsum('jc,jacb->ab', ts, self.eris.ovvv)
        Fab -= np.einsum('ja,jkbk->ab', ts, self.eris.oovo)
        Fab -= np.einsum('jc,ka,jkcb->ab', ts, ts, self.eris.oovv)

        # Fji: equation (15)
        Fji = foo.copy()
        Fji += np.einsum('jkik->ji', self.eris.oooo)
        Fji += np.einsum('ib,jb->ji', ts, fov)
        Fji += np.einsum('ib,jkbk->ji', ts, self.eris.oovo)
        Fji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        Fji += np.einsum('kb,ic,kjbc->ji', ts, ts,self.eris.oovv)
        
        # Wakic: equation (16)
        W  = self.eris.voov.copy()
        W += np.einsum('ib,akbc->akic', ts, self.eris.vovv)
        W -= np.einsum('ib,ja,jkbc->akic', ts, ts, self.eris.oovv)
        W -= np.einsum('ja,jkic->akic', ts, self.eris.ooov)
        
        # Fjb: equation (17)
        Fjb = fov.copy()
        Fjb += np.einsum('jkbk->jb', self.eris.oovo)
        Fjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        F = np.einsum('jb,jb', ts, Fjb)
        del Fjb

        # r0 intermediates
        #-------------------
        
        # Zab: equation (18)
        Zab = fvv.copy()
        Zab -= np.einsum('ja,jb->ab', ts, fov)
        Zab += np.einsum('akbk->ab', self.eris.vovo)
        Zab -= np.einsum('ka,kjbj->ab', ts, self.eris.oovo)
        
        # Zji: equation (19)
        Zji = foo.copy()
        Zji += np.einsum('jkik->ji', self.eris.oooo)
        Zji += np.einsum('kb,kjbi->ji', ts, self.eris.oovo)
        tmp = np.einsum('ic,jkbc->ijkb', ts, self.eris.ooov)
        Zji -= np.einsum('kb,ijkb->ji', ts, tmp)
        del tmp

        # Zai: equation (20)
        Zai = fvo.copy()
        Zai += np.einsum('akik->ai', self.eris.vooo)
        Zai += np.einsum('jb,jabi->ai', ts, self.eris.ovvo)
        Zai += np.einsum('jb,ic,jabc->ai', ts, ts, self.eris.ovvv)

        # Zia: equation (21)
        Zia  = np.einsum('ai->ia', Zai)
        Zia += np.einsum('ib,ab->ia', ts, Zab)
        Zia -= np.einsum('ja,ji->ia', ts, Zji)
        del Zab, Zji, Zai

        # Vexp intermediate P: equation (22)
        v_vo = vm[nocc:,:nocc]
        v_vv = vm[nocc:,nocc:]
        v_oo = vm[:nocc,:nocc]
        Pia = v_vo.copy()
        Pia += np.einsum('ab,ib->ai',v_vv,ts)
        Pia -= np.einsum('ii,ja,ib->ai',v_oo,ts,ts)
        Pia = np.einsum('ai->ia',Pia)

        return Fab, Fji, W, F, Zia, Pia
    
    def R0inter(self,ts,fsp,vm):
        '''
        Calculates the one and two particles intermediates for state m as well as the Vexp intermediate
        for the R0 equations
        
        :param ts: t1 amplitudes
        :param fsp: effective fock matrix for the state m
        :param vm: m0V potential
        :return: Fjb, Zjb, P intermediates
        '''
        
        nocc,nvir = ts.shape
        fov = fsp[:nocc,nocc:].copy()
        
        # r intermediates
        # ------------------
        
        # Fjb: equation (23)
        Fjb = fov.copy()
        Fjb += np.einsum('jkbk->jb',self.eris.oovo)
        tmp = Fjb.copy()
        Fjb += np.einsum('kc,kjcb->jb',ts,self.eris.oovv)
           
        # r0 intermediates
        # ------------------
        
        # Zjb: equation (25) --> Same as Fjb in R1inter
        # Zjb and ts are contracted here
        Zjb = tmp.copy()
        Zjb += 0.5*np.einsum('kc,jkbc->jb',ts,self.eris.oovv)
        Z = np.einsum('jb,jb',ts,Zjb)
        del Zjb
        del tmp

        # Vexp inter
        # -------------------
        vm_oo = vm[:nocc,:nocc]
        vm_ov = vm[:nocc,nocc:]
        P = np.einsum('jj',vm_oo)
        P += np.einsum('jb,jb',ts,vm_ov)
        
        return Fjb, Z, P

    def Extract_Em_r(self, rs, r0, Rinter):
        '''
        Extract Em from the largest r1 element

        :param rs: r1 amplitude of state m
        :param r0: r0 amplitude of state m
        :param Rinter: R1 intermediates
        :return: Em and index of largest r1 element
        '''

        Fab, Fji, W, F, Zia, Pia = Rinter

        # largest r1
        o, v = np.unravel_index(np.argmax(rs, axis=None), rs.shape)

        # Ria = ria*En' matrix
        Ria = np.einsum('ab,ib->ia', Fab, rs)
        Ria -= np.einsum('ji,ja->ia', Fji, rs)
        Ria += np.einsum('akic,kc->ia', W, rs)
        Rov = Ria[o,v]
        del Ria

        Rov += rs[o,v] * F
        Rov += r0 * Zia[o,v]
        Rov += Pia[o,v]

        Em = Rov/rs[o,v]

        return Em, o,v

    def rsupdate(self, rs, r0, Rinter, Em):
        '''
        Update r1 amplitudes

        :param rs: matrix of r1 amplitudes for state m
        :param r0: r0 amplitude for state m
        :param Rinter: r1 intermediates for state m
        :param Em: Energy of state m
        :return: updated list of r1 amplitudes and index of the largest ria

        '''

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

        # divide by diag
        rsnew /= (Em+diag_oo[:,None]-diag_vv)

        return rsnew

    def R1eq(self, rs, r0, Rinter):

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
        '''

        :param rs: r1 amplitude
        :param Em: energy of state m
        :param R0inter: intermediates for the R0 equation
        :return: updated r0 for state m
        '''

        Fjb, Z, P = R0inter
        F = np.einsum('jb,jb',rs,Fjb)
        r0new = F+P+(r0*Z)
        r0new /= Em

        return r0new

    def get_rov(self, ls, l0, rs, r0, r_ind):
        '''
        Extract missing ria value from orthogonality relation

        :param ls: l1 amplitudes for state m
        :param l0: l0 amplitude for state m
        :param rs: r1 amplitudes for state m
        :param r0: r0 amplitude for state m
        :param r_ind: index of missing rov amplitude
        :return: updated rov
        '''

        o, v = r_ind
        rs[o, v] = 0
        rs[o+1, v+1] = 0 # G format
        lov = ls[o, v].copy()
        if abs(lov - ls[o+1,v+1]) >= 10**-4:
            raise ValueError('l/r matrix is not in G format or a symmetry breaking occurred')
        rov = 1 - r0 * l0 - np.einsum('ia,ia', rs, ls)
        rov /= lov

        return rov

    def R0eq(self, En, t1, r1, fsp=None):
        '''
        Returns r0 term for a CCS state
        See equation 23 in Derivation_ES file

        :param En: correlation energy (En+EHF=En_tot) of the state
        :param t1: t1 amp
        :param r1: r1 amp
        :param fsp: fock matrix
        :param eris_oovo: two-particle integrals
        :param eris_oovv:
        :return: r0
        '''

        if fsp is None:
            fsp = self.fock

        nocc, nvir = t1.shape
        fov = fsp[:nocc, nocc:].copy()

        d = En - np.einsum('jb,jb', t1, fov)
        d -= np.einsum('jb,jkbk', t1, self.eris.oovo)
        d -= 0.5 * np.einsum('jb,kc,jkbc', t1, t1, self.eris.oovv)

        r0 = np.einsum('jb,jb', r1, fov)
        r0 += np.einsum('jb,jkbk', r1, self.eris.oovo)
        r0 += np.einsum('kc,jb,jkbc', r1, t1, self.eris.oovv)

        r0 /= d

        return r0

    # -----------------------------------------------------------------------------------------------
    # L1 equations
    # -----------------------------------------------------------------------------------------------

    def es_L1inter(self,ts,fsp,vm):
        '''
        Returns the intermediates for the L1 equations of state m
        :param ts: t1 amplitudes
        :param fsp: effective fock matrix containing the potential Vmm
        :param vm: coupling potential V0m
        :return:
        '''

        nocc,nvir = ts.shape

        # Fock matrix
        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()

        # l intermediates
        # -----------------

        # Fba: equation (30)
        Fba = fvv.copy()
        Fba += np.einsum('bkak->ba', self.eris.vovo)
        Fba -= np.einsum('jb,ja->ba', ts, fov)
        Fba += np.einsum('jc,jbca->ba', ts, self.eris.ovvv)
        Fba -= np.einsum('jb,jkak->ba', ts, self.eris.oovo)
        Fba -= np.einsum('jc,kb,jkca->ba', ts, ts, self.eris.oovv)

        # Fij: equation (31)
        Fij = foo.copy()
        Fij += np.einsum('ikjk->ij', self.eris.oooo)
        Fij += np.einsum('jb,ib->ij', ts, fov)
        Fij += np.einsum('kb,kibj->ij', ts, self.eris.oovo)
        Fij += np.einsum('jb,ikbk->ji', ts, self.eris.oovo)
        Fij += np.einsum('kb,jc,kibc->ij', ts, ts, self.eris.oovv)

        # Wbija: equation (32)
        W = self.eris.ovvo.copy()
        W -= np.einsum('kb,kija->bija', ts, self.eris.ooov)
        W += np.einsum('jc,bica->bija', ts, self.eris.vovv)
        W -= np.einsum('jc,kb,kica->bija', ts, ts, self.eris.oovv)

        # F: equation (33) --> same as for Rinter
        Fjb = fov.copy()
        Fjb += np.einsum('jkbk->jb',self.eris.oovo)
        Fjb += 0.5*np.einsum('kc,jkbc->jb',ts,self.eris.oovv)
        F = np.einsum('jb,jb',ts,Fjb)

        # l0 intermediate
        # ------------------

        Zia  = fov.copy()
        Zia += np.einsum('ikak->ia',self.eris.oovo)
        Zia += np.einsum('jb,jiba->ia',ts,self.eris.oovv)

        # Vexp intermediate
        # ---------------------

        P = vm[:nocc,nocc:]

        return Fba, Fij, W, F, Zia, P

    def L0inter(self, ts, fsp, vm):
        '''
        L0 intermediates for the L0 equation of state m

        :param ts: t1 amplitudes
        :param fsp: effective fock matrix containing the Vmm potential
        :param vm: V0m coupling potential
        :return: L0 intermediates
        '''
       
        nocc,nvir = ts.shape

        # Effective fock sub-matrices
        foo = fsp[:nocc, :nocc].copy()
        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()
        fvo = fsp[nocc:,:nocc].copy()

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
        Wjb = np.einsum('kc,kbcj->', ts, tmp)
        del tmp
        Wjb -= np.einsum('kb,kljl->jb', ts, self.eris.oooo)
        Wjb -= np.einsum('jv,kb,klcl->jb', ts, ts, self.eris.oovo)
        Wjb += np.einsum('jc,bkck->jb', ts, self.eris.vovo)
        Wjb += np.einsum('bkjk->jb', self.eris.vooo)

        # Z: eq (38) --> same as Z in R0 equation --> R1inter
        Zjb = fov.copy()
        Zjb += np.einsum('jkbk->jb', self.eris.oovo)
        Zjb += 0.5*np.einsum('kc,jkbc->jb', ts, self.eris.oovv)
        Z = np.einsum('jb,jb', ts, Zjb)
        del Zjb
        
        # P: eq (39)
        P  = np.einsum('ia,ia', ts, vm[:nocc,nocc:])
        P += np.sum(np.diagonal(vm[:nocc,:nocc]))

        return Fbj, Wjb, Z, P

    def Extract_Em_l(self, ls, l0, L1inter):
        '''
        Extract Em from the largest l1 element

        :param ls: l1 amplitude of state m
        :param l0: l0 amplitude of state m
        :param Linter: L1 intermediates
        :return: Em and index of largest l1 element
        '''

        Fba, Fij, W, F, Zia, P = L1inter
        
        # largest r1
        o, v = np.unravel_index(np.argmax(ls, axis=None), ls.shape)

        # Lia = lia*En' matrix
        Lia  = np.einsum('ib,ba->ia', ls, Fba)
        Lia -= np.einsum('ja,ij->ia', ls, Fij)
        Lia += np.einsum('jb,ibaj->ia', ls, W)
        Lov = Lia[o,v]
        del Lia

        Lov += ls[o,v] * F
        Lov += l0 * Zia[o,v]
        Lov += P[o,v]

        Em = Lov/ls[o,v]

        return Em, o,v

    def es_lsupdate(self, ls, l0, Em, L1inter):
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

        # divide by diag
        lsnew /= (Em+diag_oo[:,None]-diag_vv)

        return lsnew

    def es_L1eq(self, ls, l0, L1inter):
        '''
        Update the l1 amplitudes for state m

        :param ls: list of l amplitudes for the m excited state
        :param l0: l0 amplitude for state m
        :param Em: Energy of the state m
        :param L1inter: intermediates for the L1 equation of state m
        :return: updated matrix of ls amplitudes for state m
        '''

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
        F = np.einsum('jb,bj',ls,Fjb)
        W = np.einsum('jb,jb',ls,Wjb)
        l0new = F+W+P+(l0*Z)
        l0new /= Em

        return l0new

    def L0eq(self, En, t1, l1, fsp=None):
        '''
        Returns l0 term for a CCS state
        See equation 23 in Derivation_ES file

        :param En: correlation energy (En+EHF=En_tot) of the state
        :param t1: t1 amp
        :param r1: l1 amp
        :param fsp: fock matrix
        :return: l0
        '''

        nocc, nvir = t1.shape

        if fsp is None:
            fsp = self.fock

        fov = fsp[:nocc, nocc:].copy()
        fvv = fsp[nocc:, nocc:].copy()
        foo = fsp[:nocc, :nocc].copy()

        d = En
        d -= np.einsum('jb,jkbk', t1, self.eris.oovo)
        d -= 0.5 * np.einsum('jb,kc,jkbc', t1, t1, self.eris.oovv)

        l0 = np.einsum('jb,jb', l1, fov)
        l0 += np.einsum('jb,ab,ja', t1, fvv, l1)
        l0 -= np.einsum('jb,kb,kj', l1, t1, foo)
        l0 -= np.einsum('jc,kb,kc,jb', t1, t1, fov, l1)
        l0 += np.einsum('jb,bkjk', l1, self.eris.vooo)
        l0 += np.einsum('jb,kc,kbcj', l1, t1, self.eris.ovvo)
        l0 -= np.einsum('jb,kb,kljl', l1, t1, self.eris.oooo)
        l0 += np.einsum('jb,jc,bkck', l1, t1, self.eris.vovo)
        tmp = np.einsum('jb,jd->bd', l1, t1)
        l0 += np.einsum('bd,kb,lc,klcd', tmp, t1, t1, self.eris.oovv)
        del tmp
        tmp = np.einsum('jb,lb->jl', l1, t1)
        l0 -= np.einsum('jl,kc,klcj', tmp, t1, self.eris.oovo)
        del tmp
        tmp = np.einsum('jb,jl->bl', l1, t1)
        l0 += np.einsum('bl,kc,kbcl', tmp, t1, self.eris.ovvo)
        tmp = np.einsum('jb,jc->bc', l1, t1)
        l0 -= np.einsum('bc,kb,klcl', tmp, t1, self.eris.oovo)
        del tmp

        l0 /= d

        return l0

#################################
#   ECW-GCCS gradient equations #
#################################

class ccs_gradient:
    def __init__(self, eris, M_tot=None, sum_sig=1):
        # obtained from generalized T1 and Lambda1 eq

        if M_tot is None:
            self.M_tot = eris.fock.shape[0] ** 2
        else:
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
        return CC_raw_equations.T1eq(ts,self.eris,fsp=fsp)

    #################
    # L1 equations
    #################

    def L1eq(self, ts, ls, fsp):
        return CC_raw_equations.La1eq(ts, ls, self.eris, fsp=fsp)

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
        doidga = C - np.einsum('klcd,kc,la->a', self.eris.oovv, ts, ts)
        doi1 = -np.einsum('ka,kg->ag', ts, fov)
        doi2 = -np.einsum('kc,kacg->ag', ts, self.eris.ovvv)
        doi3 = -C * np.einsum('ka,kg->ag', ts, ls)
        dga1 = -np.einsum('ie,oe->io', ts, fov)
        dga2 = -np.einsum('kc,ia,kocd->aio', ts, ts, self.eris.oovv)
        dga3 = -np.einsum('kc,koci->oi', ts, self.eris.oovo)
        int1 = np.einsum('ic,oa,mg,mc->ioag', ts, ts, ls, ts)
        int2 = np.einsum('ig,ka,oe,ke->igao', ts, ts, ls, ts)
        int3 = -np.einsum('id,oagd->oagi', ts, self.eris.ovvv)
        int4 = -np.einsum('olgd,ia,la->ogia', self.eris.oovv, ts, ts)
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
        int3 = -np.einsum('ic,ka,oc,kg->iaog', ts, ts, ts, ts)

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
        doi4 = np.einsum('ja,jb,mb,mg->ag', ls, ts, ts, ls)
        dag1 = -np.einsum('ie,oe->io', ts, ls)
        dag2 = np.einsum('ib,ob->io', ls, ls)
        dag3 = -np.einsum('ib,ob->io', ls, ts)
        dag4 = np.einsum('ib,jb,je,oe->io', ls, ts, ts, ls)

        # ts and ls contractions
        int1 = C * np.einsum('ib,ob,ma,mg->ogia', ls, ts, ts, ls)
        int2 = C * np.einsum('ja,jg,ie,oe->ogia', ls, ts, ts, ls)

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
                dL[p, q] += int1[o, g, i, a] + int2[o, g, i, a] + int3[o, g, i, a] + int4[o, g, i, a] + int5[o, g, i, a]
                dL[p, q] += int6[o, g, i, a] + int7[o, g, i, a] + int8[o, g, i, a] + int9[o, g, i, a] + int10[
                    o, g, i, a]
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
        dag3 = np.einsum('kibc,kc,ob->oi', self.eris.oovv, ts, ts)
        dag4 = -np.einsum('jiob,jb->oi', self.eris.ooov, ts)
        doi1 = -C * np.einsum('ja,jg->ag', ls, ts)
        doi2 = np.einsum('ja,jg->ag', fov, ts)
        doi3 = -C * np.einsum('ja,jg->ag', ls, ts)
        doi4 = np.einsum('jgba,jb->ag', self.eris.ovvv, ts)
        doi5 = np.einsum('kjab,jb,kg->ag', self.eris.oovv, ts, ts)

        # other contraction
        int1 = C * np.einsum('ib,jb,oa,jg->iaog', ls, ts, ts, ts)
        int2 = C * np.einsum('ja,jb,ob,ig->iaog', ls, ts, ts, ts)
        int3 = -np.einsum('igba,ob->iaog', self.eris.ovvv, ts)
        int4 = -np.einsum('jiac,oc,jg->iaog', self.eris.oovv, ts, ts)
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
                    dL[p, q] += doi1[a, g] + doi2[a, g] + doi3[a, g] + doi4[a, g] + doi5[a, g]
                dL[p, q] += int1[i, a, o, g] + int2[i, a, o, g] + int3[i, a, o, g] + int4[i, a, o, g] + int5[i, a, o, g]
                dL[p, q] += C * (-ts[o, a] * ts[i, g] + ls[i, g] * ts[o, a] - ls[o, a] * ts[i, g])
                dL[p, q] += self.eris.ovov[i, g, o, a]

        return dL

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
        J00 = self.dTdt(ts, ls, fsp, L)
        J01 = self.dTdl(ts, L)
        J10 = self.dLdt(ts, ls, fsp, L)
        J11 = self.dLdl(ts, ls, fsp, L)
        J = np.block([[J00, J01], [J10, J11]])

        # Solve J.Dx=-X
        Dx = np.linalg.solve(J, -X)
        # split Dx into Dt and Dl arrays
        Dt, Dl = np.split(Dx, 2)

        # build new t and l amplitudes
        tsnew = ts + Dt.reshape(nocc, nvir)
        lsnew = ls + Dl.reshape(nocc, nvir)

        return tsnew, lsnew

    def Gradient_Descent(self, alpha, ts, ls, fsp, L):

        nocc, nvir = ts.shape

        # make T1 and L1 eq. vectors and build X
        T1 = self.T1eq(ts, fsp).flatten()
        L1 = self.L1eq(ts, ls, fsp).flatten()
        X = np.concatenate((T1, L1))

        # build Jacobian
        J00 = self.dTdt(ts, ls, fsp, L)
        J01 = self.dTdl(ts, L)
        J10 = self.dLdt(ts, ls, fsp, L)
        J11 = self.dLdl(ts, ls, fsp, L)
        J = np.block([[J00, J01], [J10, J11]])
        # print J

        # make ts and ls vectors
        ts = ts.flatten()
        ls = ls.flatten()
        tls = np.concatenate((ts, ls))

        # build new t and l amplitudes
        tlsnew = tls - alpha * np.dot(J.transpose(), X)
        tsnew, lsnew = np.split(tlsnew, 2)

        tsnew = tsnew.reshape(nocc, nvir)
        lsnew = lsnew.reshape(nocc, nvir)

        return tsnew, lsnew

if __name__ == "__main__":
    # execute only if run as a script
    # test on water

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

    mol.basis = 'sto3g'
    mol.spin = 0
    mol.build()

    # generalize HF and CC
    mgf = scf.GHF(mol)
    mgf.kernel()
    mo_occ = mgf.mo_occ
    mocc = mgf.mo_coeff[:, mo_occ > 0]
    mvir = mgf.mo_coeff[:, mo_occ == 0]
    gnocc = mocc.shape[1]
    gnvir = mvir.shape[1]
    gdim = gnocc + gnvir

    mygcc = cc.GCCSD(mgf)
    geris = Eris.geris(mygcc) #mygcc.ao2mo(mgf.mo_coeff)
    gfs = geris.fock

    mccsg = Gccs(geris)

    gts = np.random.rand(gnocc, gnvir)
    #gts = gts / np.sum(gts)
    gls = np.random.rand(gnocc, gnvir)
    #gls = gls / np.sum(gls)

    print()
    print('##########################')
    print(' Test T and L equations   ')
    print('##########################')

    T1eq_1 = mccsg.T1eq(gts, gfs)
    T1eq_2 = CC_raw_equations.T1eq(gts, geris)

    L1eq_1 = mccsg.L1eq(gts, gls, gfs)
    L1eq_2 = CC_raw_equations.La1eq(gts, gls, geris)
    
    print()
    print("--------------------------------")
    print("Difference eq with inter and eq ")
    print("--------------------------------")
    print()
    print("T1 ")
    print(np.subtract(T1eq_1,T1eq_2))
    print()
    print("L1 ")
    print(np.subtract(L1eq_1, L1eq_2))
    print()

    print("--------------------------------")
    print(" ts_update with L1 reg          ")
    print("--------------------------------")

    print('ts updated with alpha = 0.0005')
    inter = mccsg.T1inter(gts,gfs)
    ts_up = mccsg.tsupdate_L1(gts,inter,0.0005)
    print(ts_up)
    print()

    print()
    print('####################')
    print(' TEST JACOBIAN      ')
    print('####################')
    print()

    print("-------------------------------------")
    print("TEST CCS Jacobian and Newton's method")
    print("-------------------------------------")

    mgrad = ccs_gradient(geris)

    print("Lamdba = 0")
    print()

    ts_G = gts.copy()*0.1
    ls_G = gls.copy()*0.1
    ite = 50
    for i in range(ite):
       tsnew, lsnew = mgrad.Newton(ts_G, ls_G, gfs, 0)
       ts_G = tsnew
       ls_G = lsnew

    print("t and l amplitudes after {} iteration".format(ite))
    print(ts_G)
    print(ls_G)


    print()
    print("######################################")
    print(" Test gamma GCCS                      ")
    print("######################################")
    print()

    print('Symmetrized and unsymmetrized gamma_GS')
    tmp = np.subtract(mccsg.gamma(gts,gls),mccsg.gamma_unsym(gts,gls))
    print(tmp)

    print()
    print('gamma for GS')
    print('symmetrized')
    g1 = mccsg.gamma(gts,gls)
    print(g1)
    print('unsymmetrized')
    g2 = mccsg.gamma_unsym(gts,gls)
    print(g2)
    print('Difference in Ek:')
    print('DEk= ', np.subtract(utilities.Ekin(mol,g1,AObasis=False,mo_coeff=mgf.mo_coeff),
          utilities.Ekin(mol,g2,AObasis=False,mo_coeff=mgf.mo_coeff)) )
    #print(np.subtract(mccsg.gamma(gts,gls), mccsg.gamma_es(gts, gls,0,1)))

    print()
    print('trace of transition rdm1 ')
    t1 = np.random.random((gnocc,gnvir))*0.1
    r1 = np.random.random((gnocc,gnvir))*0.1
    l1 = np.random.random((gnocc,gnvir))*0.1
    # orthogonalize r and l amp
    ln,rk = utilities.ortho(mol,l1,r1)
    # get r0 and l0
    r0k = mccsg.R0eq(0.1, t1, rk)
    l0n = mccsg.L0eq(0.1, t1, ln)
    tr_rdm1 = mccsg.gamma_tr(t1, ln, rk, r0k, l0n)
    print(tr_rdm1.trace())
    print()
    
    print('trace of rdm1 for excited state - nelec')
    # normalize r and l
    c = utilities.get_norm(r1,l1)
    l1 /= c
    # get r0 and l0
    r0 = mccsg.R0eq(0.1, t1, r1)
    l0 = mccsg.L0eq(0.1, t1, l1)
    rdm1 = mccsg.gamma_es(t1, l1, r1, r0, l0)
    print(rdm1.trace()-np.sum(mol.nelec))


    print()
    print("######################################")
    print(" Test R and L intermediates           ")
    print("######################################")
    print()
    
    import CC_raw_equations
    
    vn = np.zeros_like(gfs)

    ls = np.random.random((gnocc,gnvir))*0.1
    rs = ls.copy() 
    r0 = 0.1
    l0 = 0.1
    ts = np.zeros((gnocc, gnvir))
    Rinter = mccsg.R1inter(ts, gfs, vn)    #Fab, Fji, W, F, Zia, Pia
    Linter = mccsg.es_L1inter(ts, gfs, vn) #Fba, Fij, W, F, Zia, P

    print('Difference between R1 and L1 inter for t=0  and l1=r1 (should be zero)')
    for R,L in zip(Rinter,Linter):
       print(np.subtract(R,L))

    print()
    print('Difference between R1 and L1 equations for t=0 and l1=r1 (should be zero)')
    print('with intermediates')
    print(np.subtract(mccsg.R1eq(rs,r0,Rinter),mccsg.es_L1eq(ls,l0,Linter)))
    print('raw equations')
    print(np.subtract(CC_raw_equations.R1eq(ts,rs,r0,geris),CC_raw_equations.es_L1eq(ts,ls,l0,geris)))

    print()
    print('Difference between inter and raw equations for t=0')
    print('R1 difference')
    print(np.subtract(mccsg.R1eq(rs, r0, Rinter), CC_raw_equations.R1eq(ts, rs, r0, geris)))
    print('L1 difference')
    print(np.subtract(mccsg.es_L1eq(ls, l0, Linter), CC_raw_equations.es_L1eq(ts, ls, l0, geris)))

    print()
    print('Difference between inter and raw equations for t random')
    rs = np.random.random((gnocc,gnvir))*0.1
    r0 = 0.1
    l0 = 0.18
    ts = np.random.random((gnocc,gnvir))*0.1
    Rinter = mccsg.R1inter(ts, gfs, vn)
    Linter = mccsg.es_L1inter(ts, gfs, vn)
    print('R1 difference')
    print(np.subtract(mccsg.R1eq(rs,r0,Rinter),CC_raw_equations.R1eq(ts,rs,r0,geris)))
    print('L1 difference')
    print(np.subtract(mccsg.es_L1eq(ls,l0,Linter), CC_raw_equations.es_L1eq(ts, ls, l0, geris)))

    # Note: R and L intermediates are the same for HF basis (f off diag = 0) --> lambda = 0
    #       except the Zia intermediates, which contracts with r0 and l0
    #       note anymore when lambda > 0
    # Fab, Fji, W, F, Zia, Pia
    print()
    print('Difference between R1 and L1 inter for t random and l1=r1')
    for R, L in zip(Rinter, Linter):
        print('INTER')
        print(np.subtract(R, L))