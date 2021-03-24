#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################################################
#
# ECW-CCS v1
# -----------
# Experimentaly constrained wave function coupled cluster single
# ---------------------------------------------------------------
#
# Solver_CCS for SCF and Newton's method

###################################################################

import numpy as np
import utilities
from pyscf import lib



class Solver_CCS:
    def __init__(self, mycc, VX_exp, conv='tl', conv_thres=10**-8, tsini=None, lsini=None, diis=tuple(),
                 maxiter=80, maxdiis=20, CCS_grad=None):
        '''
        
        :param mycc: class containing the CCS functions and equations 
        :param VX_exp: object containing the functions to calculate Vexp and X2
        :param conv: string of the parameter for convergence check: 'Ep' or 'tl'
        :param conv_thres: convergence threshold 
        :param tsini: initial values for ts
        :param lsini: initial values for ls
        :param diis: tuple 'rdm1', 't' and/or 'l'
        :param maxiter: max number of SCF iteration
        :param maxdiis: maximum space for DIIS
        :param CCS_grad: object containing the XCW-CCS gradient and Newton's method
        '''

        # get nocc,nvir from ccs object
        self.nocc = mycc.nocc
        self.nvir = mycc.nvir

        # ts and ls initial
        if tsini is None:
            tsini = np.zeros((self.nocc, self.nvir))
        if lsini is None:
            lsini = np.zeros((self.nocc, self.nvir))

        # DIIS option
        self.diis = diis
        self.maxdiis = maxdiis

        self.Grad = CCS_grad

        self.mycc = mycc
        self.myVexp = VX_exp

        # convergence options
        self.maxiter = maxiter
        self.conv_thres = conv_thres
        if conv == 'Ep':
            self.Conv_check = self.Ep_check
        elif conv == 'l':
            self.Conv_check = self.l_check
        elif conv == 'tl':
            self.Conv_check = self.tl_check
        else:
            raise ValueError('Accepted convergence parameter is Ep, l or tl')

        self.fock = mycc.fock
        self.tsini = tsini
        self.lsini = lsini
        self.nocc, self.nvir = tsini.shape

    #####################
    # Convergence check
    #####################

    def Ep_check(self,dic):
        ts = dic.get('ts')
        fsp = dic.get('fsp')
        Ep = self.mycc.energy_ccs(ts, fsp)
        return Ep

    def l_check(self,dic):
        ls = dic.get('ls')
        return np.linalg.norm(ls)

    def tl_check(self,dic):
        ls = dic.get('ls')
        ts = dic.get('ts')
        conv = np.linalg.norm(ts+ls)
        return conv

    #############
    # SCF method
    #############

    def SCF(self, L, ts=None, ls=None, diis=None, alpha=None):

        '''
        SCF+DISS solver for the ECW-CCS equations with L1 regularization term

        :param L: Lambda, weight of the experimental potential
        :param ts: initial t1 amplitudes
        :param ls: initial l1 amplitudes
        :param diis: tuple of 'rdm1', 't' and/or 'l'
        :param alpha: L1 regularization parameter
        :return: [0] = convergence text
                 [1] = Ep(it)
                 [2] = X2(it) list of tuple: (X2,vmax,X2_Ek)
                 [3] = conv(it)
                 [4] = final gamma_calc
                 [5] = final ts and ls

        '''

        # ts and ls are initial value

        # initialize
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        if diis is None:
            diis = self.diis
        rdm1 = self.mycc.gamma(ts, ls)

        nocc = self.nocc
        nvir = self.nvir
        dim = nocc + nvir
        mycc = self.mycc
        VXexp = self.myVexp

        # initialize loop vectors
        conv = 0.
        Dconv = 1.
        ite = 0
        X2_ite = []
        Ep_ite = []
        conv_ite = []

        # initialze diis for ts,ls, rdm1
        if diis:
            if 'rdm1' in diis:
                adiis = lib.diis.DIIS()
                adiis.space = self.maxdiis
                adiis.min_space = 2
            if 't' in diis:
                tdiis = lib.diis.DIIS()
                tdiis.space = self.maxdiis
                tdiis.min_space = 2
            if 'l' in diis:
                ldiis = lib.diis.DIIS()
                ldiis.space = self.maxdiis
                ldiis.min_space = 2

        while Dconv > self.conv_thres:

            conv_old = conv

            # update fock matrix and store X2
            # ---------------------------------
            V, X2, vmax = VXexp.Vexp_update(rdm1, L, (0,0))
            fsp = np.subtract(self.fock, V)
            X2_ite.append((X2, vmax))

            # update t amplitudes
            # ---------------------------------
            T1inter = mycc.T1inter(ts, fsp)
            if alpha is None:
                ts = mycc.tsupdate(ts, T1inter)
            else:
                ts = mycc.tsupdate_L1(ts,T1inter,alpha)
            # apply DIIS
            if 't' in diis:
                ts_vec = np.ravel(ts)
                ts = tdiis.update(ts_vec).reshape((nocc, nvir))

            # update l amplitudes
            # ------------------------------------
            L1inter = mycc.L1inter(ts, fsp)
            if alpha is None:
                ls = mycc.lsupdate(ts, ls, L1inter)
            else:
                ls = mycc.lsupdate_L1(ts, ls, L1inter, alpha)
            # apply DIIS
            if 'l' in diis:
                ls_vec = np.ravel(ls)
                ls = ldiis.update(ls_vec).reshape((nocc, nvir))

            # calculated rdm1 from ts and ls
            # --------------------------------
            rdm1 = self.mycc.gamma(ts, ls)
            # apply DIIS
            if 'rdm1' in diis:
                rdm1_vec = np.ravel(rdm1)
                rdm1 = adiis.update(rdm1_vec).reshape((dim, dim))

            # calculate E'
            # -------------------------------------------
            Ep = mycc.energy_ccs(ts, fsp)
            Ep_ite.append(Ep)

            # checking convergence
            # --------------------------------------------
            dic = {'ts': ts, 'ls': ls, 'fsp': fsp}
            conv = self.Conv_check(dic)
            if ite > 0:
                Dconv = abs(conv - conv_old)
            conv_ite.append(Dconv)

            # print convergence infos
            # --------------------------------------------
            if ite >= self.maxiter:
                Conv_text = 'Max iteration reached'
                break
            if Dconv > 10.:
                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L, ite)
                break

            ite += 1

        else:
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L, ite)

        return Conv_text, np.asarray(Ep_ite), np.asarray(X2_ite), np.asarray(conv_ite), rdm1, (ts, ls)

    ###################
    # Gradient method
    ###################

    def Gradient(self, L, method='newton', ts=None, ls=None, diis=tuple(), beta=0.1):

        '''
        Solver the ECW-CCS equations with gradient based methods

        :param L: experimental weight lambda
        :param method: 'newton' or 'descend'
        :param ts: initial ts amplitudes
        :param ls: initial ls amplitudes
        :param diis: apply diis to ('Ep','t','tl')
        :param beta: step for steepest descend
        :return: [0] = convergence text
                 [1] = Ep(it)
                 [2] = X2(it) (list of tuple: (X2,vmax,X2_Ek))
                 [3] = conv(it)
                 [4] = converged gamma_calc
                 [5] = final ts and ls
        '''

        # initialize
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        rdm1 = self.mycc.gamma(ts, ls)

        nocc = self.nocc
        nvir = self.nvir
        dim = nocc + nvir
        mycc = self.mycc
        VXexp = self.myVexp

        # initialize loop vectors
        conv = 0.
        Dconv = 1.
        ite = 0
        X2_ite = []
        Ep_ite = []
        conv_ite = []

        # initialze diis for ts,ls, rdm1
        if diis:
            if 'rdm1' in diis:
                adiis = lib.diis.DIIS()
                adiis.space = self.maxdiis
                adiis.min_space = 2
            if 't' in diis:
                tdiis = lib.diis.DIIS()
                tdiis.space = self.maxdiis
                tdiis.min_space = 2
            if 'l' in diis:
                ldiis = lib.diis.DIIS()
                ldiis.space = self.maxdiis
                ldiis.min_space = 2

        while Dconv > self.conv_thres:

            conv_old = conv

            # update fock matrix and store X2
            # ---------------------------------
            V, X2, vmax = VXexp.Vexp_update(rdm1, L, (0,0))
            fsp = np.subtract(self.fock, V)
            X2_ite.append((X2, vmax))

            # update t and l amplitudes
            # ---------------------------------
            if method == 'newton':
                ts, ls = self.Grad.Newton(ts, ls, fsp, L)
            elif method == 'descend':
                ts, ls = self.Grad.Gradient_Descent(beta, ts, ls, fsp, L)
            # apply DIIS
            if 't' in diis:
                ts_vec = np.ravel(ts)
                ts = tdiis.update(ts_vec).reshape((nocc, nvir))
            if 'l' in diis:
                ls_vec = np.ravel(ls)
                ls = ldiis.update(ls_vec).reshape((nocc, nvir))

            # calculated rdm1 from ts and ls
            # --------------------------------
            rdm1 = self.mycc.gamma(ts, ls)
            # apply DIIS
            if 'rdm1' in diis:
                rdm1_vec = np.ravel(rdm1)
                rdm1 = adiis.update(rdm1_vec).reshape((dim, dim))

            # calculate E' and Ekin             
            # -------------------------------------------
            Ep = mycc.energy_ccs(ts, fsp)
            Ep_ite.append(Ep)

            # checking convergence 
            # --------------------------------------------
            dic = {'ts': ts, 'ls': ls, 'fsp': fsp}
            conv = self.Conv_check(dic)
            conv_ite.append(conv)
            if ite > 0:
                Dconv = abs(conv - conv_old)

            # print convergence infos
            # --------------------------------------------
            if ite >= self.maxiter:
                Conv_text = 'Max iteration reached'
                break
            if Dconv > 10.:
                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L, ite)
                break

            ite += 1

        else:
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L, ite)

        return Conv_text, np.asarray(Ep_ite), np.asarray(X2_ite), np.asarray(conv_ite), rdm1, (ts, ls)

    ################################
    # ECW-CCS_L1 solver 
    ################################

    def L1_grad(self, L, alpha, chi, ts=None, ls=None, diis=tuple()):

        '''
        CCS+L1 solver as described in Ivanov et al. Molecular Physics, 115(21–22), 2017
        were lambda in the paper is alpha in the present code
        NOTE: conv_thres is also taken as the treshold for the amplitudes in the calculation of W

        :param L: lambda, weight of the experimental potential
        :param alpha: weight of the L1 term
        :param chi: step of the steepest descend
        :param ts: initial ts
        :param ls: initial ls
        :return: [0] = convergence text
                 [1] = Ep(it)
                 [2] = X2(it)
                 [3] = conv(it)
                 [4] = final gamma_calc
                 [5] = final ts and ls
        '''

        # initialize
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        rdm1 = self.mycc.gamma(ts, ls)

        # get parameters from self
        nocc = self.nocc
        nvir = self.nvir
        dim = nocc + nvir
        mycc = self.mycc
        VXexp = self.myVexp
        faa = np.diagonal(self.fock[nocc:, nocc:])
        fii = np.diagonal(self.fock[:nocc, :nocc])

        # initialize loop vectors
        conv = 0.
        Dconv = 1.
        ite = 0
        X2_ite = []
        Ep_ite = []
        conv_ite = []
        ls_norm = []

        # initialze diis for ts,ls, rdm1
        if diis:
            if 'rdm1' in diis:
                adiis = lib.diis.DIIS()
                adiis.space = self.maxdiis
                adiis.min_space = 2
            if 't' in diis:
                tdiis = lib.diis.DIIS()
                tdiis.space = self.maxdiis
                tdiis.min_space = 2
            if 'l' in diis:
                ldiis = lib.diis.DIIS()
                ldiis.space = self.maxdiis
                ldiis.min_space = 2

        while Dconv > self.conv_thres:

            conv_old = conv

            # update fock matrix and store X2
            # ---------------------------------
            V, X2, vmax = VXexp.Vexp_update(rdm1, L, (0,0))
            fsp = np.subtract(self.fock, V)
            X2_ite.append((X2, vmax))

            # calculate T and L equations
            # ---------------------------------
            Teq = mycc.T1eq(ts, fsp)
            Leq = mycc.es_L1eq(ts, ls, fsp)

            # calculate subdifferential dW
            # ----------------------------------
            dWT = utilities.subdiff(Teq,ts,alpha,thres=self.conv_thres)
            dWL = utilities.subdiff(Leq,ls,alpha,thres=self.conv_thres)

            # update t and l amplitudes and apply P_0
            # -----------------------------------------
            for i in range(nocc):
                for a in range(nvir):

                    #ts
                    Xj = ts[i,a]-chi*dWT[i,a]/(-fii[i]+faa[a])
                    tmp = Xj*ts[i,a]
                    if tmp > self.conv_thres:
                        ts[i,a] = Xj
                    elif tmp < self.conv_thres:
                        ts[i,a] = 0.

                    #ls
                    Xj = ls[i,a]-chi*dWL[i,a]/(-fii[i]+faa[a])
                    tmp = Xj*ls[i,a]
                    if tmp > self.conv_thres:
                        ls[i,a] = Xj
                    elif tmp < self.conv_thres:
                        ls[i,a] = 0.

            # apply DIIS
            if 't' in diis:
                ts_vec = np.ravel(ts)
                ts = tdiis.update(ts_vec).reshape((nocc, nvir))
            if 'l' in diis:
                ls_vec = np.ravel(ls)
                ls = ldiis.update(ls_vec).reshape((nocc, nvir))

            # Update rdm1 from ts and ls
            # --------------------------------
            rdm1 = self.mycc.gamma(ts, ls)
            # apply DIIS
            if 'rdm1' in diis:
                rdm1_vec = np.ravel(rdm1)
                rdm1 = adiis.update(rdm1_vec).reshape((dim, dim))

            # calculate E' and Ekin
            # -------------------------------------------
            Ep = mycc.energy_ccs(ts, fsp)
            Ep_ite.append(Ep)

            # checking convergence using norm(ts+ls)
            # --------------------------------------------
            dic = {'ts': ts, 'ls': ls, 'fsp': fsp}
            conv = self.Conv_check(dic)
            conv_ite.append(conv)
            if ite > 0:
                Dconv = abs(conv - conv_old)

            # print convergence infos
            # --------------------------------------------
            if ite >= self.maxiter:
                Conv_text = 'Max iteration reached'
                break
            if Dconv > 2.:
                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L, ite)
                break

            ite += 1

        else:
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L, ite)

        return Conv_text, np.asarray(Ep_ite), np.asarray(X2_ite), np.asarray(conv_ite), rdm1, (ts, ls)

#-----------------------------------------------------------------------------------------------------------------------

####################################
# CCSD SOLVER
####################################

class Solver_CCSD:
    def __init__(self, mycc, VX_exp, conv='tl', conv_thres=10**-8, tsini=None, lsini=None, tdini=None, ldini=None, diis=tuple(),
                 maxiter=50, maxdiis=15):
        '''
        Solver Class for the ECW-CCSD equations
        
        :param mycc: class containing the CCSD functions and equations
        :param VX_exp: object containing the functions to calculate Vexp and X2
        :param conv: string of the parameter for convergence check: 'Ep' or 'tl'
        :param conv_thres: convergence threshold 
        :param tsini: initial values for t1, if None = 0
        :param lsini: initial values for l1, if None = 0
        :param tdini: initial values for t2, if None taken from mp2
        :param ldini: initial values for l2, if None taken from mp2
        :param diis: tuple 'rdm1', 't' and/or 'l'
        :param maxiter: max number of SCF iteration, default = 50
        :param maxdiis: maximum space for DIIS, default = 15
        '''

        # get nocc,nvir from ccs object
        self.nocc = mycc.nocc
        self.nvir = mycc.nvir

        # fock matrix
        self.fock = mycc.fock

        # ts and ls initial
        if tsini is None:
            tsini = np.zeros((self.nocc, self.nvir))
        if lsini is None:
            lsini = np.zeros((self.nocc, self.nvir))

        # td and ld initial
        if tdini is None:
            mo_e = np.diagonal(self.fock)
            fia = mo_e[:self.nocc, None] - mo_e[None, self.nocc:]
            eijab = lib.direct_sum('ia,jb->ijab', fia, fia)
            tdini = mycc.eris.oovv / eijab
            ldini = tdini.copy()
            del fia
            del eijab
            del mo_e

        self.tsini = tsini
        self.lsini = lsini
        self.tdini = tdini
        self.ldini = ldini

        # DIIS option
        self.diis = diis
        self.maxdiis = maxdiis

        self.mycc = mycc
        self.myVexp = VX_exp

        # convergence options
        self.maxiter = maxiter
        self.conv_thres = conv_thres
        if conv == 'Ep':
            self.Conv_check = self.Ep_check
        elif conv == 'l':
            self.Conv_check = self.l_check
        elif conv == 'tl':
            self.Conv_check = self.tl_check
        else:
            raise ValueError('Accepted convergence parameter is Ep, l or tl')


    #####################
    # Convergence check
    #####################

    def Ep_check(self, dic):
        ts = dic.get('ts')
        td = dic.get('td')
        fsp = dic.get('fsp')
        Ep = self.mycc.energy(ts, td, fsp)
        return Ep

    def l_check(self, dic):
        ls = dic.get('ls')
        ld = dic.get('ld')
        arr = np.concatenate((ls.flatten(),ld.flatten()))
        norm = np.linalg.norm(arr)
        return norm

    def tl_check(self, dic):
        ls = dic.get('ls').flatten()
        ts = dic.get('ts').flatten()
        ld = dic.get('ld').flatten()
        td = dic.get('td').flatten()
        arr = np.concatenate((ls+ts,ld+td))
        norm = np.linalg.norm(arr)
        return norm

    def x2_check(self, ts, ls, fsp):
        return

    #############
    # SCF method
    #############

    def SCF(self, L, ts=None, ls=None, td=None, ld=None, alpha=None, diis=None):
        '''
        Standard SCF+DIIS solver for the GS-ECW-CCSD equations with additional L1 reg term
        
        :param L: weigth of experimental potential    
        :param ts: t1 amplitude
        :param ls: lambda amplitudes
        :param td: t2 amplitudes
        :param ld: lambda2 amplitudes
        :param alpha: L1 reg parameter 
        :param diis: tuple ('t','l','tl' and/or 'rdm1')
        :return: [0] = convergence text
                 [1] = Ep(it)
                 [2] = X2(it)
                 [3] = conv(it)
                 [4] = last gamma_calc
                 [5] = list [t1,l2,t2,l2] with final amplitudes
        '''

        # initialize
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        if td is None:
            td = self.tdini
            ld = self.ldini
        Conv_text = ''
        if diis is None:
            diis = self.diis

        nocc = self.nocc
        nvir = self.nvir
        dim = nocc + nvir
        mycc = self.mycc
        VXexp = self.myVexp

        # initialize loop vectors
        conv = 0.
        conv_ite = []
        Dconv = 1.0
        ite = 0
        X2_ite = []
        Ep_ite = []

        # initialze diis for ts,ls, rdm1
        if diis:
            if 'rdm1' in diis:
                adiis = lib.diis.DIIS()
                adiis.space = self.maxdiis
                adiis.min_space = 2
            if 't' in diis:
                ts_diis = lib.diis.DIIS()
                ts_diis.space = self.maxdiis
                ts_diis.min_space = 2
                td_diis = lib.diis.DIIS()
                td_diis.space = self.maxdiis
                td_diis.min_space = 2
            if 'l' in diis:
                ls_diis = lib.diis.DIIS()
                ls_diis.space = self.maxdiis
                ls_diis.min_space = 2
                ld_diis = lib.diis.DIIS()
                ld_diis.space = self.maxdiis
                ld_diis.min_space = 2

        while Dconv > self.conv_thres:

            conv_old = conv

            # calculated rdm1 from t and l
            # --------------------------------
            rdm1 = self.mycc.gamma(ts, td, ls, ld)
            # apply DIIS
            if 'rdm1' in diis:
                rdm1_vec = np.ravel(rdm1)
                rdm1 = adiis.update(rdm1_vec).reshape((dim, dim))

            # update fock matrix and store X2
            # ---------------------------------
            V, X2, vmax = VXexp.Vexp_update(rdm1, L, (0,0))
            fsp = np.subtract(self.fock, V)
            X2_ite.append((X2, vmax))

            # Store Ep energy
            # ----------------------------------------------
            Ep_ite.append(mycc.energy(ts, td, fsp))

            # update t amplitudes
            # ---------------------------------
            ts, td = mycc.tupdate(ts, td, fsp, alpha=alpha)
            # apply DIIS
            if 't' in diis:
                ts_vec = np.ravel(ts)
                ts = ts_diis.update(ts_vec).reshape((nocc, nvir))
                td_vec = np.ravel(td)
                td = td_diis.update(td_vec).reshape((nocc, nocc, nvir, nvir))

            # update l amplitudes
            # ------------------------------------
            ls, ld = mycc.lupdate(ts, td, ls, ld, fsp, alpha=alpha)
            # apply DIIS
            if 'l' in diis:
                ls_vec = np.ravel(ls)
                ls = ls_diis.update(ls_vec).reshape((nocc, nvir))
                ld_vec = np.ravel(ld)
                ld = ld_diis.update(ld_vec).reshape((nocc, nocc, nvir, nvir))

            # check for convergence
            # -----------------------------------
            dic = {'ts': ts, 'ls': ls, 'fsp': fsp, 'td': td, 'ld': ld}
            conv = self.Conv_check(dic)
            if ite > 0:
                Dconv = abs(conv - conv_old)
            conv_ite.append(Dconv)
            del dic

            # print convergence infos
            # --------------------------------------------
            if ite >= self.maxiter:
                Conv_text = 'Max iteration reached'
                break
            # if Ep - OLDEp > 1.:
            if Dconv > 1.0:
                Conv_text = 'Diverges for lambda = {} and alpha={} after {} iterations'.format(L, alpha, ite)
                break

            ite += 1

        else:
            Conv_text = 'Convergence reached for lambda= {} and alpha={}, after {} iteration'.format(L, alpha, ite)

        return Conv_text, np.asarray(Ep_ite), np.asarray(X2_ite), np.asarray(conv_ite), rdm1, [ts,ls,td,ld]


if __name__ == "__main__":
    from pyscf import gto, scf, cc
    import CCS, CCSD, Eris
    import exp_pot
    import gamma_exp

    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = '6-31+g**'
    mol.spin = 0
    mol.build()

    # GHF calc
    mgf = scf.GHF(mol)
    mgf.kernel()
    mo_occ = mgf.mo_occ
    mocc = mgf.mo_coeff[:, mo_occ > 0]
    mvir = mgf.mo_coeff[:, mo_occ == 0]
    gnocc = mocc.shape[1]
    gnvir = mvir.shape[1]
    gdim = gnocc + gnvir

    # PySCF Eris
    mygcc = cc.GCCSD(mgf)
    geris = Eris.geris(mygcc)
    gfs = geris.fock

    # rdm1_exp
    gexp = gamma_exp.Gexp(mol, 'HF')
    field = [0.05, 0.02, 0.]
    gexp.Vext(field)
    gexp.build()
    rdm1_exp = gexp.gamma_ao
    exp = np.full((2,2),None)
    exp[0,0] = ['mat',rdm1_exp]

    print()
    print('################')
    print('# ECW-CCS test  ')
    print('################')

    # GCCS object
    mccsg = CCS.Gccs(geris)
    # Gradient objct
    mygrad = CCS.ccs_gradient(geris)

    # Vexp object
    VXexp = exp_pot.Exp(exp, mol, mgf.mo_coeff)

    # initial ts and ls
    #tsini = np.random.rand(gnocc,gnvir)*0.01
    #lsini = np.random.rand(gnocc,gnvir)*0.01
    tsini = np.zeros((gnocc,gnvir))
    lsini = np.zeros((gnocc,gnvir))

    # convergence options
    maxiter = 80
    conv_thres = 10 ** -6
    diis = ('')  # must be tuple

    # initialise Solver_CCS Class
    Solver_CCS = Solver_CCS(mccsg, VXexp,'tl',conv_thres, tsini=tsini, lsini=lsini, diis=diis, CCS_grad=mygrad)

    # Solve for L = 0
    L = 0
    Results = Solver_CCS.SCF(L)
    #Results = Solver_CCS.Gradient(L,method='newton', alpha=0.5)
    print(Results[0])
    print()
    print('Ep')
    print(Results[1])
    print()
    print('Conv')
    print(Results[3])
    print()
    print()


    print('########################')
    print('# ECW-CCSD test         ')
    print('########################')

    # CHECK for conv=tl 10^-8
    # - L=0, alpha=0,   t1=l1=rand --> SCF converges after 18 it
    # - L=0, alpha=0,   t1=l1=0    --> SCF converges after 18 it
    # - L=0, alpha=0.1, t1=l1=0

    # GCCS object
    mccsd = CCSD.GCC(geris)

    # Vexp object
    #VXexp = exp_pot.Exp(exp,mol,mgf.mo_coeff)

    # initial ts and ls
    #tsini = np.random.rand(gnocc,gnvir)*0.1
    #lsini = np.random.rand(gnocc,gnvir)*0.1

    # convergence options
    maxiter = 50
    conv_thres = 10 ** -8
    diis = ('')  # must be tuple

    # initialise Solver_CCS Class
    Solver = Solver_CCSD(mccsd, VXexp, conv='Ep', conv_thres=conv_thres, diis=diis, maxiter=80, maxdiis=20)

    # Solve for L
    L = 0
    Results = Solver.SCF(L,alpha=None)
    print(Results[0])
    print()
    print('conv')
    print(Results[3])
    print()

    print('PySCF ECCSD difference=', mygcc.kernel()[0]-Results[1][-1])
    print()
    
    ## Solve for L = 0.05
    #L = 0.05
    #Results = Solver.SCF(L)
    #print(Results[0])
    #print()
    #print('conv')
    #print(Results[3])
    #print()

    #print('PySCF ECCSD difference=', mygcc.kernel()[0]-Results[1][-1])
    #print()

    ## Solve for L = 0.
    #L = 0
    #Results = Solver.SCF(L, alpha=0.01)
    #print(Results[0])
    #print()
    #print('conv')
    #print(Results[3])
    #print()

    #print('PySCF ECCSD difference=', mygcc.kernel()[0] - Results[1][-1])
    #print()

   #print('#####################')
   #print('# Test L1 algorithm  ')
   #print('#####################')

    ## CHECK
    ## L1_grad
    ## - lambda=0,   alpha=0,   random initial ts/ls, chi=0.5, conv=10-6 --> 'tl' converges in 8 it --> OK
    ## - lambda=0,   alpha=0.1, random initial ts/ls, chi=0.5, conv=10-6 --> 'tl' converges in 3 it --> OK
    ## SCF+L1
    ## - lambda=0, alpha=0,   random initial ts/ls, conv=10-6 --> 'tl' converges in 17 it --> OK
    ## - lambda=0, alpha=0.1, random initial ts/ls, conv=10-6 --> 'tl' converges in 5 it  --> OK


    #mccsd = CCS.Gccs(geris)

    ## Vexp object
    #VXexp = exp_pot.Exp(exp,mol,mgf.mo_coeff)

    ## convergence options
    #maxiter = 20
    #conv_thres = 10 ** -6
    #diis = ('')  # ('rdm')  # must be tuple

    ## initial ts and ls
    ##tsini = np.random.rand(gnocc, gnvir) * 0.01
    ##lsini = np.random.rand(gnocc, gnvir) * 0.01
    #tsini = np.zeros((gnocc,gnvir))
    #lsini = np.zeros((gnocc,gnvir))

    ## initialise Solver_CCS Class
    #Solver = Solver_CCS(mccsd, VXexp, 'tl', conv_thres, diis=diis, maxiter=maxiter, maxdiis=20, tsini=tsini, lsini=lsini)

    ## Solve
    #alpha = 0.
    #L = 0.
    #chi = 0.5
    ## L1 in paper
    #Results = Solver.L1_grad(L, alpha, chi=chi, diis=diis)
    ## L1+SCF
    ##Results = Solver.SCF(L,ts=tsini,ls=lsini,diis=diis,alpha=alpha)

    #print(Results[0])
    #print()
    #print('tl_conv')
    #print(Results[3])
    #print()
    #print('X2 vmax')
    #print(Results[2][-1])
