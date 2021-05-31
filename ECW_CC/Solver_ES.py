#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################################################
#
# ECW-CC
# -----------
# Experimentally constrained wave function coupled cluster
# ---------------------------------------------------------------
#
# SCF Solver_CCS for coupled T, 0L, R and L equations
#
###################################################################

import numpy as np
import copy
from pyscf import lib
#from . import utilities
import utilities
from tabulate import tabulate
#import slepc4py
#from petsc4py import PETSc
#from slepc4py import SLEPc

# print float format
format_float = '{:.4e}'

class Solver_ES:
    def __init__(self, mycc, Vexp_class, rn_ini, r0_ini, ln_ini, l0_ini, tsini=None, lsini=None, conv_var='tl', conv_thres=10 ** -6, diis=tuple(),
                 maxiter=80, maxdiis=20, tablefmt='rst'):
        '''
        Solves the ES-ECW-CCS equations for V00, Vnn, V0n and Vn0 (thus not including Vnk with n != k)
        -> only state properties and GS-ES transition properties

        :param mycc: ECW.CCS object containing T, Lambda,R and L equations
        :param Vexp_class: ECW.Vexp_class, class containing the exp data
        :param tsini: t and lambda initial values for the GS and r,l,r0,l0 initial values for all needed states
                  --> list of matrix (for r and l) and float for r0 and l0
                  --> must be numpy array
        :param conv_var: convergence criteria: 'Ep', 'l' or 'tl'
        :param conv_thres: convergence threshold
        :param tsini: initial value for t
        :param lsini: initial value for Lambda
        :param diis: tuple containing the variables on which to apply diis: rdm1, t, L, r, l
        :param maxiter: max number of SCF iteration
        :param maxdiis: diis space
        :param tablefmt: format for table printed using tabulate package (eg: 'rst' or 'latex')
        '''

        # tabulate format
        self.tablefmt = tablefmt

        # check length
        if len(rn_ini) != len(Vexp_class.exp_data)-1:
            raise ValueError('number of r matrix for the initial value must match the number of exp data')

        # get nocc,nvir from ccs object
        self.nocc = mycc.nocc
        self.nvir = mycc.nvir

        # ts and Lambda initial
        if tsini is None:
            tsini = np.zeros((self.nocc, self.nvir))
        if lsini is None:
            lsini = np.zeros((self.nocc, self.nvir))
        self.tsini = tsini.copy()
        self.lsini = lsini.copy()

        # l and r initial
        if ln_ini is None:
            ln_ini = copy.deepcopy(rn_ini)
        self.rn_ini = copy.deepcopy(rn_ini)
        self.ln_ini = copy.deepcopy(ln_ini)

        # r0 and l0 initial
        self.r0_ini = copy.deepcopy(r0_ini)
        self.l0_ini = copy.deepcopy(l0_ini)

        self.nbr_states = len(rn_ini)

        # DIIS option
        self.diis = diis
        self.maxdiis = maxdiis

        # CC object
        self.mycc = mycc

        # exp_pot object
        self.Vexp_class = Vexp_class

        # convergence options
        self.maxiter = maxiter
        self.conv_thres = conv_thres
        if conv_var == 'Ep':
            self.Conv_check = self.Ep_check
        elif conv_var == 'rl':
            self.Conv_check = self.rl_check
        elif conv_var == 'tl':
            self.Conv_check = self.tl_check
        else:
            raise ValueError('Accepted convergence parameter is Ep, tl or rl')
        self.conv_var = conv_var

        self.nocc, self.nvir = tsini.shape

    #####################
    # Convergence check
    #####################

    def Ep_check(self, dic):
        ts = dic.get('ts')
        fsp = dic.get('fsp')
        Ep = self.mycc.energy_ccs(ts, fsp)
        return Ep

    def tl_check(self, dic):
        ls = dic.get('ls')
        ts = dic.get('ts')
        return ts + ls

    def rl_check(self, dic):
        rn = dic.get('rn')
        ln = dic.get('ln')
        ans = np.zeros_like(rn[0])
        for r, l in zip(rn, ln):
            ans += (r*l)
        return ans

    #############
    # SCF method
    #############

    def SCF(self, L, ts=None, ls=None, rn=None, ln=None, r0n=None, l0n=None, diis=None, S_AO=None):
        '''
        !!!
        SCF rough solver for R and L equations: takes care of the spin symmetry but
        only works when no spatial symmetry is present
        !!!

        :param L: matrix of experimental weight
            -> shape(L)[0] = total number of states (GS+ES)
            -> typically L would be the same for properties obtained from the same experiment
        :param ts, ls, rn, ln, r0n, l0n: amplitudes
            -> ln,rn,l0 and r0 are list with length = nbr of excited states
        :param diis:
        :param S_AO: AOs overlap matrix in G format
        :return:
        '''

        Vexp_class = self.Vexp_class
        nbr_states = self.nbr_states

        # initialize r and l vectors
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        if rn is None:
            rn = self.rn_ini
            ln = self.ln_ini
        if r0n is None:
            r0n = self.r0_ini
            l0n = self.l0_ini
        rnew = [None] * self.nbr_states
        lnew = [None] * self.nbr_states
        r0new = [None] * self.nbr_states
        l0new = [None] * self.nbr_states

        # initialize total spin list
        Spin = np.zeros(nbr_states)
        
        # check length
        if L.shape != Vexp_class.Vexp.shape:
            raise ValueError('Shape of weight factor must be equal to shape of Vexp vectors:', Vexp_class.Vexp.shape)

        # diis
        if diis is None:
            diis = self.diis

        nocc = self.nocc
        nvir = self.nvir
        dim = nocc + nvir
        mycc = self.mycc

        # initialize X2 and Ep array
        X2 = np.zeros((nbr_states + 1, nbr_states + 1))
        Ep = np.zeros((nbr_states+1,2))
        
        # initialize loop vectors and printed information
        conv = 0.
        Dconv = 1.
        ite = 0
        X2_ite = []
        Ep_ite = []
        conv_ite = []

        # initialize diis for ts, lam, rs, ls and rdm1
        if 'rdm1' in diis:
            dm_diis = []
            tr_diis = []
            for n in range(self.nbr_states+1):
                tmp = lib.diis.DIIS()
                tmp.space = self.maxdiis
                tmp.min_space = 2
                dm_diis.append(tmp)
                tmp = lib.diis.DIIS()
                tmp.space = self.maxdiis
                tmp.min_space = 2
                tr_diis.append(tmp)
        if 't' in diis:
            tdiis = lib.diis.DIIS()
            tdiis.space = self.maxdiis
            tdiis.min_space = 2
        if 'lam' in diis:
            lamdiis = lib.diis.DIIS()
            lamdiis.space = self.maxdiis
            lamdiis.min_space = 2
        if 'r' in diis:
            rdiis = []
            for i in range(nbr_states):
                rdiis.append(lib.diis.DIIS())
                rdiis[i].space = self.maxdiis
                rdiis[i].min_space = 2
        if 'l' in diis:
            ldiis = []
            for i in range(nbr_states):
                ldiis.append(lib.diis.DIIS())
                ldiis[i].space = self.maxdiis
                ldiis[i].min_space = 2

        table = []
        # First line of printed table
        headers = ['ite',str(self.conv_var)]
        for i in range(nbr_states):
            if i == 0:
                headers.extend(['ES {}'.format(i+1), 'norm', 'X2_r', 'X2_l', '2S+1', 'r0', 'l0','Er', 'El'])
            else:
                headers.extend(['ES {}'.format(i + 1), 'norm', 'X2_r', 'X2_l', '2S+1', 'r0', 'l0', 'Er', 'El', 'Ortho wrt ES 1'])

        while Dconv > self.conv_thres:

            #
            # Initialize
            # ---------------------------------------

            # todo: initialize needed dm outside of loop according to exp_data
            # nbr_states = nbr of excited states
            fsp = [None] * (nbr_states+1)
            rdm1 = [None]*(nbr_states+1)
            tr_rdm1 = [None]*nbr_states
            conv_old = conv

            #
            # calculate needed rdm1 and tr_rdm1 for all states
            # -------------------------------------------------

            # GS
            if Vexp_class.exp_data[0,0] is not None:
               rdm1[0] = mycc.gamma(ts, ls)

            # ES
            for n in range(nbr_states):
                
                # calculate rdm1 for state n if diagonal exp data are present
                if Vexp_class.exp_data[n+1, n+1] is not None:
                    rdm1[n+1] = mycc.gamma_es(ts, ln[n], rn[n], r0n[n], l0n[n])
                    
                # calculate tr_rdm1 if transition exp data are present
                if Vexp_class.exp_data[0, n+1] is not None:
                    # right dm1 <Psi_k|aa|Psi_n>
                    tr_r = mycc.gamma_tr(ts, ln[n], 0, 1, l0n[n])
                    # left dm1 <Psi_n|aa|Psi_k>
                    tr_l = mycc.gamma_tr(ts, 0, rn[n], r0n[n], 1)
                    tr_rdm1[n] = list((tr_r, tr_l))

            # apply DIIS on rdm1
            if 'rdm1' in diis:
                for n in range(nbr_states+1):
                    if rdm1[n] is not None:
                        rdm_vec = rdm1[n].flatten()
                        rdm_vec = dm_diis[n].update(rdm_vec)
                        rdm1[n] = rdm_vec.reshape((dim, dim))
                    if tr_rdm1[n-1] is not None:
                        tr_vec_r = np.asarray(tr_rdm1[n-1][0])
                        tr_vec_r = tr_vec_r.flatten()
                        tr_vec_l = np.asarray(tr_rdm1[n-1][1])
                        tr_vec_l = tr_vec_l.flatten()
                        tr_vec = np.concatenate((tr_vec_r, tr_vec_l))
                        tr_vec = tr_diis[n-1].update(tr_vec)
                        tr_rdm1[n-1] = list((tr_vec[:dim*dim].reshape(dim, dim), tr_vec[dim*dim:].reshape(dim, dim)))
                        
            #
            # Update Vexp, calculate effective fock matrices and store X2,vmax
            # ------------------------------------------------------------------

            # GS
            if rdm1[0] is not None:
                V, x2, vmax = Vexp_class.Vexp_update(rdm1[0], L[0, 0], (0, 0))
                fsp[0] = np.subtract(mycc.fock,V)
                X2[0, 0] = x2
            #else:
            #    fsp[0] = fock.copy()

            # ES
            for j in range(nbr_states):
                n = j+1

                if rdm1[n] is not None:
                    V, x2, vmax = Vexp_class.Vexp_update(rdm1[n], L[j, j], (n, n))
                    fsp[n] = np.subtract(mycc.fock, V)
                    X2[n, n] = x2
                #else:
                #    fsp[n] = fock.copy()

                if tr_rdm1[j] is not None:
                    v, X2[n, 0], vmax = Vexp_class.Vexp_update(tr_rdm1[j][0], L[0, j], (n, 0))
                    v, X2[0, n], vmax = Vexp_class.Vexp_update(tr_rdm1[j][1], L[0, j], (0, n))
            del v

            X2_ite.append(X2)

            # CARREFUL WITH THE SIGN OF Vexp !
            # for transition case vn = -Vexp
            # CARREFUL: the Vexp elements are not scaled with lambda

            #
            # update t amplitudes
            # ---------------------------------------------------

            vexp = -L[0, 1:]*Vexp_class.Vexp[0, 1:]
            T1inter = mycc.T1inter(ts, fsp[0])
            ts = mycc.tsupdate(ts, T1inter, rsn=rn, r0n=r0n, vn=vexp)
            # apply DIIS

            if 't' in diis:
                ts_vec = np.ravel(ts)
                ts = tdiis.update(ts_vec).reshape((nocc, nvir))

            del T1inter

            #
            # update l (Lambda) amplitudes for the GS
            # ----------------------------------------

            L1inter = mycc.L1inter(ts, fsp[0])
            vexp = -L[1:, 0] * Vexp_class.Vexp[1:, 0]
            ls = mycc.lsupdate(ts, ls, L1inter, rsn=rn, lsn=ln, r0n=r0n, l0n=l0n, vn=vexp)
            # apply DIIS

            if 'lam' in diis:
                ls_vec = np.ravel(ls)
                ls = lamdiis.update(ls_vec).reshape((nocc, nvir))

            del vexp, L1inter

            #
            # Update En_r/En_l and r, r0, l and l0 amplitudes for each ES
            # ------------------------------------------------------------
            print()
            print('-------------------')
            print('START, ite ', ite)
            print('-------------------')

            for i in range(nbr_states):
                print()
                print('State ', i)
                print('initial r')
                print(rn[i])
                print('initial l')
                print(ln[i])
                print('initial t')
                print(ts)
                print('initial lambda')
                print(ls)
                print('initial r0 and l0')
                print(r0n[i], l0n[i])
                print()

                # todo= most element in Rinter and Linter dot not depend on Vexp -> calculate ones for all states

                #
                # R and R0 intermediates
                # ------------------------

                vexp = -L[0, i + 1] * Vexp_class.Vexp[0, i + 1] # V0n
                Rinter  = mycc.R1inter(ts, fsp[i+1], vexp)
                #R0inter = mycc.R0inter(ts, fsp[i+1], vexp)
                del vexp

                #
                # update En_r
                # ------------------------

                En_r, o, v = mycc.Extract_Em_r(rn[i], r0n[i], Rinter)
                print('Update E')
                print(En_r)
                print('o,v: ', o, v)

                #
                # update r0
                # ------------------------

                #r0new[i] = mycc.r0update(rn[i], r0n[i], En_r, R0inter)
                r0new[i] = mycc.R0eq(En_r, ts, rn[i], fsp=fsp[i+1])
                #del R0inter
                print()
                print('update r0')
                print(r0new[i])
                #
                # Update r
                # -------------------------

                rnew[i] = mycc.rsupdate(rn[i], r0n[i], Rinter, En_r, idx=[o, v])
                del Rinter
                print()
                print('UPDATE r')
                print('rnew')
                print(rnew[i])
                #
                # Get missing r ampl
                # -------------------------
                print()
                print('UPDATE rov')
                rnew[i][o, v] = mycc.get_ov(ln[i], l0n[i], rn[i], r0n[i], [o, v])

                print('rnew')
                print(rnew[i])
                
                #
                # L and L0 inter
                # ------------------------

                vexp = -L[i+1,0]*Vexp_class.Vexp[i+1,0] # Vn0
                Linter = mycc.es_L1inter(ts, fsp[i+1], vexp )
                #L0inter = mycc.L0inter(ts, fsp[i+1], vexp)
                del vexp

                #
                # Update En_l
                # ------------------------

                En_l, o, v = mycc.Extract_Em_l(ln[i], l0n[i], Linter)

                #
                # Update l0
                # ------------------------

                #l0new[i] = mycc.l0update(ln[i], l0n[i], En_l, L0inter)
                l0new[i] = mycc.L0eq(En_l, ts, ln[i], fsp=fsp[i + 1])
                #del L0inter
                print()
                print('update l0')
                print(l0new[i])

                #
                # Update l
                # ------------------------

                lnew[i] = mycc.es_lsupdate(ln[i], l0n[i], En_l, Linter, idx=[o, v])
                del Linter
                print()
                print('UPDATE l')
                print('lnew')
                print(lnew[i])
                #
                # Get missing l amp
                # ------------------------
                print()
                print('UPDATE lov')
                lnew[i][o, v] = mycc.get_ov(rn[i], r0n[i], ln[i], l0n[i], [o, v])

                print('lnew')
                print(lnew[i])
                #
                # Store excited states energies Ep = (En_r,En_l)
                # -----------------------------------------------

                Ep[i+1][0] = En_r
                Ep[i+1][1] = En_l

                #
                # Apply DIIS
                # ---------------------------------------
                if 'l' in diis:
                    ln_vec = np.ravel(lnew[i])
                    lnew[i] = ldiis[i].update(ln_vec).reshape((nocc, nvir))

                if 'r' in diis:
                    rn_vec = np.ravel(rnew[i])
                    rnew[i] = rdiis[i].update(rn_vec).reshape((nocc, nvir))

            #del Rinter, R0inter, Linter, L0inter, vexp


            #
            # Check orthonormality and spin, re-normalize vectors if norm > threshold
            # -------------------------------------------------------------------------
            #

            #ln, rn, r0n, l0n = utilities.ortho_norm(ln, rn, r0n, l0n)
            C_norm = utilities.check_ortho(ln, rn, r0n, l0n)
            print('C_norm')
            print(C_norm)
            print()

            for i in range(nbr_states):
                Spin[i] = utilities.check_spin(rn[i], ln[i])

            #
            # Store new vectors
            # -------------------------

            rn = copy.deepcopy(rnew)
            ln = copy.deepcopy(lnew)
            r0n = copy.deepcopy(r0new)
            l0n = copy.deepcopy(l0new)

            #
            # Store GS energies Ep
            # --------------------------------------------

            vexp = [-L[0, i + 1] * Vexp_class.Vexp[0, i + 1] for i in range(nbr_states)]
            Ep[0][0] = mycc.energy_ccs(ts, fsp[0], rsn=rn, r0n=r0n, vn=vexp)
            Ep_ite.append(Ep)

            #
            # checking convergence
            # --------------------------------------------

            dic = {'ts':ts, 'ls':ls, 'rn':rn, 'ln':ln}
            conv = self.Conv_check(dic)
            conv_ite.append(conv)

            if ite > 0:
                Dconv = np.linalg.norm(conv - conv_old)

            conv_ite.append(Dconv)

            #
            # print convergence infos
            # --------------------------------------------

            tmp = [ite, format_float.format(Dconv)]

            for i in range(nbr_states):

                if i == 0:
                    tmp.extend(['', format_float.format(C_norm[i, i]), X2[i, 0], X2[i, i], 2*Spin[i]+1, r0n[i], l0n[i],
                                Ep[i + 1][0], Ep[i + 1][1]])
                else:
                    C_norm_av = (C_norm[0, i] + C_norm[i, 0]) / 2
                    tmp.extend(['', format_float.format(C_norm[i, i]), X2[i, 0], X2[0, i], 2*Spin[i]+1, r0n[i],
                                l0n[i], Ep[i + 1][0], Ep[i + 1][1], format_float.format(C_norm_av)])

            table.append(tmp)

            if ite >= self.maxiter:

                Conv_text = 'Max iteration reached'
                print()
                print(Conv_text)
                print(tabulate(table, headers, tablefmt=self.tablefmt))
                break

            if Dconv > 300.:

                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L.flatten(), ite)
                print()
                print(Conv_text)
                print(tabulate(table, headers, tablefmt=self.tablefmt))
                break

            ite += 1

        else:
            print(tabulate(table, headers, tablefmt=self.tablefmt))
            print()
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L.flatten(), ite)
            print(Conv_text)
            print()
            print('Final energies: ', Ep_ite[-1])
             
        return Conv_text, dic

    #############################################
    # SCF method with iterative diagonalization
    #############################################

    def SCF_davidson(self, L, ts=None, ls=None, rn=None, ln=None, r0n=None, l0n=None, max_space=10):
        '''
        SCF + davidson solver for the coupled T,Lam,R and L equations
        Diagonalizes the effective similarly transformed Hamiltonian at each iteration for each excited states and
        extract the target eigenvector and eigenvalue

        :param L: matrix of experimental weight
            -> shape(L)[0] = total number of states (GS+ES)
            -> typically L would be the same for properties obtained from the same experiment
        :param ts, ls, rn, ln, r0n, l0n: amplitudes
            -> ln,rn,l0 and r0 are list with length = nbr of excited states
        :param max_space: maximum size of the subspace used for the Davidson algorithm
        :return:
        '''

        Vexp_class = self.Vexp_class
        nbr_states = self.nbr_states

        # initialize r and l vectors
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        if rn is None:
            rn = self.rn_ini
            ln = self.ln_ini
        if r0n is None:
            r0n = self.r0_ini
            l0n = self.l0_ini

        # initialize spin
        Spin = np.zeros(nbr_states)

        # check length
        if L.shape != Vexp_class.Vexp.shape:
            raise ValueError('Shape of weight factor must be equal to shape of Vexp vectors:', Vexp_class.Vexp.shape)

        nocc = self.nocc
        nvir = self.nvir
        mycc = self.mycc

        # initialize X2 and Ep array
        X2 = np.zeros((nbr_states + 1, nbr_states + 1))
        Ep = np.zeros((nbr_states + 1, 2))

        # initialize loop vectors and printed information
        conv = 0.
        Dconv = 1.
        ite = 0
        X2_ite = []
        Ep_ite = []
        conv_ite = []

        table = []
        # First line of printed table
        headers = ['ite', str(self.conv_var)]
        for i in range(nbr_states):
            if i == 0:
                headers.extend(['ES {}'.format(i + 1), 'norm', 'X2_r', 'X2_l', '2S+1', 'r0', 'l0', 'Er', 'El'])
            else:
                headers.extend(['ES {}'.format(i + 1), 'norm', 'X2_r', 'X2_l', '2S+1', 'r0',
                                'l0', 'Er', 'El', 'Ortho wrt ES 1'])

        while Dconv > self.conv_thres:

            #
            # Initialize
            # ---------------------------------------

            # nbr_states = nbr of excited states
            fsp = [None]*(nbr_states+1)
            rdm1 = [None]*(nbr_states+1)
            tr_rdm1 = [None] * nbr_states
            conv_old = conv

            #
            # calculate needed rdm1 and tr_rdm1 for all states
            # -------------------------------------------------

            # GS
            if Vexp_class.exp_data[0, 0] is not None:
                rdm1[0] = mycc.gamma(ts, ls)

            # ES
            for n in range(nbr_states):

                # calculate rdm1 for state n if diagonal exp data are present
                if Vexp_class.exp_data[n+1,n+1] is not None:
                   rdm1[n+1] = mycc.gamma_es(ts, ln[n], rn[n], r0n[n], l0n[n])

                # calculate tr_rdm1 if transition exp data are present
                if Vexp_class.exp_data[0, n + 1] is not None:
                    # right dm1 <Psi_k|aa|Psi_n>
                    tr_1 = mycc.gamma_tr(ts, ln[n], 0, 1, l0n[n])
                    # left dm1 <Psi_n|aa|Psi_k>
                    tr_2 = mycc.gamma_tr(ts, 0, rn[n], r0n[n], 1)
                    tr_rdm1[n] = list((tr_1, tr_2))

                del tr_1, tr_2

            #
            # Update Vexp, calculate effective fock matrices and store X2,vmax
            # ------------------------------------------------------------------

            # GS
            if rdm1[0] is not None:
                V, x2, vmax = Vexp_class.Vexp_update(rdm1[0], L[0, 0], (0, 0))
                fsp[0] = np.subtract(mycc.fock, V)
                X2[0, 0] = x2
            # else:
            #    fsp[0] = fock.copy()

            # ES
            for j in range(nbr_states):
                n = j+1

                if rdm1[n] is not None:
                    V, x2, vmax = Vexp_class.Vexp_update(rdm1[n], L[j, j], (n, n))
                    fsp[n] = np.subtract(mycc.fock, V)
                    X2[n, n] = x2
                #else:
                #    fsp[n] = fock.copy()

                if tr_rdm1[j] is not None:
                    V, X2[n, 0], vmax = Vexp_class.Vexp_update(tr_rdm1[j][0], L[0, j], (n, 0))
                    V, X2[0, n], vmax = Vexp_class.Vexp_update(tr_rdm1[j][1], L[0, j], (0, n))

            del V

            X2_ite.append(X2)

            # CARREFUL WITH THE SIGN OF Vexp !
            # for transition case vn = -Vexp
            # CARREFUL: the Vexp elements are not scaled with lambda

            #
            # update t amplitudes
            # ---------------------------------------------------

            vexp = -L[0, 1:] * Vexp_class.Vexp[0, 1:]
            T1inter = mycc.T1inter(ts, fsp[0])
            ts = mycc.tsupdate(ts, T1inter, rsn=rn, r0n=r0n, vn=vexp)

            del T1inter

            #
            # update l (Lambda) amplitudes for the GS
            # ----------------------------------------

            L1inter = mycc.L1inter(ts, fsp[0])
            vexp = -L[1:, 0] * Vexp_class.Vexp[1:, 0]
            ls = mycc.lsupdate(ts, ls, L1inter, rsn=rn, lsn=ln, r0n=r0n, l0n=l0n, vn=vexp)

            del vexp, L1inter

            #
            # Build guess r and l vector
            # ---------------------------

            vec_r = np.zeros((nbr_states, nocc*nvir))
            vec_l = np.zeros_like(vec_r)
            for i in range(nbr_states):
                vec_r[i, :] = rn[i].flatten()
                vec_l[i, :] = ln[i].flatten()

            for i in range(nbr_states):

                #
                # Right Vexp
                # ----------------------------------------------------
                vexp = -L[0, i + 1] * Vexp_class.Vexp[0, i + 1]  # V0n

                #
                # make H_i right matrix and diag terms
                # ----------------------------------------------

                Rinter = mycc.R1inter(ts, fsp[i+1], vexp)
                del vexp

                diag = np.zeros((nocc, nvir))
                for j in range(nocc):
                    for b in range(nvir):
                        diag[j, b] = Rinter[0][b, b] - Rinter[1][j, j] + Rinter[2][b, j, j, b] \
                                     + Rinter[3] + Rinter[5][i, b]

                # function to create H matrix from Ria and ria
                matvec = lambda xs: [mycc.R1eq(x.reshape(nocc, nvir), r0n[i], Rinter).flatten() for x in xs]
                # needed function for diagonal term of H
                precond = lambda r, e, r0: r / (e-diag.flatten()+1e-12)

                #
                # Apply Davidson using PySCF library
                # ------------------------------------------

                conv, de, rvec = lib.davidson_nosym1(matvec, vec_r, precond, max_space=max_space, nroots=nbr_states)

                for j in range(len(conv)):
                    if not conv[j]:
                        print('Davidson algorithm did not converged for {}th right eigenvectors '
                          'at iteration {}'.format(j+1, ite))

                # store results
                print('E_R = ', de)
                print('rn eigenvectors')
                En_r = de[i]
                rn[i] = rvec[i].reshape((nocc, nvir))
                print(rn[i])  #rn[i][0, 0])
                #rn[i][1, 1])
                print()

                #
                # Update r0
                # ----------------------------------------------

                r0n[i] = mycc.R0eq(En_r, ts, rn[i], fsp=fsp[i+1])

                #
                # Left Vexp
                # ---------------------------------------------
                vexp = -L[i + 1, 0] * Vexp_class.Vexp[i + 1, 0]  # Vn0

                #
                # make H_i left matrix and diag terms
                # ----------------------------------------------

                Linter = mycc.es_L1inter(ts, fsp[i+1], vexp)
                del vexp

                diag = np.zeros((nocc, nvir))
                for j in range(nocc):
                    for b in range(nvir):
                        diag[j, b] = Linter[0][b, b] - Linter[1][j, j] + Linter[2][b, j, j, b] \
                                     + Linter[3] + Linter[5][i, b]

                # function to create H matrix from Lia and lia
                matvec = lambda xs: [mycc.es_L1eq(x.reshape(nocc, nvir), l0n[i], Linter).flatten() for x in xs]
                # needed function for diagonal term of H
                precond = lambda l, e, l0: l / (e-diag.flatten()+1e-12)

                #
                # Apply Davidson using PySCF library
                # ------------------------------------------

                conv, de, lvec = lib.davidson_nosym1(matvec, vec_l, precond, max_space=max_space, nroots=nbr_states)

                for j in range(len(conv)):
                    if not conv[j]:
                        print('Davidson algorithm did not converged for {}th left eigenvectors '
                          'at iteration {}'.format(j+1, ite))

                # store results
                En_l = de[i]
                ln[i] = lvec[i].reshape((nocc, nvir))

                #
                # Update l0
                # ----------------------------------------------
                l0n[i] = mycc.L0eq(En_l, ts, ln[i], fsp=fsp[i+1])

                #
                # Update Energies
                # ----------------------------------------------

                Ep[i + 1][0] = En_r
                Ep[i + 1][1] = En_l

            # del Rinter, R0inter, Linter, L0inter, vexp

            #
            # Check orthonormality and spin, re-normalize vectors if norm > threshold
            # -------------------------------------------------------------------------
            #

            C_norm = utilities.check_ortho(ln, rn, r0n, l0n)
            print('C_norm')
            print(C_norm)
            print()
            #ortho = False
            #for i in range(nbr_states):
            #    if 0.99 > C_norm[i, i] > 1.01:
            #        ln[i] = ln[i] / C_norm[i, i]
            #        l0n[i] = l0n[i] / C_norm[i, i]
            #    for j in range(nbr_states):
            #        if C_norm[i, j] > 0.01:
            #            ortho = True

            #if ortho:
            #    rn, ln, r0n, l0n = utilities.ortho_es(rn, ln, r0n, l0n)

            for i in range(nbr_states):
                Spin[i] = utilities.check_spin(rn[i], ln[i])

            #
            # Store GS energies Ep
            # --------------------------------------------

            vexp = [-L[0, i + 1] * Vexp_class.Vexp[0, i + 1] for i in range(nbr_states)]
            Ep[0][0] = mycc.energy_ccs(ts, fsp[0], rsn=rn, r0n=r0n, vn=vexp)
            Ep_ite.append(Ep)

            #
            # checking convergence
            # --------------------------------------------

            dic = {'ts': ts, 'ls': ls, 'rn': rn, 'ln': ln}
            conv = self.Conv_check(dic)
            conv_ite.append(conv)

            if ite > 0:
                Dconv = np.linalg.norm(conv - conv_old)

            conv_ite.append(Dconv)

            #
            # print convergence infos
            # --------------------------------------------

            tmp = [ite, format_float.format(Dconv)]

            for i in range(nbr_states):

                if i == 0:
                    tmp.extend(['', format_float.format(C_norm[i, i]), X2[i, 0], X2[i, i], 2*Spin[i]+1, r0n[i], l0n[i],
                                Ep[i + 1][0], Ep[i + 1][1]])
                else:
                    C_norm_av = (C_norm[0, i] + C_norm[i, 0]) / 2
                    tmp.extend(['', format_float.format(C_norm[i, i]), X2[i, 0], X2[0, i], 2*Spin[i]+1, r0n[i],
                                l0n[i], Ep[i + 1][0], Ep[i + 1][1], format_float.format(C_norm_av)])

            table.append(tmp)

            if ite >= self.maxiter:
                Conv_text = 'Max iteration reached'
                print()
                print(Conv_text)
                print(tabulate(table, headers, tablefmt=self.tablefmt))
                break

            if Dconv > 30.:
                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L.flatten(), ite)
                print()
                print(Conv_text)
                print(tabulate(table, headers, tablefmt=self.tablefmt))
                break

            ite += 1

        else:
            print(tabulate(table, headers, tablefmt=self.tablefmt))
            print()
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L.flatten(), ite)
            print(Conv_text)
            print()
            print('Final energies: ', Ep_ite[-1])

        return Conv_text, dic


if __name__ == "__main__":

    from pyscf import gto, scf, cc, tdscf
    import CCS
    import exp_pot
    import Eris

    # build molecule
    mol = gto.Mole()
    #mol.atom = [
    #    [8 , (0. , 0.     , 0.)],
    #    [1 , (0. , -0.757 , 0.587)],
    #    [1 , (0. , 0.757  , 0.587)]]
    mol.atom = '''
    H 0 0 0
    H 0 0 1.
    '''
    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()

    # GHF calc
    mgf = scf.GHF(mol)
    mgf.kernel()
    mo_occ = mgf.mo_occ
    mo_energy =mgf.mo_energy
    mocc = mgf.mo_coeff[:, mo_occ > 0]
    mvir = mgf.mo_coeff[:, mo_occ == 0]
    gnocc = mocc.shape[1]
    gnvir = mvir.shape[1]
    mo_coeff = mgf.mo_coeff

    # GCCSD eris
    mygcc = cc.GCCSD(mgf)
    geris = Eris.geris(mygcc)
    gfs = geris.fock

    # GCCS object
    mccsg = CCS.Gccs(geris)

    # overlap integral
    S_AO = mol.intor('int1e_ovlp')

    # build gamma_exp for GS
    ts = np.random.random((gnocc,gnvir))*0.1
    ls = np.random.random((gnocc,gnvir))*0.1 
    GS_exp_ao = mccsg.gamma(ts, ls)
    GS_exp_ao = utilities.mo_to_ao(GS_exp_ao,mo_coeff)

    # build exp list
    exp_data = np.full((3, 3), None)
    exp_data[0, 0] = ['mat', GS_exp_ao]
    exp_data[0, 1] = ['dip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
    exp_data[0, 2] = ['dip', [0.000000, 0.000000, -0.622534]]  # DE = 0.37 au

    # Vexp object
    VXexp = exp_pot.Exp(exp_data,mol,mgf.mo_coeff)

    # initial rn, r0n and ln, l0n list
    rnini, DE = utilities.koopman_init_guess(mo_energy,mo_occ,nstates=2)
    lnini = [i * 1 for i in rnini]

    # convergence options
    maxiter = 40
    conv_thres = 10 ** -8
    diis = ('',)  # must be tuple

    # initialise Solver_CCS Class
    Solver = Solver_ES(mccsg, VXexp, rnini, conv_var='rl', ln_ini=lnini, maxiter=maxiter, diis=diis)

    # CIS calc
    mrf = scf.RHF(mol).run()
    mcis = tdscf.TDA(mrf)
    mcis.kernel(nstates=2)
    print()
    print('CIS calc')
    print('DE= ',mcis.e)
    print('c1= ',mcis.xy)
    print()
    print('initial guess')
    print('DE= ', DE)
    print('r1= ', rnini[0])
    print()

    # Solve for L = 0
    L = np.zeros_like((exp_data))
    Solver.SCF(L)