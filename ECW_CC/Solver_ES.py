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
import scipy
import copy
from pyscf import lib
import utilities
from tabulate import tabulate

# print float format
format_float = '{:.4e}'


class Solver_ES:
    def __init__(self, mycc, Vexp, rn_ini=None, tsini=None, lsini=None, val_core=None, rini_koop_idx=None
                 , conv_var='tl', conv_thres=10 ** -6, diis='', maxiter=40, maxdiis=20, mindiis=2, tablefmt='rst'):
        """
        Solves the ES-ECW-CCS equations for V00, Vnn, V0n and Vn0 (thus not including Vnk with n != k)
        -> only state properties and GS-ES transition properties

        :param mycc: ECW.CCS object containing T, Lambda,R and L equations
        :param Vexp: ECW.Vexp_class, class containing the exp data
        :param conv_var: convergence criteria: 'Ep', 'l' or 'tl'
        :param conv_thres: convergence threshold
        :param tsini: initial value for t
        :param lsini: initial value for Lambda
        :param val_core: tuple (nval, ncore)
        :param diis: list containing the variables on which to apply diis: rdm1, tl or rl or all (for tl and rl)
        :param maxiter: max number of iteration
        :param maxdiis: max diis space
        :param mindiis: ite start for diis
        :param tablefmt: format for table printed using tabulate package (eg: 'rst' or 'latex')
        """

        # CC object
        self.mycc = mycc

        # exp_pot object
        self.Vexp_class = Vexp
        self.nbr_states = Vexp.nbr_states  # total number of states involved GS+nES

        # tabulate format
        self.tablefmt = tablefmt

        # get HF information from ccs object
        self.nocc = mycc.nocc
        self.nvir = mycc.nvir
        self.dim = self.nocc + self.nvir
        self.EHF = mycc.eris.EHF

        # ts and Lambda initial
        if tsini is None:
            tsini = np.zeros((self.nocc, self.nvir))
        if lsini is None:
            lsini = np.zeros((self.nocc, self.nvir))
        self.tsini = tsini
        self.lsini = lsini

        # l and r initial values using Koopman's guess
        # Koopman initial guess
        if rn_ini is None:
            self.rn_ini, de = utilities.koopman_init_guess(np.diag(mycc.fock), mycc.eris.mo_occ,
                                                           val_core, koop_idx=rini_koop_idx)

        else:
            if len(rn_ini) != self.nbr_states - 1:
                raise ValueError('The number of given initial r vectors is not '
                                 'consistent with the given experimental data for ES')
            else:
                # extract DE from given r1 vectors
                self.rn_ini = rn_ini
                de = [utilities.get_DE(np.diag(mycc.fock), rs) for rs in rn_ini]

        # ls, l0, r0
        self.ln_ini = [i * 1 for i in self.rn_ini]
        self.r0_ini = [mycc.r0_fromE(de, np.zeros_like(tsini), r, np.zeros((self.dim, self.dim)))
                       for r, de in zip(rn_ini, de)]
        self.l0_ini = [i * 1 for i in self.r0_ini]
        self.E_ini = -np.asarray(de)  # initial excited states correlation energy

        print(' Initial Koopman energies in eV: ', -self.E_ini*27.2114)

        # DIIS option
        self.diis = diis
        self.maxdiis = maxdiis
        self.mindiis = mindiis

        # convergence options
        self.maxiter = maxiter
        self.conv_thres = conv_thres
        if conv_var == 'Ep':
            self.Conv_check = self.Ep_check
        elif conv_var == 'rl':
            self.Conv_check = self.rl_check
        elif conv_var == 'tl':
            self.Conv_check = self.tl_check
        elif conv_var == 'all':
            self.Conv_check = self.all_amp_check
        else:
            raise ValueError('Accepted convergence parameter is Ep, tl, rl or all')
        self.conv_var = conv_var

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
            ans += r + l
        return ans

    def all_amp_check(self, dic):
        ans = self.tl_check(dic) + self.rl_check(dic)
        return ans

    #############
    # SCF method
    #############

    def SCF(self, L=None, dic_amp_ini=None, diis=None, force_alpha=True, print_ite=True):
        """
        Rough SCF solver for coupled T, Lam, R and L equations: takes care of the spin symmetry

        :param force_alpha: if True, forces alpha transition for the r ans l amplitudes
        :param L: matrix of experimental weight
            -> shape(L)[0] = total number of states (GS+ES)
            -> typically L would be the same for properties obtained from the same experiment
        :param dic_amp_ini: contains ts, ls, ln, rn, r0 and l0 initial amplitudes
        :param diis: string 'GS' (apply to t and l), 'ES' (apply to rn and ln) or 'all' (apply to all amp.)
        :param print_ite: True if each iteration has to be printed
        :return:
        """

        Vexp_class = self.Vexp_class
        nbr_states = self.nbr_states  # total number of states

        if L is None:
            L = Vexp_class.L
        else:  # check L format
            L = Vexp_class.L_check(L)

        # initialize r and l vectors for iteration k-1
        if dic_amp_ini is None:
            # From Koopman guest
            ts = self.tsini
            ls = self.lsini
            rn = self.rn_ini
            ln = self.ln_ini
            r0n = self.r0_ini
            l0n = self.l0_ini
            # initial o,v indices from Koopman's excitation
            ov = [np.where(r == 1) for r in rn]
        else:
            # From given
            ts = dic_amp_ini['ts']
            ls = dic_amp_ini['ls']
            rn = dic_amp_ini['rn']
            ln = dic_amp_ini['ln']
            r0n = dic_amp_ini['r0n']
            l0n = dic_amp_ini['l0n']
            # initial o,v indices from largest rs/ls value
            ov = [None] * (nbr_states - 1)

        dic_amp = {'ts': ts, 'ls': ls, 'rn': rn, 'ln': ln}

        # initialize amplitudes for iteration k
        rnew = [None] * (nbr_states - 1)
        lnew = [None] * (nbr_states - 1)
        r0new = [None] * (nbr_states - 1)
        l0new = [None] * (nbr_states - 1)

        # initialize rdm1 and effective Fock matrix
        fsp = [None] * nbr_states
        rdm1 = [None] * nbr_states
        tr_rdm1 = [None] * (nbr_states - 1)

        # initialize total spin list
        Spin = np.zeros(nbr_states - 1)

        # diis options
        if diis is None:
            diis = self.diis

        nocc = self.nocc
        nvir = self.nvir
        mycc = self.mycc

        # initialize X2 and Ep array
        Delta = np.zeros((nbr_states, nbr_states))
        Ep = np.zeros((nbr_states, 2))  # right and left E' energies for each states

        # initialize loop vectors and printed information
        conv = 0.
        Dconv = 1.
        ite = 0
        Delta_ite = []
        Ep_ite = []
        conv_ite = []

        # initialize diis
        if diis:
            amp_diis = lib.diis.DIIS()
            amp_diis.space = self.maxdiis
            amp_diis.min_space = self.mindiis

        # Printed information
        table = []
        headers = []
        if print_ite:
            headers = ['ite', 'Dconv ' + str(self.conv_var)]  # First line of printed table
            for i in range(nbr_states - 1):
                if i == 0:
                    headers.extend(['ES {}'.format(i + 1), 'norm', 'Delta_r', 'Delta_l', '2S+1',
                                    'r0', 'l0', 'Er', 'El'])
                else:
                    headers.extend(['ES {}'.format(i + 1), 'norm', 'Delta_r', 'Delta_l', '2S+1',
                                    'r0', 'l0', 'Er', 'El', 'Ortho wrt ES 1'])

        #############
        # Main loop
        #############

        while Dconv > self.conv_thres:

            conv_old = conv

            #
            # calculate all rdm1 and tr_rdm1 for all states
            # -------------------------------------------------

            # compute GS rdm1
            rdm1[0] = mycc.gamma(ts, ls)

            # ES
            for n in range(1, nbr_states):
                # calculate rdm1 for state n <Psi_n|aa|Psi_n>
                rdm1[n] = mycc.gamma_es(ts, ln[n - 1], rn[n - 1], r0n[n - 1], l0n[n - 1])
                # right tr_rdm1 <Psi_0|aa|Psi_n>
                tr_r = mycc.gamma_tr(ts, ln[n - 1], None, None, l0n[n - 1])
                # left tr_rdm1 <Psi_n|aa|Psi_0>
                tr_l = mycc.gamma_tr(ts, ls, rn[n - 1], r0n[n - 1], 1)
                tr_rdm1[n - 1] = list((tr_r, tr_l))

            #
            # Update Vexp, calculate effective fock matrices and store Delta,vmax
            # --------------------------------------------------------------------

            # calculate GS Vexp from GS prop
            if Vexp_class.exp_data[0]:
                Delta[0, 0], vmax = Vexp_class.Vexp_update(rdm1[0], tr_rdm1, (0, 0), L=L)

            # calculate ES Vexp and update GS Vexp (if DEk) from ES prop and tr prop
            for n in range(1, nbr_states):
                # if ES prop
                if Vexp_class.exp_data[n]:
                    # transition prop
                    if 'trdip' in Vexp_class.prop_names[n] or 'trmat' in Vexp_class.prop_names[n]:
                        Delta[n, 0], vmax = Vexp_class.Vexp_update(tr_rdm1[n - 1][0], tr_rdm1[n - 1][1],
                                                                   (n, 0), L=L)  # right
                        Delta[0, n], vmax = Vexp_class.Vexp_update(tr_rdm1[n - 1][1], tr_rdm1[n - 1][0],
                                                                   (0, n), L=L)  # left
                    # ESn prop or DEk
                    else:
                        Delta[n, n], vmax = self.Vexp_class.Vexp_update(rdm1[n], rdm1[0], (n, n), L=L)
                        # Calculate effective Fock matrix for ES
                        fsp[n] = np.subtract(mycc.fock, Vexp_class.Vexp[n, n])

            # Calculate effective Fock matrix for GS
            if Vexp_class.Vexp[0, 0] is not None:
                fsp[0] = np.subtract(mycc.fock, Vexp_class.Vexp[0, 0])

            Delta_ite.append(Delta)

            #
            # update t amplitudes
            # ---------------------------------------------------
            vexp = Vexp_class.Vexp[0, 1:]  # -lambda * 0mV -> the minus sign is taken care in tsupdate function
            T1inter = mycc.T1inter(ts, fsp[0])
            ts = mycc.tsupdate(ts, T1inter, rsn=rn, r0n=r0n, vn=vexp)
            del T1inter

            #
            # update l (Lambda) amplitudes for the GS
            # ----------------------------------------

            L1inter = mycc.L1inter(ts, fsp[0])
            vexp = Vexp_class.Vexp[1:, 0]  # -lambda * 0mV -> the minus sign is taken care in lsupdate function
            ls = mycc.lsupdate(ts, ls, L1inter, rsn=rn, lsn=ln, r0n=r0n, l0n=l0n, vn=vexp)

            #
            # Apply diis to GS amplitudes
            # ----------------------------------------

            if diis == 'GS':
                vec = np.concatenate((np.ravel(ls), np.ravel(ts)))
                ls, ts = np.split(amp_diis.update(vec), 2)
                ls = ls.reshape(nocc, nvir)
                ts = ts.reshape(nocc, nvir)

            del vexp, L1inter

            #
            # Update En_r/En_l and r, r0, l and l0 amplitudes for each ES
            # ------------------------------------------------------------

            for n in range(1, nbr_states):
                # todo: most element in Rinter and Linter dot not depend on Vexp -> calculate ones for all states

                # Ria intermediates
                vexp = Vexp_class.Vexp[0, n]  # 0nV negative sign is taken care in R1inter
                Rinter = mycc.R1inter(ts, fsp[n], vexp)

                # update En_r using initial rs from Koopman's excitation
                En_r, o, v = mycc.Extract_Em_r(rn[n - 1], r0n[n - 1], Rinter, ov=ov[n - 1])

                # Update r
                rnew[n - 1] = mycc.rsupdate(rn[n - 1], r0n[n - 1], Rinter, En_r, force_alpha=force_alpha)
                del Rinter

                # Get missing r ampl
                rnew[n - 1][o, v] = mycc.get_ov(ln[n - 1], l0n[n - 1], rn[n - 1], r0n[n - 1], [o, v])

                # update r0
                r0new[n - 1] = mycc.r0_fromE(En_r, ts, rn[n - 1], vexp, fsp=fsp[n])

                # L and L0 inter
                vexp = Vexp_class.Vexp[n, 0]  # Vn0 # 0nV negative sign is taken care in es_L1inter
                Linter = mycc.es_L1inter(ts, fsp[n], vexp)
                # L0inter = mycc.L0inter(ts, fsp[n], vexp)

                # Update En_l
                En_l, o, v = mycc.Extract_Em_l(ln[n - 1], l0n[n - 1], Linter, ov=ov[n - 1])

                # Update l
                lnew[n - 1] = mycc.es_lsupdate(ln[n - 1], l0n[n - 1], En_l, Linter, force_alpha=force_alpha)
                del Linter

                # Get missing l amp
                lnew[n - 1][o, v] = mycc.get_ov(rn[n - 1], r0n[n - 1], ln[n - 1], l0n[n - 1], [o, v])

                # Update l0
                l0new[n - 1] = mycc.l0_fromE(En_l, ts, ln[n - 1], vexp, fsp=fsp[n])
                del vexp

                # Store right and left excited states energies Ep^r and Ep^l
                Ep[n, 0] = En_r
                Ep[n, 1] = En_l

            # Apply DIIS to ES amplitudes
            if diis == 'ES':
                vec = np.concatenate(([np.ravel(r) for r in rnew][0],
                                      [np.ravel(l) for l in lnew][0], r0new[0], l0new[0]))
                vec = amp_diis.update(vec)
                vec_0 = vec[-2 * len(r0new):]
                vec = np.split(vec[:2 * len(r0new)], 2 * len(rnew))
                for i in range(len(lnew)):
                    rnew[i] = vec[i].reshape((nocc, nvir))
                    lnew[i] = vec[i + len(rnew)].reshape((nocc, nvir))
                    r0new[i] = vec_0[:len(r0new)]
                    l0new[i] = vec_0[len(r0new):]
                del vec, vec_0

            #
            # Apply DIIS to all amplitudes
            # ---------------------------------------
            if diis == 'all':
                nbr_ES = len(r0new)
                # create flatten array with all amplitudes
                vec = np.concatenate((
                    np.ravel(ts), np.ravel(ls),
                    np.ravel([np.ravel(rn) for rn in rnew]), np.ravel([np.ravel(ln) for ln in lnew]),
                    np.ravel([np.ravel(r0) for r0 in r0new]), np.ravel([np.ravel(l0) for l0 in l0new])
                ))
                vec = amp_diis.update(vec)
                vec_0 = vec[-2 * nbr_ES:]  # extract r0 and l0 amplitudes
                vec = np.split(vec[:-2 * nbr_ES], 2 * nbr_ES + 2)  # extract other amp
                # extract all new amplitudes
                ts = vec[0].reshape((nocc, nvir))
                ls = vec[1].reshape((nocc, nvir))
                for i in range(nbr_ES):
                    rnew[i] = vec[2 + i].reshape((nocc, nvir))
                    lnew[i] = vec[2 + i + nbr_ES].reshape((nocc, nvir))
                    r0new[i] = vec_0[i]
                    l0new[i] = vec_0[nbr_ES + i]
                del vec

            #
            # Check orthonormality and spin, re-normalize vectors if norm > threshold
            # -------------------------------------------------------------------------
            #

            # ln, rn, r0n, l0n = utilities.ortho_norm(ln, rn, r0n, l0n)
            C_norm = utilities.check_ortho(lnew, rnew, r0new, l0new)
            for i in range(nbr_states - 1):
                Spin[i] = utilities.check_spin(rnew[i], lnew[i])

            #
            # Store new vectors
            # -------------------------
            rn = copy.deepcopy(rnew)
            ln = copy.deepcopy(lnew)
            r0n = copy.deepcopy(r0new)
            l0n = copy.deepcopy(l0new)
            dic_amp = {'ts': ts, 'ls': ls, 'rn': rn, 'ln': ln, 'r0n': r0n, 'l0n': l0n}

            #
            # Store total GS energy Ep
            # --------------------------------------------

            vexp = [Vexp_class.Vexp[0, n] for n in range(1, nbr_states)]
            Ep[0, 0] = mycc.energy_ccs(ts, fsp[0], rsn=rn, r0n=r0n, vn=vexp)
            Ep_ite.append(Ep)
            del vexp

            #
            # checking convergence
            # --------------------------------------------

            conv = self.Conv_check(dic_amp)
            conv_ite.append(conv)

            if ite > 0:
                Dconv = np.linalg.norm(conv - conv_old)

            conv_ite.append(Dconv)

            #
            # print convergence infos
            # --------------------------------------------

            if print_ite:

                tmp = [ite, format_float.format(Dconv)]

                for i in range(nbr_states - 1):

                    if i == 0:  # first excited state to be printed
                        tmp.extend(['', format_float.format(C_norm[i, i]), Delta[1, 0], Delta[0, 1], 2 * Spin[i] + 1,
                                    r0n[i], l0n[i], Ep[i + 1, 0], Ep[i + 1, 1]])
                    else:
                        C_norm_av = (C_norm[0, i] + C_norm[i, 0]) / 2
                        tmp.extend(
                            ['', format_float.format(C_norm[i, i]), Delta[i + 1, 0], Delta[0, i + 1], 2 * Spin[i] + 1,
                             r0n[i],
                             l0n[i], Ep[i + 1, 0], Ep[i + 1, 1], format_float.format(C_norm_av)])

                table.append(tmp)

            if ite >= self.maxiter:

                Conv_text = 'Max iteration reached'
                if print_ite:
                    print(tabulate(table, headers, tablefmt=self.tablefmt))
                break

            if Dconv > 10.:

                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L, ite)
                if print_ite:
                    print(tabulate(table, headers, tablefmt=self.tablefmt))
                break

            ite += 1

        else:
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L, ite)
            if print_ite:
                print(tabulate(table, headers, tablefmt=self.tablefmt))

        return Conv_text, dic_amp, Delta, Ep, rdm1[0]

    #############################################
    # SCF method with iterative diagonalization
    #############################################

    def SCF_diag(self, L, Vexp_norm2=None, ts=None, ls=None, rn=None, ln=None, r0n=None, l0n=None, max_space=10,
                 davidson=True):
        """
        SCF + davidson solver for the coupled T,Lam,R and L equations
        Diagonalizes the effective similarly transformed Hamiltonian at each iteration for each excited states and
        extract the target eigenvector and eigenvalue

        :param L: matrix of experimental weight
            -> shape(L)[0] = total number of states (GS+ES)
            -> typically L would be the same for properties obtained from the same experiment
        :param ts: initial t1 amplitudes
        :param ls: initial lambda amplitudes
        :param rn: list of initial r1 amplitudes for each excited states
        :param ln: list of initial l1 amplitudes for each excited states
        :param l0n: list of l0 values
        :param r0n: list of r0 values
        :param max_space: maximum size of the subspace used for the Davidson algorithm
        :param davidson: if true uses davidson solver, if false use standard eigenvalue solver
        :return:
        """

        Vexp_class = self.Vexp_class
        nbr_states = Vexp_class.nbr_states

        # Vexp update function
        if not Vexp_norm2:
            Vexp_update = Vexp_class.Vexp_update
        else:
            Vexp_update = Vexp_class.Vexp_update_norm2

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
            fsp = [None] * (nbr_states + 1)
            rdm1 = [None] * (nbr_states + 1)
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
                if Vexp_class.exp_data[n + 1, n + 1] is not None:
                    rdm1[n + 1] = mycc.gamma_es(ts, ln[n], rn[n], r0n[n], l0n[n])

                # calculate tr_rdm1 if transition exp data are present
                if Vexp_class.exp_data[0, n + 1] is not None:
                    # right tr_dm1 <Psi_k|aa|Psi_n>
                    tr_r = mycc.gamma_tr(ts, ln[n], None, None, l0n[n])
                    # left tr_dm1 <Psi_n|aa|Psi_k>
                    tr_l = mycc.gamma_tr(ts, ls, rn[n], r0n[n], 1.)
                    tr_rdm1[n] = list((tr_r, tr_l))
                del tr_r, tr_l

            #
            # Update Vexp, calculate effective fock matrices and store X2,vmax
            # ------------------------------------------------------------------

            # GS
            if Vexp_class.exp_data[0, 0] is not None:
                V, x2, vmax = Vexp_update(rdm1[0], (0, 0))
                fsp[0] = np.subtract(mycc.fock, L[0, 0] * V)
                X2[0, 0] = x2

            # ES
            for j in range(nbr_states):
                n = j + 1

                if Vexp_class.exp_data[n, n] is not None:
                    V, x2, vmax = Vexp_update(rdm1[n], (n, n))
                    fsp[n] = np.subtract(mycc.fock, L[j, j] * V)
                    X2[n, n] = x2
                # else:
                #    fsp[n] = fock.copy()

                if Vexp_class.exp_data[0, n] is not None:
                    # norm 2 case
                    v, X2[n, 0], vmax = Vexp_update(tr_rdm1[j][0], (n, 0), rdm1_add=tr_rdm1[j][1])  # right
                    v, X2[0, n], vmax = Vexp_update(tr_rdm1[j][1], (0, n), rdm1_add=tr_rdm1[j][0])  # left
                    # DEk case
                    # v, X2[n, 0], vmax = Vexp_update(rdm1[j], (n, 0), rdm1_add=rdm1[0])
                    # v, X2[0, n], vmax = Vexp_update(rdm1[j], (0, n), rdm1_add=rdm1[0])
            del v

            X2_ite.append(X2)

            # CARREFUL WITH THE SIGN OF Vexp !
            # for transition case vn = -Vexp

            #
            # update t amplitudes
            # ---------------------------------------------------

            vexp = -L[0, 1:] * Vexp_class.Vexp[0, 1:]  # use right Vexp
            T1inter = mycc.T1inter(ts, fsp[0])
            ts = mycc.tsupdate(ts, T1inter, rsn=rn, r0n=r0n, vn=vexp)

            del T1inter

            #
            # update l (Lambda) amplitudes for the GS
            # ----------------------------------------

            L1inter = mycc.L1inter(ts, fsp[0])
            if Vexp_class.Vexp[1, 0] is not None:
                vexp = -L[1:, 0] * Vexp_class.Vexp[1:, 0]  # use left Vexp
            else:
                vexp = -L[0, 1:] * Vexp_class.Vexp[0, 1:]  # use right Vexp
            ls = mycc.lsupdate(ts, ls, L1inter, rsn=rn, lsn=ln, r0n=r0n, l0n=l0n, vn=vexp)

            del vexp, L1inter

            #
            # Build guess r and l vector for davidson
            # -----------------------------------------

            if davidson:
                vec_r = np.zeros((nbr_states, nocc * nvir))
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
                # Apply Diagonalization procedure
                # ------------------------------------

                # Davidson using PySCF library
                if davidson:
                    # make H_ia right matrix
                    Rinter = mycc.R1inter(ts, fsp[i + 1], vexp)

                    # diagonal element
                    diag = np.zeros((nocc, nvir))
                    for j in range(nocc):
                        for b in range(nvir):
                            diag[j, b] = Rinter[0][b, b] - Rinter[1][j, j] + Rinter[2][b, j, j, b] \
                                         + Rinter[3] + Rinter[5][j, b]

                    # function to create H matrix from Ria and ria
                    matvec = lambda xs: [mycc.R1eq(x.reshape(nocc, nvir), r0n[i], Rinter).flatten() for x in xs]
                    # needed function for diagonal term of H
                    precond = lambda rs, e0, r0: rs / (e0 - diag.flatten() + 1e-12)

                    # Apply Davidson
                    conv, de, rvec = lib.davidson_nosym1(matvec, vec_r, precond, max_space=max_space,
                                                         nroots=nbr_states * 2)

                    for j in range(len(conv)):
                        if not conv[j]:
                            print('Davidson algorithm did not converged for {}th right eigenvectors '
                                  'at iteration {}'.format(j + 1, ite))

                # Diagonalization of the full right matrix using scipy
                # else:
                #    mat =
                #    scipy.linalg.eigh()

                # store results
                En_r = de[i]
                rn[i] = rvec[i].reshape((nocc, nvir))

                #
                # Update r0
                # ----------------------------------------------

                r0n[i] = mycc.r0_fromE(En_r, ts, rn[i], vexp, fsp=fsp[i + 1])

                #
                # Left Vexp
                # ---------------------------------------------
                if Vexp_class.Vexp[i + 1, 0] is not None:
                    vexp = -L[i + 1, 0] * Vexp_class.Vexp[i + 1, 0]  # Vn0
                else:
                    vexp = -L[0, i + 1] * Vexp_class.Vexp[0, i + 1]  # use V0n
                    #
                # make H_i left matrix and diag terms
                # ----------------------------------------------

                Linter = mycc.es_L1inter(ts, fsp[i + 1], vexp)

                diag = np.zeros((nocc, nvir))
                for j in range(nocc):
                    for b in range(nvir):
                        diag[j, b] = Linter[0][b, b] - Linter[1][j, j] + Linter[2][b, j, j, b] \
                                     + Linter[3] + Linter[5][j, b]

                # function to create H matrix from Lia and lia
                matvec = lambda xs: [mycc.es_L1eq(x.reshape(nocc, nvir), l0n[i], Linter).flatten() for x in xs]
                # needed function for diagonal term of H
                precond = lambda l, e, l0: l / (e - diag.flatten() + 1e-12)

                #
                # Apply Davidson using PySCF library
                # ------------------------------------------

                conv, de, lvec = lib.davidson_nosym1(matvec, vec_l, precond, max_space=max_space, nroots=nbr_states)

                for j in range(len(conv)):
                    if not conv[j]:
                        print('Davidson algorithm did not converged for {}th left eigenvectors '
                              'at iteration {}'.format(j + 1, ite))

                # store results
                En_l = de[i]
                ln[i] = lvec[i].reshape((nocc, nvir))

                #
                # Update l0
                # ----------------------------------------------
                l0n[i] = mycc.l0_fromE(En_l, ts, ln[i], vexp, fsp=fsp[i + 1])

                #
                # Update ES energies
                # ----------------------------------------------

                Ep[i + 1][0] = En_r + self.EHF
                Ep[i + 1][1] = En_l + self.EHF

            # del Rinter, R0inter, Linter, L0inter, vexp

            #
            # Check orthonormality and spin, re-normalize vectors if norm > threshold
            # -------------------------------------------------------------------------
            #

            ln, rn, r0n, l0n = utilities.ortho_norm(ln, rn, r0n, l0n, ortho=False)
            C_norm = utilities.check_ortho(ln, rn, r0n, l0n)

            for i in range(nbr_states):
                Spin[i] = utilities.check_spin(rn[i], ln[i])

            #
            # Store GS energies Ep
            # --------------------------------------------

            vexp = [-L[0, i + 1] * Vexp_class.Vexp[0, i + 1] for i in range(nbr_states)]
            Ep[0][0] = mycc.energy_ccs(ts, fsp[0], rsn=rn, r0n=r0n, vn=vexp)
            Ep_ite.append(Ep)
            del vexp

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
            # print convergence info
            # --------------------------------------------

            tmp = [ite, format_float.format(Dconv)]

            for i in range(nbr_states):

                if i == 0:  # first state
                    tmp.extend(['', format_float.format(C_norm[i, i]), X2[0, i + 1], X2[i + 1, 0],
                                2 * Spin[i] + 1, r0n[i], l0n[i], Ep[i + 1][0], Ep[i + 1][1]])
                else:
                    C_norm_av = (C_norm[0, i] + C_norm[i, 0]) / 2
                    tmp.extend(
                        ['', format_float.format(C_norm[i, i]), X2[0, i + 1], X2[i + 1, 0], 2 * Spin[i] + 1, r0n[i],
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
    from pyscf import gto, scf, cc
    import CCS
    import exp_pot
    import Eris

    # build molecule
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    # mol.atom = '''
    # H 0 0 0
    # H 0 0 1.
    # '''
    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()

    # GHF calc
    mgf = scf.GHF(mol)
    mgf.kernel()
    mo_occ = mgf.mo_occ
    mo_energy = mgf.mo_energy
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
    ts = np.random.random((gnocc, gnvir)) * 0.1
    ls = np.random.random((gnocc, gnvir)) * 0.1
    GS_exp = mccsg.gamma(ts, ls)

    # build exp list
    exp_data = [
        [['mat', GS_exp]],
        [['trdip', [0.000000, 0.523742, 0.0000]]],
        [['trdip', [0.000000, 0.000000, -0.622534]]]
    ]

    # lambda value
    L = 0.

    # Vexp object
    VXexp = exp_pot.Exp(L, exp_data, mol, mgf.mo_coeff)

    # convergence options
    maxiter = 50
    conv_thres = 10 ** -4
    diis = 'all'

    # initialise Solver_CCS Class
    Solver = Solver_ES(mccsg, VXexp, val_core=(2, 0), conv_var='rl', maxiter=maxiter, diis=diis)

    print()
    print("#####################")
    print('SCF SOLVER')
    print("#####################")
    print()

    Solver.SCF()

    print()
    print("#####################")
    print('DAVIDSON SOLVER')
    print("#####################")
    print()

    # Solver.SCF_diag()
