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
from pyscf import lib
from . import utilities
from tabulate import tabulate

# print float format
format_float = '{:.4e}'

class Solver_ES:
    def __init__(self, mycc, Vexp_class, rn_ini, r0n_ini, ln_ini=None, l0n_ini=None, tsini=None, lsini=None, conv_var='tl', conv_thres=10 ** -6, diis=tuple(),
                 maxiter=80, maxdiis=20):
        '''
        Solves the ES-ECW-CCS equations for V00, Vnn, V0n and Vn0 (thus not including Vnk with n != k)
        -> only state properties and GS-ES transition properties

        :param mycc: ECW.CCS object containing T, Lambda,R and L equations
        :param Vexp_class: ECW.Vexp_class, class containing the exp data
        :param ini: t and lambda initial values for the GS and r,l,r0,l0 initial values for all needed states
                  --> list of matrix (for r and l) and float for r0 and l0
                  --> must be numpy array
        :param conv_var: convergence criteria: 'Ep', 'l' or 'tl'
        :param conv_thres: convergence threshold
        :param tsini: initial value for t
        :param lsini: initial value for Lambda
        :param diis: tuple containing the variables on which to apply diis: rdm1, t, L, r, l
        :param maxiter: max number of SCF iteration
        :param maxdiis: diis space
        '''

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

        # ln and l0 initial
        if ln_ini is None:
            ln_ini = [i * -1 for i in r0ini]
            l0n_ini = list([0]*len(r0ini))

        self.rn_ini = rn_ini
        self.ln_ini = ln_ini
        self.r0n_ini= r0n_ini
        self.l0n_ini= l0n_ini
        self.tsini = tsini
        self.lsini = lsini

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
        self.conv_var =conv_var

        self.fock = mycc.fock
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
        return (abs(ts) + abs(ls))

    def rl_check(self, dic):
        rn = dic.get('rn')
        ln = dic.get('ln')
        ans = np.zeros_like(rn[0])
        for r,l in zip(rn,ln):
            ans += (abs(r)+abs(l))
        return ans

    #############
    # SCF method
    #############

    def SCF(self, L, ts=None, ls=None, rn=None, ln=None, r0n=None, l0n=None, diis=None, S_AO=None):
        '''

        :param L: matrix of experimental weight
            -> shape(L)[0] = total number of states (GS+ES)
            -> typically L would be the same for prop obtained from the same exp
        :param ts, ls, rn, ln, r0n, l0n: amplitudes
            -> ln,rn,l0 and r0 are list with length = nbr of excited states
        :param diis:
        :param S_AO: AOs overlap matrix in G format
        :return:
        '''

        Vexp_class = self.Vexp_class
        nbr_states = self.nbr_states

        # initialize
        if ts is None:
            ts = self.tsini
            ls = self.lsini
        if rn is None:
            rn    = self.rn_ini
            ln = self.ln_ini
            r0n = self.r0n_ini
            l0n = self.l0n_ini
        if r0n is None or l0n is None:
            raise ValueError('if r and l initial vectors are given, r0 and l0 values must also be given')

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
        fock = self.fock

        # initialize X2 and Ep array
        X2 = np.zeros((nbr_states + 1, nbr_states + 1))
        Ep = np.zeros((nbr_states+1,2))
        
        # initialize loop vectors and printed infos
        conv = 0.
        Dconv = 1.
        ite = 0
        X2_ite = []
        Ep_ite = []
        conv_ite = []

        # initialze diis for ts,ls, rdm1
        if diis:
            if 'rdm1' in diis:
                adiis = []
                for n in range(self.nbr_states+1):
                  tmp = lib.diis.DIIS()
                  tmp.space = self.maxdiis
                  tmp.min_space = 2
                  adiis.append(tmp)
            if 't' in diis:
                tdiis = lib.diis.DIIS()
                tdiis.space = self.maxdiis
                tdiis.min_space = 2
            if 'l' in diis:
                ldiis = lib.diis.DIIS()
                ldiis.space = self.maxdiis
                ldiis.min_space = 2

        table = []
        # First line of printed table
        headers = ['ite',str(self.conv_var)]
        for i in range(nbr_states):
            headers.extend(['ES {} -> norm'.format(i+1), 'r0', 'l0','Er', 'El', 'Ortho wrt ES 1'])

        while Dconv > self.conv_thres:

            # nbr_states = nbr of excited states
            fsp = [None] * (nbr_states+1)
            rdm1 = [None]*(nbr_states+1)
            tr_rdm1 = [None]*(nbr_states)
            conv_old = conv

            # check for orthonormality
            # ------------------------------
            C_norm = utilities.check_ortho(ln, rn, r0n, l0n, S_AO=S_AO)*2 # factor 2 for G format

            # calculate needed rdm1 tr_rdm1 for all states
            # -------------------------------------------------
            # GS
            if Vexp_class.exp_data[0,0] is not None:
               rdm1[0] = mycc.gamma(ts, ls)
            # ES
            for n in range(nbr_states):
                # calculate rdm1 for state n
                if Vexp_class.exp_data[n+1,n+1] is not None:
                   rdm1[n+1] = mycc.gamma_es(ts, ln[n], rn[n], r0n[n], l0n[n])
                # calculate tr_rdm1 
                if Vexp_class.exp_data[0,n+1] is not None:
                   # <Psi_k||Psi_n>
                   tr_1 = mycc.gamma_tr(ts, ln[n], 0, 1, l0n[n])
                   # <Psi_n||Psi_k> 
                   tr_2 = mycc.gamma_tr(ts, 0, rn[n], r0n[n], 1)
                   tr_rdm1[n] = (tr_1,tr_2)
                del tr_1, tr_2
            # apply DIIS on rdm1
            # todo: apply DIIS on tr_rdm1 (left or right) ?
            if 'rdm1' in diis:
                for n in range(nbr_states+1):
                  rdm_vec = np.ravel(rdm1[n])
                  rdm1[n] = adiis[n].update(rdm_vec).reshape((dim, dim))

            # Update Vexp, calculate effective fock matrices and store X2,vmax
            # ------------------------------------------------------------------
            # GS
            if rdm1[0] is not None:
                V,x2,vmax = Vexp_class.Vexp_update(rdm1[0],L[0,0],(0,0))
                fsp[0] = np.subtract(fock,V)
                X2[0,0] = x2
            else:
                fsp[0] = fock.copy()
            # ES
            for j in range(nbr_states):
                i = j+1
                if rdm1[i] is not None:
                    V,x2,vmax = Vexp_class.Vexp_update(rdm1[i],L[j,j],(i,i))
                    fsp[i] = np.subtract(fock,V)
                    X2[i,i] = x2
                else:
                    fsp[i] = fock.copy()
                if tr_rdm1[j] is not None:
                    V,X2[i,0],vmax = Vexp_class.Vexp_update(tr_rdm1[j][0],L[0,j],(i,0))
                    V,X2[0,i],vmax = Vexp_class.Vexp_update(tr_rdm1[j][1],L[0,j],(0,i))
            del V
            X2_ite.append(X2)

            # CARREFUL WITH THE SIGN OF Vexp !
            # for transition case vn = -Vexp
            # CARREFUL: the Vexp elements are not scaled with lambda
            
            # update t amplitudes
            # ---------------------------------------------------
            vexp = -L[0,1:]*Vexp_class.Vexp[0,1:]
            T1inter = mycc.T1inter(ts,fsp[0])
            ts = mycc.tsupdate(ts, T1inter, rsn=rn, r0n=r0n, vn=vexp)
            # apply DIIS
            if 't' in diis:
                ts_vec = np.ravel(ts)
                ts = tdiis.update(ts_vec).reshape((nocc, nvir))

            # update l (Lambda) amplitudes for the GS
            # ----------------------------------------
            L1inter = mycc.L1inter(ts, fsp[0])
            ls = mycc.lsupdate(ts, ls, L1inter, rsn=rn, lsn=ln, r0n=r0n, l0n=l0n, vn=vexp)
            # apply DIIS
            if 'l' in diis:
                ls_vec = np.ravel(ls)
                ls = ldiis.update(ls_vec).reshape((nocc, nvir))

            # Update En_r/En_l and r, r0, l and l0 amplitudes for each ES
            # ------------------------------------------------------------

            for i in range(nbr_states):

                # todo= most element in Rinter dot not depend on Vexp -> calculate ones for all states

                # R and R0 intermediates
                # ------------------------
                vexp = -L[0, i + 1] * Vexp_class.Vexp[0, i + 1] # V0n
                Rinter  = mycc.R1inter(ts, fsp[i], vexp)
                R0inter = mycc.R0inter(ts, fsp[i], vexp)

                # update En_r
                # ------------------------
                En_r, o,v = mycc.Extract_Em_r(rn[i], r0n[i], Rinter)

                # update r0
                # ------------------------
                r0n[i] = mycc.r0update(rn[i], r0n[i], En_r, R0inter)

                # Update r
                # -------------------------
                rn[i] = mycc.rsupdate(rn[i], r0n[i], Rinter, En_r)

                # Get missing r ampl
                # -------------------------
                if rn[0].shape[0]**2 > 4: # if more than 2 basis functions
                    rn[i][o,v] = mycc.get_rov(ln[i], l0n[i], rn[i], r0n[i], [o,v])
                    rn[i][o+1,v+1] = rn[i][o,v] # G format

                # L and L0 inter
                # ------------------------
                vexp = -L[i+1,0]*Vexp_class.Vexp[i+1,0] # Vn0
                Linter = mycc.es_L1inter(ts, fsp[i], vexp )
                L0inter = mycc.L0inter(ts, fsp[i], vexp)

                # Update En_l
                # ------------------------
                En_l, o, v = mycc.Extract_Em_l(ln[i], l0n[i], Linter)

                # Update l0
                # ------------------------
                l0n[i] = mycc.l0update(ln[i], l0n[i], En_l, L0inter)

                # Update l
                # ------------------------
                ln[i] = mycc.es_lsupdate(ln[i], l0n[i], En_l, Linter)

                # Get missing l amp
                # ------------------------
                if ln[0].shape[0] ** 2 > 4: # if more than 2 basis functions
                    ln[i][o, v] = mycc.get_rov(rn[i], r0n[i], ln[i], l0n[i], [o, v])
                    ln[i][o + 1, v + 1] = ln[i][o, v]  # G format

                # Store excited states energies Ep = (En_r,En_l)
                # -----------------------------------------------
                Ep[i+1][0] = En_r
                Ep[i+1][1] = En_l

            #del Rinter, R0inter, Linter, L0inter, vexp

            # Store GS energies Ep
            # --------------------------------------------
            vexp = [-L[1:i,1:i]*Vexp_class.Vexp[1:i,1:i] for i in range(nbr_states)]
            Ep[0][0] = mycc.energy_ccs(ts, fsp[0], rs=rn, vnn=vexp)
            Ep_ite.append(Ep)

            # checking convergence
            # --------------------------------------------
            dic = {'ts':ts, 'ls':ls, 'rn':rn, 'ln':ln}
            conv = self.Conv_check(dic)
            conv_ite.append(conv)
            if ite > 0:
                Dconv = np.linalg.norm(conv - conv_old)
            conv_ite.append(Dconv)

            # print convergence infos
            # --------------------------------------------
            tmp = [ite, format_float.format(Dconv)]
            for i in range(nbr_states):
                tmp.extend([format_float.format(C_norm[i, i]), r0n[i], l0n[i], Ep[i+1][0], Ep[i+1][1], format_float.format(C_norm[0, i])])
            table.append(tmp)

            if ite >= self.maxiter:
                Conv_text = 'Max iteration reached'
                print()
                print(Conv_text)
                print(tabulate(table, headers, tablefmt="rst"))
                break
            if Dconv > 3.:
                Conv_text = 'Diverges for lambda = {} after {} iterations'.format(L.flatten(), ite)
                print()
                print(Conv_text)
                print(tabulate(table, headers, tablefmt="rst"))
                break

            ite += 1

        else:
            print(tabulate(table, headers, tablefmt="rst"))
            print()
            Conv_text = 'Convergence reached for lambda= {}, after {} iteration'.format(L.flatten(), ite)
            print(Conv_text)
            print()
            print('Final energies: ', Ep_ite[-1])
             
        #return Conv_text, np.asarray(Ep_ite), np.asarray(X2_ite), np.asarray(conv_ite),dic


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
    # todo: add QChem transition dipole moment
    exp_data[0, 1] = ['dip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
    exp_data[0, 2] = ['dip', [0.000000, 0.000000, -0.622534]]  # DE = 0.37 au

    # Vexp object
    VXexp = exp_pot.Exp(exp_data,mol,mgf.mo_coeff)

    # initial rn, r0n and ln, l0n list
    rnini, DE = utilities.koopman_init_guess(mo_energy,mo_occ,nstates=2)
    lnini = [i * 1 for i in rnini]
    r0ini = utilities.EOM_r0(DE,np.zeros((gnocc,gnvir)),rnini,gfs,geris.oovv)
    l0ini = [i * 0 for i in r0ini]

    # convergence options
    maxiter = 40
    conv_thres = 10 ** -8
    diis = ('',)  # must be tuple

    # initialise Solver_CCS Class
    Solver = Solver_ES(mccsg, VXexp, rnini, r0ini, conv_var='rl', ln_ini=lnini, l0n_ini=l0ini, maxiter=maxiter, diis=diis)

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
    print('r0= ', r0ini[0])
    print()

    # Solve for L = 0
    L = np.zeros_like((exp_data))
    Solver.SCF(L)
