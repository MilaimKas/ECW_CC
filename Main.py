#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
 ECW-CC
 Contains the main loop over experimental weight L
 Calls the different Solver
 print results and plot X2(L)
'''


import matplotlib.pyplot as plt

# Python
import numpy as np

# PySCF
from pyscf import gto, scf, cc

# CC functions
import CCS, CCSD
# Vexp functions
import exp_pot
# gamma exp
import gamma_exp
# Solver for ECW GS equations
import Solver_GS
# additional functions
import utilities
# PySCF two electron integrals
import Eris

# Creating new molecule object
# ------------------------------

class ECW:
    def __init__(self, molecule, basis, int_thresh=1e-13, out_dir=None):
        '''
        Build the PySCF mol object and performs HF calculation

        :param molecule: string with name of molecule to be used
        :param basis: string with basis set to be used
        :param int_thresh: threshold for 2 electron integrals
        :param out_dir: path to the cube file directory (string)
        '''
        
        mol = gto.Mole()

        # Geometry

        mol_list = ['c2h4', 'h2o2', 'h2o', 'allene', 'urea']

        if molecule == 'c2h4':
            # C2H4
            mol.atom = """
            C	0.0000000	0.0000000	0.6034010
            C	0.0000000	0.0000000	-0.6034010
            H	0.0000000	0.0000000	1.6667490
            H	0.0000000	0.0000000	-1.6667490
            """
        
        elif molecule == 'h2o2':
            # H peroxyde CCSD(T)=FULL/daug-cc-pVTZ
            mol.atom = """
            O	0.0000000	0.7272250	-0.0593400
            O	0.0000000	-0.7272250	-0.0593400
            H	0.7847270	0.8942120	0.4747180
            H	-0.7847270	-0.8942120	0.4747180
            """
        
        elif molecule == 'allene':
            # Allene CH2CCH2 (geo = CCSD(T)=FULL/cc-pVTZ)
            mol.atom="""
            C	0.0000000	0.0000000	0.0000000
            C	0.0000000	0.0000000	1.3079970
            C	0.0000000	0.0000000	-1.3079970
            H	0.0000000	0.9259120	1.8616000
            H	0.0000000	-0.9259120	1.8616000
            H	0.9259120	0.0000000	-1.8616000
            H	-0.9259120	0.0000000	-1.8616000
            """
        
        elif molecule == 'h2o':
            # Water molecule
            mol.atom = [
                [8, (0., 0., 0.)],
                [1, (0., -0.757, 0.587)],
                [1, (0., 0.757, 0.587)]]
        
        elif molecule == 'urea':
            # Urea (geo. optimized B3LYP/6-31G*)
            mol.atom="""
            C1  0.0000   0.0000   0.1449
            O1  0.0000   0.0000   1.3650
            N1  -0.1309   1.1569  -0.6170
            N2  0.1309  -1.1569  -0.6170
            H1  0.0000   1.9959  -0.0667
            H2  0.3478   1.1778  -1.5093
            H3  0.0000  -1.9959  -0.0667
            H4  -0.3478  -1.1778  -1.5093
            """
        elif any(char.isdigit() for char in molecule):
            mol.atom = molecule
        else:
            print('Molecule not recognize')
            print('List of available molecules:')
            print(mol_list)
            raise ValueError()
        
        symmetry = True
        mol.unit = 'angstrom'
        
        # basis set
        mol.basis = basis
        
        # GHF calculation
        # -------------------
        
        mol.verbose = 0  # no output
        mol.charge = 0  # charge
        mol.spin = 0  # spin
        
        mol.build()  # build mol object
        natm = int(mol.natm)  # number of atoms
        
        mf = scf.GHF(mol)
        
        # option for calculation
        mf.conv_tol = 1e-09  # energy tolerence
        mf.conv_tol_grad = np.sqrt(mf.conv_tol)  # gradient tolerence
        mf.direct_scf_tol = int_thresh  # tolerence in discarding integrals
        mf.max_cycle = 100
        mf.max_memory = 1000
        
        # do scf calculation
        mf.kernel()
        
        # variables related to the MOs basis
        self.mo_coeff = mf.mo_coeff  # Molecular orbital (MO) coefficients (matrix where rows are atomic orbitals (AO) and columns are MOs)
        mo_occ = mf.mo_occ  # MO occupancy (vector with length equal to number of MOs)
        mocc = self.mo_coeff[:, mo_occ > 0]  # Only take the mo_coeff of occupied orb
        mvir = self.mo_coeff[:, mo_occ == 0]  # Only take the mo_coeff of virtual orb
        self.nocc = mocc.shape[1]  # Number of occ MOs in HF
        self.nvir = mvir.shape[1]  # Number of virtual MOS in HF
        
        # HF total energy
        self.EHF = mf.e_tot
        
        # dimension
        self.dim = self.nocc + self.nvir
        self.aosize = mol.nao_nr()  # number of AO --> size of the basis
        
        # a and b electrons
        Nele_a = mol.nelec[0]  # mol.nelec gives the number of alpha and beta ele (nalpha,nbeta)
        Nele_b = mol.nelec[1]
        Nele = Nele_a + Nele_b
        
        # HF rdm1
        self.rdm1_hf = mf.make_rdm1()
        self.mf = mf
        self.mol = mol

        # print cube file
        # ---------------------------------
        self.out_dir = out_dir
        if out_dir is not None:
            from pyscf.tools import cubegen
            # convert g to r
            rdm1_hf = utilities.convert_g_to_ru_rdm1(self.rdm1_hf)[0]
            cubegen.density(mol, out_dir+'/HF.cube', rdm1_hf, nx=80, ny=80, nz=80)

        # One and two particle integrals
        # ----------------------------------
        self.eris = Eris.geris(cc.GCCSD(mf))
        self.fock = self.eris.fock
        #fock = np.diag(mo_ene)
        # S = mf.get_ovlp() # overlap of the AOs basis
        #ccsd = cc.GCCSD(mf)
        #eris = ccsd.ao2mo(mo_coeff)

        # initialize exp_data
        # --------------------------
        self.exp_data = np.full((1, 1), None)

        # initial value for DE and r1 amp
        # ----------------------------------
        self.r_ini = []
        self.DE = []

        # Target energies
        # --------------------
        self.Eexp_GS = None
        self.Eexp_ES = [] # excitation energies
        
        print()
        print('--------------')
        print('Molecule build')
        print('--------------')
        print()

    def Build_GS_exp(self, posthf='HF', field=None, para_factor=None):
        '''
        Build "experimental" or "target" data for the GS

        :param posthf: method to calculate gamma_exp_GS
        :param field: external field ta calculate gamme_exp_GS
        :param para_factor: underfitted coefficient
        :return: update exp_data matrix
        '''

        # Build gamma_exp for the GS
        # ---------------------------
        gexp = gamma_exp.Gexp(self.mol,posthf)

        if field is not None:
           gexp.Vext(field)
        
        gexp.build()
        
        if para_factor is not None:
            gexp.underfit(para_factor)

        gamma_mo = utilities.ao_to_mo(gexp.gamma_ao, self.mo_coeff)
        # Update exp_data
        self.exp_data[0,0] = ['mat', gamma_mo]
        # Store GS exp energy
        self.Eexp_GS = gexp.Eexp

        if self.out_dir:
            fout = self.out_dir+'/target_GS.cube'
            utilities.cube(self.exp_data[0, 0][1], self.mo_coeff, self.mol, fout)
            
        print('GS data stored ')

    def Build_ES_exp_MOM(self,nbr_of_es=(1,0), field=None):
        '''
        Build excited states data from MOM calculation

        :param nbr_of_es: tuple with number of valence and core excited states from which rdm1 and tr_rdm1 have
        to be calculated
        :param field: additional external electric field to be added
        :return: updated exp_data matrix and list of initial r1 vectors
        '''

        es_exp = gamma_exp.ESexp(self.mol, Vext=field, nbr_of_states=nbr_of_es)
        es_exp.MOM()

        # store GS Eexp
        if self.Eexp_GS is not None:
            self.Eexp_GS = es_exp.Eexp_GS
            raise Warning('A Energy for the target (experimental) GS is already given, this will overwrite it')

        # store target excitation energies
        self.Eexp_ES.append(es_exp.DE_exp)

        # expand exp_data and
        i = self.exp_data.shape[0]

        for tr_rdm1 in es_exp.gamma_tr_ao:

            expand = self.exp_data.shape[0]
            self.exp_data.resize((expand,expand))
            tr_rdm1 = utilities.ao_to_mo(tr_rdm1, self.mo_coeff)
            self.exp_data[i,i] = ['mat', tr_rdm1]

            # store r1 initial guess
            self.r_ini.append(es_exp.ini_r[i])
            # store initial Koopman excitation energies
            i,a = np.argwhere(es_exp.ini_r[i])
            self.DE.append(self.fock[a,a]-self.fock[i,i])

            i += 1

    def Build_ES_exp(self,dip_list, DE_list, rini_list=None):
        '''
        Build excited states data from given transition properties

        :param dip_list: array with transition dipole moment values (x,y,z) for the target states
        :param DE_list: excitation energies for the target states
        :param rini_list: initial i->a one-electron excitation for each target states
               -> if rini are not given, they are taken from Koopman's initial guess
        :return: updated exp_data matrix
        '''

        # check length
        if len(dip_list) != len(DE_list):
            raise ValueError('length of given tdm and DE must be the same')

        i = self.exp_data.shape[0]
        for dip in dip_list:
            expand = self.exp_data.shape[0]
            self.exp_data.resize((expand,expand))
            self.exp_data[i,i] = ['dip', dip]

        for DE, rini in zip(DE_list, rini_list):
            self.Eexp_ES.append(DE)
            self.r_ini.append(rini)

        if rini_list is None:
            r1,de = utilities.koopman_init_guess(np.diag(self.fock), self.mf.mo_occ, len(dip_list))
            self.DE.append(de)
            self.r_ini.append(r1)
        else:
            if len(rini_list) != len(dip_list):
                raise ValueError('The number of given one-electron excitations is not equal to the number of given transition dipole moments')
            for j in range(len(rini_list)):
                i,a = rini_list[j]
                self.DE.append(self.fock[a, a] - self.fock[i, i])


    def CCS_GS(self, Larray ,alpha=None, method='scf', graph=True, diis=('',), nbr_cube_file=2, tl1ini=0, print_ite_info=False,
               beta=None, diis_max=15, conv='tl', conv_thres=10**-6, maxiter=40):
        '''
        Call CCS solver for the ground state using SCF+DIIS or gradient (steepest descend/Newton) method
        
        :param Larray: array of L value for which the CCS equations have to be solved
        :param alpha: L1 reg term
        :param method: SCF, newton, gradient or L1_grad
        :param beta: iteration step for the L1_grad method
        :param graph: True if a final plot X2(L) is shown
        :param diis: apply diis to rdm1 ('rdm1'), t amplitudes ('t'), l amplitudes ('l') or both ('tl')
        :param diis_max: max diis space
        :param nbr_cube_file: number of cube file to be printed for equaly spaced L values
        :param tl1ini: initial value for t1 and l1 amplitudes (0=zero, 1=perturbtaion thery, 2=random)
        :param print_ite_info: True if iteration step are to be printed
        :param conv: convergence variable ('l', 'tl' or 'Ep')
        :param conv_thres: threshold for convergence
        :param maxiter: max number of iterations
        :return: Converged results
                 [0] = convergence text
                 [1] = Ep(it)
                 [2] = X2(it) list of tuple: (X2, vmax, X2_Ek)
                 [3] = conv(it) 
                 [4] = final gamma (rdm1) calc
                 [5] = final ts and ls amplitudes
        '''

        if method == 'L1_grad' and beta is None:
            raise ValueError('A value for beta (gradient step) must be given for the L1_grad method')

        if self.exp_data.shape[0] > 1:
            raise Warning('Data for excited states have been found but a ground state solver is used, '
                          'the Vexp potential will only contained GS data')

        # initial values for ts and ls
        if tl1ini == 1:
            mo_ene = np.diag(self.fock)
            eia = mo_ene[:self.nocc, None] - mo_ene[None, self.nocc:]
            tsini = self.fock[:self.nocc, self.nocc:] / eia
            lsini = tsini.copy()
        # random number
        elif tl1ini == 2:
            tsini = np.random.rand(self.nocc//2, self.nvir//2) * 0.01
            lsini = np.random.rand(self.nocc//2, self.nvir//2) * 0.01
            tsini = utilities.convert_r_to_g_amp(tsini)
            lsini = utilities.convert_r_to_g_amp(lsini)
        # zero
        else:
            tsini = np.zeros((self.nocc, self.nvir))
            lsini = np.zeros((self.nocc, self.nvir))

        ts = tsini.copy()
        ls = lsini.copy()

        # L value at which a cube file is to be generated
        idx = np.round(np.linspace(0, len(Larray) - 1, nbr_cube_file)).astype(int)
        L_print = Larray[idx]

        # Vexp class
        VXexp = exp_pot.Exp(self.exp_data, self.mol, self.mo_coeff)
        # CCS class
        myccs = CCS.Gccs(self.eris)
        if method == 'newton' or method == 'descend':
            mygrad = CCS.ccs_gradient(self.eris)
        else:
            mygrad = None

        # CCS_GS solver
        Solve = Solver_GS.Solver_CCS(myccs, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini,
                                     diis=diis, maxdiis=diis_max, maxiter=maxiter, CCS_grad=mygrad)

        # initialize list for results
        X2_lamb = []
        Ep_lamb = []
        vmax_lamb = []
        X2_Ek = []
        Result = None
        Ep = None
        X2 = None

        print()
        print("##############################")
        print("#  Results using "+method+"   ")
        print("##############################")
        print()

        # Loop over Lambda
        for L in Larray:

            print("LAMBDA= ", L)

            if method == 'newton':
                Result = Solve.Gradient(L, ts=ts, ls=ls)
            elif method == 'descend':
                Result = Solve.Gradient(L, method=method, ts=ts, ls=ls, beta=beta)
            elif method == 'scf':
                Result = Solve.SCF(L, ts=ts, ls=ls, alpha=alpha)
            elif method == 'L1_grad':
                Result = Solve.L1_grad(L, alpha, beta, ts=ts, ls=ls)
            else:
                raise ValueError('method not recognize')
            ts, ls = Result[5]

            # print cube file for L listed in L_print in dir_cube path
            if self.out_dir:
                if L in L_print:
                    fout = self.out_dir + 'L{}'.format(int(L)) + '.cube'
                    utilities.cube(Result[4], self.mo_coeff, self.mol, fout)

            if print_ite_info:
                print('Iteration steps')
                print('ite', '      ', 'Ep', '         ', conv, '          ', 'X2')
                for i in range(len(Result[1])):
                    print(i, '  ', "{:.4e}".format(Result[1][i]), '  ', "{:.4e}".format(Result[3][i]), '  ',
                          "{:.4e}".format(Result[2][i][0]))
            
            # print convergence text
            print(Result[0])
            print()
            Ep = Result[1][-1]
            X2 = Result[2][-1][0]
            vmax = Result[2][-1][1]
            
            if graph:
                X2_lamb.append(X2)
                Ep_lamb.append(self.EHF - Ep)
                vmax_lamb.append(vmax)
                X2_Ek.append(VXexp.X2_Ek_GS)

        print("FINAL RESULTS")
        print("Ep   = ", Ep)
        print("X2   = ", X2)
        print("DEk  = ", X2_Ek[-1])
        print()
        print("EHF    = ", self.EHF)
        print("Eexp   = ", self.Eexp_GS+self.EHF)

        plot=None
        if graph:
            plot=plot_results(Larray, Ep_lamb, X2_lamb, vmax_lamb, X2_Ek)
            
        return Result, plot

    def CCSD_GS(self, Larray , alpha=None, method='SCF', graph=True, diis=('',), nbr_cube_file=2, tl1ini=0, print_ite_info=False,
                beta=None, diis_max=15, conv='tl', conv_thres=10**-6, maxiter=40):
        '''
        Call CCSD solver for the ground state using SCF+DIIS method
        
        :param Larray: array of L value for which the CCS equations have to be solved
        :param alpha: L1 reg term
        :param method: SCF, newton, gradient or L1_grad
        :param beta: iteration step for the L1_grad method
        :param graph: True if a final plot X2(L) is shown
        :param diis: apply diis to rdm1 ('rdm1'), t amplitudes ('t'), l amplitudes ('l') or both ('tl')
        :param diis_max: max diis space
        :param nbr_cube_file: number of cube file to be printed for equaly spaced L values
        :param tl1ini: initial value for t1 and l1 amplitudes (0=zero, 1=perturbtaion thery, 2=random)
        :param print_ite_info: True if iteration step are to be printed
        :param conv: convergence variable ('l', 'tl' or 'Ep')
        :param conv_thres: threshold for convergence
        :param maxiter: max number of iterations
        :return: Final converged results
                 [0] = convergence text
                 [1] = Ep(it)
                 [2] = X2(it)
                 [3] = conv(it)
                 [4] = last gamma (rdm1) calc
                 [5] = list [t1,l2,t2,l2] with final amplitudes
        '''
        
        if self.exp_data.shape[0] > 1:
            raise Warning('Data for excited states have been found but a ground state solver is used, '
                          'the Vexp potential will only contained GS data')
        
        # initial values for ts and ls
        if tl1ini == 1:
            mo_ene = np.diag(self.fock)
            eia = mo_ene[:self.nocc, None] - mo_ene[None, self.nocc:]
            tsini = self.fock[:self.nocc, self.nocc:] / eia
            lsini = tsini.copy()
        # random number
        elif tl1ini == 2:
            tsini = np.random.rand(self.nocc//2, self.nvir//2) * 0.01
            lsini = np.random.rand(self.nocc//2, self.nvir//2) * 0.01
            tsini = utilities.convert_r_to_g_amp(tsini)
            lsini = utilities.convert_r_to_g_amp(lsini)
        # zero
        else:
            tsini = np.zeros((self.nocc, self.nvir))
            lsini = np.zeros((self.nocc, self.nvir))
        
        ts = tsini.copy()
        ls = lsini.copy()
        
        # L value at which a cube file is to be generated
        idx = np.round(np.linspace(0, len(Larray) - 1, nbr_cube_file)).astype(int)
        L_print = Larray[idx]
        
        # Vexp class
        VXexp = exp_pot.Exp(self.exp_data, self.mol, self.mo_coeff)
        # CCSD class
        myccs = CCSD.GCC(self.eris)

        # CCS_GS solver
        Solve = Solver_GS.Solver_CCSD(myccs, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini,
                                     diis=diis, maxdiis=diis_max, maxiter=maxiter)
        # initialize plot
        if graph:
            fig, axs = plt.subplots(2, sharex='col')
        
        # initialize list for results
        X2_lamb = []
        Ep_lamb = []
        vmax_lamb = []
        X2_Ek = []

        # initialize double amp
        td = None
        ld = None
        Result = None
        Ep = None
        X2 = None
        
        print()
        print("#############")
        print("#  Results   ")
        print("#############")
        print()
        
        # Loop over Lambda
        for L in Larray:
        
            print("LAMBDA= ", L)

            Result = Solve.SCF(L, ts=ts, ls=ls, td=td, ld=ld, alpha=alpha)
            # Use previous amplitudes as initial guess
            ts, ls, td, ld = Result[5]
            
            # print cube file for L listed in L_print in out_dir path
            if self.out_dir:
                if L in L_print:
                    fout = self.out_dir + 'L{}'.format(int(L)) + '.cube'
                    utilities.cube(Result[4], self.mo_coeff, self.mol, fout)

            if print_ite_info:
                print('Iteration steps')
                print('ite', '      ', 'Ep', '         ', conv, '          ', 'X2')
                for i in range(len(Result[1])):
                    print(i, '  ', "{:.4e}".format(Result[1][i]), '  ', "{:.4e}".format(Result[3][i]), '  ',
                          "{:.4e}".format(Result[2][i][0]))

            # print convergence text
            print(Result[0])
            print()
            Ep = Result[1][-1]
            X2 = Result[2][-1][0]
            vmax = Result[2][-1][1]

            if graph:
                X2_lamb.append(X2)
                Ep_lamb.append(self.EHF - Ep)
                vmax_lamb.append(vmax)
                X2_Ek.append(VXexp.X2_Ek_GS)

        print("FINAL RESULTS")
        print("Ep   = ", Ep)
        print("X2   = ", X2)
        print("DEk  = ", X2_Ek[-1])
        print()
        print("EHF    = ", self.EHF)
        print("Eexp   = ", self.Eexp_GS)
        print("EHF-EP = ", self.EHF - Ep)

        plot=None
        if graph:
            plot=plot_results(Larray, Ep_lamb, X2_lamb, vmax_lamb, X2_Ek)

        return Result, plot

    #def CCS_ES(self):

def plot_results(Larray, Ep_lamb, X2_lamb, vmax_lamb, X2_Ek):
    '''
    Plot Ep, X2, vax and DEk as a function of L
    
    :param Larray: experimental weigth array 
    :param Ep_lamb: Ep array
    :param X2_lamb: X2 array
    :param vmax_lamb: vmax array
    :param X2_Ek: DEk array
    '''
    
    fig, axs = plt.subplots(2, sharex='col')
    # Plot Ep, X2 and vmax only for converged lambdas

    # Energy
    axs[0].plot(Larray, Ep_lamb, marker='o', markerfacecolor='black', markersize=8, color='grey', linewidth=1)
    # axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(2, 3), useMathText=True, useLocale=True)
    axs[0].set_ylabel('EHF-Ep', color='black')

    # X2 and vmax
    axs[1].plot(Larray, X2_lamb, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=1)
    ax2 = axs[1].twinx()
    ax2.plot(Larray, vmax_lamb, marker='o', markerfacecolor='blue', markersize=8, color='lightblue',
             linewidth=1)
    ax2.set_ylabel('Vmax', color='blue')
    axs[1].set_ylabel('X2', color='red')
    axs[1].set_xlabel('Lambda')
    # axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(2, 3), useMathText=True)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(2, 3), useMathText=True)

    # Ek difference
    ax2 = axs[0].twinx()
    ax2.plot(Larray, X2_Ek, marker='o', markerfacecolor='grey', markersize=8, color='black', linewidth=1)
    ax2.set_ylabel('Ek difference', color='grey')

    return plt


if __name__ == '__main__':

    molecule = 'h2o'
    basis = '6-31g'

    # Choose lambda array
    lambi = 0  # weight for Vexp, initial value
    lambf = 0.8  # lambda final
    lambn = 5  # number of Lambda value
    Larray = np.linspace(lambi, lambf, num=lambn)

    # Build molecules and basis
    ecw = ECW(molecule, basis)
    # Build GS exp data from HF+field
    ecw.Build_GS_exp('HF', field=[0.05, 0.01, 0.])
    # Build exp data from given 1e prop (Ek from CCSD+[0.05, 0.01, 0.]+6-311+g**)
    #ecw.exp_data[0,0] = ['Ek', 70.4 ]

    # Solve ECW-CCS/CCSD equations using SCF algorithm with given alpha
    #Results, plot = ecw.CCS_GS(Larray,graph=True, alpha=0.)
    Results, plot = ecw.CCSD_GS(Larray,graph=True, alpha=0.1)
    plot.show()





