#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
 ECW-CC
 Contains the main loop over experimental weight L
 Calls the different Solver
 print results and plot X2(L)
'''
import copy

import matplotlib.pyplot as plt

# Python
import numpy as np
from tabulate import tabulate

# PySCF
from pyscf import gto, scf, cc

# Import ECW modules
#from . import CCS, CCSD, exp_pot, gamma_exp, utilities, Eris, Solver_GS, Solver_ES
import CCS, CCSD, exp_pot, gamma_exp, utilities, Eris, Solver_GS, Solver_ES

# Global float format for print
# ------------------------------
format_float = '{:10.5e}'

# Creating new molecule object
# ------------------------------

class ECW:
    def __init__(self, molecule, basis, int_thresh=1e-13, out_dir=None, G_format=False):
        '''
        Build the PySCF mol object and performs HF calculation

        :param molecule: string with name of molecule to be used
        :param basis: string with basis set to be used
        :param int_thresh: threshold for 2 electron integrals
        :param out_dir: path to the directory where the cube file are saved (string), if None do not print cube files
        :param G_format: if True, the spin-orbital basis are obtained from a GHF calculation,
                         if False, the spin-orbital are converted from a RHF calc (a and b SO are degenerate)
        '''

        # Use generalized format
        self.G_format = G_format

        # CC class
        self.myccs = None
        self.myccsd = None

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
        
        # HF calculation
        # -------------------
        
        mol.verbose = 0  # no output
        mol.charge = 0  # charge
        mol.spin = 0  # spin
        
        mol.build()  # build mol object
        natm = int(mol.natm)  # number of atoms

        if G_format:
            mf = scf.GHF(mol)
        else:
            mf = scf.RHF(mol)

        # option for calculation
        mf.conv_tol = 1e-09  # energy tolerence
        mf.conv_tol_grad = np.sqrt(mf.conv_tol)  # gradient tolerence
        mf.direct_scf_tol = int_thresh  # tolerence in discarding integrals
        mf.max_cycle = 100
        mf.max_memory = 1000
        
        # do scf calculation
        mf.kernel()

        if not G_format:
            mf = scf.addons.convert_to_ghf(mf)

        # variables related to the MOs basis
        self.mo_coeff = mf.mo_coeff  # matrix where rows are atomic orbitals (AO) and columns are MOs
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

        # Target energies
        # --------------------
        self.Eexp_GS = None
        self.Eexp_ES = [] # excitation energies

        # Store list of Miller indices and unit=cell size
        # ------------------------------------------------
        self.h = None
        self.rec_vec = None
        
        print('*** Molecule build ***')

    def Build_GS_exp(self, prop, posthf='HF', field=None, para_factor=None, max_def=None, basis=None):
        '''
        Build "experimental" or "target" data for the GS

        :param prop: property to include in exp_data
                     - 'mat': directly use calculated rdm1 as target
                     - 'Ek', 'v1e', 'dip'
                     - ['F', h, (a, b, c)] for structure factor calculation where h=[h1,h2, ...] and hi=(hx,hy,hz)
                                           and a,bc are the lattice vector length. If not given a=b=c=10
        :param posthf: method to calculate gamma_exp_GS
        :param field: external field ta calculate gamme_exp_GS
        :param para_factor: under-fitting coefficient
        :param max_def: maximum bond length deformation in au
        :basis: basis used for the calculation of properties
        :return: update exp_data matrix
        '''

        # if 'mat' basis must be self.basis
        if prop == 'mat' and basis is not None:
            print('If rdm1 are to be compared, exp and calc rdm1 must be in the same basis')
            basis = self.mol.basis
        if prop == 'mat' and max_def is not None:
            print('If rdm1 are to be compared, the geometry for exp anc calc must be the same')
            max_def=None

        # Build gamma_exp for the GS
        # ---------------------------

        gexp = gamma_exp.Gexp(self.mol, posthf, self.G_format)

        if isinstance(max_def, float):
            gexp.deform(max_def)
        if isinstance(field, list):
           gexp.Vext(field)
        
        gexp.build()
        
        if para_factor is not None:
            gexp.underfit(para_factor)

        # Store GS exp energy
        self.Eexp_GS = gexp.Eexp

        # directly compare rdm1
        if isinstance(prop, str) and prop == 'mat':
            # Update exp_data
            gamma_mo = utilities.ao_to_mo(gexp.gamma_ao, self.mo_coeff)
            self.exp_data[0, 0] = ['mat', gamma_mo]

        # other properties (F, Ek, dip, etc)
        elif isinstance(prop, list):
            self.exp_data[0,0] = []
            for p in prop:

                # Structure Factor p=['F', F]
                if isinstance(p, list):
                    if p[0] == 'F':
                        h = p[1]
                        self.h = h
                        if len(p) > 2:
                            a = p[2][0]
                            b = p[2][1]
                            c = p[2][2]
                        else:
                            a=10.
                            b=10.
                            c=10.
                        self.rec_vec = np.asarray([a, b, c])
                        # calculate list of structure factors for each given set of Miller indices
                        F = utilities.structure_factor(gexp.mol_def, h, gexp.gamma_ao,
                                                   aobasis=True, mo_coeff=gexp.mo_coeff_def, rec_vec=self.rec_vec)
                        self.exp_data[0, 0].append(['F', F])
                    else:
                        raise SyntaxError('Input for prop must be list(prop1, prop2, ...) where prop is '
                                          'either a string (Ek, v1e, dip) or a list ['F', h, [a,b,c]] ')

                # Kinetic energy
                if p == 'Ek':
                    ek = utilities.Ekin(self.mol, gexp.gamma_ao, aobasis=True, mo_coeff=self.mo_coeff)
                    self.exp_data[0, 0].append(['Ek', ek])

                # one-electron potential
                if p == 'v1e':
                    v1e = utilities.v1e(self.mol, gexp.gamma_ao, aobasis=True, mo_coeff=self.mo_coeff)
                    self.exp_data[0, 0].append(['v1e', v1e])

                # dipole moment
                if p == 'dip':
                    dip = utilities.dipole(self.mol, gexp.gamma_ao, aobasis=False, mo_coeff=self.mo_coeff)
                    self.exp_data[0, 0].append(['dip', dip])
        else:
            raise ValueError('Prop is either mat or a list including Ek and/or v1e and/or dip')

        if self.out_dir:
            fout = self.out_dir+'/target_GS.cube'
            utilities.cube(self.exp_data[0, 0][1], self.mo_coeff, self.mol, fout)
            
        print('*** GS data stored ***')

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

    def Build_ES_exp(self, dip_list, nbr_of_states, rini_list=None):
        '''
        Build excited states data from given transition properties

        :param dip_list: list with transition dipole moment values np.array(x,y,z) for the target states
                        dip_list = list([x1,y1,z1],[x2,y2,z2], ...)
                     or dip_list =  list(list([x1r,y1r,z1r],[x1l,y1l,z1l]), list([x2r,y2r,z2r],[x2l,y2l,z2l])  ...)
                     if both left and right values are given.
        :param DE_list: excitation energies for the target states
        :param nbr_of_states: number of valence and core excited states (nval,ncore)
        :param rini_list: initial i->a one-electron excitation for each target states
               -> if rini are not given, they are taken from Koopman's initial guess
        :return: updated exp_data matrix
        '''

        # check length
        if len(dip_list) != sum(nbr_of_states):
            raise ValueError('length of given tdm must be the same as nbr of excited states')

        # Update exp_data with given dipole moments
        i = self.exp_data.shape[0]
        for dip in dip_list:
            expand = self.exp_data.shape[0]
            self.exp_data.resize((expand, expand))
            # if left and right values are given
            if isinstance(dip,list) and len(dip) == 2:
                # left transition moment
                self.exp_data[0, i] = ['dip', dip[1]]
                # right transition moment
                self.exp_data[i, 0] = ['dip', dip[0]]
            elif isinstance(dip, np.array):
                # left transition moment
                self.exp_data[0, i] = ['dip', np.sqrt(dip)]
                # right transition moment
                self.exp_data[i, 0] = ['dip', np.sqrt(dip)]
            else:
                raise ValueError('Bad format for experimental transition dipole moment. Should be a list of np.array')

        # CCS class
        if self.myccs is None:
            self.myccs = CCS.Gccs(self.eris)

        # Koopman initial guess
        if rini_list is None:
            r1, de = utilities.koopman_init_guess(np.diag(self.fock), self.mf.mo_occ, nbr_of_states)
            r0ini = [self.myccs.Extract_r0(r, np.zeros_like(r), self.fock, np.zeros_like(self.fock)) for r in r1]
            self.r_ini = r1
            self.l_ini = r1.copy()
            self.r0_ini = r0ini
            self.l0_ini = r0ini.copy()
        else:
            if len(rini_list) != len(dip_list):
                raise ValueError('The number of given initial r vectors is not '
                                 'equal to the number of given transition dipole moments')

            self.r_ini = rini_list
            self.r0_ini = [self.myccs.Extract_r0(r, np.zeros_like(r), self.fock, np.zeros_like(self.fock))
                           for r in rini_list]
            self.l_ini = copy.deepcopy(self.r_ini)
            self.l0_ini = copy.deepcopy(self.r0_ini)

        # orthogonalize and normalize vectors
        # NOTE: only a bi-orthogonal set can be constructed within CC theory -> only r[0] and l[1] will be ortho
        self.r_ini, self.l_ini, self.r0_ini, self.l0_ini = \
            utilities.ortho_norm(self.r_ini, self.l_ini, self.r0_ini, self.l0_ini)


    def CCS_GS(self, Larray ,alpha=None, Alpha=None, method='scf', graph=True, diis=('',), nbr_cube_file=2, tl1ini=0, print_ite_info=False,
               beta=None, diis_max=15, conv='tl', conv_thres=10**-6, maxiter=40, tablefmt='rst'):
        '''
        Call CCS solver for the ground state using SCF+DIIS or gradient (steepest descend/Newton) method
        
        :param Larray: array of L value for which the CCS equations have to be solved
        :param alpha: L1 reg term applied at micro-iteration
        :param Alpha: L1 reg term applied at macro-iteration
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
        :param tablefmt: tabulate format for the printed table ('rst' or 'latex' for example)
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
                          'the Vexp potential will only contain GS data')

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
        VXexp = exp_pot.Exp(self.exp_data, self.mol, self.mo_coeff, rec_vec=self.rec_vec, h=self.h)

        # CCS class
        if self.myccs is None:
            self.myccs = CCS.Gccs(self.eris)
        if method == 'newton' or method == 'descend':
            mygrad = CCS.ccs_gradient(self.eris)
        else:
            mygrad = None

        # CCS_GS solver
        Solve = Solver_GS.Solver_CCS(self.myccs, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini,
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

            # apply L1 at macro-iteration
            if Alpha is not None:
                # todo: L1 in macro-iteration does not work
                inter = self.myccs.T1inter(ts, Result[4])
                ts = self.myccs.tsupdate_L1(ts, inter, Alpha)
                inter = self.myccs.L1inter(ts, Result[4])
                ls = self.myccs.lsupdate_L1(ls, inter, Alpha)
                del inter

            # print cube file for L listed in L_print in dir_cube path
            if self.out_dir:
                if L in L_print:
                    fout = self.out_dir + 'L{}'.format(int(L)) + '.cube'
                    utilities.cube(Result[4], self.mo_coeff, self.mol, fout)

            if print_ite_info:
                print('Iteration steps')
                headers = ['ite','Ep',str(conv),'X2']
                table = []
                for i in range(len(Result[1])):
                    table.append([i, '{:.4e}'.format(Result[1][i]), "{:.4e}".format(Result[3][i]),
                          "{:.4e}".format(Result[2][i][0])])
                print(tabulate(table, headers, tablefmt=tablefmt))

            # print convergence text
            print(Result[0])
            print()
            Ep = Result[1][-1]
            X2 = Result[2][-1][0]
            print(X2)
            vmax = Result[2][-1][1]
            
            if graph:
                X2_lamb.append(X2)
                Ep_lamb.append(self.EHF - Ep)
                vmax_lamb.append(vmax)
                if VXexp.X2_Ek_GS is not None:
                    X2_Ek.append(VXexp.X2_Ek_GS)

        print("FINAL RESULTS")
        print("Ep   = "+format_float.format(Ep+self.EHF))
        print("X2   = "+format_float.format(X2))
        if VXexp.X2_Ek_GS is not None:
            print("DEk  = "+format_float.format(VXexp.X2_Ek_GS))
        print()
        print("EHF    = "+format_float.format(self.EHF))
        print("Eexp   = "+format_float.format(self.Eexp_GS))

        plot = None
        if graph:
            plot = plot_results(Larray, Ep_lamb, X2_lamb, vmax_lamb, X2_Ek=X2_Ek)

        return Result, plot

    def CCSD_GS(self, Larray , alpha=None, Alpha=None, graph=True, diis=('',), nbr_cube_file=2, tl1ini=0, print_ite_info=False,
                diis_max=15, conv='tl', conv_thres=10**-6, maxiter=40, tablefmt='rst'):
        '''
        Call CCSD solver for the ground state using SCF+DIIS method
        
        :param Larray: array of L value for which the CCS equations have to be solved
        :param alpha: L1 reg term applied at each micro-iteration
        :param Alpha: L1 reg term applied at each macro-iteration
        :param graph: True if a final plot X2(L) is shown
        :param diis: apply diis to rdm1 ('rdm1'), t amplitudes ('t'), l amplitudes ('l') or both ('tl')
        :param diis_max: max diis space
        :param nbr_cube_file: number of cube file to be printed for equaly spaced L values
        :param tl1ini: initial value for t1 and l1 amplitudes (0=zero, 1=perturbtaion thery, 2=random)
        :param print_ite_info: True if iteration step are to be printed
        :param conv: convergence variable ('l', 'tl' or 'Ep')
        :param conv_thres: threshold for convergence
        :param maxiter: max number of iterations
        :param tablefmt: tabulate format for the printed table ('rst' or 'latex' for example)
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
        VXexp = exp_pot.Exp(self.exp_data, self.mol, self.mo_coeff, rec_vec=self.rec_vec, h=self.h)

        # CCSD class
        if self.myccsd is None:
            self.myccd = CCSD.GCC(self.eris)

        # CCS_GS solver
        Solve = Solver_GS.Solver_CCSD(self.myccd, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini,
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
            # Apply L1 here
            if Alpha is not None:
                ts, td = self.myccd.tupdate(ts, td, alpha=Alpha)
                ls, ld = self.myccd.lupdate(ts, td, ls, ld, alpha=Alpha)

            # print cube file for L listed in L_print in out_dir path
            if self.out_dir:
                if L in L_print:
                    fout = self.out_dir + 'L{}'.format(int(L)) + '.cube'
                    utilities.cube(Result[4], self.mo_coeff, self.mol, fout)

            if print_ite_info:
                print('Iteration steps')
                headers = ['ite','Ep',str(conv),'X2']
                table = []
                for i in range(len(Result[1])):
                    table.append([i, '{:.4e}'.format(Result[1][i]), "{:.4e}".format(Result[3][i]),
                          "{:.4e}".format(Result[2][i][0])])
                print(tabulate(table, headers, tablefmt=tablefmt))

            # print convergence text
            print(Result[0])
            print()
            Ep = Result[1][-1]
            X2 = Result[2][-1][0]
            print(X2)
            vmax = Result[2][-1][1]

            if graph:
                X2_lamb.append(X2)
                Ep_lamb.append(self.EHF - Ep)
                vmax_lamb.append(vmax)
                if VXexp.X2_Ek_GS is not None:
                    X2_Ek.append(VXexp.X2_Ek_GS)

        print("FINAL RESULTS")
        print("Ep   = "+format_float.format(Ep+self.EHF))
        print("X2   = "+format_float.format(X2))
        if VXexp.X2_Ek_GS is not None:
            print("DEk  = "+format_float.format(VXexp.X2_Ek_GS))
        print()
        print("EHF    = "+format_float.format(self.EHF))
        print("Eexp   = "+format_float.format(self.Eexp_GS))

        plot=None
        if graph:
            plot = plot_results(Larray, Ep_lamb, X2_lamb, vmax_lamb, X2_Ek=X2_Ek)

        return Result, plot

    def CCS_ES(self, L, exp_data=None, conv_thres=10**-6, maxiter=40, diis=('')):
        '''

        :param L: matrix of lambda values (weigth of exp data)
        :param exp_data: matrix of ex
        :return:
        '''

        if exp_data is None:
            exp_data = self.exp_data
        if exp_data is None:
            raise ValueError('exp data matrix must be given')

        # initial value for r1 and r0
        # ----------------------------------
        if self.r_ini is None:
             raise ValueError('exp data must be created using ecw.Build_ES_exp')

        # CCS class
        if self.myccs is None:
            self.myccs = CCS.Gccs(self.eris)

        # Vexp class
        VXexp = exp_pot.Exp(exp_data, self.mol, self.mo_coeff)

        Solver = Solver_ES.Solver_ES(self.myccs, VXexp, self.r_ini, self.r0_ini, self.lnini, self.l0ini, conv_var='rl',
                                     conv_thres=conv_thres, maxiter=maxiter, diis=diis)

        Lambda = np.full(exp_data.shape, L)
        Solver.SCF(Lambda)

def plot_results(Larray, Ep_lamb, X2_lamb, vmax_lamb, X2_Ek=None):
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
    if X2_Ek:
        ax2 = axs[0].twinx()
        ax2.plot(Larray, X2_Ek, marker='o', markerfacecolor='grey', markersize=8, color='black', linewidth=1)
        ax2.set_ylabel('Ek difference', color='grey')

    return plt


if __name__ == '__main__':

    molecule = 'h2o'
    basis = '6-31g'

    # Choose lambda array
    lambi = 0.5  # weight for Vexp, initial value
    lambf = 0.5  # lambda final
    lambn = 1  # number of Lambda value
    Larray = np.linspace(lambi, lambf, num=lambn)

    # Build molecules and basis
    ecw = ECW(molecule, basis)

    # Build GS exp data from HF+field
    #ecw.Build_GS_exp('mat', 'HF', field=[0.05, 0.01, 0.])
    # Build exp data from given 1e prop (Ek from CCSD+[0.05, 0.01, 0.]+6-311+g**)
    #ecw.exp_data[0,0] = ['Ek', 70.4 ]

    # Build list of structure factors from CCSD+field
    prop_list = []
    h = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 2, 0], [2, 2, 0]]
    rec_vec = [5., 5., 5.]
    F_info = list(['F', h, rec_vec])
    prop_list.append('Ek')
    prop_list.append('v1e')
    prop_list.append(F_info)
    print('GS propetries: ', prop_list)

    ecw.Build_GS_exp(prop=prop_list, posthf='HF', field=[0.02, 0.01, 0], basis='6-31+g*')
    print('Exp data: ')
    print('length: ', len(ecw.exp_data))
    print(ecw.exp_data)

    # Solve ECW-CCS/CCSD equations using SCF algorithm with given alpha
    Results, plot = ecw.CCSD_GS(Larray, graph=False, alpha=0)
    # Results, plot = ecw.CCS_GS(Larray,graph=False, alpha=0.01)
    #plot.show()





