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
# from . import CCS, CCSD, exp_pot, gamma_exp, utilities, Eris, Solver_GS, Solver_ES
import CCS, CCSD, exp_pot, gamma_exp, utilities, Eris, Solver_GS, Solver_ES, exp_pot_new, Solver_ES_new

# Global float format for print
# ------------------------------
format_float = '{:10.5e}'


# Creating new molecule object
# ------------------------------

class ECW:
    def __init__(self, molecule, basis, int_thresh=1e-13, out_dir=None, U_format=False, spin=0):
        """
        Build the PySCF mol object and performs HF calculation

        :param molecule: string with name of molecule to be used
        :param basis: string with basis set to be used
        :param int_thresh: threshold for 2 electron integrals
        :param out_dir: path to the directory where the output files (cube files, results, etc)
                        are saved (string), if None do not print output files
        :param U_format: if True, the spin-orbital basis are converted from a UHF calculation,
                         if False, the spin-orbital are converted from a RHF calc (a and b SO are degenerate)
        """

        # CC class
        self.myccs = None
        self.myccsd = None

        mol = gto.Mole()

        # Geometry

        mol_list = ['h2', 'c2h4', 'h2o2', 'h2o', 'allene', 'urea']

        if molecule == 'h2':
            mol.atom = '''
            H 0 0 0
            H 0 0 0.74
            '''

        elif molecule == 'c2h4':
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
            mol.atom = """
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
            mol.atom = """
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

        self.molecule = molecule

        mol.unit = 'angstrom'

        # basis set
        mol.basis = basis

        # default method
        self.method = 'scf'
        self.diis = ''

        # HF calculation
        # -------------------

        mol.verbose = 0  # no output
        mol.charge = 0  # charge
        mol.spin = spin  # spin

        mol.build()  # build mol object

        if U_format:
            raise NotImplementedError('Using UHF reference implies different orbspin')
            # mf = scf.UHF(mol)
            # todo: make orbspin [0 1 0 1 ...]
        else:
            mf = scf.RHF(mol)

        # option for calculation
        mf.conv_tol = 1e-09  # energy tolerance
        mf.conv_tol_grad = np.sqrt(mf.conv_tol)  # gradient tolerance
        mf.direct_scf_tol = int_thresh  # tolerance in discarding integrals
        mf.max_cycle = 100
        mf.max_memory = 1000

        # do scf calculation
        mf.kernel()

        # convert in GHF format
        mf = scf.addons.convert_to_ghf(mf)
        self.mf = mf

        # variables related to the MOs basis
        self.mo_coeff = mf.mo_coeff  # matrix where rows are atomic orbitals (AO) and columns are MOs
        self.mo_occ = mf.mo_occ  # MO occupancy (vector with length equal to number of MOs)
        mocc = self.mo_coeff[:, self.mo_occ > 0]  # Only take the mo_coeff of occupied orb
        mvir = self.mo_coeff[:, self.mo_occ == 0]  # Only take the mo_coeff of virtual orb
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
        # self.mf = mf
        self.mol = mol

        # print cube file for L=0 (HF density)
        # --------------------------------------
        self.out_dir = out_dir
        if out_dir is not None:
            from pyscf.tools import cubegen
            # convert g to r
            rdm1_hf = utilities.convert_g_to_ru_rdm1(self.rdm1_hf)[0]
            cubegen.density(mol, out_dir + '/HF.cube', rdm1_hf, nx=80, ny=80, nz=80)

        # One and two particle integrals
        # ----------------------------------
        self.eris = Eris.geris(cc.GCCSD(mf))
        self.fock = self.eris.fock
        # fock = np.diag(mo_ene)
        # S = mf.get_ovlp() # overlap of the AOs basis
        # ccsd = cc.GCCSD(mf)
        # eris = ccsd.ao2mo(mo_coeff)

        # initialize exp_data
        # --------------------------
        # self.exp_data = np.full((1, 1), None)  # old format
        self.exp_data = []
        self.exp_data.append([])  # add GS
        # r and l vectors for the ES
        self.r_ini = None
        self.l_ini = None
        self.r0_ini = None
        self.l0_ini = None
        self.Ek_exp_GS = None # GS exp kinetic energy

        # Target energies
        # --------------------
        self.Eexp_GS = None
        self.Eexp_ES = []  # excitation energies

        # Store list of Miller indices and unit=cell size
        # ------------------------------------------------
        self.h = None
        self.rec_vec = None

        # initialize list of results as a function of L
        # -------------------------------------------------
        self.Larray = []
        self.Delta_lamb = []  # Relative prop., sum of prop. or rdm1 difference |A_calc-A_exp|/|A_exp|
        self.Ep_lamb = []  # EHF-Ep
        self.vmax_lamb = []  # vmax
        self.Delta_Ek = []  # Relative Ek difference |Ek_calc-Ek_exp|/|Ek_exp|

        print('*** Molecule build ***')

    def Build_GS_exp(self, prop, posthf='HF', field=None, para_factor=None, max_def=None, basis=None):
        """
        Build "experimental" or "target" data for the GS

        :param basis: basis for the calculation of exp. prop.
        :param prop: list of properties to include in exp_data
                     - 'mat': directly use calculated rdm1 as target (gamma_exp_GS or rdm1_exp_GS)
                     - 'Ek', 'v1e', 'dip'
                     - ['F', h, (a, b, c)] for structure factor calculation where h=[h1,h2, ...] and hi=(hx,hy,hz)
                                           and a,bc are the lattice vector length. If not given a=b=c=10
        :param posthf: method to calculate gamma_exp_GS
        :param field: external field ta calculate gamma_exp_GS
        :param para_factor: under-fitting coefficient
        :param max_def: maximum bond length deformation in au
        :param basis: basis used for the calculation of properties
        :return: update exp_data matrix
        """

        # if 'mat' is given, basis must = self.basis
        if basis is not None:
            if 'mat' in prop and self.mol.basis != basis:
                # todo: implement projector between different basis
                print('WARNING: If rdm1 are to be compared, target and calculated rdm1 must be in the same basis.'
                      'the {} basis will be used to calculate the target rdm1'.format(self.mol.basis))
                basis = None
        # if 'mat' is given, the geomerty must be the same (otherwise the basis is not the same)
        if 'mat' in prop and max_def is not None:
            print('WARNING: If rdm1 are to be compared, the geometry for exp anc calc must be the same')
            max_def = None

        # Build gamma_exp for the GS
        # ---------------------------

        # create gamma_exp object
        gexp = gamma_exp.Gexp(self.mol, posthf, basis=basis)

        # add geometrical deformation
        if max_def is not None:
            gexp.deform(max_def)
        # add external field
        if field is not None:
            if not isinstance(field, list):
                raise SyntaxError('External field must be a list [vx, vy, vz]')
            gexp.Vext(field)

        gexp.build()

        # add zero elements in gamma_exp to simulate under-fitting
        if para_factor is not None:
            gexp.underfit(para_factor)

        # Store GS exp energy
        self.Eexp_GS = gexp.Eexp

        if isinstance(prop, str):
            prop = list([prop])

        # loop over list of properties to calculate (F, Ek, dip, v1e)
        # self.exp_data[0, 0] = [] # old format
        for p in prop:

            # directly compare rdm1
            if p == 'mat':
                gamma_mo = utilities.ao_to_mo(gexp.gamma_ao, self.mo_coeff)
                self.Ek_exp_GS = utilities.Ekin(gexp.mol_def, gexp.gamma_ao)
                # self.exp_data[0, 0] = ['mat', gamma_mo]  # old format
                self.exp_data[0].append(['mat', gamma_mo])

            # Structure Factor p=['F', h, (a,b,c)]
            if isinstance(p, (list, np.ndarray)):
                raise NotImplementedError
                # if len(p) != 3:
                #    raise SyntaxError('If structure factors are to be calculated, '
                #                     'the correct syntax is ["F", h, (a,b,c)] where h are '
                #                     'the Miller indices h = [h1, h2, ...] and a,b,c are the reciprocal vector length')
                # else:
                #   # todo: make eris and CC object complex
                #    self.eris = Eris.geris(cc.GCCSD(self.mf), )
                #    rec_vec = np.asarray(p[2])
                #    h = p[1]
                #    # calculate list of structure factors for each given set of Miller indices
                #    F = utilities.structure_factor(gexp.mol_def, h, gexp.gamma_ao, gexp.mo_coeff_def,
                #                                   aobasis=True, rec_vec=rec_vec)
                #    # self.exp_data[0, 0].append(['F', F])  # old format
                #    self.exp_data[0].append(['F', F, h, rec_vec])

            # Kinetic energy
            if p == 'Ek':
                ek = utilities.Ekin(gexp.mol_def, gexp.gamma_ao)
                #  self.exp_data[0, 0].append(['Ek', ek])  # old format
                self.exp_data[0].append(['Ek', ek])

            # one-electron potential
            if p == 'v1e':
                v1e = utilities.v1e(gexp.mol_def, gexp.gamma_ao)
                #  self.exp_data[0, 0].append(['v1e', v1e])  # olf format
                self.exp_data[0].append(['v1e', v1e])

            # dipole moment
            if p == 'dip':
                dip = utilities.dipole(gexp.mol_def, gexp.gamma_ao)
                #  self.exp_data[0, 0].append(['dip', dip])  # olf format
                self.exp_data[0].append(['dip', dip])

        # store cube file with the target density in "out_dir"
        if self.out_dir:
            from pyscf.tools import cubegen
            fout = self.out_dir + '/target_GS.cube'
            cubegen.density(self.mol, fout, gexp.gamma_ao)

        print('*** GS data stored ***')

    def Build_ES_exp_MOM(self, nbr_of_es=(1, 0), field=None):
        """
        Build excited states data from MOM calculation

        :param nbr_of_es: tuple with number of valence and core excited states from which rdm1 and tr_rdm1 have
        to be calculated
        :param field: additional external electric field to be added
        :return: updated exp_data matrix and list of initial r1 vectors
        """

        print(" Functions not yet tested")

        es_exp = gamma_exp.ESexp(self.mol, Vext=field, nbr_of_states=nbr_of_es)
        es_exp.MOM()

        # store GS Eexp
        if self.Eexp_GS is not None:
            self.Eexp_GS = es_exp.Eexp_GS
            raise Warning('A Energy for the target (experimental) GS is already given, this will overwrite it')

        # store target excitation energies
        self.Eexp_ES.append(es_exp.DE_exp)

        # expand exp_data with tr_rdm1
        for tr_rdm1, rini in zip(es_exp.gamma_tr_ao, es_exp.ini_r):

            # store tr_rdm1 in MO basis
            tr_rdm1 = utilities.ao_to_mo(tr_rdm1, self.mo_coeff)
            self.exp_data.append(['trmat', [tr_rdm1, tr_rdm1]])  # left and right tr_rdm1 are the same -> verify

            # store r1 initial guess
            self.r_ini.append(rini)
            # store initial Koopman excitation energies
            # i, a = np.argwhere(rini)
            # self.DE.append(self.fock[a, a] - self.fock[i, i])

        del es_exp

    def Build_ES_exp_input(self, es_prop, rini_list=None, val_core=(1, 0)):
        """
        Store excited states data from given properties

        :param es_prop: list with either transition dipole moment values np.array(x,y,z)
                         or kinetic energy difference for the target states
                         len(exp_prop) = nbr of ES
                         ex: exp_prop for 2 ES = [[['dip', (x,y,z)],['DEk', value]],[['dip', (x,y,z)]]]
        :param rini_list: initial i->a one-electron excitation for each target states
               -> if rini are not given, they are taken from valence Koopman's initial guess
        :param val_core: tuple with number of valence and core excited states
        :return: updated exp_data matrix
        """

        # Update exp_data with given ES prop
        for es in es_prop:
            self.exp_data.append(es)

        # CCS class
        if self.myccs is None:
            self.myccs = CCS.Gccs(self.eris)

        # Koopman initial guess
        if rini_list is None:
            r1, de = utilities.koopman_init_guess(np.diag(self.fock), self.mo_occ, val_core)
            r0ini = [self.myccs.Extract_r0(r, np.zeros_like(r), self.fock, np.zeros_like(self.fock)) for r in r1]
            self.r_ini = r1
            self.l_ini = r1.copy()
            self.r0_ini = r0ini
            self.l0_ini = r0ini.copy()
        else:
            if len(rini_list) != len(es_prop):
                raise ValueError('The number of given initial r vectors is not '
                                 'equal to consistent with the given experimental data for ES')

    def CCS_GS(self, Larray, alpha=None, method='scf', diis='',
               nbr_cube_file=2, tl1ini=0, print_ite_info=False, beta=None, diis_max=15, conv='tl',
               conv_thres=10 ** -6, maxiter=40, tablefmt='rst'):
        """
        Call CCS solver for the ground state using the SCF+DIIS+L1 or the gradient (steepest descend/Newton) method

        :param Larray: array of L value for which the CCS equations have to be solved
        :param alpha: L1 reg term applied at micro-iteration
        :param method: SCF, newton, gradient or L1_grad
        :param beta: iteration step for the L1_grad method
        :param diis: where to apply diis, string 'tl' or 'rdm1'
        :param diis_max: max diis space
        :param nbr_cube_file: number of cube file to be printed for equally spaced L values
        :param tl1ini: initial value for t1 and l1 amplitudes (0=zero, 1=perturbation theory, 2=random)
        :param print_ite_info: True if iteration step are to be printed
        :param conv: convergence variable ('l', 'tl' or 'Ep')
        :param conv_thres: threshold for convergence
        :param maxiter: max number of iterations
        :param tablefmt: tabulate format for the printed table ('rst' or 'latex' for example)
        :return: Converged results
                 [0] = convergence text
                 [1] = Ep(it)
                 [2] = Delta(it) list of tuple: (Delta, vmax, Delta_Ek)
                 [3] = conv(it)
                 [4] = final gamma (rdm1) calc
                 [5] = final ts and ls amplitudes
        """

        self.diis = diis + ' diis_max={}'.format(diis_max)

        self.Larray = Larray

        if method == 'L1_grad' and beta is None:
            raise ValueError('A value for beta (gradient step) must be given for the L1_grad method')

        if len(self.exp_data) > 1:
            self.exp_data = self.exp_data[0]  # new format
            raise Warning('Data for excited states have been found but a ground state solver is used, '
                          'the Vexp potential will only contain GS data')
        self.method = method

        # Vexp class
        #  VXexp = exp_pot.Exp(self.exp_data, self.mol, self.mo_coeff, rec_vec=self.rec_vec, h=self.h)  # old format
        VXexp = exp_pot_new.Exp(Larray[0], self.exp_data, self.mol, self.mo_coeff, Ek_exp_GS=self.Ek_exp_GS)

        # initial values for ts and ls
        # CCSD initial values
        if tl1ini == 1:
            mo_ene = np.diag(self.fock)
            eia = mo_ene[:self.nocc, None] - mo_ene[None, self.nocc:]
            tsini = self.fock[:self.nocc, self.nocc:] / eia
            lsini = tsini.copy()
        # random number: only for debugging purpose
        elif tl1ini == 2:
            tsini = np.random.rand(self.nocc // 2, self.nvir // 2) * 0.01
            lsini = np.random.rand(self.nocc // 2, self.nvir // 2) * 0.01
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

        # CCS class
        if self.myccs is None:
            self.myccs = CCS.Gccs(self.eris)
        if method == 'newton' or method == 'descend':
            mygrad = CCS.ccs_gradient(self.eris)
        else:
            mygrad = None

        # CCS_GS solver class
        Solve = Solver_GS.Solver_CCS(self.myccs, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini,
                                     diis=diis, maxdiis=diis_max, maxiter=maxiter, CCS_grad=mygrad)

        # printed final results
        Result = None
        Ep = None
        Delta = None

        print()
        print("#######################################################")
        print("#  Results using " + method + " for CCS-GS calculation ")
        print("#######################################################")
        print()

        # ------------------ Loop over Lambda ---------------------------

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
                    fout = self.out_dir + '/L{}'.format(int(L)) + '.cube'
                    utilities.cube(Result[4], self.mo_coeff, self.mol, fout)

            if print_ite_info:
                print('Iteration steps')
                headers = ['ite', 'Ep', str(conv), 'Delta']
                table = []
                for i in range(len(Result[1])):
                    table.append([i, '{:.4e}'.format(Result[1][i]), "{:.4e}".format(Result[3][i]),
                                  "{:.4e}".format(Result[2][i][0])])
                print(tabulate(table, headers, tablefmt=tablefmt))

            # print convergence text
            print(Result[0])
            print()
            Ep = Result[1][-1]
            Delta = Result[2][-1][0]
            print('Delta = ', Delta)
            vmax = Result[2][-1][1]

            # store list for graph and output files
            self.Delta_lamb.append(100. * Delta)
            self.Ep_lamb.append(self.EHF - Ep)
            self.vmax_lamb.append(vmax)
            if VXexp.Delta_Ek_GS is not None:
                self.Delta_Ek.append(100. * VXexp.Delta_Ek_GS)

        print("FINAL RESULTS")
        print("Ep   = " + format_float.format(Ep + self.EHF))
        print("Delta   = " + format_float.format(Delta))
        if VXexp.Delta_Ek_GS is not None:
            print("Delta Ek  = " + format_float.format(VXexp.Delta_Ek_GS))
        print()
        print("EHF    = " + format_float.format(self.EHF))
        print("Eexp   = " + format_float.format(self.Eexp_GS))

        if self.out_dir:
            self.print_results()

        return Result

    def CCSD_GS(self, Larray, alpha=None, diis='', nbr_cube_file=2, tl1ini=0,
                print_ite_info=False, diis_max=15, conv='tl', conv_thres=10 ** -6, maxiter=40, tablefmt='rst'):
        """
        Call CCSD solver for the ground state using SCF+DIIS method

        :param Larray: array of L value for which the CCS equations have to be solved
        :param alpha: L1 reg term applied at each micro-iteration
        :param diis: apply diis to rdm1 diis='rdm1' or t, l amplitudes diis='tl'
        :param diis_max: max diis space
        :param nbr_cube_file: number of cube file to be printed for equally spaced L values
        :param tl1ini: initial value for t1 and l1 amplitudes (0=zero, 1=perturbation theory, 2=random)
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
        """

        self.Larray = Larray
        self.diis = diis + ' diis_max={}'.format(diis_max)

        if len(self.exp_data) > 1:
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
            tsini = np.random.rand(self.nocc // 2, self.nvir // 2) * 0.01
            lsini = np.random.rand(self.nocc // 2, self.nvir // 2) * 0.01
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
        # VXexp = exp_pot.Exp(self.exp_data, self.mol, self.mo_coeff, rec_vec=self.rec_vec, h=self.h)
        VXexp = exp_pot_new.Exp(Larray[0], self.exp_data, self.mol, self.mo_coeff, Ek_exp_GS=self.Ek_exp_GS)

        # CCSD class
        if self.myccsd is None:
            self.myccd = CCSD.GCC(self.eris)

        # CCS_GS solver
        Solve = Solver_GS.Solver_CCSD(self.myccd, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini,
                                      diis=diis, maxdiis=diis_max, maxiter=maxiter)

        # initialize double amp
        td = None
        ld = None
        Result = None
        Ep = None
        Delta = None

        print()
        print("##############################################")
        print("#  Results using SCF for CCSD- GS calculation ")
        print("##############################################")
        print()

        # Loop over Lambda
        for L in Larray:

            print("LAMBDA= ", L)

            Result = Solve.SCF(L, ts=ts, ls=ls, td=td, ld=ld, alpha=alpha)

            # Use previous amplitudes as initial guess
            ts, ls, td, ld = Result[5]
            # print(VXexp.Ek_exp_GS, VXexp.Ek_calc_GS)

            # print cube file for L listed in L_print in out_dir path
            if self.out_dir:
                if L in L_print:
                    fout = self.out_dir + '/L{}'.format(int(L)) + '.cube'
                    utilities.cube(Result[4], self.mo_coeff, self.mol, fout)

            if print_ite_info:
                print('Iteration steps')
                headers = ['ite', 'Ep', str(conv), 'X2']
                table = []
                for i in range(len(Result[1])):
                    table.append([i, '{:.4e}'.format(Result[1][i]), "{:.4e}".format(Result[3][i]),
                                  "{:.4e}".format(Result[2][i][0])])
                print(tabulate(table, headers, tablefmt=tablefmt))

            # print convergence text
            print(Result[0])
            print()
            Ep = Result[1][-1]
            Delta = Result[2][-1][0]
            print('Delta = ', Delta)
            vmax = Result[2][-1][1]

            # store array for graph or output file
            self.Delta_lamb.append(100. * Delta)
            self.Ep_lamb.append(self.EHF - Ep)
            self.vmax_lamb.append(vmax)
            if VXexp.Delta_Ek_GS is not None:
                self.Delta_Ek.append(100. * VXexp.Delta_Ek_GS)

        print()
        print("FINAL RESULTS")
        print("Ep   = " + format_float.format(Ep + self.EHF))
        print("X2   = " + format_float.format(Delta))
        if VXexp.Delta_Ek_GS is not None:
            print("DEk  = " + format_float.format(VXexp.Delta_Ek_GS))
        print()
        print("EHF    = " + format_float.format(self.EHF))
        print("Eexp   = " + format_float.format(self.Eexp_GS))

        if self.out_dir:
            self.print_results()

        return Result

    def CCS_ES(self, L, method='scf', conv='rl', exp_data=None, conv_thres=10 ** -5, maxiter=40, diis=''):
        # todo: make loop over L values
        """
        Calls the excited state solver

        :param method: scf or diagonalization method
        :param conv: convergence criteria applies to 'tl' (ES amplitudes), 'rl' (GS amplitudes) or 'Ep'
        :param diis: use diis solver on top of scf or diag
        :param maxiter: max number of iteration
        :param conv_thres: convergence threshold applied to the convegrence criteria 'conv'
        :param L: array of L values (weight of exp data) for each state and each property
        :param exp_data: list containing the experimental data for GS and ES ([[GS prop],[ES1 prop], ...])
        :return:
        """

        if exp_data is None:
            exp_data = self.exp_data
            if len(exp_data) == 1:
                raise Warning('No data for excited state')
        if exp_data is None:
            raise ValueError('exp_data list must be provided')
        nbr_ES = len(exp_data) - 1

        # initial value for r1 and r0
        if self.r_ini is None:
            print("Initial amplitudes will be taken from Koopman's guess")

        # CCS class
        if self.myccs is None:
            self.myccs = CCS.Gccs(self.eris)

        # Vexp class
        Vexp = exp_pot_new.Exp(L, exp_data, self.mol, self.mo_coeff, Ek_exp_GS=self.Ek_exp_GS)

        # check L format
        L = Vexp.L_check(L)

        # Solver class
        Solver = Solver_ES_new.Solver_ES(self.myccs, Vexp, conv_var=conv,
                                         conv_thres=conv_thres, maxiter=maxiter, diis=diis)

        print()
        print("########################################")
        print("#  Results using SCF for ES calculation ")
        print("########################################")
        print()

        if method == "scf":
            Solver.SCF(L)
        elif method == "diag":
            Solver.SCF_diag(L)
        else:
            raise SyntaxError("method not recognize. Should be a string: 'scf' or 'diag'")

    ##################################
    # Print results and infos in file
    ##################################

    def print_results(self):
        """
        create an output file in out_dir directory with following infos:
        - name of molecule
        - method used (basis, SCF, diis, etc)
        - type of experimental data
        - convergence criteria
        - Delta, vmax, EHF-Ep and Ek as a function of Lambda
        """

        header = 'molecule: {} /n method: {} /n basis: {} /n target data: {} /n'.format(
            self.molecule, self.method, self.mol.basis, self.exp_data)

        if self.Delta_Ek:
            data = np.column_stack([self.Larray, self.Delta_lamb, self.Ep_lamb, self.vmax_lamb, self.Delta_Ek])
            header += "L    Delta   EHF-Ep  vmax    Delta_Ek"
        else:
            data = np.column_stack([self.Larray, self.Delta_lamb, self.Ep_lamb, self.vmax_lamb])
            header += "L    Delta   EHF-Ep  vmax"

        np.savetxt(self.out_dir + '/output.txt', data, fmt="%10.5f", header=header)

    def plot_results(self):
        # todo: deal with ES case and neested Larray
        """
        Plot Ep, Delta, vmax and DEk as a function of L
        """

        from matplotlib import rc
        rc('text', usetex=True)

        fig, axs = plt.subplots(2, sharex='col')
        # Plot Ep, X2 and vmax only for converged lambdas

        # Energy
        axs[0].plot(self.Larray, self.Ep_lamb, marker='o', markerfacecolor='black', markersize=8,
                    color='grey', linewidth=1)
        # axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
        axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True, useLocale=True)
        axs[0].set_ylabel('E$_{HF}$-E$_p$', color='black')

        # X2 and vmax
        axs[1].plot(self.Larray, self.Delta_lamb, marker='o', markerfacecolor='red', markersize=4,
                    color='orange', linewidth=1)
        ax2 = axs[1].twinx()
        ax2.plot(self.Larray, self.vmax_lamb, marker='o', markerfacecolor='blue', markersize=8, color='lightblue',
                 linewidth=1)
        ax2.set_ylabel('V$_{max}$', color='blue')
        axs[1].set_ylabel(r'$\Delta$ (\%)', color='red')
        axs[1].set_xlabel(r'$\lambda$')
        # axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
        axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)

        # Ek difference
        if self.Delta_Ek:
            ax2 = axs[0].twinx()
            ax2.plot(self.Larray, self.Delta_Ek, marker='o', markerfacecolor='grey', markersize=8,
                     color='black', linewidth=1)
            ax2.set_ylabel(r'$\Delta$ Ek (\%)', color='grey')

        plt.show()


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
    # ecw.Build_GS_exp('mat', 'HF', field=[0.05, 0.01, 0.])
    # Build exp data from given 1e prop (Ek from CCSD+[0.05, 0.01, 0.]+6-311+g**)
    # ecw.exp_data[0,0] = ['Ek', 70.4 ]

    # Build list of structure factors, Ek and v1e from CCSD+field
    prop_list = []
    # h = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 2, 0], [2, 2, 0]]
    # rec_vec = [5., 5., 5.]
    # F_info = list(['F', h, rec_vec])
    # prop_list.append(F_info)
    prop_list.append('Ek')
    prop_list.append('v1e')

    print()
    print('GS propetries: ', prop_list)

    ecw.Build_GS_exp(prop=prop_list, posthf='HF', field=[0.02, 0.01, 0], basis='6-31+g*')
    print('Exp data: ')
    print(ecw.exp_data)

    # Solve ECW-CCS/CCS equations using SCF algorithm with given alpha and L value
    Results = ecw.CCS_GS(Larray, alpha=0)
    # ecw.plot_results()

    # Add excited state experimental data from 2 ES (taken from QChem, see gamma_exp.py)
    dip_z = (0.523742 + 0.550251)/2.
    DE = 7.6051*0.03675
    es_prop = [[['trdip', (0., 0., dip_z)]], [['DEk', DE]]]
    ecw.Build_ES_exp_input(es_prop)
    print()
    print('Exp data: ')
    print(ecw.exp_data)

    # Solve the ES equations using scf
    L = 0.15  # same weigth for all states and all prop.
    ecw.CCS_ES(L, method='scf', diis='')
