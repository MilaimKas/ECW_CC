#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
 ECW-CCS MAIN
'''


##############################################
#             import libraries               #
##############################################

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

##############################################
#             Global INPUTS                  #
##############################################

# Read Input file
#from Input import*

##############################################
#             Initialization                 #
##############################################

# Creating new molecule object
# ------------------------------

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

else:
    print('Molecule name: '+molecule+' , not recognize')
    print('List of available molecules:')
    print(mol_list)

symmetry = True
mol.unit = 'angstrom'

# basis set
mol.basis = basis

# HF calculation
# --------------

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
mo_coeff = mf.mo_coeff  # Molecular orbital (MO) coefficients (matrix where rows are atomic orbitals (AO) and columns are MOs)
mo_coeff_inv = np.linalg.inv(mo_coeff)
mo_ene = mf.mo_energy  # MO energies (vector with length equal to number of MOs)
mo_occ = mf.mo_occ  # MO occupancy (vector with length equal to number of MOs)
mocc = mo_coeff[:, mo_occ > 0]  # Only take the mo_coeff of occupied orb
mvir = mo_coeff[:, mo_occ == 0]  # Only take the mo_coeff of virtual orb
nocc = mocc.shape[1]  # Number of occ MOs in HF
nvir = mvir.shape[1]  # Number of virtual MOS in HF

# HF total energy
EHF = mf.e_tot

# dimension
dim = nocc + nvir
aosize = mol.nao_nr()  # number of AO --> size of the basis

# a and b electrons
Nele_a = mol.nelec[0]  # mol.nelec gives the number of alpha and beta ele (nalpha,nbeta)
Nele_b = mol.nelec[1]
Nele = Nele_a + Nele_b

# HF rdm1
rdm1_hf = mf.make_rdm1()

# print cube file
if printcube:
    from pyscf.tools import cubegen
    # convert g to r
    rdm1_hf = utilities.convert_g_to_ru_rdm1(rdm1_hf)[0]
    cubegen.density(mol, 'HF.cube', rdm1_hf, nx=80, ny=80, nz=80)

# One and two particle integrals
# ----------------------------------

eris = Eris.geris(cc.GCCSD(mf))
fock = eris.fock
#fock = np.diag(mo_ene)
# S = mf.get_ovlp() # overlap of the AOs basis
#ccsd = cc.GCCSD(mf)
#eris = ccsd.ao2mo(mo_coeff)

# Initialize ts and ls nocc(row) x nvir(col) matrix
# --------------------------------------------------

if tl1ini == 1:
    eia = mo_ene[:nocc, None] - mo_ene[None, nocc:]
    tsini = fock[:nocc, nocc:] / eia
    lsini = tsini.copy()
# random number
elif tl1ini == 2:
    tsini = np.random.rand(nocc, nvir)*0.01
    lsini = np.random.rand(nocc, nvir)*0.01
# zero
elif tl1ini == 0:
    tsini = np.zeros((nocc, nvir))
    lsini = np.zeros((nocc, nvir))

ts = tsini.copy()
ls = lsini.copy()

# Building gamma_exp, exp matrix and initialize print cube
# -----------------------------------------------------------

gexp = gamma_exp.Gexp(mol,posthf)

if deform_max is not None:
    gexp.deform(deform_max)

if field is not None:
   gexp.Vext(field)

gexp.build()

if para_factor is not None:
    gexp.underfit(para_factor)

exp_data = np.full((1, 1), None)
exp_data[0, 0] = ['mat', gexp.gamma_ao]

mo_coeff_def = gexp.mo_coeff_def

# initialize Vexp object
#VXexp = exp_pot_old.Exp(rdm1_exp, scale=Vscale)
VXexp = exp_pot.Exp(exp_data, mol, mo_coeff, mo_coeff_def=mo_coeff_def)

if printcube:
    fout = dir_cube + 'CCSD'
    utilities.cube(exp_data[0, 0][1], mo_coeff, mol, fout)
    idx = np.round(np.linspace(0, len(Larray) - 1, nbr_cube)).astype(int)
    L_print = Larray[idx]

# Initialize CCS and Solver_CCS class
# ------------------------------------------------------------

# Solve returns a list of 5 elements
# - text with convergence text and number of iteration
# - array of Ep
# - array of X2
# - array of conv criteria (Ep or tl)
# - final rdm1

if model == 'CCS':
    myccs = CCS.Gccs(eris)
    if method == 'newton' or method == 'descend':
        mygrad = CCS.ccs_gradient(eris)
    else:
        mygrad = None
    Solve = Solver_GS.Solver_CCS(myccs, VXexp, conv=conv, conv_thres=conv_thres, tsini=tsini, lsini=lsini, diis=diis, maxdiis=maxdiis,
                             maxiter=maxiter, CCS_grad=mygrad)
if model == 'CCSD':
    myccs = CCSD.GCC(eris)
    if method != 'scf':
        raise NotImplemented('For ECW-CCSD, only the scf solver is implemented')
    Solve = Solver_GS.Solver_CCSD(myccs, VXexp, tsini=tsini, lsini=lsini, conv=conv, conv_thres=conv_thres, diis=diis, maxdiis=maxdiis,
                                  maxiter=maxiter)
    td = None
    ld = None


##############################################
#             MAIN                           #
##############################################

# initialize plot
if graph:
    fig, axs = plt.subplots(2, sharex='col')
X2_lamb = []
Ep_lamb = []
vmax_lamb = []
X2_Ek = []

print()
print("#############")
print("# Results")
print("#############")
print()

# Loop over Lambda
for L in Larray:

    print("LAMBDA= ", L)
    print()
    
    if model == "CCS":
        if method == 'newton':
            Result = Solve.Gradient(L,ts=ts,ls=ls)
        elif method == 'descend':
            Result = Solve.Gradient(L,method=method,ts=ts,ls=ls,beta=beta)
        elif method == 'scf':
            Result = Solve.SCF(L,ts=ts,ls=ls,alpha=alpha)
        elif method == 'L1_grad':
            Result = Solve.L1_grad(L,alpha,beta,ts=ts,ls=ls)
        else:
            raise ValueError('method not recognize')
        ts,ls = Result[5]

    if model == "CCSD":
        Result = Solve.SCF(L, ts=ts, ls=ls, td=td, ld=ld, alpha=alpha)
        ts,ls,td,ld = Result[5]


    # print cube file for L listed in L_print in dir_cube path
    if printcube:
        if L in L_print:
            fout = dir_cube+'L{}'.format(int(L))
            utilities.cube(Result[4], mo_coeff, mol, fout)
    if print_ite_info:
        print('Iteration steps')
        print('ite','      ','Ep','         ',conv,'          ','X2')
        for i in range(len(Result[1])):
           print(i,'  ',"{:.4e}".format(Result[1][i]),'  ',"{:.4e}".format(Result[3][i]),'  ',"{:.4e}".format(Result[2][i][0]))

    # print convergence text
    print()
    print(Result[0])
    print()
    Ep = Result[1][-1]
    X2 = Result[2][-1][0]
    vmax = Result[2][-1][1]

    if graph:
      X2_lamb.append(X2)
      Ep_lamb.append(EHF - Ep)
      vmax_lamb.append(vmax)
      X2_Ek.append(VXexp.X2_Ek_GS)

# Print results
print("INPUT")
print('model= ','ECW-'+model)
print('method= ', method)
print('molecule and basis set: ', molecule, basis)
print("")
print("FINAL RESULTS")
print("Ep   = ", Ep)
print("X2   = ", X2)
print("DEk  = ", X2_Ek[-1])
print()
print("EHF    = ", EHF)
print("Eexp   = ",gexp.Eexp)
print("EHF-EP = ", EHF - Ep)

if graph:
    # Plot Ep, X2 and vmax only for converged lambdas

    # Energy
    axs[0].plot(Larray, Ep_lamb, marker='o', markerfacecolor='black', markersize=8, color='grey', linewidth=1)
    #axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(2, 3),useMathText=True,useLocale=True)
    axs[0].set_ylabel('EHF-Ep',color='black')

    # X2 and vmax
    axs[1].plot(Larray, X2_lamb, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=1)
    ax2 = axs[1].twinx()
    ax2.plot(Larray, vmax_lamb, marker='o', markerfacecolor='blue', markersize=8, color='lightblue', linewidth=1)
    ax2.set_ylabel('Vmax', color='blue')
    axs[1].set_ylabel('X2', color='red')
    axs[1].set_xlabel('Lambda')
    #axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(2, 3),useMathText=True)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(2, 3), useMathText=True)

    # Ek difference
    if Ek:
        ax2 = axs[0].twinx()
        ax2.plot(Larray,X2_Ek,marker='o', markerfacecolor='grey',markersize=8, color='black',linewidth=1)
        ax2.set_ylabel('Ek difference',color='grey')


    plt.show()

