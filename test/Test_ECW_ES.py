import numpy as np
from context import Main, utilities

from pyscf import scf, cc

molecule = 'h2o'
basis = '6-31++g**'

# Choose lambda
# --------------
# lamb = 0.1
# lamb = np.linspace(0., 1., 15)  # value for trdip
lamb = np.linspace(0., 0.1, 15)

# Build molecules and basis
# ------------------------------
ecw = Main.ECW(molecule, basis)#, out_dir="Documents/Post_these/ECW_results/h2o/ES/")

# Build target GS rdm1
# --------------------------------
myhf = scf.RHF(ecw.mol)
myhf.kernel()
mycc = cc.CCSD(myhf, frozen=0)
mycc.kernel()
gs_rdm1 = mycc.make_rdm1()  # mo
gs_rdm1 = np.einsum('pi,ij,qj->pq', myhf.mo_coeff, gs_rdm1, myhf.mo_coeff.conj())  # ao
gs_rdm1 = utilities.convert_r_to_g_rdm1(gs_rdm1)
gs_rdm1 = utilities.ao_to_mo(gs_rdm1, ecw.mo_coeff)

# Build ES exp data:
# ------------------
# QChem results for transition dipole moment:
# VES1 => ['trdip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
# VES2 => ['trdip', [0.000000, 0.000000, -0.622534]]  # DE = 10.06 eV
# VES3 => ['trdip', [0.000000, -0.09280, 0.00000]]    # DE = 10.81 eV
# CES1 => ['trdip', [0., 0. ,0.030970]]                # DE = 536 eV
# ecw.Build_ES_exp_input([[['trdip', [0.523742, 0, 0]]]])  # valence 1
# ecw.Build_ES_exp_input([[['trdip', [0, 0, 0.622534]]]], val_core=[1, 0], rini_koop_idx=[2])  # valence 2
# ecw.Build_ES_exp_input([[['trdip', [0, 0.030970, 0]]]], val_core=[0, 1], rini_koop_idx=[1])  # core
# ecw.Build_ES_exp_input([[['DEk', 0.28]]])  # with DE
ecw.Build_ES_exp_input([[['trdip', [0.523742, 0, 0]]], [['trdip', [0, 0, 0.622534]]]],
                       val_core=[2, 0], rini_koop_idx=[0, 2])  # 2 val states

# Solve ECW-ES-CCS equations using SCF algorithm
# -----------------------------------------------

ecw.CCS_ES(lamb, diis='all', L_loop=True, maxiter=80, target_rdm1_GS=gs_rdm1, print_ite=False)
ecw.plot_results_ES()
ecw.print_results_ES()

