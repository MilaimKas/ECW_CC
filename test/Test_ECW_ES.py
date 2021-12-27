import numpy as np
from context import Main

molecule = 'h2o'
basis = '6-31+g*'

# Choose lambda
# --------------
# lamb = 0.1
lamb = np.linspace(0, 0.4, 5)

# Build molecules and basis
# ------------------------------
ecw = Main.ECW(molecule, basis)

# Build ES exp data:
# ------------------
# QChem results:
# VES1 => ['trdip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
# VES2 => ['trdip', [0.000000, 0.000000, -0.622534]]  # DE = 10.06 eV
# VES3 => ['trdip', [0.000000, -0.09280, 0.00000]]    # DE = 10.81 eV
# CES1 => ['trdip', [0., 0. ,0.030970]]                # DE = 536 eV
ecw.Build_ES_exp_input([[['trdip', [0.523742, 0., 0.]]]])

# Solve ECW-ES-CCS equations using SCF algorithm
# -----------------------------------------------

ecw.CCS_ES(lamb, diis='all', L_loop=True, maxiter=80)
ecw.plot_results_ES()
ecw.print_results_ES()
