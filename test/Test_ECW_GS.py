import numpy as np
from context import Main

molecule = 'c2h2'
basis = '631+g**'

# Choose lambda array
# ---------------------
lambi = 0.  # weight for Vexp, initial value
lambf = 0.7  # lambda final
lambn = 8  # number of Lambda value
Larray = np.linspace(lambi, lambf, num=lambn)

# Build molecules and basis
# ------------------------------
ecw = Main.ECW(molecule, basis, out_dir="/home/milaim/Documents/ECW_results/c2h2/CCSD/631+g**")
# Build GS exp data from HF/CC+field
# ------------------------------------
# gamma_exp
# ecw.Build_GS_exp(['mat'], 'CCSDt')
ecw.Build_GS_exp(['mat'], 'CCSDt')
# list of prop
# ecw.Build_GS_exp(['v1e', 'Ek'], 'CCSD(T)') #,basis='6-311++g**')

# Directly gave experimental data for the GS
# -------------------------------------------
# ecw.exp_data[0, 0] = [['Ek', 75.], ['dip', [0., 0,02, 0,8]]] # old format
# ecw.exp_data[0] = [['Ek', 75.], ['dip', [0., 0,02, 0,8]]] # new format

# Solve ECW-CCS/CCSD equations using SCF algorithm with given alpha
# ---------------------------------------------------------------------

diis = 'tl'
Results = ecw.CCSD_GS(Larray, diis=diis, nbr_cube_file=7, HF_prop=True, diis_max=15, maxiter=100)
ecw.plot_results()
ecw.print_results()


# Results contains the following info for the last value of L:
# ite = iteration
# [0] = convergence text
# [1] = Ep(it)
# [2] = X2(it) list of tuple: (X2, vmax, X2_Ek)
# [3] = conv(it) 
# [4] = final gamma (rdm1) calc
# [5] = (ts,ls) if CCS then final ts and ls amplitudes. If CCSD then (t1,t2,l1,l2)

