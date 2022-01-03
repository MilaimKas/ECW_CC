import numpy as np
from context import Main

molecule = 'allene'
basis = '6-31+g**'

# Choose lambda array
# ---------------------
lambi = 0.  # weight for Vexp, initial value
lambf = 10  # lambda final
lambn = 11  # number of Lambda value
Larray = np.linspace(lambi, lambf, num=lambn)

# Build molecules and basis
# ------------------------------
ecw = Main.ECW(molecule, basis)#, out_dir="/Users/milaimkas/Documents/Post_these/ECW_results/allene/CCS/mat")

# Build GS exp data from HF/CC+field
# ------------------------------------
# gamma_exp
ecw.Build_GS_exp(['mat'], 'CCSD')
# list of prop
# ecw.Build_GS_exp(['dip', 'Ek'], 'CCSD(T)', basis='6-311+g**', field=[0.001, 0., 0.])
# print(ecw.exp_data)

# Directly gave experimental data for the GS
# -------------------------------------------
# ecw.exp_data[0] = [['Ek', 76.26263875454327], ['dip', [0, 0, -8.42688035e-01]]] # H2O ()CCSD(T)/6311+g**

# Solve ECW-CCS/CCSD equations using SCF algorithm with given alpha
# ---------------------------------------------------------------------

diis = 'tl'
Results = ecw.CCS_GS(Larray, diis=diis, nbr_cube_file=4, maxiter=80)
# Results, plot = ecw.CCSD_GS(Larray, graph=True, print_ite_info=False)
ecw.plot_results()

# print(Results[2])
# Results contains the following info for the last value of L:
# ite = iteration
# [0] = convergence text
# [1] = Ep(it)
# [2] = X2(it) list of tuple: (X2, vmax, X2_Ek)
# [3] = conv(it) 
# [4] = final gamma (rdm1) calc
# [5] = (ts,ls) if CCS then final ts and ls amplitudes. If CCSD then (t1,t2,l1,l2)

