import numpy as np
from context import Main

molecule = 'urea'
basis = '6-31+g*'

# Choose lambda array
# ---------------------
lambi = 0.  # weight for Vexp, initial value
lambf = 16.  # lambda final
lambn = 1  # number of Lambda value
Larray = np.linspace(lambi, lambf, num=lambn)

# Build molecules and basis
# ------------------------------
ecw = Main.ECW(molecule, basis)#, out_dir="/Users/milaimkas/Documents/Post_these/ECW_results/urea")

# Build GS exp data from HF/CC+field
# ------------------------------------
# gamma_exp
# ecw.Build_GS_exp(['mat'], 'CCSDt')
ecw.Build_GS_exp(['mat'], 'HF', field=[0.005, 0.001, 0.])
# list of prop
# ecw.Build_GS_exp(['dip', 'Ek'], 'CCSD', basis='6-311+g**')

# Directly gave experimental data for the GS
# -------------------------------------------
# ecw.exp_data[0, 0] = [['Ek', 75.], ['dip', [0., 0,02, 0,8]]] # old format
# ecw.exp_data[0] = [['Ek', 75.], ['dip', [0., 0,02, 0,8]]] # new format

# Solve ECW-CCS/CCSD equations using SCF algorithm with given alpha
# ---------------------------------------------------------------------

diis = 'tl'
Results = ecw.CCS_GS(Larray, diis=diis)
# Results, plot = ecw.CCSD_GS(Larray, graph=True, print_ite_info=False)
# Results = ecw.CCS_GS(Larray, print_ite_info=False, conv_thres=10**-5, diis=diis, maxiter=50)
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

