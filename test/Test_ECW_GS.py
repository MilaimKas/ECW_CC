import numpy as np
from context import Main

molecule = 'h2o'
basis = '6-31+g**'

# Choose lambda array
# ---------------------
lambi = 0.  # weight for Vexp, initial value
lambf = 0.05  # lambda final
lambn = 5  # number of Lambda value
Larray = np.linspace(lambi, lambf, num=lambn)

# Build molecules and basis
# ------------------------------
ecw = Main.ECW(molecule, basis)

# Build GS exp data from HF/CC+field
# ------------------------------------
# ecw.Build_GS_exp('mat', 'CCSD', field=[0.05, 0.01, 0.])  # gamma_exp
ecw.Build_GS_exp(['Ek'], 'CCSD', field=[0.005, 0.001, 0.], basis='6-311+g**')  # list of prop

# Directly gave experimental data for the GS
# -------------------------------------------
# ecw.exp_data[0, 0] = [['Ek', 75.], ['dip', [0., 0,02, 0,8]]]

# Solve ECW-CCS/CCSD equations using SCF algorithm with given alpha
# ---------------------------------------------------------------------

# Results, plot = ecw.CCSD_GS(Larray, graph=True)
# Results, plot = ecw.CCSD_GS(Larray, graph=True, print_ite_info=False)
Results, plot = ecw.CCS_GS(Larray, graph=True, print_ite_info=False, conv_thres=10**-5, diis='rdm1', maxiter=50)
plot.show()

#print(Results[2])
# Results contains the following info for the last value of L:
# ite = iteration
# [0] = convergence text
# [1] = Ep(it)
# [2] = X2(it) list of tuple: (X2, vmax, X2_Ek)
# [3] = conv(it) 
# [4] = final gamma (rdm1) calc
# [5] = (ts,ls) if CCS then final ts and ls amplitudes. If CCSD then (t1,t2,l1,l2)

