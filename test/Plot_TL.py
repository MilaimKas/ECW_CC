

# Python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# PySCF
from pyscf import gto, scf, cc

# CCS files
from context import CCS, Solver_GS, exp_pot, Eris

##################
# Build molecule
##################

mol = gto.Mole()

mol.atom = """
H 0. 0. 0.
H 0. 0. 1.
"""

symmetry = True
mol.unit = 'angstrom'
mol.basis = 'sto3g'

mol.build()
mrf = scf.RHF(mol)
mrf.kernel()
mgf = scf.addons.convert_to_ghf(mrf)
#mgf = scf.GHF(mol)
#mgf.kernel()

##########
# ERIS
##########

ccsd = cc.GCCSD(mgf)
geris = Eris.geris(ccsd)

mygccs = CCS.Gccs(geris)

# print eris

# Create rdm1_exp
# GCCS
t0 = np.asarray([[0.01, 0.], [0., 0.]])
l0 = np.asarray([[0.01, 0.], [0., 0.]])
rdm1_exp = mygccs.gamma(t0, l0)

############################
# vector t, l for plotting
############################

# number of 2D grid points
step = 0.0005
max = 0.02
N = round(max/step)
t_grid = np.zeros((N, 2, 2))

# Create grid of t and l values
tli = 0
for i in range(N):
    t_grid[i, 0, 0] = tli
    tli += step
l_grid = t_grid.copy()

# Lambda value
Larray = [0, 1, 1.5, 2.2]

################
# Solve T1/L1
################

conv = 10**-5

# Vexp Object
exp_data = np.full((1, 1), None)
exp_data[0, 0] = ['mat', rdm1_exp]
VX_exp = exp_pot.Exp(exp_data, mol, mgf.mo_coeff)

# Gradient object
mygrad = CCS.ccs_gradient(geris)

# Solver_CCS objects
# returns: text, Ep, X2, T1, L1, t1, l1
Solver = Solver_GS.Solver_CCS(mygccs, VX_exp, 'Ep', conv, CCS_grad=mygrad)

# L1 alpha value
alpha = 0.001

# List results from different Solver
Result_scf = []
Result_scf_L1 = []
Result_scf_rdm = []
Result_scf_t_l = []
Result_grad = []
Result_des = []

# array of ts and ls amplitudes
ts_scf_rdm = np.zeros((2, 2))
ls_scf_rdm = np.zeros((2, 2))
ts_scf_t_l = np.zeros((2, 2))
ls_scf_t_l = np.zeros((2, 2))
ts_scf_L1 = np.zeros((2, 2))
ls_scf_L1 = np.zeros((2, 2))
ts_grad = np.zeros((2, 2))
ls_grad = np.zeros((2, 2))
ts_des = np.zeros((2, 2))
ls_des = np.zeros((2, 2))
ts_scf = np.zeros((2, 2))
ls_scf = np.zeros((2, 2))

# Result_ is a list: [L][result of solver]
# where [result of solver] = [conv text][Ep][X2][conv_ite][rdm1][t1_te][l1_ite]
# except conv text and rdm1 all array contain the result for each iteration

for L in Larray:
  #Result_scf.append(Solver.SCF(L, ts=ts_scf, ls=ls_scf, diis=tuple(), store_ite=True))
  Result_scf.append(Solver.SCF(L, ts=ts_grad, ls=ls_grad, diis=tuple(), store_ite=True))
  Result_scf_L1.append(Solver.SCF(L, ts=ts_scf_L1, ls=ls_scf_L1, diis=tuple(), alpha=alpha, store_ite=True))
  #Result_scf_rdm.append(Solver.SCF(L, ts=ts_scf_rdm, ls=ls_scf_rdm, diis=('rdm1',), store_ite=True))
  Result_scf_rdm.append(Solver.SCF(L, ts=ts_grad, ls=ls_grad, diis=('rdm1',), store_ite=True))
  Result_scf_t_l.append(Solver.SCF(L, ts=ts_scf_t_l, ls=ls_scf_t_l, diis=('t', 'l'), store_ite=True))
  Result_grad.append(Solver.Gradient(L, method='newton', ts=ts_grad, ls=ls_grad, store_ite=True))
  Result_des.append(Solver.Gradient(L, beta=0.1, method='descend', ts=ts_des, ls=ls_des, store_ite=True))

  #ts_scf         = Result_scf[-1][5][-1, :, :]
  #ls_scf         = Result_scf[-1][6][-1, :, :]
  #ts_scf_rdm     = Result_scf_rdm[-1][5][-1, :, :]
  #ls_scf_rdm     = Result_scf_rdm[-1][6][-1, :, :]
  ts_scf_L1       = Result_scf_L1[-1][5][-1, :, :]
  ls_scf_L1       = Result_scf_L1[-1][6][-1, :, :]
  ts_scf_t_l = Result_scf_t_l[-1][5][-1, :, :]
  ls_scf_t_l = Result_scf_t_l[-1][6][-1, :, :]
  ts_grad        = Result_grad[-1][5][-1, :, :]
  ls_grad        = Result_grad[-1][6][-1, :, :]
  ts_des = Result_des[-1][5][-1, :, :]
  ls_des = Result_des[-1][6][-1, :, :]

##################
# Plot T1/L1 map
##################

# Initialize Plot
pl_num = 0
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
step = 0.005  # step for the x,y ticks

for L in Larray:
    T1_vec = np.zeros((N, N))
    L1_vec = np.zeros((N, N))
    cost_vec = np.zeros((N, N))  # cost function
    X2 = np.zeros((N, N))
    j = 0
    for ti in t_grid:
        i = 0
        for li in l_grid:
            rdm1 = mygccs.gamma(ti, li)
            Vexp = np.subtract(rdm1_exp, rdm1)

            # GCCS
            fsp = np.subtract(geris.fock, L*Vexp)
            T1 = mygccs.T1eq(ti, fsp)
            L1 = mygccs.L1eq(ti, li, fsp)
            cost_vec[i, j] = np.sqrt(np.sum(T1**2+np.sum(L1)**2))
            T1_vec[i, j] = T1[0, 0]

            i += 1

        j += 1

    t_vec = t_grid[:, 0, 0]
    l_vec = l_grid[:, 0, 0]

    X, Y = np.meshgrid(t_vec, l_vec)
    i, j = np.unravel_index([pl_num], (2, 2))
    i = int(i)
    j = int(j)
    tit = '$\lambda$ = %1.3f' % L

    # Plot style
    axs[i, j].set_title(tit, fontsize=14)
    axs[i, j].xaxis.set_ticks(np.arange(np.min(t_vec), np.max(t_vec), step))
    axs[i, j].yaxis.set_ticks(np.arange(np.min(l_vec), np.max(l_vec), step))
    axs[i, j].tick_params(axis="x", labelsize=12)
    axs[i, j].tick_params(axis="y", labelsize=12)
    axs[i, j].set_xlim([np.min(t_vec), np.max(t_vec)])
    axs[i, j].set_ylim([np.min(l_vec), np.max(l_vec)])
    axs[i, j].contour(X, Y, cost_vec, 20, colors='grey', linewidths=0.3)
    axs[i, j].pcolormesh(X, Y, cost_vec, cmap='ocean', shading='auto')

    # Plot Result of solvers
    # plot line points of iteration for t[0,0] and l[0,0]

    # SCF
    axs[i, j].plot(Result_scf[pl_num][5][:, 0, 0], Result_scf[pl_num][6][:, 0, 0], marker='o',
                      alpha=0.8, linewidth=0.5, markerfacecolor='orange', color='red')
    #axs[i, j].plot(Result_scf[pl_num][5][-1, 0, 0], Result_scf[pl_num][6][-1, 0, 0],
    #                  marker='x', color='red', markersize=8)

    # Grad descend
    #axs[i, j].plot(Result_des[pl_num][5][:, 0, 0], Result_des[pl_num][6][:, 0, 0], marker='o',
    #                  alpha=0.8, linewidth=0.5, markerfacecolor='grey', color='black')
    #axs[i, j].scatter(Result_scf_t_l[pl_num][5][-1, 0, 0], Result_scf_t_l[pl_num][6][-1, 0, 0], marker='o', color='grey', s=40)

    # newton
    print('t-l')
    print(np.subtract(Result_grad[pl_num][5][:, :, :], Result_grad[pl_num][6][:, :, :]))

    axs[i, j].plot(Result_grad[pl_num][5][::2, 0, 0], Result_grad[pl_num][6][::2, 0, 0], marker='o', markersize=10,
                      alpha=0.8, linewidth=0.5, markerfacecolor='white', color='black')
    #axs[i, j].plot(Result_grad[pl_num][5][-1, 0, 0], Result_grad[pl_num][6][-1, 0, 0],
    #                  marker='x', color='black', markersize=8)

    # DIIS
    axs[i, j].plot(Result_scf_rdm[pl_num][5][:-2:4, 0, 0], Result_scf_rdm[pl_num][6][:-2:4, 0, 0], marker='o', color='green'
                   , linewidth=0.5, markerfacecolor='yellow')
    #axs[i, j].plot(Result_scf_rdm[pl_num][5][-1, 0, 0], Result_scf_rdm[pl_num][6][-1, 0, 0], marker='x', color='yellow'
    #               , linewidth=0.5, markersize=8)

    # L1 reg
    #print(Result_scf_L1[pl_num][5][-1, :, :])
    #axs[i, j].plot(Result_scf_L1[pl_num][5][:, 0, 0], Result_scf_L1[pl_num][6][:, 0, 0], marker='o', color='green'
    #               , linewidth=0.5, markerfacecolor='yellow')
    #axs[i, j].plot(Result_scf_rdm[pl_num][5][-1, 0, 0], Result_scf_rdm[pl_num][6][-1, 0, 0], marker='x', color='yellow'
    #               , linewidth=0.5, markersize=8)

    ## Plot t0 and l0
    axs[i, j].plot(t0[0, 0], l0[0, 0], 'x', color='grey', markersize=12, markeredgewidth=3, markerfacecolor='white')

    # Insets
    if pl_num != 0:
        axins = inset_axes(axs[i, j], width="40%", height="40%", loc=2)
        axins.set_facecolor('black')
        axins.patch.set_alpha(0.5)
        axins.plot(Result_grad[pl_num][3][1:], color='white', linewidth=2)
        axins.plot(Result_scf[pl_num][3][1:], color='orange', linewidth=2)
        axins.plot(Result_scf_rdm[pl_num][3][1:], color='green', linewidth=2)
        axins.set_xticks([])
        axins.set_yticks([])
        if pl_num == 1:
            axins.set_ylim([-0.001, 0.012])
        elif pl_num == 2:
            axins.set_ylim([-0.001, 0.05])
        elif pl_num == 3:
            axins.set_ylim([-0.001, 0.02])


    pl_num += 1

# X,Y,Z labels
fig.text(0.9, 0.95, '$\sqrt{\sum T_1^2+\sum \Lambda_1^2}$', ha='center', fontsize=14)
fig.text(0.5, 0.01, '$t_1[0,0]$', ha='center', fontsize=14)
fig.text(0.01, 0.5, '$\Lambda_1[0,0]$', va='center', rotation='vertical', fontsize=14)

# better layout
fig.tight_layout(pad=1.5, rect=(0.03, 0.03, 0.99, 0.99))

# add colorbar
cbar = fig.colorbar(axs[-1, -1].pcolormesh(X, Y, cost_vec, cmap='ocean', shading='auto'),
                    ax=axs.ravel().tolist(), shrink=0.95)
cbar.ax.tick_params(labelsize=12)

plt.show()
#plt.savefig('/home/milaim/Documents/Post_doc_DESY/XCW_CCS/Code/DEBUG/Test_H2/GCCS_scf_grad_tlini_2.png', format='png')
#plt.savefig('/home/blob.eps', format='eps')
