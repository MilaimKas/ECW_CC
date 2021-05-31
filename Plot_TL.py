

# Python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# PySCF
from pyscf import gto, scf, cc

# CCS files
import CCS
import Solver_GS
import exp_pot

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
mgf = scf.GHF(mol)
mrf.kernel()
mgf.kernel()

rfock = np.diag(mrf.mo_energy)
gfock = np.diag(mgf.mo_energy)

##########
# ERIS
##########

#myCC = cc.RCCSD(mrf)
#myCC.run()
#print(myCC.make_rdm1())

reris = cc.RCCSD(mrf).ao2mo(mrf.mo_coeff)
geris = cc.GCCSD(mgf).ao2mo(mgf.mo_coeff)

myccs = CCS.Rccs(reris)
mygccs = CCS.Gccs(geris)

# print eris

# Create rdm1_exp
# GCCS
t0=np.asarray([[0.01,0.],[0.,0.]])
l0=np.asarray([[0.01,0.],[0.,0.]])
rdm1_exp = mygccs.gamma(t0,l0)
# RCCS
#t0 = np.asarray([[0.01]])
#l0 = np.asarray([[0.01]])
#rdm1_exp = myccs.gamma(t0,l0)

print(rdm1_exp)

############################
# vector t, l for plotting
############################

N = 20

step = 0.0005
max = 0.02
N = round(max/step)#*2
t = np.zeros((N,2,2)) # GCCS
#t = [] # RCCS
l = t.copy()
tli = 0 #-max
for i in range(N):
  t[i,0,0] = tli # GCCS
  #t.append(np.asarray([[tli]])) # RCCS
  tli += step

l = t.copy()

# Lambda value
Larray = [0,1,3,5]

################
# Solve T1/L1
################
conv = 10**-6
# VX Object
exp_data = np.full((1,1),None)
exp_data[0,0] = ['mat', rdm1_exp]
VX_exp = exp_pot.Exp(exp_data)

# Gradient object
mygrad = CCS.ccs_gradient(geris)

# Solver_CCS objects
# returns: text, Ep, X2, T1, L1, t1, l1
Solver = Solver_GS.Solver_CCS(mygccs, VX_exp,'tl', conv, CCS_grad=mygrad)

# Solve for L in Larray
Result_scf = []
Result_scf_rdm = []
Result_scf_t = []
Result_scf_rdm_t = []
Result_scf_rdm_t_l = []
Result_grad = []
Result_des = []

# initial ts, ls
ts_scf_rdm=np.zeros((2,2))
ls_scf_rdm=np.zeros((2,2))
ts_scf_t=np.zeros((2,2))
ls_scf_t=np.zeros((2,2))
ts_scf_rdm_t=np.zeros((2,2))
ls_scf_rdm_t=np.zeros((2,2))
ts_scf_rdm_t_l=np.zeros((2,2))
ls_scf_rdm_t_l=np.zeros((2,2))
ts_grad= np.zeros((2,2))
ls_grad= np.zeros((2,2))
#ts_grad[0] = 0.015
#ls_grad[0] = 0.015
ts_des=ts_grad.copy() #np.zeros((2,2))
ls_des=ls_grad.copy() #np.zeros((2,2))
ts_scf= ts_grad.copy() #np.zeros((2,2))
ls_scf= ts_grad.copy() #np.zeros((2,2))

# Result_ is a list: [L][result of solver]
# where [result of solver] = [conv text][Ep][X2][t1][l1][T1][L1]
# except [conv text] all array contains the result for each iteration

for L in Larray:
  ts_des = np.zeros((2,2))
  ls_des = np.zeros((2,2))
  ts_scf = np.zeros((2,2))
  ls_scf = np.zeros((2,2))
  Result_scf.append(Solver.SCF(L,ts=ts_scf,ls=ls_scf,diis=tuple()))
  #Result_scf_rdm.append(Solver_CCS.SCF(L, ts=ts_scf_rdm, ls=ls_scf_rdm, diis=('rdm1',)))
  #Result_scf_t.append(Solver_CCS.SCF(L, ts=ts_scf_t, ls=ls_scf_t, diis=('t',)))
  #Result_scf_rdm_t.append(Solver_CCS.SCF(L, ts=ts_scf_rdm_t, ls=ls_scf_rdm_t, diis=('rdm','t')))
  #Result_scf_rdm_t_l.append(Solver_CCS.SCF(L, ts=ts_scf_rdm_t_l, ls=ls_scf_rdm_t_l, diis=('rdm','t','l')))
  Result_grad.append(Solver.Gradient(L,method='newton',ts=ts_grad,ls=ls_grad))
  Result_des.append(Solver.Gradient(L, method='descend', ts=ts_des, ls=ls_des))

  ts_scf         = Result_scf[-1][3][-1]
  ls_scf         = Result_scf[-1][4][-1]
  #ts_scf_rdm     = Result_scf_rdm[-1][3][-1]
  #ls_scf_rdm     = Result_scf_rdm[-1][4][-1]
  #ts_scf_t       = Result_scf_t[-1][3][-1]
  #ls_scf_t       = Result_scf_t[-1][4][-1]
  #ts_scf_rdm_t   = Result_scf_rdm_t[-1][3][-1]
  #ls_scf_rdm_t   = Result_scf_rdm_t[-1][4][-1]
  #ts_scf_rdm_t_l = Result_scf_rdm_t_l[-1][3][-1]
  #ls_scf_rdm_t_l = Result_scf_rdm_t_l[-1][4][-1]
  ts_grad        = Result_grad[-1][3][-1]
  ls_grad        = Result_grad[-1][4][-1]
  ts_des = Result_des[-1][3][-1]
  ls_des = Result_des[-1][4][-1]

##################
# Plot T1/L1 map
##################

# Initialize Plot
pl_num = 0
fig, axs = plt.subplots(2,2,figsize=(10,8))
step =0.005 # step for the x,y ticks


for L in Larray:
  T1_vec = np.zeros((N,N))
  L1_vec = np.zeros((N,N))
  cost_vec = np.zeros((N,N))
  X2 = np.zeros((N,N))
  j = 0
  for ti in t:
      i = 0
      for li in l:
        rdm1 = mygccs.gamma(ti, li)
        Vexp = np.subtract(rdm1_exp,rdm1)

        # RCCS
        #fsp  = np.subtract(rfock,L*Vexp)
        #T1_vec[i,j] = (myccs.T1eq(ti,fsp))
        #Foo_l, Fvv_l = myccs.L1inter(ti,fsp)
        ##L1_vec[i,j] = (myccs.lsupdate(ti,li,Foo_l,Fvv_l,fsp,es_L1eq=True))
        #L1_vec[i, j] = (myccs.es_L1eq(ti, li, fsp))
        #cost_vec[i, j] = np.sqrt((T1_vec[i,j]**2+L1_vec[i,j]**2))
        ##X2[i,j] = np.sum(Vexp**2)

        # GCCS
        fsp = np.subtract(gfock,L*Vexp)
        T1 = mygccs.T1eq(ti,fsp)
        L1 = mygccs.L1eq(ti,li,fsp)
        cost_vec[i,j] = np.sqrt(np.sum(T1**2+np.sum(L1)**2))
        T1_vec[i,j] = T1[0,0]

        i += 1

      j += 1

  # RCCS
  #t_vec = t.copy()
  #l_vec = l.copy()

  # GCCS
  t_vec = t[:,0,0]
  l_vec = l[:,0,0]

  X, Y = np.meshgrid(t_vec, l_vec)
  i,j = np.unravel_index([pl_num],(2,2))
  i = int(i)
  j = int(j)
  tit = '$\lambda$ = %1.3f' % L

  axs[i, j].set_title(tit, fontsize=14)
  axs[i, j].xaxis.set_ticks(np.arange(np.min(t_vec), np.max(t_vec), step))
  axs[i, j].yaxis.set_ticks(np.arange(np.min(l_vec), np.max(l_vec), step))
  axs[i, j].tick_params(axis="x", labelsize=12)
  axs[i, j].tick_params(axis="y", labelsize=12)
  axs[i, j].set_xlim([np.min(t_vec), np.max(t_vec)])
  axs[i, j].set_ylim([np.min(l_vec), np.max(l_vec)])

  # RCCS
  # plot cost function
  #axs[i,j].contour(X,Y,cost_vec,50,colors='black',linewidths=0.65)
  #axs[i,j].pcolormesh(X,Y,cost_vec,cmap='ocean')
  ## Plot T1
  #axs[i,j].set_title(tit, fontsize='small')
  #axs[i,j].contour(X,Y,T1_vec,10)
  #axs[i,j].pcolormesh(X,Y,T1_vec)
  ## Plot L1
  #axs[i,j].set_title(tit, fontsize='small')
  #axs[i,j].contour(X,Y,L1_vec,10)
  #axs[i,j].pcolormesh(X,Y,L1_vec)

  # GCCS
  axs[i,j].contour(X,Y,cost_vec,20,colors='black',linewidths=0.5)
  axs[i,j].pcolormesh(X,Y,cost_vec,cmap='ocean')
  # Plot Result of solvers
  # SCF
  axs[i,j].plot(Result_scf[pl_num][3][:,0,0],Result_scf[pl_num][4][:,0,0],marker='o',color='red',
                linewidth=0.5,markerfacecolor='orange',alpha=0.8)
  # newton
  axs[i, j].plot(Result_grad[pl_num][3][:, 0, 0], Result_grad[pl_num][4][:, 0, 0], marker='o',color='grey'
                 ,linewidth=0.5,markerfacecolor='black')
  # DIIS
  #axs[i, j].plot(Result_scf_rdm_t[pl_num][3][:, 0, 0], Result_scf_rdm_t[pl_num][4][:, 0, 0], marker='o',
  #               color='indigo',linewidth=0.5,markerfacecolor='deeppink', alpha=1.0)
  # gradient descend
  #axs[i, j].plot(Result_des[pl_num][3][:, 0, 0], Result_des[pl_num][4][:, 0, 0], marker='o',color='green'
  #               ,linewidth=0.5,markerfacecolor='yellow')

  ## Plot t0 and l0
  axs[i,j].plot(t0[0,0],l0[0,0],'x',color='black',markersize=10,markeredgewidth=5)

  pl_num += 1

# X,Y,Z labels
fig.text(0.9,0.95,'$Z=\sqrt{\sum T_1^2+\sum L_1^2}$',ha='center',fontsize=14)
fig.text(0.5, 0.01, 't1 amplitudes', ha='center',fontsize=14)
fig.text(0.01, 0.5, 'l1 amplitudes', va='center', rotation='vertical',fontsize=14)

# better layout
fig.tight_layout(pad=1.5,rect=(0.03, 0.03, 0.99, 0.99))

# add colorbar
cbar = fig.colorbar(axs[-1,-1].pcolormesh(X,Y,cost_vec,cmap='ocean'), ax=axs.ravel().tolist(), shrink=0.95)
cbar.ax.tick_params(labelsize=12)

print('Calculated rdm1')
print()
print('no diis')
#print(Result_scf[0][-1])
print()
print('diis full')
#print(Result_scf_rdm_t_l[0][-1])
print()
print('Newton')
print(Result_grad[0][-1])
print(Result_grad[-1][3][-1])
print(Result_grad[-1][4][-1])

plt.show()
#plt.savefig('/home/milaim/Documents/Post_doc_DESY/XCW_CCS/Code/DEBUG/Test_H2/GCCS_scf_grad_tlini_2.png', format='png')
#plt.savefig('/home/milaim/Documents/blob.png',format='png')