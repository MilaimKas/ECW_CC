import copy
import numpy as np
from pyscf import gto, scf, cc, tdscf

from context import Eris, utilities, CCS, Solver_ES, exp_pot, exp_pot_new, Solver_ES_new

# build molecule
mol = gto.Mole()
#mol.atom = [
#    [8 , (0. , 0.     , 0.)],
#    [1 , (0. , -0.757 , 0.587)],
#    [1 , (0. , 0.757  , 0.587)]]
mol.atom = '''
H 0 0 0
H 0 0 1.
'''

mol.basis = '6-31g'
mol.spin = 0
mol.build()

# GHF calc
mf = scf.RHF(mol).run()
mgf = scf.addons.convert_to_ghf(mf)
#mgf = scf.GHF(mol).run()
mo_occ = mgf.mo_occ
mo_energy =mgf.mo_energy
mocc = mgf.mo_coeff[:, mo_occ > 0]
mvir = mgf.mo_coeff[:, mo_occ == 0]
gnocc = mocc.shape[1]
gnvir = mvir.shape[1]
mo_coeff = mgf.mo_coeff

# GCCSD eris
mygcc = cc.GCCSD(mgf)
geris = Eris.geris(mygcc)
gfs = geris.fock

# GCCS object
mccsg = CCS.Gccs(geris)

##########################
# build gamma_exp for GS
##########################

nbr_states = 2  # GS + ES

# ts and ls amplitudes
rcc = cc.RCCSD(mf)
rcc.kernel()
ts = rcc.t1
ls, ld = rcc.solve_lambda()
ts = utilities.convert_r_to_g_amp(ts)
ls = utilities.convert_r_to_g_amp(ls)
GS_exp = mccsg.gamma(ts, ls)
GS_Ek_exp = utilities.Ekin(mol, GS_exp)

# initial rn and ln from Koopman
rnini, DE = utilities.koopman_init_guess(mo_energy, mo_occ, nstates=(nbr_states-1, 0))
#rnini[0][0, 0] = 0.7
#rnini[0][1, 1] = 0.7
lnini = copy.deepcopy(rnini)
vm0 = np.zeros_like(gfs)
# initial r0 and l0
#r0ini = [mccsg.Extract_r0(r, ts, gfs, np.zeros_like(gfs)) for r in rnini]
#l0ini = [mccsg.Extract_l0(l, ts, gfs, np.zeros_like(gfs)) for l in lnini]
r0ini = [mccsg.r0_fromE(de, ts, r, vm0) for r, de in zip(rnini, DE)]
l0ini = [mccsg.r0_fromE(de, ts, l, vm0) for l, de in zip(lnini, DE)]
# normalize
# rnini, lnini, r0ini, l0ini = utilities.ortho_norm(rnini, lnini, r0ini, l0ini, ortho=False)

# build exp list
# exp_data = {np.full((nbr_states, nbr_states), None)}  # old format
exp_data = [] # new format
exp_data.append([])  # add GS

# build target tr_rdm1 and dip for ES
ES_trrdm_r = []
ES_trrdm_l = []
ES_dip_r = []
ES_dip_l = []
for i in range(nbr_states-1):
    exp_data.append([])

    # ES rdm1
    #ES_rdm1 = mccsg.gamma_es(ts, lnini[i], rnini[i], r0ini[i], l0ini[i])
    #ES_Ek = utilities.Ekin(mol, ES_rdm1)

    # ES tr rdm1
    # right: r=0 and r0=1
    ES_trrdm_r.append(mccsg.gamma_tr(ts, lnini[i], None, None, l0ini[i]))
    ES_dip_r.append(utilities.dipole(mol, ES_trrdm_r[-1], g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None))
    #exp_data[i+1, 0] = ['mat', ES_trrdm_r[-1]]  # old format

    # left: l=lambda and l0=1
    ES_trrdm_l.append(mccsg.gamma_tr(ts, ls, rnini[i], r0ini[i], 1.))
    ES_dip_l.append(utilities.dipole(mol, ES_trrdm_l[-1], g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None))
    # exp_data[0, i + 1] = ['mat', ES_trrdm_l[-1]]  # old format
    exp_data[i+1].append(['trmat', [ES_trrdm_r[-1], ES_trrdm_l[-1]]])  # new format

    # store squared norm of transition dipole
    #exp_data[0, i+1] = ['dip', ES_dip_l[-1]*ES_dip_r[-1]]
    #exp_data[i + 1, 0] = ['dip', ES_dip_l[-1] * ES_dip_r[-1]]

    # store Ek difference
    #exp_data[0, i+1] = ['DEk', ES_Ek-GS_Ek_exp]
    #exp_data[i + 1, 0] = ['DEk', ES_Ek-GS_Ek_exp]

# dip transition moment for H2O/6-311+g** with Qchem
#exp_data[0, 1] = ['dip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
#exp_data[0, 2] = ['dip', [0.000000, 0.000000, -0.622534]]  # DE = 10.06 eV
#exp_data[0, 3] = ['dip', [0.000000, -0.09280, 0.00000]]    # DE = 10.81 eV
#exp_data[0,3] = ['dip', [0., 0. ,0.030970]]                # DE = 536 eV

# Vexp object
L = 3.
VXexp = exp_pot_new.Exp(L, exp_data, mol, mgf.mo_coeff)

# convergence options
maxiter = 50
conv_thres = 10 ** -5
diis = 'all'
conv = 'rl'

# initialise Solver_CCS Class
Solver = Solver_ES_new.Solver_ES(mccsg, VXexp, conv_var=conv, conv_thres=conv_thres, maxiter=maxiter,
                             diis=diis, mindiis=2)

# CIS calc
mrf = scf.RHF(mol).run()
mcis = tdscf.TDA(mrf)
mcis.kernel(nstates=1)

print()
print('EHF= ', mgf.e_tot)
print()
print('CIS calc')
print('E= ', mcis.e_tot)
print('mu =', mcis.transition_dipole())
print('coeff= ', mcis.xy[0])
print()
print('initial guess')
print('E= ', Solver.E_ini)
print('r0= ', Solver.r0_ini)
print('l0= ', Solver.l0_ini)
print('rn= ', Solver.rn_ini)
print('ln= ', Solver.ln_ini)
print()

# orthogonalize and normalize initial vectors
print('Initial ortho')
print(utilities.check_ortho(rnini, lnini, r0ini, l0ini))
print()

# print('Difference between left and right gamma_exp')
# for i in range(0, nbr_states-1):
#     print(np.sum(np.subtract(ES_trrdm_l[i], ES_trrdm_r[i])))
#     print(np.sum(np.subtract(ES_trrdm_l[i], ES_trrdm_r[i])))
# print()

# Solve for L
result = Solver.SCF()# ,Vexp_norm2=True)
# result = Solver.SCF_diag(L, Vexp_norm2=True)

print()
print('Final ts and ls')
print(result[1].get('ts'))
print(result[1].get('ls'))

print()
print('Final rs and ls')
print(result[1].get('rn'))
print(result[1].get('ln'))

print()
print("Final max difference between GS rdm1 and target GS rdm1")
print(np.max(np.subtract(CCS.gamma_CCS(result[1].get('ts'), result[1].get('ls')), GS_exp)))
print("Final sum difference between GS rdm1 and target GS rdm1")
print(np.sum(np.abs(np.subtract(CCS.gamma_CCS(result[1].get('ts'), result[1].get('ls')), GS_exp))))

print()
print('TEST final dipole moment')
ES_trrdm_r = mccsg.gamma_tr(result[1].get('ts'), result[1].get('ln')[0], None, None, 0.)
ES_trrdm_l = mccsg.gamma_tr(result[1].get('ts'), result[1].get('ls'), result[1].get('rn')[0], 0., 1.)
#ES_trrdm_r = mccsg.gamma_tr(np.zeros_like(ts), lnini[0], None, None, 0.)
#ES_trrdm_l = mccsg.gamma_tr(np.zeros_like(ts), np.zeros_like(ls), rnini[0], 0., 1.)
print(utilities.dipole(mol, ES_trrdm_r, g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None) *
      utilities.dipole(mol, ES_trrdm_l, g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None))


