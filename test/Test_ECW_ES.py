import copy
import numpy as np
from pyscf import gto, scf, cc, tdscf

from context import Eris, utilities, CCS, Solver_ES, exp_pot

# build molecule
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
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
#ts = np.random.random((gnocc//2,gnvir//2))
#ts = utilities.convert_r_to_g_amp(ts)
ts = np.zeros((gnocc, gnvir))
ts[1, 1] = 0.1
ls = ts.copy()
GS_exp = mccsg.gamma(ts, ls)

# initial rn and ln from Koopman
rnini, DE = utilities.koopman_init_guess(mo_energy, mo_occ, nstates=(nbr_states-1, 0))
lnini = copy.deepcopy(rnini)
vm0 = np.zeros_like(gfs)
# initial r0 and l0
#r0ini = [mccsg.Extract_r0(r, ts, gfs, np.zeros_like(gfs)) for r in rnini]
#l0ini = [mccsg.Extract_l0(l, ts, gfs, np.zeros_like(gfs)) for l in lnini]
r0ini = [mccsg.r0_fromE(de, ts, r, vm0) for r, de in zip(rnini, DE)]
l0ini = [mccsg.r0_fromE(de, ts, l, vm0) for l, de in zip(lnini, DE)]
# normalize
rnini, lnini, r0ini, l0ini = utilities.ortho_norm(rnini, lnini, r0ini, l0ini, ortho=False)

# build exp list
exp_data = np.full((nbr_states, nbr_states), None)
# build target tr_rdm1 and dip for ES
ES_trrdm_r = []
ES_trrdm_l = []
ES_dip_r = []
ES_dip_l = []
for i in range(nbr_states-1):

    # right: r=0 and r0=1
    ES_trrdm_r.append(mccsg.gamma_tr(ts, lnini[i], None, None, l0ini[i]))
    ES_dip_r.append(utilities.dipole(mol, ES_trrdm_r[-1], g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None))
    exp_data[i+1, 0] = ['mat', ES_trrdm_r[-1]]

    # left: l=lambda and l0=1
    ES_trrdm_l.append(mccsg.gamma_tr(ts, ls, rnini[i], r0ini[i], None))
    ES_dip_l.append(utilities.dipole(mol, ES_trrdm_l[-1], g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None))
    exp_data[0, i+1] = ['mat', ES_trrdm_l[-1]]

# dip transition moment for H2O/6-311+g** with Qchem
#exp_data[0, 1] = ['dip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
#exp_data[0, 2] = ['dip', [0.000000, 0.000000, -0.622534]]  # DE = 10.06 eV
#exp_data[0, 3] = ['dip', [0.000000, -0.09280, 0.00000]]    # DE = 10.81 eV
#exp_data[0,3] = ['dip', [0., 0. ,0.030970]]                # DE = 536 eV

# Vexp object
VXexp = exp_pot.Exp(exp_data, mol, mgf.mo_coeff)

# convergence options
maxiter = 100
conv_thres = 10 ** -5
#diis = ('r', 'l')  # must be tuple
#diis = ('rdm1')
diis = ('')
conv = 'rl'

# initialise Solver_CCS Class
Solver = Solver_ES.Solver_ES(mccsg, VXexp, rnini, r0ini, lnini, l0ini,
                             conv_var=conv, conv_thres=conv_thres, maxiter=maxiter, diis=diis, mindiis=2)

# CIS calc
mrf = scf.RHF(mol).run()
mcis = tdscf.TDA(mrf)
mcis.kernel(nstates=3)

print()
print('EHF= ', mgf.e_tot)
print()
print('CIS calc')
print('DE= ', mcis.e)
print()
print('initial guess')
print('DE= ', DE)
print('r0= ', r0ini)
print('l0= ', l0ini)
print()

# orthogonalize and normalize initial vectors
print('Initial ortho')
print(utilities.check_ortho(rnini, lnini, r0ini, l0ini))
print()

print('Difference between left and right gamma_exp')
for i in range(0, nbr_states-1):
    print(np.sum(np.subtract(ES_trrdm_l[i], ES_trrdm_r[i])))
    print(np.sum(np.subtract(ES_trrdm_l[i], ES_trrdm_r[i])))
print()

# Solve for L
L = np.full(exp_data.shape, 0)
L[0, 0] = 0
result = Solver.SCF(L)
#result = Solver.SCF_diag(L)

print('Final ts and ls')
print(result[1].get('ts')[0, 0])
print(result[1].get('ls')[0, 0])
