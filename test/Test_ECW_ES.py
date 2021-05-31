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

mol.basis = '6-31+g*'
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

nbr_states = 3

# build gamma_exp for GS
#ts = np.random.random((gnocc//2,gnvir//2))
#ts = utilities.convert_r_to_g_amp(ts)
ts = np.zeros((gnocc, gnvir))
ts[1, 1] = 0.1
ls = ts.copy()
GS_exp = mccsg.gamma(ts, ls)

# initial rn, r0n and ln, l0n list
rnini, DE = utilities.koopman_init_guess(mo_energy, mo_occ, nstates=(nbr_states-1, 0))
lnini = copy.deepcopy(rnini)
r0ini = [mccsg.Extract_r0(r, np.zeros_like(rnini[0]), gfs, np.zeros_like(gfs)) for r in rnini]
l0ini = copy.deepcopy(r0ini)
rnini, lnini, r0ini, l0ini = utilities.ortho_norm(rnini, lnini, r0ini, l0ini)

# build target tr_rdm1 for ES
ES_trrdm_1 = mccsg.gamma_es(ts, lnini[0], rnini[0], r0ini[0], l0ini[0])
ES_trrdm_2 = mccsg.gamma_es(ts, lnini[1], rnini[1], r0ini[1], l0ini[1])
ES_dip_1 = utilities.dipole(mol, ES_trrdm_1, g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None)
ES_dip_2 = utilities.dipole(mol, ES_trrdm_2, g=True, aobasis=False, mo_coeff=mo_coeff, dip_int=None)

# build exp list
exp_data = np.full((nbr_states, nbr_states), None)
#exp_data[0, 0] = ['mat', GS_exp]
# dip transition moment for H2O/6-311+g** with Qchem
exp_data[0, 1] = ['mat', ES_trrdm_1]
exp_data[0, 2] = ['mat', ES_trrdm_2]
#exp_data[0, 1] = ['dip', [0.000000, 0.523742, 0.0000]]     # DE = 0.28 au
#exp_data[0, 2] = ['dip', [0.000000, 0.000000, -0.622534]]  # DE = 10.06 eV
#exp_data[0, 3] = ['dip', [0.000000, -0.09280, 0.00000]]    # DE = 10.81 eV
#exp_data[0,3] = ['dip', [0., 0. ,0.030970]]                # DE = 536 eV

# Vexp object
VXexp = exp_pot.Exp(exp_data, mol, mgf.mo_coeff)

# convergence options
maxiter = 50
conv_thres = 10 ** -5
diis = ('r', 'l')  # must be tuple
#diis = ('rdm1')
#diis = ('')
conv = 'rl'

# initialise Solver_CCS Class
Solver = Solver_ES.Solver_ES(mccsg, VXexp, rnini, r0ini, lnini, l0ini,
                             conv_var=conv, conv_thres=conv_thres, maxiter=maxiter, diis=diis)


# CIS calc
mrf = scf.RHF(mol).run()
mcis = tdscf.TDA(mrf)
mcis.kernel(nstates=3)

print('EHF= ', mgf.e_tot)
print()
print('CIS calc')
print('DE= ', mcis.e)
print()
print('initial guess')
print('DE= ', DE)
print('r0= ', r0ini)
print()

# orthogonalize and normalize initial vectors
print('Initial ortho')
print(utilities.check_ortho(rnini, lnini, r0ini, l0ini))
print()

print('Initial r vectors')
print(rnini)
print()

#print('Exp data')
#print(exp_data)
#print()

# Solve for L
L = np.full((exp_data.shape), 0)
L[0, 0] = 0
result = Solver.SCF(L)
#result = Solver.SCF_davidson(L)

print('Final ts and ls')
print(result[1].get('ts')[0, 0])
print(result[1].get('ls')[0, 0])
