import numpy as np
from pyscf import gto, scf, cc
from context import Eris, CCS, gamma_exp, utilities

# build molecule
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '6-31g'
mol.spin = 0
mol.build()

# GHF calc
mf = scf.RHF(mol).run()
mgf = scf.addons.convert_to_ghf(mf)
mo_occ = mgf.mo_occ
mo_energy =mgf.mo_energy
mocc = mgf.mo_coeff[:, mo_occ > 0]
mvir = mgf.mo_coeff[:, mo_occ == 0]
nocc = mocc.shape[1]
nvir = mvir.shape[1]
mo_coeff = mgf.mo_coeff

# GCCSD eris
mygcc = cc.GCCSD(mgf)
geris = Eris.geris(mygcc)
gfs = geris.fock

# GCCS object
mccsg = CCS.Gccs(geris)

# Build target density matrix gamma_exp_mo
gamma_exp = gamma_exp.Gexp(mol, 'CCSD')
gamma_exp.Vext((0.05, 0.01, 0.01))
gamma_exp.build()
gamma_exp_ao = gamma_exp.gamma_ao
gamma_exp_mo = utilities.ao_to_mo(gamma_exp_ao, mo_coeff)

# Initial ts and ls amplitudes
ts = np.zeros(nocc, nvir)
ls = np.zeros(nocc, nvir)

# Initial gamma_calc
gamma_calc = mccsg.gamma(ts, ls)

# Initial effective fock matrix f' with weight L
L = 0.
fsp = L*(np.subtract(gamma_exp, gamma_calc))

# Function to calculate T1 and L1 values and the Jacobian matrix
mygrad = CCS.ccs_gradient(geris)
T1 = mygrad.T1eq(ts, fsp)
L1 = mygrad.L1eq(ts, ls, fsp)
J = mygrad.Jacobian(ts, ls, fsp, L)

# CHANGES