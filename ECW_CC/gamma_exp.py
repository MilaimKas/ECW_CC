#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

 ECW-CC
 -----------

 Gexp: ground state
 Class for the simulated "target" or "experimental" one-electron reduced density matrix
 - Start from HF
 - Using CCSD or CCSD(T)
 - Adding external static field
 - Add geometry deformation
 - simulate under-fitting

 ESexp: excited state
 Class for the simulated experimental reduced transition density matrix
 or reduced density matrix for some excited states
 - MOM methods: to be tested
 - EOM-CCSD: to implement (L equations not implemented for EOM-EE)

 returns gamma_exp for excited states, initial guess for r and E'n

"""

# QChem H2O/AVTZ EOM-CCSD and CVS-EOM-CCSD calculation results:
# ----------------------------------------------------------------------
#
# Careful with the definition of the axes
#
#  EOMEE transition 1/A
#  Total energy = -76.05418671 a.u.  Excitation energy = 7.6051 eV. 
#  Transition dipole moment(a.u.):
#  GS->ES: (X 0.000000, Y 0.523742, Z 0.0000)
#  ES->GS: (X 0.000000, Y 0.550251, Z 0.00000)
#   Amplitude    Transitions between orbitals
#   0.5614       5 (A) A                   ->    6 (A) A
#   0.5614       5 (A) B                   ->    6 (A) B
#   0.2731       5 (A) A                   ->    8 (A) A
#   0.2731       5 (A) B                   ->    8 (A) B
#   0.2498       5 (A) A                   ->    10 (A) A
#   0.2498       5 (A) B                   ->    10 (A) B
#  -0.1278       5 (A) A                   ->    18 (A) A
#  -0.1278       5 (A) B                   ->    18 (A) B
#
#  EOMEE transition 3/A
#  Total energy = -75.96762546 a.u.  Excitation energy = 9.9605 eV. 
#  Transition dipole moment(a.u.):
#  GS->ES: (X 0.000000, Y 0.000000, Z -0.622534)
#  ES->GS: (X 0.000000, Y 0.000000, Z -0.649058)
#   Amplitude    Transitions between orbitals
#  -0.5624       4 (A) A                   ->    6 (A) A
#  -0.5624       4 (A) B                   ->    6 (A) B
#  -0.2618       4 (A) A                   ->    10 (A) A
#  -0.2618       4 (A) B                   ->    10 (A) B
#  -0.2321       4 (A) A                   ->    8 (A) A
#  -0.2321       4 (A) B                   ->    8 (A) B
#  -0.1207       5 (A) A                   ->    9 (A) A
#  -0.1207       5 (A) B                   ->    9 (A) B
#   0.1129       4 (A) A                   ->    18 (A) A
#   0.1129       4 (A) B                   ->    18 (A) B
#
#  EOMEE transition 4/A
#  Total energy = -75.93639427 a.u.  Excitation energy = 10.8104 eV. 
#  Transition dipole moment(a.u.):
#  GS->ES: (X 0.000000, Y -0.092803, Z 0.0000000)
#  ES->GS: (X 0.000000, Y -0.097482, Z 0.0000000)
#   Amplitude    Transitions between orbitals
#   -0.5859       5 (A) A                   ->    8 (A) A
#  -0.5859       5 (A) B                   ->    8 (A) B
#   0.2410       5 (A) A                   ->    6 (A) A
#   0.2410       5 (A) B                   ->    6 (A) B
#   0.1682       5 (A) A                   ->    16 (A) A
#   0.1682       5 (A) B                   ->    16 (A) B
#   0.1656       5 (A) A                   ->    10 (A) A
#   0.1656       5 (A) B                   ->    10 (A) B
#
#  CVS-EOMEE transition 1/A1
#  Total energy = -56.67790457 a.u.  Excitation energy = 534.8605 eV. 
#  Transition dipole moment(a.u.):
#  GS->ES: (X 0.000000, Y 0.000000, Z 0.030970)
#  ES->GS: (X 0.000000, Y 0.000000, Z 0.031222)
#   Amplitude    Transitions between orbitals
#  -0.4703       1 (A) A                   ->    6 (A) A
#  -0.4703       1 (A) B                   ->    6 (A) B
#  -0.3000       1 (A) A                   ->    8 (A) A
#  -0.3000       1 (A) B                   ->    8 (A) B
#  -0.2595       1 (A) A                   ->    10 (A) A
#  -0.2595       1 (A) B                   ->    10 (A) B
#   0.1825       1 (A) A                   ->    18 (A) A
#   0.1825       1 (A) B                   ->    18 (A) B
#
# ------------------------------------------------------------------------
#
###################################################################

import numpy as np
import utilities

from pyscf import scf, gto, cc


class Gexp:
    def __init__(self, mol, method, basis=None):
        """
        Returns the rdm1 in AOs basis (G format) from a RHF, RCCSD or RCCSD(T) calculation with deformed geometry
        and/or additional external static field
        --> called 'experimental' rdm1 (or 'target' rdm1 or gamma)

        :param mol: PySCF mol object
        :param basis: basis set for the calculation, if not given, uses mol.basis
        :param method: string: 'HF', 'CCSD' or 'CCSD(T)'
        """

        # deformed HF object
        self.mol_def = gto.mole.copy(mol)
        if basis is not None:
            if not isinstance(basis, str):
                raise ValueError('basis must be a string')
            self.mol_def.basis = basis
            self.mol_def.build()
        self.mf_def = scf.RHF(self.mol_def)
        self.mo_coeff_def = None
        self.nocc = None
        self.nvir = None

        # Exp or target rdm1
        self.gamma_ao = None  # in AOs basis R format

        self.method = method
        self.mycc = None

        # energies
        self.EHF_def = 0
        self.ECCSD_def = 0
        self.ECCSD_t_def = 0
        self.Eexp = 0

    def deform(self, def_max):
        """
        Apply a geometry deformation on the molecule and update mol_def object

        :param def_max: max value of a deformed bond length
        """

        autoang = 0.529177

        natm = self.mol_def.natm
        dq = (np.random.random_sample(natm * 3) * 2 - 1) * def_max
        new_coord = np.zeros((natm, 3))

        atom_name = []
        for i in range(natm):
            atom_name.append(mol._atom[i][0])
            for j in range(3):
                new_coord[i, j] = autoang * (self.mol_def.atom_coords()[i, j] + dq[i + j])

        # parse new coordinates to mol_def object
        self.mol_def.atom = []
        for i in range(natm):
            self.mol_def.atom.append([atom_name[i][0], [new_coord[i, 0], new_coord[i, 1], new_coord[i, 2]]])

        # rebuild integral
        self.mol_def.build()

        # re-initialize mf object
        self.mf_def = scf.RHF(self.mol_def)

    def Vext(self, field):
        """
        Add an external static electric field on the one-electron operator

        :param field: (x,y,z) 3 components of the field in au
        """

        self.mol_def.set_common_orig([0, 0, 0])

        # set the origin of the dipole at center of charge
        # --> check center of charge, is it a general expression ?
        # charges = self.mol_def.atom_charges()
        # coords = self.mol_def.atom_coords()
        # nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        # self.mol_def.set_common_orig_(nuc_charge_center)

        h_def = (self.mol_def.intor('cint1e_kin_sph') + self.mol_def.intor('cint1e_nuc_sph')  # Ekin_e + Ve-n
                 + np.einsum('x,xij->ij', field,
                             self.mol_def.intor('cint1e_r_sph', comp=3)))  # <psi|E.r|psi> dipole int.(comp ?)
        # h_def = scipy.linalg.block_diag(h_def, h_def)  # make hcore in SO basis
        self.mf_def.get_hcore = lambda *args: h_def  # pass the new one electron hamiltonian
        self.mol_def.incore_anyway = True  # force to use new h1e even when memory is not enough

    def build(self):
        """
        Perform HF,CCSD or CCSD(T) calculation on mol_def
        """

        # HF calculation with new mol_def object
        self.mf_def.conv_tol = 1e-09  # energy tolerance
        self.mf_def.conv_tol_grad = np.sqrt(1e-09)  # gradient tolerance
        self.mf_def.direct_scf_tol = 10e-13  # tolerance in discarding integrals
        self.mf_def.max_cycle = 100
        self.mf_def.max_memory = 1000

        # update mf_def
        self.mf_def.kernel()
        self.mo_coeff_def = self.mf_def.mo_coeff
        self.nocc = np.count_nonzero(self.mf_def.mo_occ > 0)
        self.nvir = np.count_nonzero(self.mf_def.mo_occ == 0)
        self.EHF_def = self.mf_def.e_tot
        self.Eexp = self.EHF_def

        # rdm1 in AOs R format
        tmp_rdm1 = self.mf_def.make_rdm1()

        # CCSD calculation
        if self.method == 'CCSD':

            mycc = cc.CCSD(self.mf_def, frozen=0)
            mycc.set(max_cycle=100)
            mycc.set(diis_space=10)
            self.ECCSD_def = mycc.kernel()[0]
            self.Eexp = self.ECCSD_def + self.EHF_def

            tmp_rdm1 = mycc.make_rdm1()  # in deformed MOs basis R format

        # CCSD(T) calculation
        elif self.method in ['CCSD(T)', 'CCSD(t)', 'CCSDT', 'CCSDt', 'ccsd(t)', 'ccsdt']:

            from pyscf.cc import ccsd_t_lambda_slow
            from pyscf.cc import ccsd_t_rdm_slow

            # moc = self.mf_def.mo_coeff

            # CCSD calc
            mycc = cc.CCSD(self.mf_def, frozen=0)
            mycc.set(max_cycle=100)
            mycc.set(diis_space=15)
            self.ECCSD_def, t1, t2 = mycc.kernel()
            eris = mycc.ao2mo()
            self.ECCSD_t_def = self.ECCSD_def + mycc.ccsd_t()
            self.Eexp = self.ECCSD_t_def + self.EHF_def

            # Solve Lambda
            l1, l2 = ccsd_t_lambda_slow.kernel(mycc, eris, t1, t2, verbose=0)[1:]

            # get rdm1
            tmp_rdm1 = ccsd_t_rdm_slow.make_rdm1(mycc, t1, t2, l1, l2, eris=eris)  # in def MOs R format

        elif self.method != 'HF':
            raise ValueError('Method not recognized')

        # convert to AO R format
        self.gamma_ao = utilities.mo_to_ao(tmp_rdm1, self.mo_coeff_def)  # in AOs R format
        # self.gamma_ao = utilities.convert_r_to_g_rdm1(self.gamma_ao_R)    # in AOs G format

    def underfit(self, para_factor):
        """
        Update gamma_ao
        Randomly distributed 0 elements --> under fitting n_exp is the final number of experimental parameters
        the number of optimized parameters is nocc*nvir*2 (t1+l1)

        :param para_factor: ratio between given exp elements in rdm1_exp and rdm1
        """

        import random

        dim = self.mo_coeff_def.shape[0]

        n_exp = int(round(dim ** 2 - (para_factor * (self.nocc * self.nvir * 2))))
        indice = random.sample(range(dim ** 2), n_exp)  # random indices fro the flattened gamma_ao matrix
        self.gamma_ao = self.gamma_ao.flatten()
        for i in indice:
            self.gamma_ao[i] = 0
        self.gamma_ao = np.reshape(self.gamma_ao, (dim, dim))


##############################
# Excited state case
##############################

class ESexp:
    def __init__(self, mol, Vext=None, nbr_of_states=(1, 0)):
        """
        Class to build ES rdm and tdm
        Contains:
           - MOM method: with tr_rdm1 GS->ES core/valence transition
           - EOM method: with rdm1 and tr_rdm1 for GS,ES and Gs->ES valence transition

        :param mol: PySCF mol object
        :param Vext: External static potential Vext=(vx,vy,vz)
        :param nbr_of_states: number of valence and core excited states (n_v,n_c)

        """

        self.mf = scf.RHF(mol)
        self.mol = mol

        self.nbr_of_states = nbr_of_states
        # list of rdm1 in AOs for all excited states
        self.gamma_ao = []
        # list of the transition rdm1 in AOs for all excited states
        self.gamma_tr_ao = []
        # rdm1 for the GS
        self.gamma_ao_gs = None

        # Apply external static field
        if Vext is not None:
            self.mol.set_common_orig([0, 0, 0])

            h_def = (self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')  # Ekin_e + Ve-n
                     + np.einsum('x,xij->ij', Vext,
                                 self.mol.intor('cint1e_r_sph', comp=3)))  # <psi|E.r|psi> dipole int.
            # make h1e in G format
            # h_def = scipy.linalg.block_diag(h_def, h_def)
            self.mf.get_hcore = lambda *args: h_def  # pass the new one electron hamiltonian
            # self.mol.incore_anyway = True            # force to use new h1e even when memory is not enough

        self.mf.kernel()
        self.mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        mocc = self.mo_coeff[:, mo_occ > 0]
        mvir = self.mo_coeff[:, mo_occ == 0]
        self.nocc = mocc.shape[1]
        self.nvir = mvir.shape[1]
        self.Eexp_GS = self.mf.e_tot
        self.DE_exp = []

        # list of initial r1 vectors and related excitation energies
        self.ini_r = [np.zeros((self.nocc, self.nvir))] * sum(nbr_of_states)

    def MOM(self):
        # todo: test the function and add property calculation
        # todo: verify if left and right tr_rdm1 are the same
        """
        Use MOM appraoch to calculate rdm1 and tr_rdm1 for a set of valence and core ES

        :return:
        """

        # !!!!!!!!
        # MOM method not implemented for GHF/RHF
        # Use UHF format
        # !!!!!!!!

        nao = (self.nvir + self.nocc)
        homo = self.mol.nelectron // 2 - 1
        lumo = homo + 1
        # mo_coeff in UHF format
        mo_coeff = np.asarray([np.zeros((nao, nao)), np.zeros((nao, nao))])
        mo_coeff[0, :, :] = self.mo_coeff[:, :]
        mo_coeff[1, :, :] = self.mo_coeff[:, :]

        # MOM calculation
        # ------------------

        # build GS mo_occ
        # mo_occ in UHF format
        moc = np.zeros((2, nao))
        moc[0, :self.mol.nelec[0]] = 1.
        moc[1, :self.mol.nelec[1]] = 1.

        # loop over valence excited states
        for v in range(self.nbr_of_states[0]):

            # orbital occupation in U format
            moc = np.zeros((2, nao))
            moc[0, :self.mol.nelec[0]] = 1
            moc[1, :self.mol.nelec[1]] = 1

            moc[0, homo] = 0.0
            moc[0, lumo + v] = 1.0
            # store initial r vector for the transition
            self.ini_r[v][homo, v] = 1.0

            # ES initial density matrix in UHF format
            es_mf = scf.UHF(self.mol)
            dm = es_mf.make_rdm1(mo_coeff, moc)

            # Apply MOM algorithm
            scf.addons.mom_occ(es_mf, mo_coeff, moc)

            # Start new SCF with new density matrix
            es_mf.scf(dm)

            # store new mo coeff and excitation energies
            es_mo_coeff = es_mf.mo_coeff
            self.DE_exp.append(es_mf.e_tot - self.Eexp_GS)

            # Calculate rdm1
            # -----------------

            # uhf rdm1
            uhf_ao = es_mf.make_rdm1(es_mo_coeff)

            # convert to GHF
            ghf_ao = utilities.convert_u_to_g_rdm1(uhf_ao)
            self.gamma_ao.append(['val', ghf_ao])

            # Convert coeff and moc to G format
            # ----------------------------------
            mo_coeff_g = utilities.convert_r_to_g_coeff(self.mo_coeff)
            es_mo_coeff = utilities.convert_u_to_g_coeff(es_mo_coeff)
            moc = utilities.convert_u_to_g_moc(moc)

            # Calculate transition density matrix
            # -------------------------------------

            # orthogonalize state n with GS
            TcL, TcR = utilities.ortho_SVD(self.mol, es_mo_coeff, mo_coeff_g)

            # express tdm in canonical MOs basis
            tdm = utilities.tdm_slater(TcL, TcR, moc)
            self.gamma_tr_ao.append(['val', tdm])

        # loop over core excited states
        for c in range(self.nbr_of_states[1]):
            moc = np.zeros((2, nao))

            moc[0, 0] = 0.0
            moc[0, lumo + c] = 1.0
            self.ini_r[self.nbr_of_states[0] + c][0, c] = 1.0

            # ES initial density matrix in UHF format
            dm = scf.uhf.make_rdm1(mo_coeff, moc)
            es_mf = scf.UHF(mol)

            # Apply MOM algorithm
            scf.addons.mom_occ(es_mf, mo_coeff, moc)

            # Start new SCF with new density matrix
            es_mf.scf(dm)

            # store new mo coeff and excitation energies
            es_mo_coeff = es_mf.mo_coeff
            self.DE_exp.append(es_mf.e_tot - self.Eexp_GS)

            # Calculate rdm1
            # -----------------

            # uhf rdm1
            uhf_ao = es_mf.make_rdm1(es_mo_coeff)

            # convert to GHF
            ghf_ao = utilities.convert_u_to_g_rdm1(uhf_ao)
            self.gamma_ao.append(['core', ghf_ao])

            # Convert coeff and moc to G format
            # ----------------------------------
            mo_coeff_g = utilities.convert_r_to_g_coeff(self.mo_coeff)
            es_mo_coeff = utilities.convert_u_to_g_coeff(es_mo_coeff)
            moc = utilities.convert_u_to_g_moc(moc)

            # Calculate transition density matrix
            # -------------------------------------

            # orthogonalize state n with GS
            TcL, TcR = utilities.ortho_SVD(self.mol, es_mo_coeff, mo_coeff_g)

            # express tdm in canonical MOs basis
            tdm = utilities.tdm_slater(TcL, TcR, moc)
            self.gamma_tr_ao.append(['core', tdm])

    def EOM(self, nbr_ES):
        """
        PySCF RCCSD-EOM calculation
        returns the Ek for each states
        """

        raise NotImplementedError

        import CCSD
        from pyscf import cc

        # Do RCCSD calculation using self.mf
        mycc = cc.RCCSD(self.mf)
        E, t1, t2 = mycc.kernel()
        # convert t into amplitudes
        # t1, t2 = mycc.vector_to_amplitudes(t)
        # convert into G format
        t1 = cc.addons.spatial2spin(t1)
        t2 = cc.addons.spatial2spin(t2)

        nroots = self.nbr_of_states[0]
        myeom = cc.eom_rccsd.EOMEESinglet(mycc)

        # Do singlet excitation EOM(2,2) and store excitation energies and r amplitudes
        DE_r, rn = myeom.kernel()  # mycc.eomee_ccsd_singlet(nroots=nroots)


if __name__ == "__main__":
    # Define Water molecule

    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = 'sto3g'
    mol.spin = 0
    mol.build()

    # RHF calculation

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-09  # energy tolerance
    mf.conv_tol_grad = np.sqrt(mf.conv_tol)  # gradient tolerance
    mf.direct_scf_tol = 1e-13  # tolerance in discarding integrals
    mf.max_cycle = 100
    mf.max_memory = 1000

    mf.kernel()

    # convert to GHF
    mfg = scf.addons.convert_to_ghf(mf)

    # GHF gamma_mo
    gamma_pred_ao = mfg.make_rdm1()
    mo_coeff = mfg.mo_coeff
    gamma_pred = utilities.ao_to_mo(gamma_pred_ao, mo_coeff)

    # GS gamma_exp
    gexp = Gexp(mol, 'CCSD(T)')
    gexp.deform(0.05)
    field = [0.05, 0.02, 0.]
    gexp.Vext(field)
    gexp.build()
    gamma_exp_ao = gexp.gamma_ao

    # exp rdm1 in canonical MOs --> only valid if basis_def = basis
    # gamma_exp_mo = utilities.ao_to_mo(gamma_exp_ao, mo_coeff)

    # apply underfitting
    # gexp.underfit(0.5)
    # gamma_exp_ao_under = gexp.gamma_ao      # underfitted exp rdm1 in AOs
    # gamma_exp_mo_under = utilities.ao_to_mo(gamma_exp_ao_under,mo_coeff) # underfitted exp rdm1 in MOs

    print()
    print('###########')
    print('#  GS exp  ')
    print('###########')
    print()
    print('method=', gexp.method,
          'EHF=', mfg.e_tot, 'EHF_def=', gexp.EHF_def, 'Eexp=', gexp.Eexp)
    # print()
    # print("Test under fitting with rho = 0.5")
    # print("number of 0 elements in gamma_exp        = ", np.count_nonzero(gamma_exp_mo_under==0.0))
    # print("total number of elements in gamma_exp    = ", gamma_exp_mo_under.shape[0]**2)
    # print()
    # print('-------------------------------------')
    # print()

    print()
    print('###########')
    print('#  ES exp  ')
    print('###########')
    print()
    print(' MOM for the 2 first valence ES and first core ES')
    print()

    # two valence and one core ES
    es_exp = ESexp(mol, nbr_of_states=(2, 1))
    es_exp.MOM()
