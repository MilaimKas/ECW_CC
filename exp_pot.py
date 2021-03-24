#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################################################
#
# ECW-CCS v1
# -----------
# Experimentally constrained wave function coupled cluster single
# ---------------------------------------------------------------
# Calculate Vexp potential and X2 statistic from either rdm1_exp or
# one electron properties
#
# todo: so far exp_pot is compatible with Solver_CCS and exp_pot_old, however Chi2 function is obsolete
# todo: implement list of exp
# todo: V_Ek is store even if rdm1 is compared
#
#
###################################################################

import numpy as np
import utilities

class Exp:
   def __init__(self, exp_data, mol, mo_coeff, mo_coeff_def=None):
       '''
       Class containing the experimental potentials Vnm

       :param exp_data: nn square matrix of list ('text',F) where text indicates which property or rdm1 is given
                   text = 'mat','Ek','dip','v1e'
                   Vexp = exp-calc
                   Vexp[0,0] -> GS
                   Vexp[n,0] and Vexp[0,n] -> transition case
                   Vexp[n,n] -> ES
                   'mat' is either a rdm1 or transition rdm1
                     -> must be given in AOs basis
                     -> must be given in G format
       :param mol: PySCF molecule object
       :param mo_coeff: "canonical" MOs coefficients
       :param mo_coeff_def: MOs coefficients for the exp rdm1 ("deformed" one)
       '''

       self.nbr_of_states = len(exp_data[0]) # total nbr of states: GS+ES
       self.exp_data = exp_data
       self.mo_coeff = mo_coeff
       self.mo_coeff_def = mo_coeff_def
       self.mol = mol
       
       # store necessary AOs integrals
       self.Ek_int = None
       self.dip_int = None
       self.v1e_int = None
       self.charge_center = None
       for n in exp_data:
           # dipole integrals
           if n[0] == 'dip' and self.dip_int is None:
               charges = mol.atom_charges()
               coords = mol.atom_coords()
               self.charge_center = np.einsum('z,zr->r', charges, coords) / charges.sum()
               # calculate integral -> 3 components
               with mol.with_common_orig(self.charge_center):
                   self.dip_int = mol.intor_symmetric('int1e_r', comp=3)
           # coulomb integrals
           if n[0] == 'v1e' and self.v1e_int is None:
               self.v1e_int = mol.intor_symmetric('int1e_nuc')
           # Kinetic integrals
           if n[0] == 'Ek' and self.Ek_int is None:
               self.Ek_int = mol.intor_symmetric('int1e_kin')

       # calculate Ek_exp from rdm1_exp for the GS if rdm1 given
       # initialize Ek_calc_GS
       if self.exp_data[0, 0][0] == 'mat':
           if self.Ek_int is None:
               self.Ek_int = mol.intor_symmetric('int1e_kin')
           self.Ek_exp_GS = utilities.Ekin(mol, exp_data[0, 0][1], g=True, AObasis=True, Ek_int=self.Ek_int)
           self.Ek_calc_GS = None
           self.X2_Ek_GS = None

       # initialize Vexp potential 2D list
       self.Vexp = np.full((self.nbr_of_states,self.nbr_of_states),None)

   def Vexp_update(self, rdm1, L, index):

       '''
       Update the Vexp[index] element of the Vexp matrix for a given rdm1_calc

       :param rdm1: calculated rdm1 or tr_rdm1 in MO basis
       :param L: exp weight
       :param index: nm index of the potential Vexp
            -> index = (0,0) for GS
            -> index = (n,n) for prop. of excited state n
            -> index = (0,n) for transition prop. of excited state n
       :return: (positive) Vexp(index) potential
       '''

       n, m = index
       k, l = np.sort(index)

       X2 = None
       vmax = None
       
       # check if prop or mat comparison
       check = self.exp_data[k,l][0]

       if check == 'mat' and index == (0,0):
           #print('GS',n,m)
           self.Vexp[0,0] = np.subtract(self.exp_data[0, 0][1], rdm1)
           X2 = np.sum(self.Vexp[0,0] ** 2)
           vmax = np.max(self.Vexp[0,0] ** 2)
           # calculate Ek_calc
           self.Ek_calc_GS = utilities.Ekin(self.mol, rdm1, g=True, AObasis=False, mo_coeff=self.mo_coeff, Ek_int=self.Ek_int)
           self.X2_Ek_GS = self.Ek_exp_GS-self.Ek_calc_GS

       elif check == 'mat' and index != (0,0):
           #print('ES', n, m)
           self.Vexp[n,m] = np.subtract(self.exp_data[k,l][1], rdm1)
           X2 = np.sum(self.Vexp[n,m] ** 2)
           vmax = np.max(self.Vexp[n,m] ** 2)

       elif check == 'Ek':
           # calculate Ek_calc
           Ek_calc = utilities.Ekin(self.mol, rdm1, g=True, AObasis=False, mo_coeff=self.mo_coeff, Ek_int=self.Ek_int)
           self.Vexp[n,m] = (self.exp_data[k,l][1] - Ek_calc)
           X2 = np.abs(self.Vexp[n,m])
           vmax = X2.copy()
           self.Vexp[n,m] *= rdm1

       elif check == 'v1e':
           # calculate v1e
           v1e_calc = utilities.v1e(self.mol, rdm1, g=True, AObasis=False, mo_coeff=self.mo_coeff, v1e_int=self.v1e_int)
           self.Vexp[n,m] = (self.exp_data[k,l][1] - v1e_calc)
           X2 = np.abs(self.Vexp[n,m])
           vmax = X2.copy()
           self.Vexp[n,m] *= rdm1
           
       elif check == 'dip':
           dip_calc = utilities.dipole(self.mol,rdm1,g=True,AObasis=False,mo_coeff=self.mo_coeff, dip_int=self.dip_int)
           X2 = 0
           for i in range(3):
              self.Vexp[n,m] = 0
              self.Vexp[n,m] += (self.exp_data[k,l][1][i] - dip_calc[i])
              X2 = self.Vexp[n,m]**2
           self.Vexp[n,m] *= 1/3
           self.Vexp[n,m] *= rdm1
           X2 *= 1 / 3
           vmax = X2.copy()
       else:
           raise ValueError('Exp list must contain information on the type of properties given: "mat", "dip", "Ek" or "v1e"')

       return self.Vexp[n,m]*L, X2, vmax


if __name__ == "__main__":
  # execute only if run as a script
  # test on water

  from pyscf import gto,scf,cc
  import gamma_exp
  import CCS

  mol = gto.Mole()
  mol.atom = [
      [8 , (0. , 0.     , 0.)],
      [1 , (0. , -0.757 , 0.587)],
      [1 , (0. , 0.757  , 0.587)]]

  mol.basis = '6-31g'
  mol.spin = 0
  mol.build()
  mf = scf.GHF(mol)
  mf.kernel()
  mo_occ   = mf.mo_occ
  mo_coeff = mf.mo_coeff
  mo_coeff_inv = np.linalg.inv(mo_coeff)
  mocc     = mo_coeff[:,mo_occ>0]
  mvir     = mo_coeff[:,mo_occ==0]
  nocc     = mocc.shape[1]
  nvir     = mvir.shape[1]

  rdm1_ao = mf.make_rdm1()
  # in MOs (should be diag with 1s and 0s) --> OK
  rdm1_mo_GS = np.einsum('pi,ij,qj->pq',mo_coeff_inv,rdm1_ao,mo_coeff_inv.conj())

  mycc = cc.GCCSD(mf)
  eris = mycc.ao2mo(mf.mo_coeff)
  fs = eris.fock

  # build gamma_exp
  #mycc.kernel()
  #rdm1_exp = mycc.make_rdm1()
  gexp = gamma_exp.Gexp(mol,'HF')
  gexp.Vext([0.05,0.02,0.0])
  gexp.build()
  GS_exp_ao = gexp.gamma_ao  # exp rdm1 in AOs
  mo_coeff_def = gexp.mo_coeff_def
  
  # build exp list
  exp = np.full((2,2),None)
  exp[0,0] = ['mat',GS_exp_ao]
  exp[0,1] = ['dip',[0,0,0.8]]
  exp[1,1] = ['Ek',75.97]

  L = 1000
  
  # initialize Vexp object
  myVexp = Exp(exp,mol,mo_coeff)

  print()
  print('nocc and nvir')
  print(nocc,nvir)
  print()
  print('Initial Vexp matrix')
  print(myVexp.Vexp)
  print()

  print(" Experimental potential and X2 ")

  #######
  # GS
  #######

  V, X2, vmax = myVexp.Vexp_update(rdm1_mo_GS,L,(0,0))

  print()
  print('GS --> exp[0,0]')

  print(exp[0,0][0])
  print('Vexp')
  print()
  print('X2, vmax =', X2,vmax)
  print('Vexp shape=', V.shape)
  print()
  print('DEk')
  print()
  print('Ek_calc =', myVexp.Ek_calc_GS)
  print('Ek_exp =', myVexp.Ek_exp_GS)
  print('X2: ', myVexp.X2_Ek_GS)
  print()


  ########
  # ES
  ########

  V, X2, vmax = myVexp.Vexp_update(rdm1_mo_GS, L, (1, 1))

  print('ES --> exp[1,1]')

  print(exp[1,1])
  print('Vexp')
  print()
  print('X2, vmax =', X2, vmax)
  print('Vexp shape=', V.shape)
  print()

  V, X2, vmax = myVexp.Vexp_update(rdm1_mo_GS, L, (0, 1))

  print('ES --> exp[0,1]')

  print(exp[0,1])
  print('Vexp')
  print()
  print('X2, vmax =', X2, vmax)
  print('Vexp shape=', V.shape)
  print()
