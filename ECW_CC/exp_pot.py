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
#
###################################################################

import numpy
import numpy as np
#from . import utilities
import utilities

class Exp:
    def __init__(self, exp_data, mol, mo_coeff, mo_coeff_def=None, a=None, b=None, c=None):
        '''
        Class containing the experimental potentials Vnm
        math: Vexp = 2/M sum_i^{M} (Aexp_i-Acalc_i)/sig_i * Ai_pq
              Acalc_i = sum_pq gamma_pq * Ai_pq

        :param exp_data: nn square matrix of list ('text',A) where 'text' indicates which property or if a rdm1 is given
                    text = 'mat','Ek','dip','v1e','F'
                    Vexp = exp-calc
                    Vexp[0,0] -> GS
                    Vexp[n,0] and Vexp[0,n] -> transition case
                    Vexp[n,n] -> ES
                    Vexp[n,m] = ('text', A) or list of prop (('text', prop1),('text', prop2),('text', prop3), ...)
                                Note that dipole has 3 components
                                Note that structure factor ['F', F] where F=[[h1,F1],[h2,F2], ...] and h=(hx,hy,hz)
                    'mat' is either a rdm1 or transition rdm1
                      -> must be given in MOs basis
                      -> must be given in G (spin-orbital basis) format
                    ''
        :param mol: PySCF molecule object
        :param mo_coeff: "canonical" MOs coefficients
        :param mo_coeff_def: MOs coefficients for the exp rdm1 ("deformed" one)
        :param a: reciprocal lattice length
        :param b: reciprocal lattice length
        :param c: reciprocal lattice length
        '''

        self.nbr_of_states = len(exp_data[0])  # total nbr of states: GS+ES
        self.exp_data = exp_data
        self.mo_coeff = mo_coeff
        self.mo_coeff_def = mo_coeff_def
        self.mol = mol

        # store necessary AOs and MOs integrals
        # ---------------------------------------

        self.Ek_int = None
        self.dip_int = None
        self.v1e_int = None
        self.F_int = None
        self.h = None
        self.charge_center = None
        self.check = []  # list of prop or mat for each state and transitions
        self.dic_int = {}  # dictionary containing needed MO integrals Ai_pq matrix

        for n in exp_data.flatten():
            # Build check list

            if n is not None:

                # if list of properties are given for a given state
                if isinstance(n[0], list):

                    # construct list of prop names
                    tmp_list = []
                    for m in n:
                      tmp_list.append(m[0])
                    self.check.append(tmp_list)

                    # only one property given
                elif isinstance(n[0], str):
                    self.check.append(n[0])

                # structure factor
                if 'F' in self.check[-1] and self.F_int is None:
                    self.h = n[1][:, 1]  # list of (hx, hy, hz)
                    F_mo, self.F_int = utilities.FT_MO(mol, self.h, mo_coeff, a=a, b=b, c=c)
                    self.dic_int['F'] = F_mo  # F_mo.shape = (nbr_h_pts, dim, dim)
                
                # dipole integrals
                if 'dip' in self.check[-1] and self.dip_int is None:
                    charges = mol.atom_charges()
                    coords = mol.atom_coords()
                    self.charge_center = np.einsum('z,zr->r', charges, coords) / charges.sum()
                    # calculate integral -> 3 components
                    with mol.with_common_orig(self.charge_center):
                        self.dip_int = mol.intor_symmetric('int1e_r', comp=3)  # AO basis
                        dip_int_mo = utilities.convert_aoint(self.dip_int, mo_coeff)  # MO spin-orbital basis
                        self.dic_int['dip'] = dip_int_mo

                # coulomb integrals
                if 'v1e' in self.check[-1] and self.v1e_int is None:
                    self.v1e_int = mol.intor_symmetric('int1e_nuc')
                    v1e_int_mo = utilities.convert_aoint(self.v1e_int, mo_coeff)
                    self.dic_int['v1e']= v1e_int_mo

                # Kinetic integrals
                if 'Ek' in self.check[-1] and self.Ek_int is None:
                    self.Ek_int = mol.intor_symmetric('int1e_kin')
                    Ek_int_mo = utilities.convert_aoint(self.Ek_int, mo_coeff)
                    self.dic_int['Ek'] = Ek_int_mo
            else:
                self.check.append(None)

        # calculate Ek_exp from rdm1_exp for the GS if rdm1 given
        # ---------------------------------------------------------

        # initialize Ek_calc_GS
        if self.exp_data[0, 0] is not None and self.exp_data[0, 0][0] == 'mat':
            if self.Ek_int is None:
                self.Ek_int = mol.intor_symmetric('int1e_kin')
            self.Ek_exp_GS = utilities.Ekin(mol, exp_data[0, 0][1], aobasis=False,
                                            ek_int=self.Ek_int, mo_coeff=self.mo_coeff)
        self.Ek_calc_GS = None
        self.X2_Ek_GS = None

        # initialize Vexp potential 2D list
        # ------------------------------------
        self.Vexp = np.full((self.nbr_of_states, self.nbr_of_states), None)


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
        k, l = np.sort(index)  # exp_data only contains right value (upper triangle)

        X2 = 0.
        vmax = 0.

        # check if prop or mat comparison
        # -> check_idx = index to retrive the name of the property from self.check list
        #check = self.exp_data[k,l][0]
        check_idx = np.ravel_multi_index((k, l), self.exp_data.shape)

        # Ground state case
        # -------------------

        if index == (0, 0):

            # if only one data

            if isinstance(self.check[check_idx], str):
                prop = self.check[check_idx]
                if not isinstance(prop, str):
                    raise ValueError('The name of the given exp prop must be a string')

                # use given target rdm1

                if prop == 'mat':
                    self.Vexp[0, 0] = np.subtract(self.exp_data[0, 0][1], rdm1)
                    X2 = np.sum(abs(self.Vexp[0, 0]))
                    vmax = np.max(abs(self.Vexp[0, 0]))
                    # calculate Ek_calc
                    self.Ek_calc_GS = utilities.Ekin(self.mol, rdm1, aobasis=False,
                                                     mo_coeff=self.mo_coeff, ek_int=self.Ek_int)
                    self.X2_Ek_GS = (self.Ek_exp_GS - self.Ek_calc_GS)**2

                # use given single experimental property

                else:
                    calc_prop = self.calc_prop(prop, rdm1)  # use calc_prop function sum_pq rdm_pq*Aint_pq
                    exp_prop = self.exp_data[0, 0]
                    self.Vexp[0, 0] = np.zeros_like(rdm1)

                    # if prop = dip or F
                    if isinstance(exp_prop[1], list):

                        # if prop = dip
                        if len(exp_prop[1]) == 3:
                            for d_calc, d_exp, d_int in zip(calc_prop, exp_prop[1], self.dic_int[prop]):
                                self.Vexp[0, 0] += (d_exp - d_calc)*d_int
                                X2 += (d_exp - d_calc)**2
                            X2 /= 3.
                            self.Vexp[0, 0] *= 2/3
                            vmax = np.max(abs(self.Vexp[0, 0]))

                        # structure factor
                        else:
                            for F_exp, F_calc, F_int_mo in zip(exp_prop[1][:, 1], calc_prop, self.dic_int[prop]):
                                self.Vexp[0, 0] += (F_exp - F_calc)*F_int_mo
                                X2 += (F_exp - F_calc)**2
                            self.Vexp[0, 0] *= 2/(len(self.h))
                            vmax = np.max(abs(self.Vexp[0, 0]))

                    # other prop (Ek or v1e)
                    elif isinstance(exp_prop[1], float):
                        self.Vexp[0, 0] = (exp_prop[1] - calc_prop)*self.dic_int[prop]  # self.dic_int = Aint matrix
                        X2 = self.Vexp[0, 0]**2
                        vmax = np.max(abs(self.Vexp[0, 0]))

            # use given list of properties

            elif isinstance(self.check[check_idx], list):
                self.Vexp[0, 0] = np.zeros_like(rdm1)
                X2 = 0.
                M = 0. # nbr of prop

                for exp_prop in self.exp_data[0, 0]:
                    calc_prop = self.calc_prop(exp_prop[0], rdm1)

                    # dip case -> 3 components
                    if isinstance(calc_prop, list) and len(calc_prop) == 3:
                        for d_calc, d_exp, j in zip(calc_prop, exp_prop[1], [0, 1, 2]):
                            self.Vexp[0, 0] += (d_exp - d_calc)*self.dic_int[exp_prop[0]][j]
                            X2 += (d_exp - d_calc)**2
                            M += 1

                    # structure factor
                    elif isinstance(calc_prop, list) and len(calc_prop) == len(self.h):
                        for F_exp, F_calc, F_int_mo in zip(exp_prop[1][:, 1], calc_prop, self.dic_int['F']):
                            self.Vexp[0, 0] += (F_exp - F_calc) * F_int_mo
                            X2 += (F_exp - F_calc) ** 2
                            M += 1
                    # other prop
                    elif isinstance(calc_prop, float):
                        self.Vexp[0, 0] += (exp_prop[1] - calc_prop)*self.dic_int[exp_prop[0]]
                        X2 += (exp_prop[1] - calc_prop) ** 2
                        M += 1
                    else:
                        raise ValueError('Wrong format for calculated property '
                                         '{}: must be a list or a float'.format(calc_prop))
                X2 /= float(M)
                vmax = np.max(abs(self.Vexp[0, 0]))
                self.Vexp[0, 0] *= 2/float(M)

            else:
                raise ValueError('Wrong format for exp_data')

            return self.Vexp[0, 0]*L, X2, vmax
        
        else:

            # Excited state case
            # --------------------

            # if one exp data

            if isinstance(self.check[check_idx], str):
                prop = self.check[check_idx]
                if not isinstance(prop, str):
                    raise ValueError('The name of the given exp prop must be a string')

                # direct comparison between rdm1

                if prop == 'mat':
                    self.Vexp[n, m] = np.subtract(self.exp_data[k, l][1], rdm1)
                    X2 = np.sum(abs(self.Vexp[n, m]))
                    vmax = np.max(abs(self.Vexp[n, m]))

                # use given single experimental property

                else:
                    calc_prop = self.calc_prop(prop, rdm1)
                    exp_prop = self.exp_data[k, l]
                    self.Vexp[n, m] = np.zeros_like(rdm1)
                    X2 = 0.
                    # if prop = dip
                    if isinstance(exp_prop[1], list):
                        for d_calc, d_exp, j in zip(calc_prop, exp_prop[1], [0,1,2]):
                            self.Vexp[n, m] += (d_exp - d_calc)*self.dic_int[prop][j]
                            X2 += (d_exp - d_calc)**2
                        self.Vexp[n, m] /= 3.
                    elif isinstance(exp_prop[1], float):
                        self.Vexp[n, m] = (exp_prop[1] - calc_prop)*self.dic_int[prop]
                        X2 = np.abs(self.Vexp[n, m])
                    else:
                        raise ValueError('Wrong format for calculated property {}: '
                                         'must be a list or a float'.format(calc_prop))
                    vmax = np.max(abs(self.Vexp[n, m]))

            # use given list of properties

            elif isinstance(self.check[check_idx], list):
                self.Vexp[n, m] = np.zeros_like(rdm1)
                X2 = 0.
                M = 0.
                for exp_prop in self.exp_data[k, l]:
                    calc_prop = self.calc_prop(exp_prop[0], rdm1)
                    # dip case -> 3 components
                    if isinstance(exp_prop[1], list):
                        for d_calc, d_exp, j in zip(calc_prop, exp_prop[1], [0,1,2]):
                            self.Vexp[n, m] += (d_exp - d_calc)*self.dic_int[exp_prop[0]][j]
                            X2 += (d_exp - d_calc)**2
                            M += 1
                    elif isinstance(exp_prop[1], float):
                        self.Vexp[n, m] += (exp_prop[1] - calc_prop)*self.dic_int[exp_prop[0]]
                        X2 += (exp_prop[1] - calc_prop) ** 2
                        M += 1
                    else:
                        raise ValueError('Wrong format for calculated property {}: '
                                         'must be a list or float'.format(calc_prop))
                    X2 /= float(M)
                    self.Vexp[n, m] *= 2/float(M)
                    vmax = np.max(abs(self.Vexp[n, m]))

            return self.Vexp[n ,m]*L, X2, vmax

    def calc_prop(self, prop, rdm1, g_format=True):
        '''
        Calculate prop using given rdm1

        :param prop: one-electron prop to calculate -> 'Ek', 'v1e' or 'dip'
        :param rdm1: reduced one body density matrix in MO basis in G format
        :return: calculated one-electron property
        '''

        ans = 0.

        if prop == 'Ek':
            ans = utilities.Ekin(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff, ek_int=self.Ek_int)

        elif prop == 'v1e':
            ans = utilities.v1e(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                v1e_int=self.v1e_int)

        elif prop == 'dip':
            ans = utilities.dipole(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                   dip_int=self.dip_int)
            ans = list(ans)

        elif prop == 'F':
            if self.h is None:
                raise ValueError('A list of Miller indices must be given')
            ans = utilities.structure_factor(self.mol, self.h, rdm1, aobasis=False, mo_coeff=self.mo_coeff,
                                             F_int=self.F_int)

        else:
            raise NotImplementedError('The possible properties are: Ek, v1e, dip and F')

        return ans

if __name__ == "__main__":
    # execute only if run as a script
    # test on water

    from pyscf import gto,scf,cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mf = scf.addons.convert_to_ghf(mf)
    mo_occ   = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mo_coeff_inv = np.linalg.inv(mo_coeff)
    mocc = mo_coeff[:,mo_occ>0]
    mvir = mo_coeff[:,mo_occ==0]
    nocc = mocc.shape[1]
    nvir = mvir.shape[1]
    dim = nocc+nvir

    # build exp list for 3 states: 1 GS + 2 ES
    nbr_os_states = 3
    exp_data = np.full((nbr_os_states,nbr_os_states),None)
    GS_exp_mo = np.random.random((dim//2,dim//2))
    GS_exp_mo = utilities.convert_r_to_g_rdm1(GS_exp_mo)
    exp_data[0,0] = ['mat', GS_exp_mo]
    exp_data[0,1] = ['dip', [0.,0.,0.8]]
    exp_data[1,1] = [['Ek', 75.97],['dip', [0.1,0.2,0.]],['v1e', -70.]]
    exp_data[0,2] = ['dip', [0.5, 0.2, 0.]]

    L = 0

    # initialize Vexp object
    myVexp = Exp(exp_data,mol,mo_coeff)

    print()
    print('nocc and nvir')
    print(nocc,nvir)
    print()
    print('Initial Vexp matrix')
    print(myVexp.Vexp)
    print()
    
    print('###############################')
    print(" Experimental potential and X2 ")
    print('###############################')
    
    #######
    # GS
    #######

    # GS calc rdm1
    rdm1_GS = np.random.random((dim//2,dim//2))
    rdm1_GS = utilities.convert_r_to_g_rdm1(rdm1_GS)

    V, X2, vmax = myVexp.Vexp_update(rdm1_GS, L, (0, 0))
    
    print('--------')
    print('GS case')
    print('--------')
    print()
    print('Prop = {}'.format(exp_data[0,0][1]))
    print()
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print('Vexp shape =', V.shape)
    print()
    print('DEk')
    print()
    print('Ek_calc =', myVexp.Ek_calc_GS)
    print('Ek_exp =', myVexp.Ek_exp_GS)
    print('X2_Ek: ', myVexp.X2_Ek_GS)
    print()

    print('--------')
    print('ES case')
    print('--------')

    V, X2, vmax = myVexp.Vexp_update(rdm1_GS, L, (1, 1))

    print('ES 1,1')
    print('prop = {}'.format(exp_data[1,1]))
    print()
    print('Vexp')
    print()
    print('X2, vmax =', X2, vmax)
    print('Vexp shape=', V.shape)
    print()

    V, X2, vmax = myVexp.Vexp_update(rdm1_GS, L, (0, 1))

    print('ES 0,1')
    print('prop = {}'.format(exp_data[0,1][1]))
    print()
    print('Vexp')
    print()
    print('X2, vmax =', X2, vmax)
    print('Vexp shape=', V.shape)
    print()
