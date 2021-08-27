#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Calculate Vexp potential and X2 statistic from either rdm1_exp or one electron properties
"""

# todo: add std deviation sig_j
# todo: Problem with complex number: when using structure factor, amplitudes, fock matrix and <||> must be complex


import numpy
import numpy as np
# from . import utilities
import utilities


class Exp:
    def __init__(self, exp_data, mol, mo_coeff, mo_coeff_def=None, rec_vec=None, h=None):
        """
        Class containing the experimental potentials Vnm
        math: Vexp = 2/M sum_i^{M} (Aexp_i-Acalc_i)/sig_i * Ai_pq
              Acalc_i = sum_pq gamma_pq * Ai_pq

        :param exp_data: nn square matrix of list ('text',A) where 'text' indicates which property or if a rdm1 is given
                    text = 'mat','Ek','dip','v1e','F'
                    Vexp = exp-calc
                    Vexp[0,0] -> GS prop
                    Vexp[n,0] and Vexp[0,n] -> transition case
                    Vexp[n,n] -> ES prop
                    Vexp[n,m] = ('text', A) or list of prop (('text', prop1),('text', prop2),('text', prop3), ...)
                                Note that dipole has 3 components ['dip': [x,y,z]]
                                Note that structure factor ['F', F] where F=[[h1,F1],[h2,F2], ...] and h=(hx,hy,hz)
                    'mat' is either a rdm1 or transition rdm1
                      -> must be given in MOs basis
                      -> must be given in G (spin-orbital basis) format

                    Note: the upper triangle corresponds to left properties or dm1
                          the lower triangle corresponds to right properties or dm1
                    ''
        :param mol: PySCF molecule object
        :param mo_coeff: "canonical" MOs coefficients
        :param mo_coeff_def: MOs coefficients for the exp rdm1 ("deformed" one)
        :param rec_vec: array of reciprocal lattice lengths (a,b,c)
        :param h: list of Miller indices
        """

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
        # todo: write integrals on external file to save memory ?
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
                    if rec_vec is None or h is None:
                        raise ValueError('If F are to be calculated, Miller indices and unit cell size must be given')
                    F_mo, self.F_int = utilities.FT_MO(mol, h, mo_coeff, rec_vec)
                    self.h = h
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
                    self.dic_int['v1e'] = v1e_int_mo

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

        # initialize Vexp potential, X2 and vmax 2D list
        # ------------------------------------------------
        self.Vexp = np.full((self.nbr_of_states, self.nbr_of_states), None)
        # self.X2 = np.zeros_like(self.Vexp)
        # self.vmax = np.zeros_like(self.X2)

    def Vexp_update(self, rdm1, index, rdm1_add=None):
        """
        Update the Vexp[index] element of the Vexp matrix for a given rdm1_calc
        Here, norm of expectation value are compared

        :param rdm1: calculated left(mn) or right(nm) rdm1 or tr_rdm1 in MO basis
        :param L: exp weight
        :param index: nm index of the potential Vexp
             -> index = (0,0) for GS
             -> index = (n,n) for prop. of excited state n
             -> index = (0,n) and (n,0) for left and right transition prop. of excited state n
                        if prop are given (not mat.), the square or the norm is take |prop|^2 = prop_l*prop_r
        :return: (positive) Vexp(index) potential
        """

        n, m = index

        X2 = 0.
        vmax = 0.

        # check if prop, lis of prop or matrix comparison
        # -> check_idx = index to retrieve the name of the property from self.check list
        check_idx = np.ravel_multi_index((n, m), self.exp_data.shape)

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
                    # calculate kinetic energy Ek_calc
                    self.Ek_calc_GS = utilities.Ekin(self.mol, rdm1, aobasis=False,
                                                     mo_coeff=self.mo_coeff, ek_int=self.Ek_int)
                    # store Chi squared value for Ek
                    self.X2_Ek_GS = (self.Ek_exp_GS - self.Ek_calc_GS) ** 2

                # use given single experimental property

                else:
                    calc_prop = self.calc_prop(prop, rdm1)  # use calc_prop function sum_pq rdm_pq*Aint_pq
                    exp_prop = self.exp_data[0, 0]
                    self.Vexp[0, 0] = np.zeros_like(rdm1)

                    # if prop = dip or F
                    if isinstance(exp_prop[1], (list, np.ndarray)):

                        # if prop = dip
                        if len(exp_prop[1]) == 3:
                            for d_calc, d_exp, d_int in zip(calc_prop, exp_prop[1], self.dic_int[prop]):
                                self.Vexp[0, 0] += np.abs((d_exp - d_calc)) * d_int
                                X2 += (d_exp - d_calc) ** 2
                            X2 /= 3.
                            self.Vexp[0, 0] *= 2 / 3
                            vmax = np.max(abs(self.Vexp[0, 0]))

                        # structure factor
                        else:
                            for F_exp, F_calc, F_int_mo in zip(exp_prop[1][:, 1], calc_prop, self.dic_int[prop]):
                                self.Vexp[0, 0] += np.abs((F_exp - F_calc)) * F_int_mo
                                X2 += (F_exp - F_calc) ** 2
                            self.Vexp[0, 0] *= 2 / (len(self.h))
                            vmax = np.max(abs(self.Vexp[0, 0]))

                    # other prop (Ek or v1e)
                    elif isinstance(exp_prop[1], float):
                        # self.dic_int = Aint matrix
                        self.Vexp[0, 0] = np.abs((exp_prop[1] - calc_prop)) * self.dic_int[prop]
                        X2 = self.Vexp[0, 0] ** 2
                        vmax = np.max(abs(self.Vexp[0, 0]))

            # use given list of properties

            elif isinstance(self.check[check_idx], (list, np.ndarray)):
                self.Vexp[0, 0] = np.zeros_like(rdm1)
                X2 = 0.
                M = 0.  # nbr of prop

                # loop over properties
                for exp_prop in self.exp_data[0, 0]:
                    calc_prop = self.calc_prop(exp_prop[0], rdm1)

                    # dip case -> 3 components
                    if isinstance(calc_prop, (list, np.ndarray)) and len(calc_prop) == 3:
                        for d_calc, d_exp, j in zip(calc_prop, exp_prop[1], [0, 1, 2]):
                            self.Vexp[0, 0] += np.abs((d_exp - d_calc)) * self.dic_int[exp_prop[0]][j]
                            X2 += (d_exp - d_calc) ** 2
                            M += 1

                    # structure factor
                    elif isinstance(calc_prop, (list, np.ndarray)) and len(calc_prop) == len(self.h):
                        # loop over structure factors
                        for F_exp, F_calc, F_int_mo in zip(exp_prop[1][:], calc_prop, self.dic_int['F']):
                            self.Vexp[0, 0] += np.abs((F_exp - F_calc)) * F_int_mo
                            X2 += np.abs((F_exp - F_calc)) ** 2
                            M += 1

                    # other prop: Ek or v1e
                    elif isinstance(calc_prop, float):
                        self.Vexp[0, 0] += np.abs((exp_prop[1] - calc_prop)) * self.dic_int[exp_prop[0]]
                        X2 += (exp_prop[1] - calc_prop) ** 2
                        M += 1
                    else:
                        raise ValueError('Wrong format for calculated property '
                                         '{}: must be a list or a float'.format(calc_prop))
                X2 /= float(M)
                vmax = np.max(abs(self.Vexp[0, 0]))
                self.Vexp[0, 0] *= 2 / float(M)

            else:
                raise SyntaxError('Wrong format for exp_data')

            return self.Vexp[0, 0], X2, vmax

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
                    self.Vexp[n, m] = np.subtract(self.exp_data[n, m][1], rdm1)
                    X2 = np.sum(abs(self.Vexp[n, m]))
                    vmax = np.max(abs(self.Vexp[n, m]))

                # use given single experimental property

                else:
                    exp_prop = self.exp_data[n, m]
                    self.Vexp[n, m] = np.zeros_like(rdm1)
                    X2 = 0.
                    # if prop = dip
                    if isinstance(exp_prop[1], (list, np.ndarray)):
                        calc_prop = self.calc_prop(prop, rdm1)
                        for d_calc, d_exp, j in zip(calc_prop, exp_prop[1], [0, 1, 2]):
                            self.Vexp[n, m] += np.abs((d_exp - d_calc)) * self.dic_int[prop][j]
                            X2 += (d_exp - d_calc) ** 2
                        self.Vexp[n, m] /= 3.
                    # Difference in Ek: DEk=Ek_GS-Ek_ES
                    elif exp_prop[0] == "DEk":
                        if rdm1_add is None:
                            raise ValueError('GS rdm1 must be given if DEk is to calculated')
                        calc_prop = self.calc_prop("Ek", np.subtract(rdm1, rdm1_add))
                        self.Vexp[n, m] = np.abs((exp_prop[1] - calc_prop)) * self.dic_int['Ek']
                        X2 = (exp_prop[1] - calc_prop) ** 2
                    # Ek or v1e
                    elif isinstance(exp_prop[1], float):
                        calc_prop = self.calc_prop(prop, rdm1)
                        self.Vexp[n, m] = np.abs((exp_prop[1] - calc_prop)) * self.dic_int[prop]
                        X2 = (exp_prop[1] - calc_prop) ** 2
                    else:
                        raise ValueError('Wrong format for calculated property {}: '
                                         'must be a list or a float'.format(exp_prop[0]))
                    vmax = np.max(abs(self.Vexp[n, m]))

            # use given list of properties

            elif isinstance(self.check[check_idx], list):
                self.Vexp[n, m] = np.zeros_like(rdm1)
                X2 = 0.
                M = 0.
                for exp_prop in self.exp_data[n, m]:
                    calc_prop = self.calc_prop(exp_prop[0], rdm1)
                    # dip case -> 3 components
                    if isinstance(exp_prop[1], (list, np.ndarray)):
                        for d_calc, d_exp, j in zip(calc_prop, exp_prop[1], [0, 1, 2]):
                            self.Vexp[n, m] += np.abs((d_exp - d_calc)) * self.dic_int[exp_prop[0]][j]
                            X2 += (d_exp - d_calc) ** 2
                            M += 1
                    # Difference in Ek: DEk=Ek_GS-Ek_ES
                    elif exp_prop[1] == "DEk":
                        if rdm1_add is None:
                            raise ValueError('GS rdm1 must be given if DEk is to calculated')
                        calc_prop = self.calc_prop('DEk', np.subtract(rdm1, rdm1_add))
                        exp_prop = self.exp_data[n, m]
                        self.Vexp[n, m] = np.abs((exp_prop - calc_prop)) * self.dic_int['Ek']
                        X2 += (exp_prop - calc_prop) ** 2
                        M += 1
                    # Ek and V1e
                    elif isinstance(exp_prop[1], float):
                        self.Vexp[n, m] += np.abs((exp_prop[1] - calc_prop)) * self.dic_int[exp_prop[0]]
                        X2 += (exp_prop[1] - calc_prop) ** 2
                        M += 1
                    else:
                        raise ValueError('Wrong format for calculated property {}: '
                                         'must be a list or float'.format(calc_prop))
                    X2 /= float(M)
                    self.Vexp[n, m] *= 2 / float(M)
                    vmax = np.max(abs(self.Vexp[n, m]))

            return self.Vexp[n, m], X2, vmax

    def Vexp_update_norm2(self, rdm1, index, rdm1_add=None):
        """
        Update the Vexp[index] element of the Vexp matrix for a given rdm1_calc
        assumes that square norm for the expectation value are compared

        rdm can be a transition rdm. In this case the right and left tr_rdm must be given

        :param rdm1: rdm1 or nm tr_rdm1 in MO basis, if not given rdm1_add=rdm1
        :param rdm1_add: transpose rdm1 or left (mn) tr_rdm1 in MO basis
        :param index: nm index of the potential Vexp
             -> index = (0,0) for GS
             -> index = (n,n) for prop. of excited state n
             -> index = (0,n) and (n,0) for left and right transition prop. of excited state n
                        if prop are given (not mat.), the square or the norm is taken as |prop|^2 = prop_l*prop_r
        :return: (positive) Vexp(index) potential
        """

        n, m = index
        k, l = np.sort(index)  # index of the upper triangle exp_data

        X2 = 0.
        vmax = 0.

        # check if prop or mat comparison
        # -> check_idx = index to retrieve the name of the property from self.check list
        check_idx = np.ravel_multi_index((k, l), self.exp_data.shape)

        # Ground state case: nm = 00 and rdm1-left = rdm1.transpose
        # ------------------------------------------------------------

        if index == (0, 0):

            if rdm1_add is None:
                rdm1_add = rdm1.transpose()

            # if only one data

            if isinstance(self.check[check_idx], str):
                prop = self.check[check_idx]
                if not isinstance(prop, str):
                    raise ValueError('The name of the given exp prop must be a string')

                # use given target rdm1

                if prop == 'mat':
                    raise NotImplementedError
                    # self.Vexp[0, 0] = np.subtract(self.exp_data[0, 0][1], rdm1)
                    # X2 = np.sum(abs(self.Vexp[0, 0]))
                    # vmax = np.max(abs(self.Vexp[0, 0]))
                    # # calculate Ek_calc
                    # self.Ek_calc_GS = utilities.Ekin(self.mol, rdm1, aobasis=False,
                    #                                 mo_coeff=self.mo_coeff, ek_int=self.Ek_int)
                    # self.X2_Ek_GS = (self.Ek_exp_GS - self.Ek_calc_GS) ** 2

                # use given single experimental property

                else:
                    calc_prop, A_scale = self.calc_prop(prop, rdm1, rdm1_2=rdm1_add)  # returns A.g1*A.g2, A.g2
                    exp_prop = self.exp_data[0, 0]
                    self.Vexp[0, 0] = np.zeros_like(rdm1)

                    # if prop = dip or F
                    if isinstance(exp_prop[1], (list, np.ndarray)):

                        # if prop = dip
                        if len(exp_prop[1]) == 3:
                            for d_calc, d_exp, d_int, A in zip(calc_prop, exp_prop[1], self.dic_int[prop], A_scale):
                                self.Vexp[0, 0] += (d_exp - d_calc) * d_int * A
                                X2 += (d_exp - d_calc) ** 2
                            X2 /= 3.
                            self.Vexp[0, 0] *= 2 / 3
                            vmax = np.max(abs(self.Vexp[0, 0]))

                        # structure factor
                        else:
                            for F_exp, F_calc, F_int_mo, A \
                                    in zip(exp_prop[1][:, 1], calc_prop, self.dic_int[prop], A_scale):
                                self.Vexp[0, 0] += (F_exp - F_calc) * F_int_mo * A
                                X2 += (F_exp - F_calc) ** 2
                            self.Vexp[0, 0] *= 2 / (len(self.h))
                            vmax = np.max(abs(self.Vexp[0, 0]))

                    # other prop (Ek or v1e)
                    elif isinstance(exp_prop[1], float):
                        # self.dic_int = Aint matrix
                        self.Vexp[0, 0] = (exp_prop[1] - calc_prop) * self.dic_int[prop] * A_scale
                        X2 = (exp_prop[1] - calc_prop) ** 2
                        vmax = np.max(abs(self.Vexp[0, 0]))

            # use given list of properties

            elif isinstance(self.check[check_idx], (list, np.ndarray)):
                self.Vexp[0, 0] = np.zeros_like(rdm1)
                X2 = 0.
                M = 0.  # nbr of prop

                for exp_prop in self.exp_data[0, 0]:
                    calc_prop, A_scale = self.calc_prop(exp_prop[0], rdm1, rdm1_2=rdm1_add)

                    # dip case -> 3 components
                    if isinstance(calc_prop, (list, np.ndarray)) and len(calc_prop) == 3:
                        for d_calc, d_exp, j, A in zip(calc_prop, exp_prop[1], [0, 1, 2], A_scale):
                            self.Vexp[0, 0] += (d_exp - d_calc) * self.dic_int[exp_prop[0]][j] * A
                            X2 += (d_exp - d_calc) ** 2
                            M += 1

                    # structure factor
                    elif isinstance(calc_prop, (list, np.ndarray)) and len(calc_prop) == len(self.h):
                        # loop over structure factors
                        for F_exp, F_calc, F_int_mo, A in zip(exp_prop[1][:], calc_prop, self.dic_int['F'], A_scale):
                            self.Vexp[0, 0] += (F_exp - np.abs(F_calc) ** 2) * F_int_mo * A
                            X2 += (F_exp - F_calc) ** 2
                            M += 1
                    # other prop
                    elif isinstance(calc_prop, float):
                        self.Vexp[0, 0] += (exp_prop[1] - calc_prop) * self.dic_int[exp_prop[0]] * A_scale
                        X2 += (exp_prop[1] - calc_prop) ** 2
                        M += 1
                    else:
                        raise ValueError('Wrong format for calculated property '
                                         '{}: must be a list or a float'.format(calc_prop))
                X2 /= float(M)
                vmax = np.max(abs(self.Vexp[0, 0]))
                self.Vexp[0, 0] *= 2 / float(M)

            else:
                raise ValueError('Wrong format for exp_data')

            return self.Vexp[0, 0], X2, vmax

        else:

            # Excited state case
            # --------------------

            if rdm1_add is None:
                rdm1_add = rdm1.copy()

            # if one exp data

            if isinstance(self.check[check_idx], str):

                # check type of property
                prop = self.check[check_idx]
                if not isinstance(prop, str):
                    raise ValueError('The name of the given exp prop must be a string')

                # direct comparison between rdm1
                if prop == 'mat':
                    raise NotImplementedError
                    # self.Vexp[k, l] = np.subtract(self.exp_data[k, l][1], rdm1_r*rdm1_l)
                    # X2 = np.sum(abs(self.Vexp[k, l]))
                    # vmax = np.max(abs(self.Vexp[k, l]))

                # use given single experimental property
                else:
                    calc_prop, A_scale = self.calc_prop(prop, rdm1, rdm1_2=rdm1_add)
                    exp_prop = self.exp_data[k, l]
                    self.Vexp[n, m] = np.zeros_like(rdm1)
                    X2 = 0.
                    # if prop = dip
                    if isinstance(exp_prop[1], (list, np.ndarray)):
                        for d_calc, d_exp, j, A in zip(calc_prop, exp_prop[1], [0, 1, 2], A_scale):
                            self.Vexp[n, m] += (d_exp - d_calc) * self.dic_int[prop][j] * A
                            X2 += (d_exp - d_calc) ** 2
                        self.Vexp[n, m] /= 3.
                    elif isinstance(exp_prop[1], float):
                        self.Vexp[n, m] = (exp_prop[1] - calc_prop) * self.dic_int[prop] * A_scale
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
                # loop over properties
                for exp_prop in self.exp_data[k, l]:
                    # returns |A|^2, A if rdm1_2 is given
                    calc_prop, A_scale = self.calc_prop(exp_prop[0], rdm1, rdm1_2=rdm1_add)
                    # dip case -> 3 components
                    if isinstance(exp_prop[1], list):
                        for d_calc, d_exp, j, A in zip(calc_prop, exp_prop[1], [0, 1, 2], A_scale):
                            self.Vexp[n, m] += (d_exp - d_calc) * self.dic_int[exp_prop[0]][j] * A
                            X2 += (d_exp - d_calc) ** 2
                            M += 1
                    elif isinstance(exp_prop[1], float):
                        self.Vexp[k, l] += (exp_prop[1] - calc_prop) * self.dic_int[exp_prop[0]] * A_scale
                        X2 += (exp_prop[1] - calc_prop) ** 2
                        M += 1
                    else:
                        raise ValueError('Wrong format for calculated property {}: '
                                         'must be a list or float'.format(calc_prop))
                    X2 /= float(M)
                    self.Vexp[n, m] *= 2 / float(M)
                    vmax = np.max(abs(self.Vexp[n, m]))

            return self.Vexp[n, m], X2, vmax

    #def Vexp_update_new(self, rdm1, tr_rdm1, index):
    #    """
    #
    #    :param rdm1: list of rdm1
    #    :param tr_rdm1: list of tr_rdm1
    #    :param index: index of Vexp element to update
    #    :return:
    #    """
    #
    #    n, m = index
    #
    #    X2 = 0.
    #    vmax = 0.
    #
    #    # check if prop, list of prop or matrix comparison
    #    # -> check_idx = index to retrieve the name of the property from self.check list
    #    # can be either a list or a string
    #    check_idx = np.ravel_multi_index((n, m), self.exp_data.shape)
    #
    #    # single property
    #    if isinstance(self.check[check_idx], str):
    #        prop = self.check[check_idx]
    #
    #        # if norm squared prop (such as transition dipole moment)
    #        if "2" in prop:
    #            if index == (0, 0):
    #                self.Vexp[], self.X2[], self.vmax[] = Vexp_update_norm2(rdm1[0], rdm1[0].transpose(), prop)
    #        else:
    #            self.Vexp[], self.X2[], self.vmax[] = Vexp_update(rdm1[0], rdm1[0].transpose(), prop)

    def calc_prop(self, prop, rdm1, g_format=True, rdm1_2=None):
        """
        Calculate A**2 and/or A using given rdm1

        :param g_format: True if rdm1 given in Generalized spin-orbit format
        :param prop: one-electron prop to calculate -> 'Ek', 'v1e' or 'dip'
        :param rdm1: reduced one body density matrix in MO basis in G format
        :param rdm1_2: left rdm1, if given, the norm squared of the prop is calculated using both right and left rdm1
                        where rdm1_2 is the rdm1^mn if Vnm
        :return: calculated one-electron property A**2 and/or A
        """

        if prop == 'Ek':
            ans1 = utilities.Ekin(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                  ek_int=self.Ek_int)
            if rdm1_2 is not None:
                ans2 = utilities.Ekin(self.mol, rdm1_2.transpose(), g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                      ek_int=np.conj(self.Ek_int))
                return ans1 * ans2, ans2
            else:
                return ans1

        elif prop == 'v1e':
            ans1 = utilities.v1e(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                 v1e_int=self.v1e_int)
            if rdm1_2 is not None:
                ans2 = utilities.v1e(self.mol, rdm1_2.transpose(), g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                     v1e_int=np.conj(self.v1e_int))
                return ans1 * ans2, ans2
            else:
                return ans1

        elif prop == 'dip':
            ans1 = utilities.dipole(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                    dip_int=self.dip_int)
            if rdm1_2 is not None:
                ans2 = utilities.dipole(self.mol, rdm1_2.transpose(), g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                        dip_int=np.conj(self.dip_int))
                return list(ans1 * ans2), list(ans2)
            else:
                return list(ans1)

        elif prop == 'F':
            if self.h is None:
                raise ValueError('A list of Miller indices must be given')
            ans1 = utilities.structure_factor(self.mol, self.h, rdm1, aobasis=False, mo_coeff=self.mo_coeff,
                                              F_int=self.F_int)
            if rdm1_2 is not None:
                ans2 = utilities.structure_factor(self.mol, self.h, rdm1_2.transpose(), aobasis=False,
                                                  mo_coeff=self.mo_coeff, F_int=np.conj(self.F_int))
                return list(ans1 * ans2), list(ans2)
            else:
                return list(ans1)

        else:
            raise NotImplementedError('The possible properties are: Ek, v1e, dip and F')


if __name__ == "__main__":
    # execute only if run as a script
    # test on water

    from pyscf import gto, scf
    import CCS

    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mf = scf.addons.convert_to_ghf(mf)
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mo_coeff_inv = np.linalg.inv(mo_coeff)
    mocc = mo_coeff[:, mo_occ > 0]
    mvir = mo_coeff[:, mo_occ == 0]
    nocc = mocc.shape[1]
    nvir = mvir.shape[1]
    dim = nocc + nvir

    # build simulated properties given ts and rs amplitudes
    nbr_os_states = 3
    exp_data = np.full((nbr_os_states, nbr_os_states), None)
    # amplitudes for GS (ts) and 2 ES (rs1, rs2)
    ts = np.random.random((nocc, nvir)) * 0.01
    rs1 = np.zeros_like(ts)
    rs1[0, 0] = 1.
    rs2 = np.zeros_like(ts)
    rs2[0, 3] = 1.
    # GS rdm1
    gamma_exp = CCS.gamma_unsym_CCS(ts, ts)
    exp_data[0, 0] = ['mat', gamma_exp]
    # ES 1 rdm1
    gamma_exp_es1 = CCS.gamma_es_CCS(ts, rs1, rs1, 0., 0.)
    Ek_11 = utilities.Ekin(mol, gamma_exp_es1, aobasis=False, mo_coeff=mo_coeff)
    dip_11 = utilities.dipole(mol, gamma_exp_es1, aobasis=False, mo_coeff=mo_coeff)
    v1e_11 = utilities.v1e(mol, gamma_exp_es1, aobasis=False, mo_coeff=mo_coeff)
    exp_data[1, 1] = [['Ek', Ek_11], ['dip', dip_11], ['v1e', v1e_11]]
    # tr-ES 1
    gamma_exp_tr_es1 = CCS.gamma_tr_CCS(ts, rs1, 0., 0., 0.)
    exp_data[0, 1] = ['dip', utilities.dipole(mol, gamma_exp_tr_es1, aobasis=False, mo_coeff=mo_coeff)]
    # tr-ES 2
    gamma_exp_tr_es2 = CCS.gamma_tr_CCS(ts, rs2, 0., 0., 0.)
    exp_data[0, 2] = ['dip', utilities.dipole(mol, gamma_exp_tr_es2, aobasis=False, mo_coeff=mo_coeff)]

    print()
    print('Check gamma traces (should be 0)')
    print(np.trace(gamma_exp) - 10., np.trace(gamma_exp_es1) - 10.)
    print(np.trace(gamma_exp_tr_es1), np.trace(gamma_exp_tr_es2))
    print()

    print()
    print('Exp data')
    print(np.asarray([['mat', 'dip', ''], ['', 'Ek, dip, v1e', ''], ['', '', 'dip']]))
    print()

    L = 0

    # initialize Vexp object
    myVexp = Exp(exp_data, mol, mo_coeff)

    print('#############################################')
    print(" |Aexp-Acalc|: Experimental potential and X2 ")
    print('#############################################')

    # calculated gamma: slightly different from gamma_exp
    rs1_calc = np.zeros_like(ts)
    rs2_calc = np.zeros_like(ts)
    rs1_calc[0, 0] = 0.98
    rs1_calc[0, 3] = np.sqrt(1 - 0.98 ** 2)
    rs2_calc[0, 0] = np.sqrt(1 - 0.98 ** 2)
    rs2_calc[0, 3] = 0.98
    gamma_calc = CCS.gamma_unsym_CCS(ts * 1.15, ts * 1.15)
    gamma_calc_es1 = CCS.gamma_es_CCS(ts * 1.15, rs1_calc, rs1_calc, 0., 0.)
    gamma_calc_tr_es1 = CCS.gamma_tr_CCS(ts * 1.15, rs1_calc, 0., 0., 0.)
    gamma_calc_tr_es2 = CCS.gamma_tr_CCS(ts * 1.15, rs2_calc, 0., 0., 0.)

    #######
    # GS
    #######

    V, X2, vmax = myVexp.Vexp_update(gamma_calc, (0, 0))

    print('--------')
    print('GS case')
    print('--------')
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

    V, X2, vmax = myVexp.Vexp_update(gamma_calc_es1, (1, 1))

    print('ES 1,1')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()

    V, X2, vmax = myVexp.Vexp_update(gamma_calc_es1, (0, 1))

    print('ES 0,1')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()

    V, X2, vmax = myVexp.Vexp_update(gamma_calc_tr_es2, (0, 2))

    print('ES 0,2')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()

    print('#####################################')
    print(" A**2: Experimental potential and X2 ")
    print('#####################################')
    print()

    # The value for X2 are large because the input data are Ek, v1e and dip
    # whereas the function calculates their norm squared

    print('---------')
    print('ES 0,1')
    print('---------')

    V, X2, vmax = myVexp.Vexp_update_norm2(gamma_calc_tr_es1, gamma_calc_tr_es1, (0, 1))
    print('X2=', X2)
