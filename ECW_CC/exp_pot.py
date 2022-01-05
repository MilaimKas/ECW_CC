"""
 Calculate Vexp potential and difference between exp and calc. from either rdm1_exp or one electron properties
"""


import numpy
import numpy as np
# from . import utilities
import utilities


class Exp:
    def __init__(self, L, exp_data, mol, mo_coeff, Ek_exp_GS=None):
        """
        Class containing the experimental potentials Vnm
        Vnm is a matrix containing the Vexp potential associated to the state nn or the transition potential 0n, n0
        In the present implementation only diagonal Vnn and off-diagonal V0n, Vn0 cases are account for,
        i.e only state properties and transition with GS.

        :param exp_data: list of n elements, where n=nbr of states. Each element contains the exp properties for
                         the given state.
                         exp_data = [[GS],[ES1],...,[ESn]]
                         GS, ESn = ['prop1', value], ['prop2', value]
                    prop1 = 'mat','Ek','dip','v1e','F', 'DEkn'
                             => 'mat': value = given rdm1 in MO basis G format
                             => 'trmat': value = tuple with given left and right tr_rdm1
                             => 'Ek': value = kinetic energy
                             => 'dip': value = mu_x, mu_y, mu_z dipole moment
                             => 'trdip': value = mu_x, mu_y, mu_z transition dipole moment
                             => 'F': value = [F, h, (a,b,c)] structure factor, miller index and reciprocal vector
                             => 'v1e': value = one electron potential
                             => 'DEKn': value = Ek difference between GS and ESn
        :param mol: PySCF molecule object
        :param mo_coeff: "canonical" MOs coefficients
        :param Ek_exp_GS: experimental Ek for the GS
        :param L: array of L values for each state and each set of properties
        """

        self.nbr_states = len(exp_data)  # total nbr of states: GS+ES
        self.exp_data = exp_data
        self.mo_coeff = mo_coeff
        self.mol = mol
        self.prop_calc = []

        # check L format
        self.L = self.L_check(L)

        self.charge_center = None

        # store necessary AOs and MOs integrals
        # ---------------------------------------

        # AOs integrals
        self.Ek_int = None
        self.dip_int = None
        self.v1e_int = None
        self.F_int = None
        # dictionary containing needed MO integrals (Ai_pq matrix)
        self.dic_int = {}

        self.prop_names = []  # list of prop 'name' for each state

        # Build check list and store MOs int
        for i in range(len(exp_data)):

            self.prop_names.append([])

            for prop in exp_data[i]:

                # structure factor
                if prop[0] == 'F':
                    if len(prop) < 4:
                        raise SyntaxError('If Structure factors are to be calculated, '
                                          'Structure factors, Miller indices and reciprocal '
                                          'vectors must be given in exp_data: '
                                          '['F', F, h, rec_vec]')
                    # F_mo.shape = (nbr_h_pts, dim, dim)
                    if self.F_int is None:
                        self.dic_int['F'], self.F_int = utilities.FT_MO(mol, prop[2], mo_coeff, prop[3])
                    self.h = prop[2]
                    self.rec_vec = prop[3]

                # dipole integrals for "dip" or "trdip"
                if ('dip' in prop[0] or 'trdip' in prop[0]) and self.dip_int is None:
                    charges = mol.atom_charges()
                    coords = mol.atom_coords()
                    # store common charge center for further calculation
                    self.charge_center = np.einsum('z,zr->r', charges, coords) / charges.sum()
                    # calculate integral -> 3 components
                    with mol.with_common_orig(self.charge_center):
                        self.dip_int = mol.intor_symmetric('int1e_r', comp=3)  # AO basis
                        self.dic_int['dip'] = utilities.convert_aoint(self.dip_int, mo_coeff)  # MO spin-orbital basis

                # coulomb integrals
                if 'v1e' in prop[0] and self.v1e_int is None:
                    self.v1e_int = mol.intor_symmetric('int1e_nuc')
                    v1e_int_mo = utilities.convert_aoint(self.v1e_int, mo_coeff)
                    self.dic_int['v1e'] = v1e_int_mo

                # Kinetic integrals Ek or DEk
                if 'Ek' in prop[0] and self.Ek_int is None:
                    self.Ek_int = mol.intor_symmetric('int1e_kin')
                    Ek_int_mo = utilities.convert_aoint(self.Ek_int, mo_coeff)
                    self.dic_int['Ek'] = Ek_int_mo

                self.prop_names[i].append(prop[0])

        # store idx of DEk weight for the GS
        self.DEk_GS_idx = None
        for i in range(len(self.prop_names[0])):
            if 'DEk' in self.prop_names[0][i]:
                self.DEk_GS_idx = i

        # initialize Ek_calc_GS
        # -------------------------------------------------
        self.Ek_exp_GS = Ek_exp_GS
        self.Ek_calc_GS = None
        self.Delta_Ek_GS = None

        # initialize Vexp potential
        # ------------------------------------------------
        self.Vexp = np.full((self.nbr_states, self.nbr_states), None)

    def Vexp_update(self, rdm1, rdm1_add, index, L=None):
        """
        Update the Vexp[index] element of the Vexp matrix for given rdm1
        and calculates the relative difference Delta.

        math:

        For target = rdm1 and GS prop (norm)
        Vexp^nm  = 2/M sum_i^{M} (|Aexp_i-sum_pq gamma^nm_pq * Ai_pq|)/sig_i * Ai_rs

        For target = tr_rdm1, trdip or ES prop (norm squared)
        Vexp^nm = 2/M sum_i^{M} (|Aexp_i-sum_pquv gamma^mn_pq * gamma^nm_vu * Ai_pq * Ai_uv.conj |)/sig_i
                   * Ai_rs.conj * sum_pq gamma^mn_pq Ai_pq

        Delta = sum_ij |gamma_exp_ij - gamma_calc_ij| / (sum_ij |gamma_exp_ij|
         or sum_i|A_exp,i-A_calc,i|/A_exp,i

        :param rdm1: calculated rdm1 or left(mn) or right(nm) tr_rdm1 in MO basis
        :param rdm1_add: additional rdm1. Can be left or right tr_rdm1 or rdm1 needed for DEk
        :param L: exp weight, must be given for each state and set of prop
        :param index: nm index of the potential Vexp
             -> index = (0,0) for GS
             -> index = (n,n) for prop. of excited state n
             -> index = (0,n) and (n,0) for left and right transition prop. of excited state n
        :return:
        """

        n, m = index

        # initialize
        self.Vexp[n, m] = np.zeros_like(rdm1)
        Delta = 0.
        vmax = 0.
        self.prop_calc = []

        # experimental weight
        if L is None:
            L = self.L
        else:
            L = self.L_check(L)

        # check if prop, list of prop or matrix comparison
        # -> st_idx = index to retrieve the name of the property from given state
        st_idx = np.max(index)

        i = 0  # loop index

        # Loop over properties of the given state n or m
        # Update corresponding Vexp element

        for prop in self.prop_names[st_idx]:

            if prop == 'mat':

                # ---------------- given GS or ES exp rdm1 ---------------------------

                if index == (0, 0):
                    diff = np.subtract(self.exp_data[0][i][1], rdm1)
                    self.Vexp[0, 0] += L[st_idx][i] * diff
                    Delta += np.sum(abs(diff))/np.sum(abs(self.exp_data[0][i][1]))
                    vmax += np.max(abs(diff))
                    # calculate kinetic energy Ek_calc
                    if self.Ek_exp_GS is not None:
                        self.Ek_calc_GS = utilities.Ekin(self.mol, rdm1, aobasis=False,
                                                         mo_coeff=self.mo_coeff, ek_int=self.Ek_int, g=True)
                        # store Delta value for Ek
                        self.Delta_Ek_GS = np.abs(self.Ek_exp_GS - self.Ek_calc_GS)/np.abs(self.Ek_exp_GS)

                elif n == m:
                    diff = np.subtract(self.exp_data[n][i][1], rdm1)
                    self.Vexp[n, n] += L[st_idx][i] * diff
                    Delta += np.sum(abs(diff))/(np.sum(abs(self.exp_data[n][i][1])))
                    vmax += np.max(abs(diff))

            # ------------ given ES left and right exp tr_rdm1 ------------------

            if prop == 'trmat' and n != m:
                # todo: verify definition of Delta
                if n == 0:  # left
                    diff = np.subtract(self.exp_data[st_idx][i][1][0], rdm1)
                    self.Vexp[n, m] += (L[st_idx][i]) * diff
                elif m == 0:  # right
                    diff = np.subtract(self.exp_data[st_idx][i][1][1], rdm1)
                    self.Vexp[n, m] += (L[st_idx][i]) * diff
                else:
                    raise ValueError("Only transition properties between GS and ES are implemented: m or n must be = 0")
                avg = sum(abs(self.exp_data[st_idx][i][1][1]))+sum(abs(self.exp_data[st_idx][i][1][0]))
                Delta += np.sum(abs(diff))/(avg/2.)
                vmax += np.max(abs(diff))

            # ------------------- properties Aexp,i -----------------------

            # Kinetic energy
            if (prop == 'Ek' or prop == 'v1e') and n == m:
                calc_prop = self.calc_prop(prop, rdm1)  # use calc_prop function sum_pq rdm_pq*Aint_pq
                exp_prop = self.exp_data[st_idx][i][1]
                diff = np.abs(exp_prop - calc_prop) * self.dic_int[prop]
                self.Vexp[n, n] += L[st_idx][i] * diff
                Delta += np.abs(exp_prop - calc_prop)/np.abs(exp_prop)
                vmax += np.max(abs(diff))
                self.prop_calc.append([prop, calc_prop])

            # Kinetic energy difference
            if 'DEk' in prop and n == m and n != 0:
                # only update when calling for the ES
                Drdm1 = np.subtract(rdm1, rdm1_add)  # ESn_rdm1 - GS_rdm1
                calc_prop = self.calc_prop('Ek', Drdm1)  # use calc_prop function sum_pq rdm_pq*Aint_pq
                exp_prop = self.exp_data[st_idx][i][1]
                v_tmp = np.abs(exp_prop - calc_prop) * self.dic_int['Ek']
                # update excited states Vexp with corresponding L
                self.Vexp[n, m] += L[st_idx][i] * v_tmp
                # update ground state Vexp with corresponding L
                if self.DEk_GS_idx is not None:
                    self.Vexp[0, 0] += L[0][self.DEk_GS_idx] * v_tmp
                else:  # if L not given for GS DEk, use ES value
                    self.Vexp[0, 0] += L[st_idx][i] * v_tmp
                Delta += np.abs(exp_prop - calc_prop)/np.abs(exp_prop)
                vmax += np.max(np.abs(v_tmp))
                self.prop_calc.append([prop, calc_prop])

            # dipole moment
            if prop == 'dip' and n == m:
                calc_prop = self.calc_prop('dip', rdm1)  # use calc_prop function sum_pq rdm_pq*Aint_pq
                exp_prop = self.exp_data[st_idx][i][1]
                delta = 0.
                for d_calc, d_exp, d_int in zip(calc_prop, exp_prop, self.dic_int[prop]):
                    diff = np.abs(d_exp - d_calc) * d_int
                    self.Vexp[n, m] += L[st_idx][i] * diff
                    delta += np.abs(d_exp - d_calc)
                    vmax += np.max(np.abs(diff))
                # delta /= 3.
                Delta += delta/(np.sum(np.abs(exp_prop)))
                self.prop_calc.append([prop, calc_prop])

            # transition dipole moment
            if prop == 'trdip' and n != m:
                calc_prop, A_scale = self.calc_prop('dip', rdm1, rdm1_add=rdm1_add)
                exp_prop = self.exp_data[st_idx][i][1]
                # self.Vexp[n, m] = np.zeros_like(rdm1)
                delta = 0.
                for d_calc, d_exp, j, A in zip(calc_prop, exp_prop, [0, 1, 2], A_scale):
                    diff = np.abs(d_exp - d_calc) * self.dic_int['dip'][j] * A
                    self.Vexp[n, m] += L[st_idx][i] * diff
                    delta += np.abs(d_exp - d_calc)
                    vmax += np.max(np.abs(diff))
                # delta /= 3.
                Delta += delta/(np.sum(np.abs(exp_prop)))
                self.prop_calc.append([prop, calc_prop])

            # structure factor
            if prop == 'F' and n == m:
                calc_prop = utilities.structure_factor(self.mol, self.h, rdm1,
                                                       aobasis=False, mo_coeff=self.mo_coeff,
                                                       F_int=self.F_int, rec_vec=self.rec_vec)
                exp_prop = self.exp_data[st_idx][i][1]
                # x2 = 0.
                delta = 0.
                for F_exp, F_calc, F_int_mo in zip(exp_prop, calc_prop, self.dic_int[prop]):
                    diff = np.abs((F_exp - F_calc)) * F_int_mo
                    self.Vexp[n, n] += L[st_idx][i] * (2. / (len(self.h))) * diff
                    # x2 += np.abs(F_exp - F_calc) ** 2
                    delta += np.abs(F_exp - F_calc)/np.abs(F_exp)
                    vmax += np.max(np.abs(diff))
                # X2 += x2
                Delta += delta
                self.prop_calc.append([prop, calc_prop])

            i += 1

        return Delta, vmax

    def calc_prop(self, prop, rdm1, g_format=True, rdm1_add=None):
        """
        Calculate A**2 and/or A using given rdm1

        :param g_format: True if rdm1 given in Generalized spin-orbit format
        :param prop: one-electron prop to calculate -> 'Ek', 'v1e' or 'dip'
        :param rdm1: reduced one body density matrix in MO basis in G format
        :param rdm1_add: left rdm1, if given, the norm squared of the prop is calculated using both right and left rdm1
                        where rdm1_2 is the rdm1^mn if Vnm
        :return: calculated one-electron property (A**2 and/or A)
        """

        if prop == 'Ek':
            ans1 = utilities.Ekin(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                  ek_int=self.Ek_int)
            if rdm1_add is not None:
                ans2 = utilities.Ekin(self.mol, rdm1_add.transpose(), g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                      ek_int=np.conj(self.Ek_int))
                return ans1 * ans2, ans2
            else:
                return ans1

        elif prop == 'v1e':
            ans1 = utilities.v1e(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                 v1e_int=self.v1e_int)
            if rdm1_add is not None:
                ans2 = utilities.v1e(self.mol, rdm1_add.transpose(), g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                     v1e_int=np.conj(self.v1e_int))
                return ans1 * ans2, ans2
            else:
                return ans1

        elif prop == 'dip':
            ans1 = utilities.dipole(self.mol, rdm1, g=g_format, aobasis=False, mo_coeff=self.mo_coeff,
                                    dip_int=self.dip_int)
            if rdm1_add is not None:
                ans2 = utilities.dipole(self.mol, rdm1_add.transpose(), g=g_format, aobasis=False,
                                        mo_coeff=self.mo_coeff, dip_int=np.conj(self.dip_int))
                return list(ans1 * ans2), list(ans2)
            else:
                return list(ans1)

        else:
            raise NotImplementedError('The possible properties are: Ek, v1e and dip')

    def print_cube(self):
        """
        Function to print a cube file of the given Vexp potential

        :return:
        """

        raise NotImplementedError

    def L_check(self, L):
        """
        Check the format of the given experimental weight L

        :param L:
        :return:
        """

        # if only one value is given, all prop are weighted with this value
        if isinstance(L, (float, int)):
            L_list = []
            for st in self.exp_data:
                L_list.extend([[float(L)]*len(st)])
            return L_list
        # if array or list is given, check the size and shape
        elif isinstance(L, (list, np.ndarray)):
            if len(L) != self.nbr_states:
                raise SyntaxError('Given constrain weight length does not equal the number of states. '
                                  'You might have forgotten to put L_loop = True.')
            # if only 1 value of L per states, all prop are wighted equally
            i = 0
            for st, l in zip(self.exp_data, L):
                if len(st) != len(l) and len(l) == 1:
                    print('Warning: all properties for state {} will be wighted equally '.format(i))
                    for j in range(len(st)-1):
                        L[i].extend(l[0])
                elif len(st) != len(l):
                    raise SyntaxError("Wrong syntax for L list")
                i += 1

            return L


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

    # HF calculation
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
    nbr_states = 3  # total nbr of states GS and ES
    exp_data = []
    # amplitudes for GS (ts) and 2 ES (rs1, rs2)
    ts = np.random.random((nocc, nvir)) * 0.01
    ls = ts.copy()
    rs1 = np.zeros_like(ts)
    rs1[0, 0] = 1.
    ls1 = rs1.copy()
    rs2 = np.zeros_like(ts)
    rs2[0, 3] = 1.
    ls2 = rs2.copy()
    # GS rdm1
    gamma_exp = CCS.gamma_unsym_CCS(ts, ts)
    Ek_GS = utilities.Ekin(mol, gamma_exp, aobasis=False, mo_coeff=mo_coeff)
    # rdm1
    gamma_exp_es1 = CCS.gamma_es_CCS(ts, ls1, None, None, 0.)
    gamma_exp_es2 = CCS.gamma_es_CCS(ts, ls1, None, None, 0.)
    # tr rdm1
    tr_gamma_exp_es1_r = CCS.gamma_tr_CCS(ts, ls1, None, None, 0.)
    tr_gamma_exp_es1_l = CCS.gamma_tr_CCS(ts, ls, rs1, 0., 1.)
    tr_gamma_exp_es2_r = CCS.gamma_tr_CCS(ts, ls2, None, None, 0.)
    tr_gamma_exp_es2_l = CCS.gamma_tr_CCS(ts, ls, rs2, 0., 1.)
    # ES1 prop
    Ek_1 = utilities.Ekin(mol, gamma_exp_es1, aobasis=False, mo_coeff=mo_coeff)
    tr_dip_1 = utilities.dipole(mol, tr_gamma_exp_es1_r, aobasis=False, mo_coeff=mo_coeff)
    tr_dip_1 *= utilities.dipole(mol, tr_gamma_exp_es1_l, aobasis=False, mo_coeff=mo_coeff)
    v1e_1 = utilities.v1e(mol, gamma_exp_es1, aobasis=False, mo_coeff=mo_coeff)
    exp_data.append([['mat', gamma_exp], ['DEk1', np.abs(Ek_1 - Ek_GS)]])  # GS prop
    exp_data.append([['DEk1', Ek_1-Ek_GS], ['trdip', tr_dip_1], ['v1e', v1e_1]])  # ES1 prop
    # ES2 prop
    Ek_2 = utilities.Ekin(mol, gamma_exp_es2, aobasis=False, mo_coeff=mo_coeff)
    tr_dip_2 = utilities.dipole(mol, tr_gamma_exp_es2_r, aobasis=False, mo_coeff=mo_coeff)
    tr_dip_2 *= utilities.dipole(mol, tr_gamma_exp_es2_l, aobasis=False, mo_coeff=mo_coeff)
    exp_data.append([['Ek', Ek_2], ['trdip', tr_dip_2]])

    print()
    print('Check gamma traces (should be 0)')
    print(np.trace(gamma_exp) - 10., np.trace(gamma_exp_es1) - 10.)
    print(np.trace(tr_gamma_exp_es1_r), np.trace(tr_gamma_exp_es2_l))
    print()

    L = [[0.1, 0.1], [0.04, 0.07, 0.06], [0.05, 0.01]]

    # initialize Vexp object
    myVexp = Exp(L, exp_data, mol, mo_coeff)

    # calculated gamma: slightly different from gamma_exp
    rs1_calc = np.zeros_like(ts)
    rs2_calc = np.zeros_like(ts)
    rs1_calc[0, 0] = 0.98
    rs1_calc[0, 3] = np.sqrt(1 - 0.98 ** 2)
    rs2_calc[0, 0] = np.sqrt(1 - 0.98 ** 2)
    rs2_calc[0, 3] = 0.98
    gamma_calc = CCS.gamma_unsym_CCS(ts * 1.15, ts * 1.15)
    gamma_calc_es1 = CCS.gamma_es_CCS(ts * 1.15, rs1_calc, None, None, 0.)
    gamma_calc_es2 = CCS.gamma_es_CCS(ts * 1.15, rs2_calc, None, 0., 0.)
    tr_gamma_calc_es1_l = CCS.gamma_tr_CCS(ts * 1.15, rs1_calc, None, None, 0.)
    tr_gamma_calc_es2_l = CCS.gamma_tr_CCS(ts * 1.15, rs2_calc, None, None, 0.)
    tr_gamma_calc_es1_r = CCS.gamma_tr_CCS(ts * 1.15, ls * 1.15, rs1_calc, 0., 1.)
    tr_gamma_calc_es2_r = CCS.gamma_tr_CCS(ts * 1.15, ls * 1.15, rs2_calc, 0., 1.)

    #############
    # update
    #############

    print()
    print('------------')
    print('List of prop')
    print('------------')
    print()
    print(myVexp.prop_names)
    print()

    print('--------')
    print('GS case')
    print('--------')
    X2, vmax = myVexp.Vexp_update(gamma_calc, gamma_calc, (0, 0))
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()

    print('--------')
    print('ES case')
    print('--------')

    X2, vmax = myVexp.Vexp_update(gamma_calc_es1, gamma_calc, (1, 1))
    print('ES 1,1')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()
    X2, vmax = myVexp.Vexp_update(tr_gamma_calc_es1_l, tr_gamma_calc_es1_r, (0, 1))
    print('ES 0,1')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()

    X2, vmax = myVexp.Vexp_update(gamma_calc_es2, gamma_calc_es2, (2, 2))
    print('ES 2,2')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()
    X2, vmax = myVexp.Vexp_update(tr_gamma_calc_es1_l, tr_gamma_calc_es2_r, (0, 2))
    print('ES 0,2')
    print('Vexp')
    print('X2, vmax =', X2, vmax)
    print()

