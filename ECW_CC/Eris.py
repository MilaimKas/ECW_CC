
from pyscf import cc,scf,ao2mo
import numpy as np

class PySCF_geris():
    '''
    physics eris PySCF class from GCC
    '''

    def __init__(self,gmf):
        '''

        :param gmf: HF PySCF class
        '''

        # convert to GHF
        if not isinstance(gmf, scf.ghf.GHF):
            gmf = scf.addons.convert_to_ghf(gmf)

        self.eris = cc.GCCSD(gmf).ao2mo(gmf.mo_coeff)
     
class geris():
    '''
    two electron integrals from PySCF ao2mo
    see pyscf.cc.gccsd._make_eris_incore function
    '''

    def __init__(self, mycc, int_thresh=10 ** -13, dir_cont=False):
        '''
        PySCF electron repulsion integrals in physics notation
        see cc.gccsd.eris

        <pq||rs> = <pq|rs> - <pq|sr>

        :param mycc: PySCF CC class
        :param int_thresh: threshold for the two electron integrals
        '''
        self.threshold = int_thresh

        # convert to GHF
        if not isinstance(mycc, cc.gccsd.GCCSD):
            mycc = cc.addons.convert_to_gccsd(mycc)

        eris = cc.gccsd._PhysicistsERIs()
        eris._common_init_(mycc)
        nocc = eris.nocc
        nao, nmo = eris.mo_coeff.shape  # in GCC nao = nmo = nbr of spin orbitals
        assert (eris.mo_coeff.dtype == np.double)
        # in GCC mo_coeff = [[mo_a],[mo_b]]
        mo_a = eris.mo_coeff[:nao // 2]
        mo_b = eris.mo_coeff[nao // 2:]
        orbspin = eris.orbspin

        # Assumes RCC -> GCC orbspin format [1,0,1,0,1,0, ...]
        if dir_cont:
            # eri_ao.shape = (nao, nao, nao, nao) --> full two electron integrals
            #eri_ao = mycc.mol.intor('int2e', aosym='s1')
            # spinor AO basis
            eri_ao = gto.getints('int2e', mol._atm, mol._bas, mol._env, aosym='s1')
            # convert mo to R format
            #mo = mo_a[:, ::2]
            mo = eris.mo_coeff
            # direct contraction between eri_ao and mo
            tmp1 = np.einsum('sd, abcd->sabc', mo, eri_ao)
            tmp2 = np.einsum('rc, sabc->rsab', mo, tmp1)
            tmp3 = np.einsum('qb, rsab->qrsa', mo.conj(), tmp2)
            eri = np.einsum('pa, qrsa->pqrs', mo.conj(), tmp3)
            del tmp1, tmp2, tmp3
            self.orbspin = orbspin
            # convert to G format
            #orb_mask = np.zeros((nmo,nmo,nmo,nmo), dtype=bool)
            #print(orb_mask.shape)
            #for p in range(nmo):
            #    for q in range(nmo):
            #        for r in range(nmo):
            #            for s in range(nmo):
            #                if p % 2 == 0 and q % 2 == 0 and r % 2 == 0 and s % 2 == 0:
            #                    orb_mask[p, q, r, s] = True
            #                elif p % 2 != 0 and q % 2 != 0 and r % 2 == 0 and s % 2 == 0:
            #                    orb_mask[p, q, r, s] = True
            #                elif p % 2 != 0 and q % 2 != 0 and r % 2 != 0 and s % 2 != 0:
            #                    orb_mask[p, q, r, s] = True
            #                elif p % 2 == 0 and q % 2 == 0 and r % 2 != 0 and s % 2 != 0:
            #                    orb_mask[p, q, r, s] = True
            #eri = np.zeros((nmo,nmo,nmo,nmo)).ravel()
            #eri[orb_mask.ravel()] = eri_R.ravel()
            #eri = eri.reshape(nmo,nmo,nmo,nmo)
            # make pqrs-pqsr
            eri = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
            low_values_flags = abs(eri) < self.threshold  # Where values are low
            eri[low_values_flags] = 0  # All low values set to 0

        # Construct MO integral in spin-orbital format
        else:
            # mol.intor('int2e', aosym='s8') -> AO in spatial format
            eri_ao = mycc._scf._eri
            # with or without orbspin gives same result
            if orbspin is None:
                eri = ao2mo.kernel(eri_ao, mo_a)   # alpha integral aa|aa
                eri += ao2mo.kernel(eri_ao, mo_b)  # beta integral bb|bb
                eri1 = ao2mo.kernel(eri_ao, (mo_a, mo_a, mo_b, mo_b))  # aa|bb integrals
                eri += eri1
                eri += eri1.T
            else:
                # mo_a.shape = (nao // 2, nao)
                mo = mo_a + mo_b
                # eri.shape = 2 (alpha and beta 2e int) -> contract mo_a and mo_b with eri
                eri = ao2mo.full(eri_ao, mo)
                # 8-fold symmetry
                if eri.size == nmo ** 4:  # if mycc._scf._eri is a complex array
                    sym_forbid = (orbspin[:, None] != orbspin).ravel()  # create 2x2 matrix of orbspin mask
                else:  # 4-fold symmetry
                    sym_forbid = (orbspin[:, None] != orbspin)[np.tril_indices(nmo)]
                # make integrals between alpha and beta 0
                eri[sym_forbid, :] = 0
                eri[:, sym_forbid] = 0
            self.orbspin = orbspin

            if eri.dtype == np.double:
                eri = ao2mo.restore(1, eri, nmo)  # transform eri in s1 symmetry -> shape = 1

            eri = eri.reshape(nmo, nmo, nmo, nmo)  # reshape into nmo (nbr of spin orb.)

            eri = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)  # make pqrs-pqsr
            low_values_flags = abs(eri) < self.threshold  # Where values are low
            eri[low_values_flags] = 0  # All low values set to 0

        self.fock = np.diag(eris.mo_energy)
        self.oooo = eri[:nocc, :nocc, :nocc, :nocc].copy()
        self.ooov = eri[:nocc, :nocc, :nocc, nocc:].copy()
        self.oovv = eri[:nocc, :nocc, nocc:, nocc:].copy()
        self.ovov = eri[:nocc, nocc:, :nocc, nocc:].copy()
        self.ovvo = eri[:nocc, nocc:, nocc:, :nocc].copy()
        self.ovvv = eri[:nocc, nocc:, nocc:, nocc:].copy()
        self.vvvv = eri[nocc:, nocc:, nocc:, nocc:].copy()
        self.vooo = eri[nocc:, :nocc, :nocc, :nocc].copy()
        self.vovo = eri[nocc:, :nocc, nocc:, :nocc].copy()
        self.oovo = eri[:nocc, :nocc, nocc:, :nocc].copy()
        self.vovv = eri[nocc:, :nocc, nocc:, nocc:].copy() 
        self.oovo = eri[:nocc, :nocc, nocc:, :nocc].copy() 
        self.vvoo = eri[nocc:, nocc:, :nocc, :nocc].copy()
        self.vvvo = eri[nocc:, nocc:, nocc:, :nocc].copy()
        self.oovo = eri[:nocc, :nocc, nocc:, :nocc].copy()
        self.voov = eri[nocc:, :nocc, :nocc, nocc:].copy()
        # additional
        self.ovoo = eri[:nocc, nocc:, :nocc, :nocc].copy()

        self.nocc = nocc



if __name__ == "__main__":

    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    #mol.atom = '''
    #H 0 0 0
    #H 0 0 1.
    #'''

    mol.basis = '6-31g*'
    mol.spin = 0
    mol.build()
    mfr = scf.RHF(mol)
    mfr.kernel()
    mo = mfr.mo_coeff
    mf = scf.addons.convert_to_ghf(mfr)
    mycc = cc.GCCSD(mf)

    print()
    print('----------------')
    print('Test ERIS class')
    print('----------------')
    print()

    eris_1 = geris(mycc)
    eris_2 = geris(mycc, dir_cont=True)

    print('orbspin')
    print(eris_1.orbspin)
    print()

    print('Some values for i -> a')
    nocc = mycc.nocc
    # beta -> beta transition in G format
    i = nocc-1
    a = 1
    print('ikik')
    ikik = np.einsum('ikik->i', eris_1.oooo)[i]
    print('eris 1', ikik)
    print('eris 2', np.einsum('ikik->i', eris_2.oooo)[i])
    print('akak')
    akak = np.einsum('akak->a', eris_1.vovo)[a]
    print('eris 1', akak)
    print('eris 2', np.einsum('akak->a', eris_2.vovo)[a])
    print('aiia')
    aiia = eris_1.voov[a, i, i, a]
    print('eris 1', aiia)
    print('eris 2', eris_2.voov[a, i, i, a])
    print('akik -> r0 term')
    akik = np.einsum('akik->ia', eris_1.vooo)[i, a]
    print('eris 1', akik)
    print('eris 2', np.einsum('akik->ia', eris_2.vooo)[i, a])
    print()
    print('Excitation energy')
    print('geris')
    print('Koopman', eris_1.fock[nocc+a, nocc+a]-eris_1.fock[i, i])
    print('CIS', eris_1.fock[nocc+a, nocc+a]-eris_1.fock[i, i]+aiia)

    print()
    print('Additional tests')
    print('mycc._scf._eri.shape')
    print(mycc._scf._eri.shape)
    print('apply ao2mo.restore --> into s1')
    print(ao2mo.restore('s1', mycc._scf._eri, mol.nao).shape)
    print('mol.intor s1')
    print(mol.intor('int2e', aosym='s1').shape)
    print('gto.getints s1')
    print(gto.getints('int2e', mol._atm, mol._bas, mol._env, aosym='s1' ).shape)

    print()
    print(' Test symmetries')
    print('o1o2v1v2 - (-o2o1v1v2) = ', np.max(np.subtract(eris_1.oovv, -1*eris_1.oovv.transpose(1, 0, 2, 3))))
    print('o1o2v1v2 - (-o1o2v2v1) = ', np.max(np.subtract(eris_1.oovv, -1 * eris_1.oovv.transpose(0, 1, 3, 2))))
    print('o1o2v1v2 - o2o1v2v1 = ', np.max(np.subtract(eris_1.oovv, eris_1.oovv.transpose(1, 0, 3, 2))))

