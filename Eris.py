
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
        if not isinstance(gmf,scf.ghf.GHF):
            gmf = scf.addons.convert_to_ghf(gmf)

        self.eris = cc.GCCSD(gmf).ao2mo(gmf.mo_coeff)

class geris():
    '''
    two electron integrals from PySCF ao2mo
    see pyscf.cc.gccsd._make_eris_incore function
    '''

    def __init__(self,mycc, threshold=10**-8):
        '''
        PySCF electron repulsion integrals in physics notation
        see cc.gccsd.eris

        <pq||rs> = <pq|rs> - <pq|sr>

        :param mycc: PySCF CC class
        :param threshold: threshold for the two electron integrals
        '''

        self.threshold = threshold

        # convert to GHF
        if not isinstance(mycc, cc.gccsd.GCCSD):
            mycc = cc.addons.convert_to_gccsd(mycc)

        eris = cc.gccsd._PhysicistsERIs()
        eris._common_init_(mycc)
        nocc = eris.nocc
        nao, nmo = eris.mo_coeff.shape
        assert (eris.mo_coeff.dtype == np.double)
        mo_a = eris.mo_coeff[:nao // 2]
        mo_b = eris.mo_coeff[nao // 2:]
        orbspin = eris.orbspin

        if orbspin is None:
            eri = ao2mo.kernel(mycc._scf._eri, mo_a)
            eri += ao2mo.kernel(mycc._scf._eri, mo_b)
            eri1 = ao2mo.kernel(mycc._scf._eri, (mo_a, mo_a, mo_b, mo_b))
            eri += eri1
            eri += eri1.T
        else:
            mo = mo_a + mo_b
            eri = ao2mo.kernel(mycc._scf._eri, mo)
            if eri.size == nmo ** 4:  # if mycc._scf._eri is a complex array
                sym_forbid = (orbspin[:, None] != orbspin).ravel()
            else:  # 4-fold symmetry
                sym_forbid = (orbspin[:, None] != orbspin)[np.tril_indices(nmo)]
            eri[sym_forbid, :] = 0
            eri[:, sym_forbid] = 0

        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

        eri = eri.reshape(nmo, nmo, nmo, nmo)
        eri = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
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

    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()
    mf = scf.GHF(mol)
    mf.kernel()
    mycc = cc.GCCSD(mf)

    print()
    print('----------------')
    print('Test ERIS class')
    print('----------------')
    print()

    pyscf_eris = PySCF_geris(mf)
    eris = geris(mycc)

    print('oovv difference')
    print(np.sum(np.subtract(pyscf_eris.eris.oovv,eris.oovv)))
    print('oooo difference')
    print(np.sum(np.subtract(pyscf_eris.eris.oooo, eris.oooo)))
    print('ovvv difference')
    print(np.sum(np.subtract(pyscf_eris.eris.ovvv, eris.ovvv)))