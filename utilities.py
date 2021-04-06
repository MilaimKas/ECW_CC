#!/usr/bin/python
# -*- coding: utf-8 -*-

###################################################################
#
# ECW-CCS v1
# -----------
# Experimentaly constrained wave function coupled cluster single
# ---------------------------------------------------------------
#
# Tools:
# - get closest ci coefficient
# - Print Natural orbitals using CC one particle rdm1
# - Print Cube file: electron density on a grid
# - Calculation of Ekin and V1e from rdm1
# - convert G rdm1 into R and U
#
###################################################################

import pyscf
from pyscf import scf, gto, tdscf,cc
from pyscf.tools import cubegen
from pyscf.tools import molden
import scipy
import numpy as np

#######################################
# L1 regularization related functions
#######################################

def subdiff(eq,var,alpha,thres=10**-8, R_format=False):

    '''
    Calculates the sub-gradient value of a functional
    see equations (17) in Ivanov et al, Molecular Physics, 115(21–22)

    :param eq: Jacobian matrix of the functional (T,L equations) given in amp format
    :param var: matrix of variables (t or l amplitudes) given in amp format
    :param alpha: L1 threshold
    :return: subdifferential W in amp format
    '''
    
    # check shape
    if eq.shape != var.shape:
        raise ValueError('equations and variables matrices must have the same shape')

    # transform in R format
    if R_format:
        eq = convert_g_to_r_amp(eq)
        var = convert_g_to_r_amp(var)

    # initialize sub-gradient W
    dW = np.zeros_like(eq)

    # check for non zero elements in var
    ind = np.argwhere(np.abs(var) > thres)
    for ix in ind:
        dW[tuple(ix)] = eq[tuple(ix)]+alpha*np.sign(var[tuple(ix)])

    # zero elements in var
    ind = np.argwhere(var <= thres)
    for ix in ind:
        ix = tuple(ix)
        if eq[ix] < -alpha:
            dW[ix] = eq[ix]+alpha
        elif eq[ix] > alpha:
            dW[ix] = eq[ix]-alpha
        else:
            dW[ix] = 0.

    # back transform into G format
    if R_format:
        dW = convert_r_to_g_amp(dW)

    return dW

def prox_l1(x_J, alpha):
    '''
    proximal point mapping method of the L1 regularization approach

    :param x_J: matrix of parameters
    :param alpha: proximal point factor
    :return: matrix of the proximal point function dh
    '''

    dim1, dim2 = x_J.shape
    ans = np.zeros((dim1, dim2))
    # sign_tJ = np.sign(t_J)
    for i in range(dim1):
        for j in range(dim2):
            if x_J[i, j] > alpha:
                ans[i][j] = x_J[i][j] - alpha
            elif x_J[i][j] < -alpha:
                ans[i][j] = x_J[i][j] + alpha
            else:
                ans[i][j] = 0
    return ans

###############################################
# Use TDHF methods to get initial r amplitudes
###############################################

def get_init_r(mol, roots=10):

    import scipy.spatial.distance

    '''
    Performs TDHF calculation and extract coeff for initial r value
    :param mol: mol object
    :param roots: number of states to print
    '''

    # RHF calc
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    #mtda = tdscf.rhf.TDA(mf)
    #mtda.nroots = roots_max
    #e,ci = mtda.kernel()

    # TDHF calc
    mtdhf = tdscf.TDHF(mf)
    mtdhf.nroots = roots
    mtdhf.kernel()

    # calculate transition dipole moments
    tdms = mtdhf.transition_dipole()
    r_ini = mtdhf.xy[:][0]

    return r_ini, tdms

#############################################
# Functions to deal with G, R and U format
#############################################

def convert_r_to_g_amp(amp):
    # todo: sign issue with R <-> G conversion
    '''
    Converts amplitudes in restricted format into generalized, spin-orbital format
    amp must be given in nocc x nvir shape
    NOTE: PySCF as several functions to perform the conversion within CC, CI or EOM classes: spatial2spin()

    :param amp: amplitudes in restricted format
    :return: amp in generalized spin-orb format
    '''

    if amp.ndim == 2:
        g_amp = np.zeros((amp.shape[0]*2,amp.shape[1]*2))
        for i in range(amp.shape[0]):
            for j in range(amp.shape[1]):
                a = amp[i,j]
                g_amp[i*2:i*2+2,j*2:j*2+2] = np.diag(np.asarray([a,a]))
    elif amp.ndim == 4:
        g_amp = cc.addons.spatial2spin(amp)

    #else:
    #    raise ValueError('amp must be either dim 2 or 4')

    return g_amp

def convert_g_to_r_amp(amp):
    # todo: sign issue with R <-> G conversion
    '''
    Converts G format amplitudes into R format

    :param amp:CC single or double amplitudes
    :return:
    '''

    if amp.ndim == 2:
       tmp = np.delete(amp, np.s_[1::2], 0)
       r_amp = np.delete(tmp, np.s_[1::2], 1)
    elif amp.ndim == 4:
        dim = amp.shape[0]+amp.shape[2]
        orbspin = np.zeros(dim, dtype=int)
        orbspin[1::2] = 1
        r_amp = cc.addons.spin2spatial(amp, orbspin)[1] # t2ab

    else:
        raise ValueError('amp dimension must be 2 or 4')

    return r_amp

def convert_g_to_ru_rdm1(rdm1_g):
    '''
    Transform generalised rdm1 to R and U rdm1

    :param rdm1_g: one-electron reduced density matrix in AOs basis
    :return: rdm1 in R and U format
    '''

    nao = rdm1_g.shape[0]//2

    rdm_a = rdm1_g[:nao,:nao]
    rdm_b = rdm1_g[nao:,nao:]

    rdm_u = (rdm_a,rdm_b)

    rdm_r = rdm_a + rdm_b

    return rdm_r,rdm_u

def convert_u_to_g_rdm1(rdm_u):
    """
    convert U rdm1 to G rdm1

    :param rdm_u: unrestricted format rdm1 in AOs basis
    :return:
    """

    nao, nao = rdm_u[0].shape

    rdm_g = np.zeros((nao * 2, nao * 2))
    rdm_g[::2, ::2] = rdm_u[0]
    rdm_g[1::2, 1::2] = rdm_u[1]

    return rdm_g

def convert_r_to_g_rdm1(rdm_r):
    """
    convert R rdm1 to G rdm1

    :param rdm_r: restricted format rdm1 in AOs basis
    :return:
    """

    nao, nao = rdm_r.shape

    rdm_g = np.zeros((nao * 2, nao * 2))
    rdm_g[::2, ::2] = rdm_r
    rdm_g[1::2, 1::2] = rdm_r

    return rdm_g


#####################
# Quantum chemistry
#####################

def cis_rdm1(c1):
    '''
    Calculates the cis rdm1 given a set of c1 coefficients

    :param c1: CIS or TDA coeff
    :return: oo and vv contribution to RHF MO rdm1
    '''
    doo  =2-np.einsum('ia,ka->ik', c1.conj(), c1)
    dvv  = np.einsum('ia,ic->ac', c1, c1.conj())
    return doo, dvv

def ao_to_mo(rdm1_ao, mo_coeff):
    '''
    transform given rdm1 from AOs to MOs basis

    :param rdm1_ao: one-particle reduced density matrix given in AOs basis
    :param mo_coeff: MOs coefficients
    :return: rdm1 in MOs basis
    '''

    # check dimension
    if rdm1_ao.shape != mo_coeff.shape:
        raise ValueError('Rdm1 and MOs coefficients must have the same dimension')
    mo_coeff_inv = np.linalg.inv(mo_coeff)
    rdm1_mo = np.einsum('pi,ij,qj->pq', mo_coeff_inv, rdm1_ao, mo_coeff_inv.conj())
    
    return rdm1_mo

def mo_to_ao(rdm1_mo, mo_coeff):
    '''
    transform given rdm1 from MOs to AOs basis

    :param rdm1_mo: one-particle reduced density matrix given in MOs basis
    :param mo_coeff: MOs coefficients
    :return: rdm1 in AOs basis
    '''

    rdm1_ao = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1_mo, mo_coeff.conj())

    return rdm1_ao

def get_norm(rs, ls, r0=0, l0=0):
    '''
    Return the normalization factor between two sets of amplitudes

    :param rs: set of amplitudes
    :param ls: set of amplitudes
    :return:
    '''

    # check shape
    if rs.shape != ls.shape:
        raise ValueError('Shape of both set of amplitudes must be the same')

    C = abs(l0*r0)+np.sum(abs(rs*ls))

    return C

def ortho(mol,cL,cR):
    '''
    Orthogonalize two vectors using singular value decomposition
    See Molecular Physics, 105(9), 1239–1249. https://doi.org/10.1080/00268970701326978

    :param mol: PySCF mol object for AOs overlap matrix or overlap matrix
    :param cL,cR: set of MOs coefficients for bra (Left) and ket (Right) vectors
    :return:
    '''

    # Get AOs overlap
    if isinstance(mol, gto.Mole):
        S_AO = mol.intor('int1e_ovlp')
    elif isinstance(mol, np.ndarray):
        S_AO = mol
    else:
        raise ValueError('AOs overlap must be a ndarray or a PySCF Mole class')

    # check shape
    if S_AO.shape != cL.shape:
        raise ValueError('MOs coefficients and AO overlap matrix must be the same size')

    # Build MOs overlap matrix
    S = np.einsum('mp,nq,mn->pq',cL.conj(),cR,S_AO)

    # perform svd
    u,S_sv,v = np.linalg.svd(S)

    # build transformation matrices TL and TR
    S_sv = np.sqrt(np.linalg.inv(np.diag(S_sv)))
    TL = np.dot(u,S_sv)
    TR = np.dot(v.conj().T,S_sv)

    # transform L and R basis
    newcL = np.dot(cL,TL)
    newcR = np.dot(cR,TR)

    return newcL, newcR

def check_ortho(ln, rn, l0n, r0n, thres_ortho=10**-2,S_AO=None):
    '''
    Check the norm for a list of r and l vectors

    :param ln,rn: list of r and l vectors
    :param l0n,r0n: list of r0 and l0 value
    :param thres_ortho: threshold for C=<rk|ln>
    :param S_AO: either PySCF.mol object or AO overlap matrix
    :return:
    '''

    nbr_of_states = len(rn)

    # check length
    if nbr_of_states != len(ln):
        raise ValueError('r and l list of vectors must be the same length')

    # initialize matrix of norm
    C_norm = np.zeros((nbr_of_states,nbr_of_states))

    for k in range(nbr_of_states):
        for l in range(nbr_of_states):
            C_norm[k, l] = get_norm(rn[k], ln[l], r0=r0n[k], l0=l0n[l])
            # otthogonalize vectors
            # todo: here a set of vectors have to be orthogonalize! Not MOs
            #if l != k and C_norm[k, l] > thres_ortho:
            #    if S_AO is not None:
            #        ln[l], rn[k] = ortho(S_AO,ln[l],rn[k])
            #        C_norm[k, l] = get_norm(rn[k], ln[l], r0=r0n[k], l0=l0n[l])

    return C_norm

def koopman_init_guess(mo_energy,mo_occ,nstates=1):
    '''
    Generates list of koopman guesses for r1 vectors in G format
    The guess is obtained in the r format to avoid breaking symmetry

    :param mo_energy: MOs energies
    :param: mo_occ: occupation array
    :param nstates: number of states
    :return: list of r_ini and koopman's excitation
    '''

    # convert to R format
    mo_energy = mo_energy[0::2]
    mo_occ = mo_occ[0::2]
    occidx = np.where(mo_occ == 1)[0]
    viridx = np.where(mo_occ == 0)[0]
    nocc = occidx.shape[0]
    nvir = viridx.shape[0]
    e_ia = mo_energy[viridx] - mo_energy[occidx, None]

    nov = e_ia.size
    if nstates > nov:
        raise Warning('The size of the basis is smaller than the number of requested states')
    nroot = min(nstates, nov)
    x0 = [] # np.zeros((nroot, nov))
    DE = []
    e_ia = e_ia.ravel()
    idx = np.argsort(e_ia)

    for i in range(nroot):
        tmp = np.zeros(nov)
        tmp[idx[i]]   = 1
        tmp = tmp.reshape((nocc,nvir))
        tmp = convert_r_to_g_amp(tmp)*0.5
        x0.append(tmp)  # Koopmans' excitations
        DE.append(e_ia[idx[i]])

    return x0, DE

def tdm_slater(TcL, TcR, occ_diff):
    '''
    Express a bi-orthogonal transition density matrix in AOs basis

    math:
    <Tphi_L|Tphi_R> = delta_{pq}
    gamma_ao = TcL Tgamma (TcR)^dag
    see Werner 2007 DOI:10.1080/00268970701326978

    :param TcL: left transformed orbital (T=tilde)
    :param TcR: right transformed orbital (T=tilde)
    :param occ_diff: occupation difference in MO spin-orbitals basis between the two Slater states
           --> ex: a single excitation i->a for i=2 and a=5, occ_diff = [0,0,1,0,0,1,0]
    :return: transition density matrix in AOs basis
    '''
    
    Tgamma = np.diag(occ_diff)
    gamma_ao = np.einsum('pi,ij,qj->pq',TcL,Tgamma,TcR.conj())
    
    return gamma_ao

def EOM_r0(DE, t1, r1, fsp, eris_oovv, r2=None):
    '''
    Returns the r0 amplitudes for n excited states
    see Bartlett's book: Figure 13.2 page 439 (R equations)

    :param DE: list of excitation energies for n ES
    :param t1: t1 amplitudes
    :param r1: list of r1 amplitudes for n ES
    :param r2: list of r2 amplitudes for n ES
    :param fsp: fock matrix
    :param eris_oovv: two-electron integral <oo||vv>
    :return: list of r0 amp for n ES
    '''

    nbr_of_states = len(r1)
    nocc,nvir = r1[0].shape

    if r2 is None:
        r2 = [np.zeros((nocc,nocc,nvir,nvir))]*nbr_of_states

    Xia = fsp[:nocc, nocc:]
    Xia += np.einsum('me,imae->ia', t1, eris_oovv)
    r0n = []

    for n in range(nbr_of_states):
       r0 = np.einsum('ld,ld',Xia,r1[n])
       r0 += 0.25*np.einsum('lmde,lmde',eris_oovv,r2[n])
       r0 /= DE[n]
       r0n.append(r0)

    return r0n


#################################
# Printing density and orbitals
#################################

def printNO(rdm1,mf,mol,fout):
    # todo: extend for transition rdm1 -> NTOs. Use SVD since tr_rdm1 is not squared
    '''
    Calculates the natural orbitals and prints them in Molden format

    :param rdm1: reduced one-particle density matrix
    :param mf: PySCF HF object (U,G or R format), must be the same format as rdm1
    :param mol: PySCF mol object
    :param fout: output molden file
    :return:
    '''

    mo_ene = mf.mo_energy
    fout = fout + '.molden'

    # diagonalize rdm1 to yield NOs expansion coeff in MO basis
    # occupation results are generated in ascending order
    no_occ, no = scipy.linalg.eigh(rdm1)

    # reorder occ and no in descending order
    no_occ = no_occ[::-1]
    no = no[:, ::-1]

    # Express NOs in AOs basis
    # LCAO coefficient for the NOs
    no_coeff = mf.mo_coeff.dot(no)

    # print molden format with MOs energy (...)
    with open(fout, 'w') as f1:
       molden.header(mol, f1)
       molden.orbital_coeff(mol, f1, no_coeff, ene=mo_ene,occ=no_occ)


def cube(rdm1,mo_coeff,mol,fout, g=True):
    '''

    :param rdm1: reduced one-particle density matrix in MOs basis
    :param mo_coeff: LCAO coefficients of the MOs
    :param mol: PySCF mol object
    :param fout: name of the cube output file
    :param g: True if G format, false if R format
    :return:
    '''

    fout = fout+'.cube'

    # express rdm1 in AOs
    rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff,rdm1,mo_coeff.conj())

    # convert generalized rdm1 to restricted format
    if g:
        rdm1 = convert_g_to_ru_rdm1(rdm1)[0]

    cubegen.density(mol,fout,rdm1)

def diff_cube(file1,file2,out):
    '''
    Takes the difference between two cube files

    :param file1: name of first file
    :param file2: name of second file
    :param out: name of output cube file
    :return:
    '''

    # see Diff_cube file

    import sys

    initial_line = 14

    file1 = open(file1)
    file2 = open(file2)
    file_out = open(out+'.cube','w')

    f1 = file1.readlines()
    f2 = file2.readlines()

    string_out = ''
    for i in range(initial_line):
        string_out += f1[i]
    for i in range(initial_line,len(f1)):
        line1 = f1[i].split()
        line2 = f2[i].split()
    for j in range(len(line1)):
        string_out += str(float(line1[j])-float(line2[j]))
        string_out += ' '
    string_out += ("\n")

    print >> file_out,string_out


###########################
# One electron properties
###########################

def Ekin(mol,rdm1,g=True,AObasis=True,mo_coeff=None,Ek_int=None):
    '''

    :param mol: PySCF mol object
    :param rdm1: reduced one-particle density matrix
    :param g: True if rdm1 given in G format, False if given in R format
    :param AObasis: True if rdm1 given in AO basis, False if given in MOs basis
    :param mo_coeff: MOs coefficients
    :param Ek_int: Kinetic energy AO integrals
    :return:
    '''

    # dm1 must be in AOs basis
    if AObasis is False:
        if mo_coeff is None:
            raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
        rdm1 = np.einsum('pi,ij,qj->pq',mo_coeff,rdm1,mo_coeff.conj())

    # convert GHF rdm1 to RHF rdm1
    if g:
        rdm1_g = convert_g_to_ru_rdm1(rdm1)[0]

    # Ek integral in AO basis
    if Ek_int is None:
        Ek_int = mol.intor_symmetric('int1e_kin')

    # Ekin of the electrons
    Ek = np.einsum('ij,ji',Ek_int,rdm1_g)

    return Ek


def v1e(mol,rdm1,g=True,AObasis=True, mo_coeff=None,v1e_int=None):
    '''
    Calculates the one-electron potential Ve for a given rdm1

    :param mol: PySCF mol object
    :param rdm1: reduced one-particle density matrix
    :param g: True if rdm1 given in G format, False if given in R format
    :param AObasis: True if rdm1 given in AO basis
    :param mo_coeff: MOs coefficients
    :return:
    '''

    # rdm1 must be in AOs basis
    if AObasis is False:
       if mo_coeff is None:
           raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
       rdm1 = np.einsum('pi,ij,qj->pq',mo_coeff,rdm1,mo_coeff.conj())

    # convert GHF rdm1 to RHF rdm1
    if g:
       rdm1 = convert_g_to_ru_rdm1(rdm1, mol.nao)[0]

    # v1e integral in AO basis
    if v1e_int is None:
        v1e_int = mol.intor_symmetric('int1e_nuc')

    # VeN --> one electron coulombic potential
    v1e = np.einsum('ij,ji',v1e_int,rdm1)

    return v1e

def dipole(mol,rdm1,g=True,AObasis=True,mo_coeff=None,dip_int=None):
    '''
    Calculates the dipole or transition dipole moment vector for a given rdm1
    or transition rdm1

    :param mol: PySCF mol object
    :param rdm1: one-particle reduced density matrix
    :param g: True if rdm1 given in G format
    :param AObasis: True if rdm1 given in AOs basis
    :param mo_coeff: MOs coefficients
    :param dip_int: dipole integral in AOs basis
    :return:

    '''
    
    # rdm1 must be in AOs basis
    if AObasis is False:
       if mo_coeff is None:
           raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
       rdm1 = np.einsum('pi,ij,qj->pq',mo_coeff,rdm1,mo_coeff.conj())
    
    # convert GHF rdm1 to RHF rdm1
    if g:
       rdm1 = convert_g_to_ru_rdm1(rdm1)[0]

    # dipole integral
    if dip_int is None:
        # define center of charge
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        charge_center = np.einsum('z,zr->r', charges, coords) / charges.sum()
        # calculate integral -> 3 components
        with mol.with_common_orig(charge_center):
            dip_int = mol.intor_symmetric('int1e_r', comp=3)

    # contract rdm1 and dip_int
    ans = np.einsum('xij,ji->x', dip_int, rdm1)

    return ans



if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = 'sto3g'
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mfr = scf.RHF(mol)
    mfg = scf.GHF(mol)
    mfr.kernel()
    mfg.kernel()

    rdm1_g = mfg.make_rdm1()
    rdm1_r = mfr.make_rdm1()

    print("###########################")
    print('# Convert rdm1_G to R and U')
    print("###########################")
    print()
    print('Max difference between rdm1_R and rdm1_R from rdm1_G ')
    print(np.max(np.subtract(rdm1_r, convert_g_to_ru_rdm1(rdm1_g)[0])))
    print()

    print()
    print("####################")
    print('# Test get_ini_r ')
    print("####################")
    print()

    # QChem EOM-CCSD calculation of the first singlet states
    exp = []
    exp.append((0,0,0.622537))
    exp.append((0,0,0.056985))
    r, tdm_diff = get_init_r(mol,roots=10)

    print(tdm_diff)

    print()
    print('######################')
    print('# Test amp conversion ')
    print('######################')
    print()

    from pyscf import ci

    print('RCI to GCI coeff')
    myrci = ci.RCISD(mfr)
    myrci.kernel()
    c = myrci.ci
    co,c1,c2 = myrci.cisdvec_to_amplitudes(c)
    c1_g = convert_r_to_g_amp(c1)
    print('rci shape= ',c1.shape)
    print('gci shape= ',c1_g.shape)
    cg = pyscf.cc.addons.spatial2spin(c1)
    print('Difference with PySCF function')
    print(np.subtract(cg,c1_g))

    print()
    print('########################')
    print('# Test L1 sub-gradient')
    print('########################')
    print()

    import CCS
    from pyscf import cc
    import Eris

    # fock matrix and eris
    mo_occ = mfg.mo_occ
    mocc = mfg.mo_coeff[:, mo_occ > 0]
    mvir = mfg.mo_coeff[:, mo_occ == 0]
    nocc = mocc.shape[1]
    nvir = mvir.shape[1]
    mygcc = cc.GCCSD(mfg)
    eris = Eris.geris(mygcc)
    fock = eris.fock
    
    # random R t1 and t2
    ts=np.random.random((nocc//2,nvir//2))*0.1
    td=np.random.random((nocc//2,nocc//2,nvir//2,nvir//2))*0.1
    ts = convert_r_to_g_amp(ts)
    td = convert_r_to_g_amp(td)

    # T1 eq
    import CC_raw_equations
    T1,T2 = CC_raw_equations.T1T2eq(ts,td,eris)

    # print sub-gradient
    alpha = 0.
    print('alpha=0')
    W1 = subdiff(T1,ts,alpha)
    W2 = subdiff(T2,td,alpha)
    print('W1-T1')
    print(np.sum(np.subtract(W1,T1)))
    print('W2-T2')
    print(np.sum(np.subtract(W2,T2)))
    
    print()
    print('################################')
    print('# Test SVD for orbital rotation ')
    print('################################')
    print()

    # create MOs coeff
    dim = mol.nao
    cL = np.random.random((dim,dim))
    cR = np.random.random((dim,dim))

    cL,cR = ortho(mol,cL,cR)
    S_AO = mol.intor('int1e_ovlp')

    print('check orthogonality --> OK')
    print(np.einsum('mp,nq,mn->pq',cL,cR,S_AO))
    print('norm=', get_norm(cL,cR))


    print()
    print('################################')
    print('# Test tr_rdm1 for Slater       ')
    print('################################')
    print()
    
    print()
    print('################################')
    print('# Test Koopman\'s guess         ')
    print('################################')
    print()
    c0, DE = koopman_init_guess(mfg.mo_energy,mfg.mo_occ,1)
    print(DE)
    print(c0[0])