#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ECW.utilities

contains list of useful functions
"""

import copy

import numpy
from pyscf import scf, gto, tdscf, cc, lib
from pyscf.gto import ft_ao
from pyscf.tools import cubegen
from pyscf.tools import molden
import scipy
import numpy as np

#######################################
# L1 regularization related functions
#######################################


def subdiff(eq,var,alpha, R_format=False):
    # todo: R_format's conversion does not work
    """
    Calculates the sub-gradient value of a functional
    see equations (17) in Ivanov et al, Molecular Physics, 115(21–22)

    :param eq: Jacobian matrix of the functional (T,L equations) given in amp format
    :param var: matrix of variables (t or l amplitudes) given in amp format
    :param alpha: L1 threshold
    :return: subdifferential W in amp format
    """
    
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
    ind = np.argwhere(np.abs(var) > 0.)
    for ix in ind:
        dW[tuple(ix)] = eq[tuple(ix)]+alpha*np.sign(var[tuple(ix)])

    # zero elements in var
    ind = np.argwhere(var <= 0.)
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
    """
    proximal point mapping method of the L1 regularization approach

    :param x_J: matrix of parameters
    :param alpha: proximal point factor
    :return: matrix of the proximal point function dh
    """

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
    """
    Performs TDHF calculation and extract coeff for initial r value
    :param mol: mol object
    :param roots: number of states to print
    """

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
    """
    Converts amplitudes in restricted format into generalized, spin-orbital format
    amp must be given in nocc x nvir shape
    !!! Convert into [0 1 0 1] spin format !!!
    NOTE: PySCF as several functions to perform the conversion within CC, CI or EOM classes: spatial2spin()

    :param amp: amplitudes in restricted format
    :return: amp in generalized spin-orb format
    """

    if amp.ndim == 2:
        g_amp = np.zeros((amp.shape[0]*2, amp.shape[1]*2))
        for i in range(amp.shape[0]):
            for j in range(amp.shape[1]):
                a = amp[i, j]
                g_amp[i*2:i*2+2, j*2:j*2+2] = np.diag(np.asarray([a, a]))  #np.asarray([[0,a],[a,0]])
    elif amp.ndim == 4:
        g_amp = cc.addons.spatial2spin(amp)
    else:
        raise ValueError('amplitudes must be 2 or 4 dim')
    return g_amp


def convert_g_to_r_amp(amp, orbspin=None):
    """
    Converts G format amplitudes into R format
    !!! Only works for G amplitudes with [0 1 0 1] spin format !!!

    :param orbspin: array of 0 and 1 defining the alpha and beta nature of the MO coeff
    :param amp:CC single or double amplitudes
    :return:
    """

    if amp.ndim == 2:
        tmp = np.delete(amp, np.s_[1::2], 0)
        r_amp = np.delete(tmp, np.s_[1::2], 1)
    # use PySCF function for doubles
    elif amp.ndim == 4:
        dim = amp.shape[0]+amp.shape[2]
        if orbspin is None:
            orbspin = np.zeros(dim, dtype=int)
            orbspin[1::2] = 1
        r_amp = cc.addons.spin2spatial(amp, orbspin)[1]  # t2ab

    else:
        raise ValueError('amp dimension must be 2 or 4')

    return r_amp


def convert_g_to_ru_rdm1(rdm1_g):
    """
    Transform generalised rdm1 to R and U rdm1

    :param rdm1_g: one-electron reduced density matrix in AOs basis
    :return: rdm1 in R and U format
    """

    nao = rdm1_g.shape[0]//2

    rdm_a = rdm1_g[:nao, :nao]
    rdm_b = rdm1_g[nao:, nao:]

    rdm_u = (rdm_a, rdm_b)

    rdm_r = rdm_a + rdm_b

    return rdm_r, rdm_u


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
    # todo: check that the trace remains the same
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


def convert_r_to_g_coeff(mo_coeff):
    # todo: trace of the transformed rdm1 is not = nelec
    """
    Convert mo_coeff in spatial format into spin-orbital format

    example:
                      phi_a,1 phi_b,1 phi_a,2 phi_b,2
                      --------------------------------
                Xa,1 | c       0        c       0
                Xa,2 | c       0        c       0
    new_coeff = Xb,1 | 0       c        0       c
                Xb,2 | 0       c        0       c


    :param mo_coeff: mo_coeff in spatial format (R format)
    :return: mo_coeff in spin-orbital format (G format)
    """

    dim = mo_coeff.shape[0]*2
    new_coeff = np.zeros((dim, dim))
    new_coeff[0:dim//2, 0::2] = mo_coeff
    new_coeff[dim//2:, 1::2] = mo_coeff

    return new_coeff


def convert_g_to_r_coeff(mo_coeff):
    """
    convert MO coeff from spin-orbital (G) to spatial (R) format

    :param mo_coeff: MO coeff in G format
    :return:
    """

    dim = mo_coeff.shape[0] // 2
    new_coeff = mo_coeff[:dim,0::2]

    return new_coeff


def convert_aoint(int_ao, mo_coeff):
    """
    Transform AO integrals into spin-orbital (G) MO integrals

    :param int_ao: matrix A_mu,nu
    :param mo_coeff: mo_coeff in G format
    :return:
    """

    # dipole case
    if int_ao.shape[0] == 3:
        dim = mo_coeff.shape[0]
        int_mo = np.zeros((3, dim, dim))
        for int, i in zip(int_ao, [0, 1, 2]):
            tmp = ao_to_mo(int, convert_g_to_r_coeff(mo_coeff))
            # R -> G
            int_mo[i, :, :] = convert_r_to_g_rdm1(tmp)

    else:
        int_mo = ao_to_mo(int_ao, convert_g_to_r_coeff(mo_coeff))
        # R -> G
        int_mo = convert_r_to_g_rdm1(int_mo)

    return int_mo

#####################
# Quantum chemistry
#####################


def cis_rdm1(c1):
    """
    Calculates the cis rdm1 given a set of c1 coefficients

    :param c1: CIS or TDA coeff
    :return: oo and vv contribution to RHF MO rdm1
    """

    doo = 2-np.einsum('ia,ka->ik', c1.conj(), c1)
    dvv = np.einsum('ia,ic->ac', c1, c1.conj())

    return doo, dvv


def ao_to_mo(rdm1_ao, mo_coeff):
    """
    transform given rdm1 from AOs to MOs basis.
    Both have to be given in the same format: either R (spatial) or G (spin-orbital)

    :param rdm1_ao: one-particle reduced density matrix given in AOs basis
    :param mo_coeff: MOs coefficients
    :return: rdm1 in MOs basis
    """

    # check dimension
    if rdm1_ao.shape != mo_coeff.shape:
        raise ValueError('Rdm1 and MOs coefficients must have the same dimension')

    mo_coeff_inv = np.linalg.inv(mo_coeff)
    rdm1_mo = np.einsum('pi,ij,qj->pq', mo_coeff_inv, rdm1_ao, mo_coeff_inv.conj())
    
    return rdm1_mo


def mo_to_ao(rdm1_mo, mo_coeff):
    """
    transform given rdm1 from MOs to AOs basis

    :param rdm1_mo: one-particle reduced density matrix given in MOs basis
    :param mo_coeff: MOs coefficients
    :return: rdm1 in AOs basis
    """

    if rdm1_mo.shape != mo_coeff.shape:
        raise ValueError('rdm1 and mo coeff must have the same size')
    rdm1_ao = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1_mo, mo_coeff.conj())

    return rdm1_ao


def koopman_init_guess(mo_energy, mo_occ, nstates=(1, 0), core_ene_thresh=10.):
    """
    Generates list of koopman guesses for r1 vectors in G format
    The guess is obtained in the restricted R format to avoid breaking symmetry

    :param mo_energy: MOs energies
    :param: mo_occ: occupation array
    :param nstates: number of states valence and core excited states
    :param core_ene_thresh: energy threshold for the definition of core
    :return: list of r_ini and koopman's excitation
    """

    # convert to R format
    mo_energy = mo_energy[0::2]
    mo_occ = mo_occ[0::2]
    occidx = np.where(mo_occ == 1)[0]
    viridx = np.where(mo_occ == 0)[0]
    nocc = occidx.shape[0]
    nvir = viridx.shape[0]
    ncore = np.where(abs(mo_energy[:nocc]) > core_ene_thresh)[0].shape[0]
    e_ia = mo_energy[viridx] - mo_energy[occidx, None]

    x0 = [] # np.zeros((nroot, nov))
    DE = []
    eia_val = e_ia[ncore:, :].ravel()
    eia_core = e_ia[:ncore, :].ravel()
    if nstates[0] > eia_val.size or nstates[1] > eia_core.size :
        raise Warning('The size of the basis is smaller than the number of requested states')

    # Valence
    nroot = min(nstates[0], eia_val.size)
    idx = np.argsort(eia_val)
    nocc_val = nocc-ncore
    for i in range(nroot):
        tmp = np.zeros(eia_val.size)
        tmp[idx[i]] = 1
        tmp = tmp.reshape((nocc_val, nvir))
        tmp = np.vstack((np.zeros((ncore, nvir)), tmp))
        tmp = convert_r_to_g_amp(tmp)
        id = tuple(np.transpose(np.nonzero(tmp))[0])
        tmp[id] = 0
        x0.append(tmp)  # Koopmans' excitations
        DE.append(eia_val[idx[i]])

    # Core
    nroot = min(nstates[1], eia_core.size)
    idx = np.argsort(eia_core)
    for i in range(nroot):
        tmp = np.zeros(eia_core.size)
        tmp[idx[i]] = 1
        tmp = tmp.reshape((ncore, nvir))
        tmp = np.vstack((tmp, np.zeros((nocc_val, nvir))))
        tmp = convert_r_to_g_amp(tmp)
        id = np.transpose(np.nonzero(tmp))
        tmp[id[0]] = 0
        x0.append(tmp)  # Koopmans' excitations
        DE.append(eia_core[idx[i]])

    return x0, DE


def get_DE(mo_energy, rs):
    """
    get the energy difference from the largest element of a amplitude vector

    :param mo_energy: array of MO energies
    :param rs: vector amplitudes
    """

    nocc, nvir = rs.shape
    eia = mo_energy[nocc:] - mo_energy[:nocc, None]
    idx = np.unravel_index(np.argmax(rs), (nocc, nvir))

    return eia[idx]


def tdm_slater(TcL, TcR, occ_diff):
    """
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
    """
    
    Tgamma = np.diag(occ_diff)
    gamma_ao = np.einsum('pi,ij,qj->pq', TcL, Tgamma, TcR.conj())
    
    return gamma_ao


def EOM_r0(DE, t1, r1, fsp, eris_oovv, r2=None):
    """
    Returns the r0 amplitudes for n excited states for the EOM case
    see Bartlett's book: Figure 13.2 page 439 (R equations)

    :param DE: list of excitation energies for n ES
    :param t1: t1 amplitudes
    :param r1: list of r1 amplitudes for n ES
    :param r2: list of r2 amplitudes for n ES
    :param fsp: fock matrix
    :param eris_oovv: two-electron integral <oo||vv>
    :return: list of r0 amp for n ES
    """

    nbr_of_states = len(r1)
    nocc,nvir = r1[0].shape

    if r2 is None:
        r2 = [np.zeros((nocc, nocc, nvir, nvir))]*nbr_of_states

    Xia = fsp[:nocc, nocc:]
    Xia += np.einsum('me,imae->ia', t1, eris_oovv)
    r0n = []

    for n in range(nbr_of_states):
       r0 = np.einsum('ld, ld', Xia, r1[n])
       r0 += 0.25*np.einsum('lmde, lmde', eris_oovv, r2[n])
       r0 /= DE[n]
       r0n.append(r0)

    return r0n


def check_spin(amp_r, amp_l):
    """
    Calculates the total spin of a CC vector in spin-orbital format
    In our simplest case, the elements corresponding to a->a, b->b, b->a and a->b transition
    are fixed in the amplitudes matrix

    ex: r=[[aa,ab],[ba,bb]]

    :param vec: ci/r/l vector in G format
    :return: S
    """

    # vector of alpha (0) and beta(1) spin
    spin_mat = np.zeros_like(amp_r)
    spin_mat[::2, 1::2] = -1
    spin_mat[1::2, 0::2] = 1

    # total spin
    S = np.einsum('ia,ia,ia', amp_r, amp_l, spin_mat)

    return S


def spin_square(rdm1, mo_coeff, ovlp=1):
    # todo: verify and test
    """
    Converts rdm1 and mo_coeff in U format and
    uses the PySCF function in fci.spin_op to calculate spin squared for a WF in G format

    :param rdm1: density matrix in G format
    :param mo_coeff: MO coefficients in G format
    :param S: AO overlap (1 if degenerate)
    :return:
    """
    from functools import reduce
    #from pyscf.fci.spin_op import spin_square_general
    #spin_square_general(dma, dmb, dmaa, dmab, dmbb, mo_coeff, s)

    # convert to U format
    dm1a, dm1b = convert_g_to_ru_rdm1(rdm1)[1]
    nao = mo_coeff.shape[0]//2

    mo_coeff_a = mo_coeff[:nao, 0::2]
    mo_coeff_b = mo_coeff[nao:, 1::2]

    #
    # PySCF spin_square function for single case
    #

    # projected overlap matrix elements for partial trace
    if isinstance(ovlp, np.ndarray):
        ovlpaa = reduce(np.dot, (mo_coeff_a.T, ovlp, mo_coeff_a))
        ovlpbb = reduce(np.dot, (mo_coeff_b.T, ovlp, mo_coeff_b))
    else:
        ovlpaa = np.dot(mo_coeff_a.T, mo_coeff_a)
        ovlpbb = np.dot(mo_coeff_b.T, mo_coeff_b)
    
    ssz = (np.einsum('ji,ij->', dm1a, ovlpaa)
        + np.einsum('ji,ij->', dm1b, ovlpbb)) *.25
    ssxy =(np.einsum('ji,ij->', dm1a, ovlpaa)
         + np.einsum('ji,ij->', dm1b, ovlpbb)) * .5
    ss = ssxy + ssz

    s = np.sqrt(ss+.25) - .5
    multip = s*2+1

    return multip

#################
# linear algebra
#################


def get_norm(rs, ls, r0, l0):
    """
    Return the linear product between two sets of amplitudes
    c = |<Psi_r|Psi_l>|**2

    :param rs: set of amplitudes
    :param ls: set of amplitudes
    :return:
    """

    # check shape
    if rs.shape != ls.shape:
        raise ValueError('Shape of both set of amplitudes must be the same')
    c = l0*r0.conjugate()+np.sum(rs.conjugate()*ls)

    return c


def ortho_QR(Mvec):
    """
    Use numpy QR factorisation to orthogonalize a set of vectors
    math: Mvec = Q.R
    wher Q has orthonormal column vectors and R is a a upper triangular matrix

    :param Mvec: NxM matrix where the columns are the vectors to orthogonalize
    :return:
    """
    Q, R = np.linalg.qr(Mvec)
    return Q


def ortho_SVD(mol, cL, cR):
    """
    Orthogonalize two vectors using singular value decomposition
    See Molecular Physics, 105(9), 1239–1249. https://doi.org/10.1080/00268970701326978

    :param mol: PySCF mol object for AOs overlap matrix or overlap matrix
    :param cL,cR: set of MOs coefficients for bra (Left) and ket (Right) vectors
    :return:
    """

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


def ortho_GS(U, eps=1e-12):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns
    will be 0.

    Args:
        U (numpy.array): d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-12).

    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    """

    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]  # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if np.linalg.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.  # set the small entries to 0
        else:
            V[i] /= np.linalg.norm(V[i])

    return V.T


def check_ortho(rn, ln, r0n, l0n):
    """
    Check the norm for a list of r and l vectors
    diagonal are the normalization constant for state k -> <Psi_k|Psi_k>
    off-diagonal are the averaged orthogonalization constant for state ij -> (<Psi_k|Psi_l>+<Psi_l|Psi_k>)/2

    :param ln: list of l vectors
    :param rn: list of r vectors
    :param l0n: list of l0 values
    :param r0n: list of r0 values
    :return:
    """

    nbr_of_states = len(rn)

    # check length
    if nbr_of_states != len(ln):
        raise ValueError('r and l list of vectors must be the same length')

    # initialize matrix of norm
    C_norm = np.zeros((nbr_of_states, nbr_of_states))

    for k in range(nbr_of_states):
        for l in range(nbr_of_states):
             c_l = get_norm(rn[k], ln[l], r0n[k], l0n[l])
             c_r = get_norm(rn[l], ln[k], r0n[l], l0n[k])
             C_norm[k, l] = (c_l+c_r)/2.

    return C_norm


def ortho_es(rn, ln, r0n, l0n):
    """
    Orthogonalizes rn and ln vectors

    :param rn: list of r1 vectors
    :param ln: list of l1 vectors
    :param r0n: list of r0n values
    :param l0n: list of l0n values
    :return: orthogonalized (r0, rn) and (l0, ln)
    """

    nocc, nvir = rn[0].shape
    nbr_states = len(rn)

    # matrix of r/l vectors as column
    Matvec_r = np.zeros((nocc*nvir+1, nbr_states))
    Matvec_l = np.zeros((nocc*nvir+1, nbr_states))

    # amplitude to vector into Matvec matrix
    for j in range(nbr_states):
        Matvec_r[1:, j] = rn[j].flatten().copy()
        Matvec_l[0, j] = r0n[j]
        Matvec_l[1:, j] = ln[j].flatten().copy()
        Matvec_l[0, j] = l0n[j]

    # right and left vectors
    new_Matvec_r = ortho_QR(Matvec_r)
    new_Matvec_l = ortho_QR(Matvec_l)
    ortho_r0n = []
    ortho_l0n = []
    ortho_rn = []
    ortho_ln = []

    # vector to amplitude
    for i in range(nbr_states):
        ortho_ln.append(new_Matvec_l[1:, i].reshape(nocc, nvir))
        ortho_rn.append(new_Matvec_r[1:, i].reshape(nocc, nvir))
        ortho_r0n.append(new_Matvec_r[0, i])
        ortho_l0n.append(new_Matvec_l[0, i])

    return ortho_rn, ortho_ln, ortho_r0n, ortho_l0n


def biortho_es(r1, l1, r0, l0):
    """
    Orthogonormalized rn with ln vectors to construct a bi-orthonormal set
    math: <Psi_n|Psi_k> = 0

    :param r1: r1 amplitudes for state k
    :param l1: l1 amplitudes for state n
    :param r0: r0 value for state k
    :param l0: l0 value for state n
    :return: orthogonalized (r0, rn) and (l0, ln)
    """

    nocc, nvir = r1.shape

    # decompose Matvec
    Matvec = np.zeros((nocc*nvir+1, 2))
    Matvec[1:, 0] = r1.flatten().copy()
    Matvec[0, 0] = r0
    Matvec[1:, 1] = l1.flatten().copy()
    Matvec[0, 1] = l0

    new_Matvec = ortho_QR(Matvec)

    new_r = new_Matvec[1:, 0].reshape(nocc, nvir)
    new_l = new_Matvec[1:, 1].reshape(nocc, nvir)
    new_r0 = new_Matvec[0, 0]
    new_l0 = new_Matvec[0, 1]

    return new_r, new_l, new_r0, new_l0


def ortho_norm(rn, ln, rn0, ln0, ortho=True):
    """
    normalize vectors r and l
    orthogonalize r and l if biorthogonal set (len(rn)=2)

    math: <Psi_n|Psi_k> = dnk

    :param rn: list of r1 amplitudes for each state
    :param ln: list of l1 amplitudes for each state
    :param rn0: list of r0 amplitudes for each state
    :param ln0: list of l0 amplitudes for each state
    :param ortho: True if a biorthogonalization is to be performed
    :return:
    """

    C_norm = check_ortho(rn, ln, rn0, ln0)

    ln_new = copy.deepcopy(ln)
    rn_new = copy.deepcopy(rn)
    ln0_new = copy.deepcopy(ln0)
    rn0_new = copy.deepcopy(rn0)

    # check if orthogonal and orthogonalizes if bi-basis
    if len(rn) == 2 and ortho is True:
        for c in np.tril(C_norm, -1).flatten():
            if c > 0.001 or c < -0.001:
                # right
                rn_new[0], ln_new[1], rn0_new[0], ln0_new[1] = \
                    biortho_es(rn_new[0], ln_new[1], rn0_new[0], ln0_new[1])
                # left
                rn_new[1], ln_new[0], rn0_new[1], ln0_new[0] = \
                    biortho_es(rn_new[1], ln_new[0], rn0_new[1], ln0_new[0])
                C_norm = check_ortho(rn_new, ln_new, rn0_new, ln0_new)
                break

    # normalize
    for i in range(len(ln)):
        if 0.999 > C_norm[i, i] or C_norm[i, i] > 1.001:
            ln_new[i] = ln_new[i] / C_norm[i, i]
            ln0_new[i] = ln0_new[i] / C_norm[i, i]

    return rn_new, ln_new, rn0_new, ln0_new

#################################
# Printing density and orbitals
#################################


def printNO(rdm1, mf, mol, fout):
    # todo: extend for transition rdm1 -> NTOs. Use SVD since tr_rdm1 is not squared
    """
    Calculates the natural orbitals and prints them in Molden format

    :param rdm1: reduced one-particle density matrix
    :param mf: PySCF HF object (U,G or R format), must be the same format as rdm1
    :param mol: PySCF mol object
    :param fout: output molden file
    :return:
    """

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
    """

    :param rdm1: reduced one-particle density matrix in MOs basis
    :param mo_coeff: LCAO coefficients of the MOs
    :param mol: PySCF mol object
    :param fout: name of the cube output file
    :param g: True if G format, false if R format
    :return:
    """

    fout = fout+'.cube'

    # express rdm1 in AOs
    rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff,rdm1,mo_coeff.conj())

    # convert generalized rdm1 to restricted format
    if g:
        rdm1 = convert_g_to_ru_rdm1(rdm1)[0]

    cubegen.density(mol,fout,rdm1)


def diff_cube(file1,file2,out):
    """
    Takes the difference between two cube files

    :param file1: name of first file
    :param file2: name of second file
    :param out: name of output cube file
    :return:
    """

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

    print >> file_out, string_out


###########################
# One electron properties
###########################


def Ekin(mol, rdm1, g=True, aobasis=True, mo_coeff=None, ek_int=None):
    """

    :param mol: PySCF mol object
    :param rdm1: reduced one-particle density matrix
    :param g: True if rdm1 given in G format, False if given in R format
    :param aobasis: True if rdm1 given in AO basis, False if given in MOs basis
    :param mo_coeff: MOs coefficients in same format as rdm1
    :param ek_int: Kinetic energy AO integrals
    :return:
    """

    # dm1 must be in AOs basis
    if aobasis is False:
        if mo_coeff is None:
            raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
        rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1, mo_coeff.conj())

    # convert GHF rdm1 to RHF rdm1
    if g:
        rdm1 = convert_g_to_ru_rdm1(rdm1)[0]

    # Ek integral in AO basis
    if ek_int is None:
        ek_int = mol.intor_symmetric('int1e_kin')

    # Ekin of the electrons
    Ek = np.einsum('ij,ji', ek_int, rdm1)

    return Ek


def v1e(mol, rdm1, g=True, aobasis=True, mo_coeff=None, v1e_int=None):
    """
    Calculates the one-electron potential Ve for a given rdm1

    :param mol: PySCF mol object
    :param rdm1: reduced one-particle density matrix
    :param g: True if G format, False if given in R format
    :param aobasis: True if rdm1 given in AO basis
    :param mo_coeff: MOs coefficients
    :return:
    """

    # rdm1 must be in AOs basis
    if aobasis is False:
       if mo_coeff is None:
           raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
       rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1, mo_coeff.conj())

    # convert GHF rdm1 to RHF rdm1
    if g:
       rdm1 = convert_g_to_ru_rdm1(rdm1)[0]

    # v1e integral in AO basis
    if v1e_int is None:
        v1e_int = mol.intor_symmetric('int1e_nuc')

    # VeN --> one electron coulombic potential
    v1e = np.einsum('ij,ji',v1e_int,rdm1)

    return v1e


def dipole(mol, rdm1, g=True, aobasis=True, mo_coeff=None, dip_int=None):
    """
    Calculates the dipole or transition dipole moment vector for a given rdm1
    or transition rdm1

    :param mol: PySCF mol object
    :param rdm1: one-particle reduced density matrix
    :param g: True if rdm1 given in G format
    :param aobasis: True if rdm1 given in AOs basis
    :param mo_coeff: MOs coefficients
    :param dip_int: dipole integral in AOs basis
    :return:

    """
    
    # rdm1 must be in AOs basis
    if aobasis is False:
        if mo_coeff is None:
            raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
        rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1, mo_coeff.conj())
    
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


def structure_factor(mol, h, rdm1, g=True, aobasis=True, mo_coeff=None, F_int=None,
                     rec_vec=np.asarray([10., 10., 10.])):
    """
    Calculates the structure factors for a given rdm1 and list of Miller indices

    :param mol: PySCF mol object
    :param rdm1: one-particle reduced density matrix
    :param g: True if rdm1 given in G format
    :param aobasis: True if rdm1 given in AOs basis
    :param mo_coeff: MOs coefficients
    :param F_int: Fourier transform over AO basis
    :param rec_vec: reciprocal lattice lengths (a,b,c)
    :return: array of structure factors F corresponding to Miller indices in h

    """

    # rdm1 must be in AOs basis
    if aobasis is False:
        if mo_coeff is None:
            raise ValueError('mo_coeff must be given if rdm is not in AOs basis')
        rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1, mo_coeff.conj())

    # convert GHF rdm1 to RHF rdm1
    if g:
        rdm1 = convert_g_to_ru_rdm1(rdm1)[0]

    # concert h in numpy array
    h = np.asarray(h)

    if F_int is None:
        F_int = FT_MO(mol, h, mo_coeff, rec_vec)[0]

    # contract rdm and Fint
    ans = np.einsum('hij,ji->h', F_int, rdm1)

    return ans


def FT_MO(mol, h, mo_coeff, rec_vec=np.asarray([10., 10., 10.])):
    """
    Calculates the FT over AO, transforms it into MO basis in G format

    math: F_pq = <p|F(h)|q>

    :param mol: PySCF molecular object
    :param h: Miller indices, list of triples [[h1x,h1y,h1z],[h2x,h2y,h2z], ...]
    :param rec_vec: reciprocal lattice length (a,b,c)
    :return: Fmo_pq and Fao_ij
    """

    # convert to spatial (R) basis
    if mo_coeff.shape[0] != mol.nao:
        mo_coeff = convert_g_to_r_coeff(mo_coeff)
    mo_coeff_inv = np.linalg.inv(mo_coeff)

    # convert h to numpy array
    if not isinstance(h, np.ndarray):
        h = np.asarray(h)

    # reciprocal lattice
    rec_vec = np.diag(rec_vec)
    rec_vec = scipy.linalg.inv(rec_vec)

    # build k-vectors
    gv = 2*np.pi * np.dot(h, rec_vec)

    # gs = number of h point in each directions
    gs = None

    ft_ao = gto.ft_ao.ft_aopair(mol, gv, None, 's1', rec_vec, h, gs)  # s1 = nosymm
    ft_mo = np.einsum('pi,hij,qj->hpq', mo_coeff_inv, ft_ao, mo_coeff_inv.conj())

    return ft_mo, ft_ao


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
    mfr.kernel()
    mfg = scf.convert_to_ghf(mfr)

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

    import pyscf
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

    from pyscf import cc
    import Eris

    # fock matrix and eris
    new_mfg = scf.addons.convert_to_ghf(mfr)
    mo_occ = new_mfg.mo_occ
    mocc = new_mfg.mo_coeff[:, mo_occ > 0]
    mvir = new_mfg.mo_coeff[:, mo_occ == 0]
    nocc = mocc.shape[1]
    nvir = mvir.shape[1]
    mygcc = cc.GCCSD(new_mfg)
    eris = Eris.geris(mygcc)
    fock = eris.fock
    
    # random R t1 and t2
    ts = np.random.random((nocc//2,nvir//2))*0.1
    #t_save = ts.copy()
    td = np.random.random((nocc//2,nocc//2,nvir//2,nvir//2))*0.1
    ts = convert_r_to_g_amp(ts)
    td = convert_r_to_g_amp(td)
    #print('Test t1')
    #print(np.subtract(t_save, convert_g_to_r_amp(ts)))

    # T1 eq
    import CC_raw_equations
    T1,T2 = CC_raw_equations.T1T2eq(ts, td, eris)

    print('Test T1')
    print(np.subtract(convert_g_to_r_amp(convert_r_to_g_amp(T1)), T1))

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
    print('Test G->R->G conversion')
    print('alpha=0')
    W1 = subdiff(T1, ts, alpha)
    W2 = subdiff(T2, td, alpha)
    W1_new = subdiff(T1, ts, alpha, R_format=True)
    W2_new = subdiff(T2, td, alpha, R_format=True)
    print('W1')
    print(np.sum(np.subtract(W1,W1_new)))
    print('W2')
    print(np.sum(np.subtract(W2,W2_new)))
    print('alpha=0.1')
    alpha = 0.1 
    W1 = subdiff(T1, ts, alpha)
    W2 = subdiff(T2, td, alpha)
    W1_new = subdiff(T1, ts, alpha, R_format=True)
    W2_new = subdiff(T2, td, alpha, R_format=True)
    print('W1')
    print(np.sum(np.subtract(W1,W1_new)))
    print('W2')
    print(np.sum(np.subtract(W2,W2_new)))
    print()
    
    print()
    print('################################')
    print('# Test SVD for orbital rotation ')
    print('################################')
    print()

    # create MOs coeff
    dim = mol.nao
    cL = np.random.random((dim,dim))
    cR = np.random.random((dim,dim))

    cL,cR = ortho_SVD(mol, cL, cR)
    S_AO = mol.intor('int1e_ovlp')

    print('check orthogonality --> OK')
    print(np.einsum('mp,nq,mn->pq',cL,cR,S_AO))
    print('norm=', get_norm(cL,cR, 0, 0))

    print()
    print('################################')
    print('# QR decomposition              ')
    print('################################')
    print()

    n = 5
    vec_1 = np.random.rand(n)
    vec_2 = np.random.rand(n)
    Mvec = np.zeros((5, 2))
    Mvec[:, 0] = vec_1
    Mvec[:, 1] = vec_2
    ans = ortho_QR(Mvec)
    # check if new vectors are ortho
    print('before: ', np.sum(vec_1 * vec_2))
    print('after: ', np.sum(ans[:, 0] * ans[:, 1]))

    print()
    print('###################################')
    print('# GS (Gram=Schmidt) decomposition  ')
    print('###################################')
    print()

    ans = ortho_GS(Mvec)
    # check if new vectors are ortho
    print('before: ', np.dot(vec_1, vec_2))
    print('after: ', np.dot(ans[:, 0], ans[:, 1]))


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

    c0, DE = koopman_init_guess(mfg.mo_energy, mfg.mo_occ, nstates=(2,2))
    print('DE valence= ', DE[:2])
    print('DE core = ', DE[2:])
    print('r1 valence= ', c0[0])
    print('r1 core= ', c0[2])
    
    print()
    print('################################')
    print('# Test spin                     ')
    print('################################')
    print()

    S1 = check_spin(c0[0], c0[0])
    rs = np.random.random((c0[0].shape))
    ls = rs.copy()
    norm = check_ortho([rs], [ls], [0], [0])
    ls = ls/norm
    print('norm = ', check_ortho([rs], [ls], [0], [0]))
    S2 = check_spin(rs, ls)
    print('2S+1= ', 2*S1+1, 2*S2+1)

    print()
    print('################################')
    print('# Test ortho and norm           ')
    print('################################')
    print()

    rs = []
    ls = []
    rs.append(np.random.random((3, 4)))
    rs.append(np.random.random((3, 4)))
    ls.append(np.random.random((3, 4)))
    ls.append(np.random.random((3, 4)))
    r0 = list([0.1, 0.2])
    l0 = list([0.05, 0.07])

    print('Initial ortho/nor')
    print(check_ortho(rs, ls, r0, l0))
    print()

    print('orthogonalized rn and ln vectors')
    rs, ls, r0, l0 = ortho_es(rs, ls, r0, l0)
    print('rn vector ortho ? -> YES')
    print(np.sum(rs[0].flatten()*rs[1].flatten())+r0[0]*r0[1])
    print('ln vectors ortho ? -> YES')
    print(np.sum(ls[0].flatten() * ls[1].flatten())+l0[0]*l0[1])
    print('rn-lk ortho ? -> NO')
    print(np.sum(rs[0].flatten() * ls[1].flatten())+r0[0]*l0[1])
    print(np.sum(ls[0].flatten() * rs[1].flatten())+l0[0]*r0[1])
    print()

    print("bi-orthogonalize rn and lk")
    rs[0], ls[1], r0[0], l0[1] = biortho_es(rs[0], ls[1], r0[0], l0[1])
    rs[1], ls[0], r0[1], l0[0] = biortho_es(rs[1], ls[0], r0[1], l0[0])
    print('rn vector ortho ? -> NO')
    print(np.sum(rs[0].flatten()*rs[1].flatten())+r0[0]*r0[1])
    print('ln vectors ortho ? -> NO')
    print(np.sum(ls[0].flatten() * ls[1].flatten())+l0[0]*l0[1])
    print('rn-lk ortho ? -> YES')
    print(np.sum(rs[0].flatten() * ls[1].flatten())+r0[0]*l0[1])
    print('rk-ln ortho ? -> YES')
    print(np.sum(ls[0].flatten() * rs[1].flatten())+l0[0]*r0[1])
    print()

    print('ortho_norm function ')
    rs, ls, r0, l0 = ortho_norm(rs, ls, r0, l0)
    print(check_ortho(rs, ls, r0, l0))
    print()

    print()
    print('################################')
    print('# Fourier transform             ')
    print('################################')
    print()

    h = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 2, 0], [2, 2, 0]]
    ft_mo, ft_ao = FT_MO(mol, h, mfr.mo_coeff)
    print('FT over AO and MO shape')
    print(ft_mo.shape)
    print(ft_mo.shape)
