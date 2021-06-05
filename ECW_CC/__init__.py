'''
ECW-CC
experimentally constrained wave function coupled cluster.

The code is based on the PySCF quantum chemistry package
Author= Milaim Kas
Theory and equations: Stasis Chuchurca and Milaim Kas

Theory
--------

The method allows to solve the coupled cluster equations for an effective Hamiltonian H+L*Vexp
where Vexp is a potential that compares calculated and given one-electron properties. For increasing value of the weight
L, the obtained WF gives the best fitted one-electron property, thus minimising |calc-exp|^2.
- for ground state case: L1-ECW-CCS or L1-ECW-CCSD where L1 stands for L1 regularized solution.
- for excited state: ECW-CCS

The following n functionals have to be minimized:
J_n = <Psi_n|H|Psi_n> + L*Vexp + |Psi_n|_1
Leading to n Schrödinger equations to be solved:
E|Psi_n> = H|Psi_n> + sum_{m} Vexp^{nm}|Psi_m>
Different cases can be distinguished:
- Vexp^{nn} potentials contain one-electron properties related to state n
- Vexp^{nm} potentials contain one-electron transition properties related to the n->m transition

The Gs case corresponds to Vexp = Vexp^{00}
ES case correspond to Vexp = Vexp^{nm} and Vexp^{nn}

The Couples Cluster formalism is applied to solve the set of couples SE.

See Theory.pdf file for more detailed.

How to use
-----------

    >>> import numpy as np
    >>> molecule = 'h2o'
    >>> basis = '6-31g'

    >>> # Choose lambda array
    >>> lambi = 0.5  # weight for Vexp, initial value
    >>> lambf = 0.5  # lambda final
    >>> lambn = 1  # number of Lambda value
    >>> Larray = np.linspace(lambi, lambf, num=lambn)
    >>> ecw = ECW(molecule, basis)
    *** Molecule build ***
    >>> ecw.Build_GS_exp('HF', field=[0.05, 0.01, 0.])
    *** GS data stored ***
    >>> Results, plot = ecw.CCS_GS(Larray,graph=False, alpha=0.01)
    <BLANKLINE>
    ##############################
    #  Results using scf
    ##############################
    <BLANKLINE>
    LAMBDA=  0.5
    Convergence reached for lambda= 0.5, after 8 iteration
    <BLANKLINE>
    FINAL RESULTS
    Ep   = -7.59840e+01
    X2   = 4.57699e-03
    DEk  = 2.24583e-06
    <BLANKLINE>
    EHF    = -7.59839e+01
    Eexp   = -7.59860e+01
'''
