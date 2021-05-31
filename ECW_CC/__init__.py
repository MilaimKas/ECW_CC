'''

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
