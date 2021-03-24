#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

 ECW-CCS
 -------
 Experimentally constrained wave function coupled cluster single
 ---------------------------------------------------------------

 Solves the ECW-CCS equations with an effective one particle
 Fock operator that includes simulated experimental values.
 - One and two particle integrals obtained from PySCF gccsd modules
 - All properties and variables express in spin orbital gaussian basis
 - Solving T1 and L1 by SCF and/or Gradient method
 - Add the L1 regularization term with parameter alpha

 --> see GS-XCW-CCS pdf file for the equations

 Author: Milaim Kas
 Equations: Stasis Chuchurca and Milaim Kas

'''

import numpy as np

##############################################
#             Global INPUTS                  #
##############################################


# QC related options
# -------------------

ghf = True  # use GHF as basis, RHF otherwise
basis = '6-31+g**'
molecule = 'h2o' # choose molecule


# Solver
# -------------------

# solve either ECW-CCS ('CCS') or ECW-CCSD ('CCSD') equations
model = 'CCSD'

# Method: string
# - 'scf': with (alpha) or without (alpha=None) L1 reg
# - 'newton'
# - 'descend'
# - 'L1_grad': beta and alpha must be given
# Note: for CCSD, only the scf method is available
method = 'scf'

# L1 reg parameter (for 'scf' or 'L1_grad' method)
alpha = 0

# gradient descend constant (for 'descend' or 'L1_grad')
beta = 0.01

# DIIS options (for 'scf', 'L1_grad' method)
# tuple of variable strings on which DIIS is apply: ('rdm1','l','t')
# if ('') no DIIS
# recomended 'rdm1'
diis = ('')
maxdiis = 20 # DIIS max space

# choice of the initial t1 and l1 (1='1/fii-faa',2='rand',0='0')
tl1ini = 0


# Options for lambda
# -------------------

lambi = 0  # weight for Vexp, initial value
lambf = 0.1 # lambda final
lambn = 5  # number of Lambda value
Larray = np.linspace(lambi, lambf, num=lambn)


# Convergence options
# --------------------------

conv = 'tl'
conv_thres = 0.5*10 ** -6  # convergence threshold

# Maximum nbr of iteration
maxiter = 80

Vscale = False


# Options for building gamma_exp
# ------------------------------

# print Kinetic energy expectation value with X2
Ek = True

para_factor = None  # fraction between nbr of exp data (dim^2)
# and parameters (t1+l1)
# if para_factor > 1 --> under fitting
# if para_factor < 1 --> over  fitting (more
# parameters than exp data)
# para_factor = None dont change gamma_exp

# xyz component of the external field
# None if no field
field = [0.05, 0.01, 0.0]

# - max geometry deformation in ang
deform_max = None

# method for the calculation of gamma_exp
# 'HF','CCSD' or 'CCSD(T)'
posthf = 'HF'
# include core in CCSD calculation or not
core = False


# Other
# -----------------
# Threshold for the 2e integrals
int_thresh = 1e-13


# Printing options
# -----------------
# Plot X2 and Ep as a function of lambda
graph = True
print_ite_info = True

# print cube file for gamma_exp, ite=0(HF)
# 2 values of L and for largest calculated value of L
printcube = False
# number of cube files to print
nbr_cube = 4
# path to the directory
dir_cube = ''

# calculate NOs and print molden file
# printmolden = False

exec(open("./Main.py").read())

