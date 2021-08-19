
ECW-CC: experimentally constrained wave function coupled cluster.
==================================================================

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

Notation
---------
In order to stick with the notation used in the literature, several confusion may occur.
These are shown in the following points:

- "rdm1" and "gamma" are used indistinctly for the one-body reduced density matrix. Where "_calc" or "_pred" is added for the calculated one.
- "T1"/"t1" are the singles T equations and the singles t amplitudes, respectively.
- "L1"/"l1" may refer to the singles L or Lambda equations and the singles l or lambda amplitudes, respectively. When prone to confusion, "es_L1" may is used to specify the excited state case and/or "Lam" for the ground state.
- "L1 reg" stands for the L1 regularization method, where "alpha" is the coefficient of the L1 regularization term.
- "L" or "lambda" stands for the weight of the experimental potential.