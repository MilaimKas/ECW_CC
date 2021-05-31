'''
allows the use of the tests files without expecting the package to be installed in site-packages
'''

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#print(sys.path)
import Eris, utilities, CCS, exp_pot, gamma_exp, Solver_ES, Solver_GS, Main