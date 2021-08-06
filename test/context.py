"""
allows the use of the tests files without expecting the package to be installed in site-packages
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(sys.path)
import Eris, utilities, CCS, Solver_ES, exp_pot, Solver_GS, gamma_exp, Main
