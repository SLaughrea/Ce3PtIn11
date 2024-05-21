import numpy as np

from muesr.engines.clfc import locfield # Does the sum and returns the results
from muesr.core import Sample           # The object that contains the information
from muesr.engines.clfc import find_largest_sphere # A sphere centered at the muon is the correct summation domain
from muesr.i_o import load_cif          # To load crystal structure information from a cif file
from muesr.utilities import mago_add, show_structure # To define the magnetic structure and show it
import matplotlib.pyplot as P
np.set_printoptions(suppress=True,precision=4)       # to set displayed decimals in results

