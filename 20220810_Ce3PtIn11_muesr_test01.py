"""
Following documentation from https://muesr.readthedocs.io/en/latest/

"""


import muesr
import spglib
from muesr.core import Sample
from muesr.core.atoms import Atoms

#create sample object
mysample = Sample()

# load lattice structure from *.cif file
from muesr.i_o import load_cif
load_cif(mysample, "C:/Users/User/OneDrive - Universite de Montreal\Masters/GSASII/Ce3PtIn11.cif")

# not sure where to add the muon exactly
mysample.add_muon([0.1,0,0])

print(mysample)

# check if symmetry information was loaded
from muesr.utilities import muon_find_equiv
muon_find_equiv(mysample)

# define magnetic structure
from muesr.utilities.ms import mago_add


########## when I run this line I get an error:
"""
Traceback (most recent call last):
  File "C:\ProgramData\Anaconda3\lib\site-packages\qtconsole\base_frontend_mixin.py", line 138, in _dispatch
    handler(msg)
  File "C:\ProgramData\Anaconda3\lib\site-packages\spyder\plugins\ipythonconsole\widgets\debugging.py", line 278, in _handle_input_request
    return super(DebuggingWidget, self)._handle_input_request(msg)
  File "C:\ProgramData\Anaconda3\lib\site-packages\qtconsole\frontend_widget.py", line 512, in _handle_input_request
    self._readline(msg['content']['prompt'], callback=callback, password=msg['content']['password'])
  File "C:\ProgramData\Anaconda3\lib\site-packages\qtconsole\console_widget.py", line 2422, in _readline
    self._show_prompt(prompt, newline=False, separator=False)
TypeError: _show_prompt() got an unexpected keyword argument 'separator'
"""
# the code is supposed to keep running and prompt the user for information.

#
mago_add(mysample)






