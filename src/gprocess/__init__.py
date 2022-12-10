# linrary descripiton 
"""#############################################################################################################
The library 'gprocess' contains the following package.
$ core
- implements main class. 
- implements the set of methods included in the main class i.e. .fit, .pred
- implements the set of modules required for the methods above.

$ optimisation
- implements the set of optimisatin routine required for the .fit method

$ pyd
- subpackage containing the sub-module written in c++ code complied as .pyd files

#############################################################################################################"""
# initialsing the packages.
from .core import *
from .optimisation import * 
from .pyd import *
