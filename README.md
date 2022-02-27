# gprocess

## description
'gprocess' is an original library designed to impplement the machine library algorithm called 'Gaussian Process.' The library is written in python and c++ from scrath, in a sense that it does not rely on any external libraries except NumPy.

## packages contained
- core
Core package contains a set of core modules required for the main Python class defined for this library i.e. GProcess. The package includes the class implementation itself as well as modules for numerical methods, likelihood and prediction, etc.

- optimisation 
Optimisation package implements multiple optimisation routines used for hyper-parameter tuning. So far conjugate gradient with line search method (CGL) and scaled conjugate gradient method are implemented.

- pyd
Pyd package contains the set of compiled binary files that are originally written in c++. These are experimental scripts to make computation faster. Running these programs on Python environment requires pybind11. 
