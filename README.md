<h1 align="center">
<img src="/logo/gprocess.png" width="500">
</h1><br>

## Overview
'gprocess' is an original library designed to implement the machine learning algorithm 'Gaussian Process.' The library is written in python and c++ from scrath, meaning that it does not rely on any external python libraries other than NumPy.

## Requirement

## Usage

## packages contained
- core

Core package contains a set of core modules required for the main Python class defined for this library i.e. GProcess. The package includes the class implementation itself as well as modules for numerical methods, likelihood and prediction, etc.

- optimisation 

Optimisation package implements multiple optimisation routines used for hyper-parameter tuning. So far conjugate gradient with line search method (CGL) and scaled conjugate gradient method (SCG) are implemented.

- pyd

Pyd package contains the set of compiled binary files that are originally written in c++. These are experimental scripts to make computation faster. Running these programs on Python environment requires pybind11. 
