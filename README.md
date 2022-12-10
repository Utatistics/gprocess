<h1 align="center">
<img src="/logo/gprocess.png" width="300">
</h1><br>

<p>
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
<img src="https://img.shields.io/badge/-Linux-6C6694.svg?logo=linux&style=flat">
<img src="https://img.shields.io/badge/-Windows-0078D6.svg?logo=windows&style=flat">
<p>

## Overview
gprocess is an original library designed to implement the machine learning algorithm 'Gaussian Process'. The initial goal of the proejct includes the hands-on implementation of the ML algorithm and relevant optimazation methods, and therefore the gprocess does not rely on any existing ML library or frameworks. 

## Requirement
The library is implemented in python and c++ and requires NumPy. See pyproject.toml for more detailed dependencies.

## Description
gprocess contains the following packges

### core
contains a set of core modules required for the main Python class defined for this library (i.e. GProcess). The package includes the class implementation itself as well as modules for numerical methods, likelihood and prediction, etc.

### optimisation
implements multiple optimisation routines used for hyper-parameter tuning: conjugate gradient with line search method (CGL) and scaled conjugate gradient method (SCG) are implemented.

### pyd
the set of compiled binary files that are originally written in c++; these are experimental scripts to make computation faster. Running these programs on Python environment requires pybind11.

## License
gprocess is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
