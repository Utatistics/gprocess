# module doscription
"""*****************************************************************************************
$ This module implements functions related to likelihood required for the main class 'GProcess'
$ Inner dependency:
- gprocess.core.kernel

*****************************************************************************************"""
# initialising the library
import gprocess

# outer dependency 
import numpy as np

# testing linking
def test_likelihood():
    print("Hello, I'm likelihood from core!")
    
"""*********************************
# likelihood function implementation
*********************************"""
# analytiaclly obtained likelihood function.
def get_L(params, X, y, kernel):
    K_theta = gprocess.get_K(params, X, kernel=kernel)
    K_theta_inv = np.linalg.inv(K_theta) # C++ IMPLEMENTATION!!
    det = np.linalg.det(K_theta) 
    if det == 0.0: # avoid divergence 
        return -np.inf
    else:
        return -np.log(det) - y.T @ K_theta_inv @ y # a positive definite and its dterminant

# negative likelihood function 
def get_L_neg(params, X, y, kernel):
    return - get_L(params, X, y, kernel)

# analytically derived gradient vector of likelihood function.
def get_L_delta(params, X, y, kernel): 
    K_theta = gprocess.get_K(params, X, kernel=kernel)
    K_theta_inv = np.linalg.inv(K_theta) 
    L_delta = np.zeros_like(params)
    for d in range(len(params)):
        K_delta = gprocess.get_K_delta(params, d, X, kernel=kernel)
        L_delta[d] = -np.trace(K_theta_inv @ K_delta) + (K_theta_inv @ y).T @ K_delta @ (K_theta_inv @ y) 
    return L_delta

