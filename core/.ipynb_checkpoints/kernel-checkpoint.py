# module doscription
"""*************************************************************************************
$ This module contains a set of functions generally required for the main class 'GProcess'
$ Inner dependency:
- gprocess.core.numerical

*************************************************************************************"""
# initialising the library
import gprocess

# outer dependency
import functools
import numpy as np

# test linking
def test_kernel():
    print("Hello, I'm kernel from core!")
    
"""*****************************
# kernel function implementation 
*****************************"""
# defining RBF kernel
def kronecker_delta(x_i,x_j):
    if x_i == x_j:
        return 1
    else:
        return 0
    
# kernel functions
def kernel_rbf(params, x_i, x_j): # MAY REQUIRE NORMALISATION 
    theta_0 = params[0]
    theta_1 = params[1]
    return theta_0 * np.exp(-np.linalg.norm(x_i - x_j) / theta_1)

def kernel_rbf_linear(params, x_i, x_j): # MAY REQUIRE NORMALISATION 
    theta_0 = params[0]
    theta_1 = params[1]
    theta_2 = params[2]
    delta = kronecker_dekta(x_i, x_j)
    return theta_0 * np.exp(-np.linalg.norm(x_i - x_j) / theta_1) + (theta_2 * delta)

def kernel_exponential(param, x_i, x_j):
    return np.exp(-abs(x_i - x_j) / theta)

def kernel_periodic(params, x_i, x_j):
    theta_0 = params[0]
    theta_1 = params[1]
    return np.exp(theta_0 * np.cos(abs(x_i - x_j) / theta_1))

def kernel_linear(param, x_i, x_j):
    return np.dot(x_i, x_j)

"""***************************
# kernel matrix implementation 
***************************"""
# main kernel matrix
def get_K(params, X, **kwargs): # takes a parameter vector
    kernel = kwargs.get('kernel','rbf_kernel') 
    n = len(X) # the number of data points 
    K = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j < i:
                continue
            if kernel == 'rbf_kernel':
                K[i,j] = kernel_rbf(params,x_i=X[i],x_j=X[j])
            if kernel == 'exponential':
                K[i,j] = kernel_exponential(params,x_i=X[i],x_j=X[j])
            if kernel == 'periodic':
                K[i,j] = kernel_periodic(params,x_i=X[i],x_j=X[j])      
    return K + K.T - np.diag(K.diagonal())

# diffrenced kernel matrix
def get_K_delta(params, d, X, **kwargs): # kernel matrix differenciated wrt 'd' th parameter
    kernel = kwargs.get('kernel','rbf_kernel')
    n = len(X) # the number of data points
    K_delta = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j < i:
                continue
            if kernel == 'rbf_kernel':
                kernel_rbf_fixed = functools.partial(kernel_rbf,x_i=X[i],x_j=X[j])
                K_delta[i,j] = gprocess.numerical_diff_partial(f=kernel_rbf_fixed,x=params,dim=d)
            if kernel == 'exponential':
                kernel_exponential_fixed = functools.partial(kernel_exponential,x_i=X[i],x_j=X[j])
                K_delta[i,j] = gprocess.numerical_diff_partial(f=kernel_exponential_fixed,x=params,dim=d)
            if kernel == 'periodic':
                kernel_periodic_fixed = functools.partial(kernel_periodic,x_i=X[i],x_j=X[j])
                K_delta[i,j] = gprocess.numerical_diff_partial(f=kernel_periodic_fixed,x=params,dim=d)       
    return K_delta + K_delta.T - np.diag(K_delta.diagonal())

# off-diaonal kernel matrix 
def get_K_off_diag(params, Xi, Xj,**kwargs): # takes a parameter vector
    kernel = kwargs.get('kernel','rbf_kernel') 
    ni,nj = len(Xi),len(Xj)
    K = np.zeros([ni,nj]) # non-symmetrical dimension 
    for i in range(ni):
        for j in range(nj):
            if kernel == 'rbf_kernel':
                K[i,j] = kernel_rbf(params,x_i=Xi[i],x_j=Xj[j])
            if kernel == 'exponential':
                K[i,j] = kernel_exponential(params,x_i=Xi[i],x_j=Xj[j])
            if kernel == 'periodic':
                K[i,j] = kernel_periodic(params,x_i=Xi[i],x_j=Xj[j])      
    return K

