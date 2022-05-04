# module doscription
"""*******************************************************************************************************
$ This module contains a set of numerical functions required for:
 - the module 'kernel' contained in in the package 'core'
 - the module 'cgl' and 'scg' contained in the package 'optimisation'
 
*******************************************************************************************************"""
# initialising the library
import gprocess

# outer dependency 
import functools 
import numpy as np 

# teset linking
def test_numerical():
    print("Hello, I'm numerical from core!")

# numerical methods 
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_diff2(f, x):
    h = 1e-4
    df0 = numerical_diff(f,x + h) 
    df1 = numerical_diff(f,x - h)
    return (df0 - df1) / (2 * h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size): # x.size returns the dimension of x. 
        tmp = x[i]
        x[i] = tmp + h
        f0 = f(x)
        x[i] = tmp - h
        f1 = f(x)
        grad[i] = (f0 - f1) / (2 * h)
        x[i] = tmp # reassign the original value to x[i].
    return grad 

def numerical_diff_partial(f, x, dim): # x is a input vector
    h = 1e-4
    tmp = x.copy()
    tmp[dim] += h
    x0 = tmp
    tmp = x.copy()
    tmp[dim] -= h
    x1 = tmp
    return (f(x0) - f(x1)) / (2 * h)

def numerical_diff2_partial(f, x, dim0, dim1):
    h = 1e-4
    tmp = x.copy()
    tmp[dim1] += h
    x0 = tmp
    tmp = x.copy()
    tmp[dim1] -= h 
    x1 = tmp
    df0 = numerical_diff_partial(f, x0, dim0) 
    df1 = numerical_diff_partial(f, x1, dim0) 
    return (df0 - df1) / (2 * h)

def numerical_hessian(f, x):
    n = x.shape[0]
    H = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j < i:
                continue
            H[i,j] = numerical_diff2_partial(f, x, i, j)
    return H + H.T - np.diag(H.diagonal())
