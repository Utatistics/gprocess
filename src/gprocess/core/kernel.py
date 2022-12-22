import gprocess
import functools
import numpy as np

    
def kronecker_delta(x_i: np.ndarray, x_j: np.ndarray) -> int:
    """
    defining RBF kernel

    """
    if x_i == x_j:
        return 1
    else:
        return 0
    
def kernel_rbf(params: dict, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    """
    MAY REQUIRE NORMALISATION
    """
    theta_0 = params['theta_0']
    theta_1 = params['theta_1'] 
    return theta_0 * np.exp(-np.linalg.norm(x_i - x_j) / theta_1)

def kernel_rbf_linear(params: dict, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray: 
    """
    MAY REQUIRE NORMALISATION
    """
    theta_0 = params['theta_0']
    theta_1 = params['theta_1']
    theta_2 = params['theta_2']

    delta = kronecker_delta(x_i, x_j)
    return theta_0 * np.exp(-np.linalg.norm(x_i - x_j) / theta_1) + (theta_2 * delta)

def kernel_exponential(param: dict, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    return np.exp(-abs(x_i - x_j) / param)

def kernel_periodic(params: dict, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    theta_0 = params['theta_0']
    theta_1 = params['theta_1']
    return np.exp(theta_0 * np.cos(abs(x_i - x_j) / theta_1))

def kernel_linear(param: dict, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    return np.dot(x_i, x_j)

def get_K(params: dict, X: np.ndarray, kernel='rbf_kernel') -> np.ndarray:
    """
    compute main kernel matrix

    takes a parameter vector
    """
    n = len(X) # the number of data points 
    K = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if j < i:
                continue
            if kernel == 'rbf_kernel':
                K[i,j] = kernel_rbf(params,x_i=X[i],x_j=X[j])
            elif kernel == 'rbf_kernel_linear':
                K[i,j] = kernel_rbf_linear(params,x_i=X[i],x_j=X[j])
            elif kernel == 'exponential':
                K[i,j] = kernel_exponential(params,x_i=X[i],x_j=X[j])
            elif kernel == 'periodic':
                K[i,j] = kernel_periodic(params,x_i=X[i],x_j=X[j])      
    return K + K.T - np.diag(K.diagonal())

def get_K_delta(params: dict, X: np.ndarray, d: int, kernel='rbf_kernel') -> np.ndarray:
    """
    kernel matrix differenciated wrt 'd' th parameter
    """
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

def get_K_off_diag(params: dict, Xi: np.ndarray, Xj: np.ndarray, kernel='rbf_kernel') -> np.ndarray:
    """
    off-diaonal kernel matrix 
    """
    ni,nj = len(Xi),len(Xj)
    K = np.zeros([ni, nj]) # asymmetrical dimension 
    for i in range(ni):
        for j in range(nj):
            if kernel == 'rbf_kernel':
                K[i,j] = kernel_rbf(params,x_i=Xi[i],x_j=Xj[j])
            elif kernel == 'rbf_kernel_linear':
                K[i,j] = kernel_rbf_linear(params,x_i=Xi[i],x_j=Xj[j])
            elif kernel == 'exponential':
                K[i,j] = kernel_exponential(params,x_i=Xi[i],x_j=Xj[j])
            elif kernel == 'periodic':
                K[i,j] = kernel_periodic(params,x_i=Xi[i],x_j=Xj[j])      
    return K

