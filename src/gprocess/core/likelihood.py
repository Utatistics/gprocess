import numpy as np 
from gprocess.core.kernel import get_K, get_K_delta
from gprocess.core.matrix import Matrix

    
def get_L(matrix: Matrix, params: dict, kernel: str) -> np.float64:
    """analytiaclly obtained likelihood function.

    Args
    ----
    matrix : Matrix
        dict-like object containing matrices
    
    params : dict
        dict onject containig the parameters
    
    kernel : str
                kernel option -> 'rbf_kernel', 'rbf_kernel_linear', 'exponential, 'periodic', 'kernel_linear'        

    Returns
    -------
        : np.float64
        the computed value of likelihood for given params
    """

    K_theta = get_K(params=params, X=matrix['X'], kernel=kernel)
    K_theta_inv = np.linalg.inv(K_theta) # C++ IMPLEMENTATION!!
    det = np.linalg.det(K_theta) 
    if det == 0.0: # avoid divergence 
        return -np.inf
    else:
        return -np.log(det) - matrix['y'].T @ K_theta_inv @ matrix['y'] # a positive definite and its dterminant


def get_L_ng(matrix: Matrix, params: dict, kernel: str) -> np.float64:
    """negative likelihood function 

    Args
    ----
    matrix : Matrix
        dict-like object containing matrices
    
    params : dict
        dict onject containig the parameters
    
    kernel : str
                kernel option -> 'rbf_kernel', 'rbf_kernel_linear', 'exponential, 'periodic', 'kernel_linear'        

    Returns
    -------
        : np.float64
        the computed value of negative likelihood for given params
    """

    return - get_L(matrix=matrix, params=params, kernel=kernel)


def get_L_delta(matrix: Matrix, params: dict, kernel: str) -> np.ndarray:
    """analytically derived gradient vector of likelihood function.

    Args
    ----
    matrix : Matrix
        dict-like object containing matrices
    
    params : dict
        dict onject containig the parameters
    
    kernel : str
        kernel option -> 'rbf_kernel', 'rbf_kernel_linear', 'exponential, 'periodic', 'kernel_linear'   

    Returns
    -------
    L_delta : np.ndarray
        gradient vector

    """
 
    K_theta = get_K(params=params, X=matrix['X'], kernel=kernel)
    K_theta_inv = np.linalg.inv(K_theta) 
    L_delta = np.zeros_like(params)
    for d in range(len(params)):
        K_delta = get_K_delta(params=params, X=matrix['X'], d=d, kernel=kernel)
        L_delta[d] = -np.trace(K_theta_inv @ K_delta) + (K_theta_inv @ matrix['y']).T @ K_delta @ (K_theta_inv @ matrix['y']) 
    
    return L_delta


def get_mle(matrix: Matrix) -> np.float64:
    """returns the value of the maximum likelihood estimator

    Args
    ----
    matrix : Matrix
        dict-like object containing matrices

    Returns
    -------

    """

    return (-np.log(np.linalg.det(matrix.K00)) - matrix['y'].T @ np.linalg.inv(matrix['K00']) @ matrix['y'])[0][0]