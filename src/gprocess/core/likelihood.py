import gprocess
from gprocess.core.kernel import get_K
import numpy as np

    
def get_L(params: dict, X: np.ndarray, y: np.ndarray, kernel: str) -> np.float64:
    """analytiaclly obtained likelihood function.

    Args
    ----

    Returns
    -------

    """
    K_theta = get_K(params=dict, X=X, kernel=kernel)
    K_theta_inv = np.linalg.inv(K_theta) # C++ IMPLEMENTATION!!
    det = np.linalg.det(K_theta) 
    if det == 0.0: # avoid divergence 
        return -np.inf
    else:
        return -np.log(det) - y.T @ K_theta_inv @ y # a positive definite and its dterminant

def get_L_neg(params: dict, X: np.ndarray, y: np.ndarray, kernel: str) -> np.float64:
    """negative likelihood function 


    Args
    ----

    Returns
    -------

    """
    return - get_L(params, X, y, kernel)

def get_L_delta(params: dict, X: np.ndarray, y: np.ndarray, kernel: str) -> np.ndarray:
    """analytically derived gradient vector of likelihood function.

    Args
    ----

    Returns
    -------

    """
 
    K_theta = gprocess.get_K(params, X, kernel=kernel)
    K_theta_inv = np.linalg.inv(K_theta) 
    L_delta = np.zeros_like(params)
    for d in range(len(params)):
        K_delta = gprocess.get_K_delta(params, d, X, kernel=kernel)
        L_delta[d] = -np.trace(K_theta_inv @ K_delta) + (K_theta_inv @ y).T @ K_delta @ (K_theta_inv @ y) 
    return L_delta

