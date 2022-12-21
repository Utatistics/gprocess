import numpy as np
import pandas as pd 

 
def rmnorm(mu: int, cov: int):
    """
    generate sampling from multicaraite Gaussian

    """
    n = len(mu)
    L = np.linalg.cholesky(cov)
    x = np.random.normal(loc=0,scale=1,size=n)
    return L @ x + mu


def generate_parameters(n: int):
    """
    generate GP paramters
    """
    mu = np.random.normal(0, 1, n)
    C = np.identity(n)
    for i in range(n):
        mu[i] = np.random.gamma(n) # mean vector
        for j in range(n): 
            if i>j:
                continue
            elif i==j:
                C[i,i] = i**4 + 2*i  + abs(np.random.normal())
            else:
                C[i,j] = np.random.uniform(-1,1,1)

    C = C + C.T - np.diag(C.diagonal()) # covariance mattix
    return mu, C
