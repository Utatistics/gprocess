import functools
import numpy as np

from gprocess.core.numerical import numerical_gradient
from gprocess.core.likelihood import get_L_neg

def scaled_conjugate_gradient(theta_init: np.ndarray, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    """scaled conjugate gradient method  


    Args
    ----

    Returns
    -------

    """

    lmd_init = kwargs.get('lmd_init',1e-5)
    scale = kwargs.get('scale',1e-4) # constant parameter 
    epsilon =  kwargs.get('epsilon',1e-5) # for convergence check 
    
    kernel = kwargs.get('kernel','rbf_kernel')
    likelihood_f_neg = functools.partial(get_L_neg, X=X, y=y, kernel=kernel) # evaluates to scalar
    E1 = functools.partial(numerical_gradient, f=likelihood_f_neg) # evaluates to vector 

    # scg routine implementation
    N = len(theta_init)
    k = 1 # setting the iteration index to 1
   
    while True:
        print('==== scg routine initiated; k = {} ===='.format(k))
        
        # step 0: initial iteration.
        if k == 1: 
            theta_current = theta_init
            r_current = - E1(x=theta_current) # global negative likelihood function evaluated at kth iter.
            m_current = r_current # search direction i.e. conjugate vector
            lmd = lmd_init # scale parameter 
            lmd_hat = 0 # raised scale parameter 
            success = True # indicator
        else: 
            pass
        
        # step 1: recalibrating the Hesian to be semi-positive definite
        if success == True: # occures only if lmd gets scaled 
            sigma = scale / np.linalg.norm(m_current)
            s = (E1(x=theta_current + (m_current * sigma)) - E1(x=theta_current)) / sigma
            delta = m_current.T @ s
            
            if delta <= 0: #  ensuring the Hessian matrix to be positive definite.
                lmd_hat = 2 * (lmd -  (delta / (np.linalg.norm(m_current) ** 2)))
                delta = - delta + lmd * (np.linalg.norm(m_current) ** 2)
                lmd = lmd_hat
            else:
                pass
                
        # step 2: regulating 'lmd' i.e. the scale paramter to ensure the quality of approximation 
        mu = m_current.T @ r_current
        alpha = mu / delta # calculate the step size.
        l_sum = (likelihood_f_neg(theta_current) - likelihood_f_neg(theta_current + (alpha * m_current)))
        cmp = (2 * delta * l_sum / (mu ** 2))[0][0] # comparison parameter indicating the quality of approximation
        
        if cmp  >= 0.75: # reducing the scale parameter
            lmd *= .2
        elif cmp  <= 0.25: # increase the scale parameter
            lmd += (delta * (1 - cmp) /  np.linalg.norm(m_current)**2)
        else: 
            lmd_hat = lmd
            success = False
        
       # step 3: computing conjugate gradient after scaling 
        theta = theta_current + alpha * m_current # update theta
        r = - E1(x=theta)
        lmd_hat = 0
        success = True 
        if k % N == 0:
            m = r # RESTART ALGORITHM?
        else:
            beta = (np.linalg.norm(r) ** 2 - r.T @ r_current) / mu
            m = r + beta * m_current
            
        # step 4: checking for convergence
        if np.linalg.norm(r) > epsilon:   
            sigma_current = sigma
            r_current = r 
            m_current = m
            likelihood_current = - likelihood_f_neg(theta_current)[0][0]
            likelihood = - likelihood_f_neg(theta)[0][0] # update likelihood
            theta_current = theta
            if k > 1:
                print('likelihood at k = {}: {:.3f}'.format(k, likelihood))
            k += 1
            print('theta_current: {}'.format(theta_current))
            print('convergence not achieved...\n')
            continue
        else:
            print('optimisation successfully converged.\n')
            break # breaking the loop upon ocnversion 
            
    return theta
