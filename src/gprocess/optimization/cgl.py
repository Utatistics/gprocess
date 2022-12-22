import functools
import numpy as np
from typing import Callable

from gprocess.core.matrix import Matrix
from gprocess.core.numerical import numerical_diff, numerical_hessian
from gprocess.core.likelihood import get_L, get_L_delta


def line_search(f: Callable[[np.float64], np.float64], t_init: np.float64, h_init: np.float64, epsilon: np.float64) -> np.float64: 
    """obtain the param t via gradient search routine

    Args
    ----
    f : function


    Returns
    -------
    t : np.float64

    """

    t = t_init 
    h = h_init

    print('line search initialised...')
    while abs(numerical_diff(f, t) < epsilon):
        print('numerical diff evaluated!')
        h = np.sign(numerical_diff(f, h)) * abs(h)    
        T = t
        Th = t + h 
        if f(T) < f(Th):
            while f(T) <= f(Th):
                h = 2 * h
                T = Th
                Th = T + h
            t = T
            h = h / 2
        else:
            while f(T) >= f(Th):
                h = h / 2
                Th = Th - h
            t = Th
            h = 2 * h 
    print('line search completed!')    
    return t


def conjugate_gradient(matrix: Matrix, theta_init: np.ndarray, **kwargs) -> np.ndarray:
    """the standard conjugate gradient method with line search

    Args
    ----
     matrix : Matrix
        dict-like object containing matrices
    
    theta_init : np.ndarray
        initial value of theta

    Returns
    -------
    theta : np.ndarray 
        optimized theta 

    """

    kernel = kwargs.get('kernel', 'rbf_kernel')
    delta =  kwargs.get('delta', 1e-5)
    iter_max = kwargs.get('iter_max', 100)
    
    cnt = 1
    theta_current = theta_init
    m_current = np.ones_like(theta_init) # should not be set to 0 vector
    while cnt <= iter_max:
        # step 1: compute quadratic approximation 
        print('==== iteration {}: theta_current = {} ===='.format(cnt, theta_current))
        L_delta = get_L_delta(matrix=matrix, params=theta_current, kernel=kernel) # output is a vector     
        likelihood_f = functools.partial(get_L, matrix=matrix, kernel=kernel)
        H = numerical_hessian(f=likelihood_f, x=theta_current)
        if cnt == 1:
            alpha = 0
        else:
            alpha = - (m_current @ H @ L_delta) / (m_current @ H @ m_current)
        m = L_delta + (alpha * m_current) # obtain conjugate vector
        
        # step 2: conductthe line search
        linesearch_f = lambda t: likelihood_f(params=theta_current + t * m)[0][0] ## extract scalar 
        t_init,h_init,epsilon = 1e-4,1e-4,1e-4
        t = line_search(f=linesearch_f, t_init=t_init, h_init=h_init, epsilon=epsilon) # line search routine
        theta = theta_current + t * m
        
        # step 3: check for convergence
        d = np.sqrt((t * m[0])**2 + (t*m[1])**2)
        if d < delta:
            print("optimisation successfully converged.")
            break
        if np.isnan(theta[0]):
            print("optimisation failed due to vanishing gradinet.")
            break
        
        # step 4: update the paramteres
        m_current = m # update m_current  
        theta_current = theta # update theta
        
        likelihood = likelihood_f(params=theta_current)[0][0]
        print('likelihood at k = {}: {}'.format(cnt, likelihood),'\n')
        cnt += 1
        
    return theta
