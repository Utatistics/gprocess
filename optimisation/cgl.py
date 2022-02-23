# module doscription
"""*******************************************************************************************************
$ This module implements the optimisaiton routine 'cgl' required for .fit method in the main class 'GProcess'.
$ Inner dependency:
- gprocess.core.numerical
- gprocess.core.likelihood

*******************************************************************************************************"""
# initialising the library
import gprocess

# outer dependency 
import functools
import numpy as np

# test linking
def test_cgl():
    print("Hello, I'm cgl from optimisation!")

"""*************************
# line search implementation 
*************************"""
def line_search(f, t_init, h_init, epsilon): # obtain the param t via gradient search routine 
    t = t_init 
    h = h_init
    print('line search initialised...')
    while abs(gprocess.numerical_diff(f, t) < epsilon):
        print('numerical diff evaluated!')
        h = np.sign(gprocess.numerical_diff(f, h)) * abs(h)    
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

"""*************************
# conjugate gradient method  
*************************"""
def conjugate_gradient(theta_init, X, y, **kwargs): # the standard conjugate gradient method with line search
    kernel = kwargs.get('kernel','rbf_kernel')
    delta =  kwargs.get('delta',1e-5)
    iter_max = kwargs.get('iter_max',100)
    
    cnt = 1
    theta_current = theta_init
    m_current = np.ones_like(theta_init) # should not be set to 0 vector
    while cnt <= iter_max:
        L_delta = gprocess.get_L_delta(theta_current, X, y, kernel) # output is a vector     
        likelihood_f = functools.partial(gprocess.get_L, X=X, y=y, kernel=kernel)
        H = gprocess.numerical_hessian(f=likelihood_f, x=theta_current)
        if cnt == 1:
            alpha = 0
        else:
            alpha = - (m_current @ H @ L_delta) / (m_current @ H @ m_current)
        m = L_delta + (alpha * m_current) # obtain conjugate vector
        linesearch_f = lambda t: likelihood_f(params=theta_current + t * m)[0][0] ## extract scalar 
        t_init,h_init,epsilon = 1e-4,1e-4,1e-4
        print('t_init: ',t_init)
        t = line_search(f=linesearch_f,t_init=t_init,h_init=h_init,epsilon=epsilon) # line search routine
        print('t: ',t)
        theta = theta_current + t * m
        
        d = np.sqrt((t * m[0])**2 + (t*m[1])**2) # check convergence
        if d < delta:
            print("optimisation successfully converged.")
            break
        if np.isnan(theta[0]):
            print("optimisation failed due to vanishing gradinet.")
            break
        
        m_current = m # update m_current  
        theta_current = theta # update theta 
        print('==== iteration {}: theta_current = {} ===='.format(cnt, theta))
        cnt += 1        
    return theta
