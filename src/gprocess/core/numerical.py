import numpy as np 


def numerical_diff(f: function, x: np.float64) -> np.float64:
    """numerical diffrentiation
    
    Args
    ----
    f : function
        univariate function to be evaluated
    x : np.float64
        point of evaluation 

    Returns
    -------
    
    """
    
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_diff2(f: function, x):
    """numerical 2nd-order diffrentiation
    
    Args
    ----
    f : function
        univariate function to be evaluated
    x : np.float64
        point of evaluation 

    Returns
    -------
    
    """

    h = 1e-4
    df0 = numerical_diff(f,x + h) 
    df1 = numerical_diff(f,x - h)
    return (df0 - df1) / (2 * h)

def numerical_gradient(f: function, x: np.ndarray) -> np.ndarray:
    """numerical 2nd-order diffrentiation
    
    Args
    ----
    f : function
        multivariate function to be evaluated
    x : np.float64
        point of evaluation 

    Returns
    -------
    
    """

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

def numerical_diff_partial(f: function, x: np.ndarray, dim: int) -> np.ndarray: # x is a input vector
    """numerical 2nd-order diffrentiation
    
    Args
    ----
    f : function
        univariate function to be evaluated
    x : np.float64
        point of evaluation 

    Returns
    -------
    
    """

    h = 1e-4
    tmp = x.copy()
    tmp[dim] += h
    x0 = tmp
    tmp = x.copy()
    tmp[dim] -= h
    x1 = tmp
    return (f(x0) - f(x1)) / (2 * h)

def numerical_diff2_partial(f: function, x: np.ndarray, dim0: int, dim1: int) -> np.ndarray:
    """numerical 2nd-order diffrentiation
    
    Args
    ----
    f : function
        univariate function to be evaluated
    x : np.float64
        point of evaluation 

    Returns
    -------
    
    """

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

def numerical_hessian(f: function, x: np.ndarray) -> np.ndarray:
    """numerical 2nd-order diffrentiation
    
    Args
    ----
    f : function
        univariate function to be evaluated
    x : np.float64
        point of evaluation 

    Returns
    -------
    
    """

    n = x.shape[0]
    H = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j < i:
                continue
            H[i,j] = numerical_diff2_partial(f, x, i, j)
    
    return H + H.T - np.diag(H.diagonal())
