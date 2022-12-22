import numpy as np
import pandas as pd
from gprocess.core.matrix import Matrix

    

def get_xpred(matrix: Matrix) -> pd.DataFrame:
    """return prediction interval for the given data points

    Args
    ----
    matrix : Matrix
        dict-like object containing matrices 

    Returns
    -------
    pred_df : pd.DataFrame
        dataframe containing the result of prediction inverval

    """    
    x = np.concatenate([matrix['X'], matrix['X_test']])
    gp_mu = matrix['K10'] @ np.linalg.inv(matrix['K00']) @ matrix['y']
    gp_cov = matrix['K11'] - (matrix['K10'] @ np.linalg.inv(matrix['K00']) @ matrix['K01'])
    dim = x.shape[1]
    indices = ['x' + str(i) for i in range(dim)]
    
    pred_df = pd.DataFrame(x, columns=indices)
    pred_df['mid'] = np.concatenate([matrix['y'], gp_mu])
    pred_df['lwr'] = np.concatenate([matrix['y'], gp_mu - 1.96*np.diag(gp_cov).reshape(-1,1)])
    pred_df['upr'] = np.concatenate([matrix['y'], gp_mu + 1.96*np.diag(gp_cov).reshape(-1,1)])
    
    return pred_df.sort_values(indices) 


def get_ypred(matrix: Matrix) -> pd.DataFrame:
    return



def variational_bayesian_method():
    pass 

def inducing_variable_method():
    pass 

def SoD():
    """
    subset of data approximation
    """
    pass  