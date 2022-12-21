import numpy as np
import pandas as pd

class GPpred:
    def __init__(self) -> None:
        pass

    def gp_pred(X,X_test,y,K00,K01,K10,K11):
        """
        """ 
        x = np.concatenate([X,X_test])
        gp_mu = K10 @ np.linalg.inv(K00) @ y
        gp_cov = K11 - (K10 @ np.linalg.inv(K00) @ K01)
        dim = x.shape[1]
        indices = ['x' + str(i) for i in range(dim)]
        
        predDataFrame = pd.DataFrame(x,columns=indices)
        predDataFrame['mid'] = np.concatenate([y,gp_mu])
        predDataFrame['lwr'] = np.concatenate([y,gp_mu - 1.96*np.diag(gp_cov).reshape(-1,1)])
        predDataFrame['upr'] = np.concatenate([y,gp_mu + 1.96*np.diag(gp_cov).reshape(-1,1)])
        predDataFrame = predDataFrame.sort_values(indices) 
        
        return predDataFrame

    def get_y_pred(predDataFrame):
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