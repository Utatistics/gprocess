# module doscription
"""********************************************************************************************************
This module contains a set of computational algorithms implementaiton generally required for the main class 'GProcess' *see input_pred_routine().
********************************************************************************************************"""
import numpy as np
import pandas as pd
# from gprocess.module_kernel import *

def test_pred():
    print("Hello, I'm pred from core!")
    
# def variational_bayesian_method():
# def inducing_variable_method():
# def SoD(): # subset of data approximation

def gp_pred(X,X_test,y,K00,K01,K10,K11): # 'pred()' will be called by 'input_routine_pred()' in main.py
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