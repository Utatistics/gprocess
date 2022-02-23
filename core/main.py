# module doscription
"""****************************************************
$ The module 'main' implements the main class 'GProcess' 
$ Inner dependency:
- gprocess.core.kernel
- gprocess.core.pred

- gprocess.optimisation *
****************************************************"""
# initialising the library
import gprocess

# outer dependency
import time
import numpy as np

# test linking
def test_main():
    print("Hello, I'm main from core!")
    
"""*************************
# main class implementation 
*************************"""
class GProcess:
    def __init__(self, X, y, **kwargs): # initialise instance.
        self.kernel = kwargs.get('kernel','rbf_kernel') # set rbf kernel as a default kernel
        self.std = kwargs.get('standardise',False) # set rbf kernel as a default kernel
        self.X = X
        self.y = y
        if self.std == True:
            self.X_std = std(X)
            self.y_std = std(y)
            
    def fit(self, **kwargs): # hyper parameter tuning.
        self.method = kwargs.get('method','cgl')
        # setting initial value for optimisation routine    
        if self.kernel == 'rbf_kernel':
            self.theta_init = kwargs.get('theta_init',np.array([1.25,.7])) # should not be set to 0 vector.
        elif self.kernel == 'rbf_kernel_linear':
            self.theta_init = kwargs.get('theta_init',np.array([1.25,.7,1.]))   
        elif self.kernel == 'exponential':
            self.theta_init = kwargs.get('theta_init',np.array([1.]))
        elif self.kernel == 'periodic':
            self.theta_init = kwargs.get('theta_init',np.array([1.,1.])) 
        
        # callign oprimisation routine 
        start_time = time.time()
        if self.std == False: 
            input_routine_fit(self,self.X,self.y,**kwargs) 
        else:
            input_routine_fit(self,self.X_std,self.y_std,**kwargs) 
        print('\noptimisation time: {:.2f} sec.'.format(time.time() - start_time))
        
        # method chain 
        return self
               
    def pred(self,X_test): # prediction.
        self.X_test = X_test
        if self.std == True:
            self.X_test_std = std(X_test)
            
        if self.std == False:
            input_routine_pred(self,self.X_test)
        else:
            input_routine_pred(self,self.X_test_std)
        
        # method chain
        return self 
    
"""**************************
# sub functions within class 
**************************"""
# standardisation function. *see .__init__
def std(x):
    return (x - np.nanmean(x,axis=0)) / np.nanstd(x,axis=0)

# general input routine. *see .fit
def input_routine_fit(self,X,y,**kwargs): # return tuened hyper-parameters 'theta' and compute kernel matrix 
    if self.method == 'cgl':
        self.theta = gprocess.conjugate_gradient(self.theta_init,X,y,kernel=self.kernel,**kwargs) 
    if self.method == 'scg':
        self.theta = gprocess.scaled_conjugate_gradient(self.theta_init,X,y,kernel=self.kernel,**kwargs)
    
    self.K00 = gprocess.get_K(params=self.theta,X=X,kernel=self.kernel)  # using tuned hyper-parameters
    self.mle = (-np.log(np.linalg.det(self.K00)) - self.y.T @ np.linalg.inv(self.K00) @ self.y)[0][0]

# general input routine. *see .pred
def input_routine_pred(self,X_test):
    self.K11 = gprocess.get_K(params=self.theta,X=X_test,kernel=self.kernel) # using tuned hyper-parameters
    self.K01 = gprocess.get_K_off_diag(self.theta,Xi=self.X,Xj=X_test) # see module_pred.py 
    self.K10 = self.K01.T
    self.predDataFrame = gprocess.gp_pred(self.X,X_test,self.y,self.K00,self.K01,self.K10,self.K11)
    self.y_pred = gprocess.get_y_pred(self.predDataFrame)
    
