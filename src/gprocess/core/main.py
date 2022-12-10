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
    
# class implementation 
class GProcess:
    def __init__(self, X, y, **kwargs): # initialise instance.
        self.kernel = kwargs.get('kernel','rbf_kernel') # set rbf kernel as a default kernel
        self.X = X
        self.y = y
                    
    def fit(self, **kwargs): # hyper parameter tuning.
        self.method = kwargs.get('method','cgl')
        
        # intial values setting   
        if self.kernel == 'rbf_kernel':
            self.theta_init = kwargs.get('theta_init',np.array([1.25,.7])) # should not be set to 0 vector.
        elif self.kernel == 'rbf_kernel_linear':
            self.theta_init = kwargs.get('theta_init',np.array([1.25,.7,1.]))   
        elif self.kernel == 'exponential':
            self.theta_init = kwargs.get('theta_init',np.array([1.]))
        elif self.kernel == 'periodic':
            self.theta_init = kwargs.get('theta_init',np.array([1.,1.])) 
        
        # calling oprimisation routine 
        start_time = time.time()
        
        if self.method == 'cgl':
            self.theta = gprocess.conjugate_gradient(self.theta_init,self.X,self.y,kernel=self.kernel,**kwargs) 
        if self.method == 'scg':
            self.theta = gprocess.scaled_conjugate_gradient(self.theta_init,self.X,self.y,kernel=self.kernel,**kwargs)
        
        self.K00 = gprocess.get_K(params=self.theta,X=self.X,kernel=self.kernel)  # using tuned hyper-parameters
        self.mle = (-np.log(np.linalg.det(self.K00)) - self.y.T @ np.linalg.inv(self.K00) @ self.y)[0][0]
      
        print('\noptimisation time: {:.2f} sec.'.format(time.time() - start_time))
        
        # method chain 
        return self
               
    def pred(self,X_test): # prediction.
        self.X_test = X_test
        
        self.K11 = gprocess.get_K(params=self.theta,X=self.X_test,kernel=self.kernel) # using tuned hyper-parameters
        self.K01 = gprocess.get_K_off_diag(self.theta,Xi=self.X,Xj=self.X_test) # see module_pred.py 
        self.K10 = self.K01.T
        self.predDataFrame = gprocess.gp_pred(self.X,self.X_test,self.y,self.K00,self.K01,self.K10,self.K11)
        self.y_pred = gprocess.get_y_pred(self.predDataFrame)
        
        # method chain
        return self 
    