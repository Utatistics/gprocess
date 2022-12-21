import time
import numpy as np

from gprocess.optimization.cgl import conjugate_gradient
from gprocess.optimization.scg import scaled_conjugate_gradient
from gprocess.core.kernel import get_K, get_K_off_diag
from gprocess.core.prediction import GPpred

class GProcess:
    """main class implementing the method chain to allow for the one-stop use of algorithm.


    """
    def __init__(self, X: np.ndarray, y: np.ndarray, kernel='rbf_kernel') -> None:
        """initialize with two np.ndarary 

        Args
        ----
        X : np.ndarray
            2-dimensional array for training data
        y : np.ndarray
            1-dimensional array for prediction date
        kernel : string
            kernel option; 'rbf_kernel', 'rbf_kernel_linear', 'exponential, 'periodic', 'kernel_linear'
        """
        
        self.X = X
        self.y = y
        self.kernel = kernel
                    
   
    def _init_params(self):
        """for kernel, temporary non-zero initial values are set.

        """
  
        if self.kernel == 'rbf_kernel':
            self.theta_init = np.array([1.25,.7])
        elif self.kernel == 'rbf_kernel_linear':
            self.theta_init = np.array([1.25,.7,1.])  
        elif self.kernel == 'exponential':
            self.theta_init = np.array([1.])
        elif self.kernel == 'periodic':
            self.theta_init = 'theta_init',np.array([1.,1.]) 
        
    
    def fit(self, method='cfl'):
        """run hyper-parameter tuning for kernel, by calling oprimisation routine 
        
        Args
        ----
        methods : str
            optimization algorithms option; 'cgi', 'scg' 
        """

        start_time = time.time()
        self._init_params()
        self.method = method
        
        if self.method == 'cgl':
            self.theta = conjugate_gradient(self.theta_init,self.X,self.y,kernel=self.kernel) 
        if self.method == 'scg':
            self.theta = scaled_conjugate_gradient(self.theta_init,self.X,self.y,kernel=self.kernel)
        
        self.K00 = get_K(params=self.theta,X=self.X,kernel=self.kernel)  # using tuned hyper-parameters
        self.mle = (-np.log(np.linalg.det(self.K00)) - self.y.T @ np.linalg.inv(self.K00) @ self.y)[0][0]
      
        print("\noptimisation time: {:.2f} sec.".format(time.time() - start_time))
         
        return self
               

    def pred(self,X_test):
        """run prediction

        Args
        ----
        X_test : np.ndarray
            2-dimensional array for training data
        """
        
        self.X_test = X_test
        gppred = GPpred()

        self.K11 = get_K(params=self.theta,X=self.X_test,kernel=self.kernel) # using tuned hyper-parameters
        self.K01 = get_K_off_diag(self.theta,Xi=self.X,Xj=self.X_test) # see module_pred.py 
        self.K10 = self.K01.T
        self.predDataFrame = gppred.gp_pred(self.X,self.X_test,self.y,self.K00,self.K01,self.K10,self.K11)
        self.y_pred = gppred.get_y_pred(self.predDataFrame)
        
        # method chain
        return self 
    