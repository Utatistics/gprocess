import time
import numpy as np

from gprocess.core.matrix import Matrix
from gprocess.optimization.cgl import conjugate_gradient
from gprocess.optimization.scg import scaled_conjugate_gradient
from gprocess.core.kernel import get_K, get_K_off_diag
from gprocess.core.likelihood import get_mle
from gprocess.core.prediction import get_xpred, get_ypred


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
            1-dimensional array for prediction data
        kernel : string
            kernel option -> 'rbf_kernel', 'rbf_kernel_linear', 'exponential, 'periodic', 'kernel_linear'
        
        """
        
        self.kernel = kernel

        self.mat = Matrix()
        self.mat['X'] = X
        self.mat['y'] = y
                    
   
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
            self.theta_init = np.array([1.,1.]) 
        
    
    def fit(self, method='cgl'):
        """run hyper-parameter tuning for kernel, by calling oprimisation routine 
        
        Args
        ----
        methods : str
            optimization algorithms option -> 'cgi', 'scg' 

        Returns
        -------
        self : GProcess
            enabling method chain
        
        """

        start_time = time.time()
        self._init_params()
        self.method = method
        
        if self.method == 'cgl':
            self.theta = conjugate_gradient(matrix=self.mat, theta_init=self.theta_init, kernel=self.kernel) 
        if self.method == 'scg':
            self.theta = scaled_conjugate_gradient(matrix=self.mat, theta_init=self.theta_init, kernel=self.kernel)
        
        self.mat['K00'] = get_K(matrix=self.mat, params=self.theta, kernel=self.kernel)  # using tuned hyper-parameters
        self.mle = get_mle(matrix=self.mat)
      
        print("\noptimisation time: {:.2f} sec.".format(time.time() - start_time))
         
        return self
               

    def pred(self, Xt: np.ndarray):
        """run prediction

        Args
        ----
        X_test : np.ndarray
            2-dimensional array for training data

        Returns
        -------
        self : GProcess
            enabling method chain
        
        """

        self.mat['Xt'] = Xt
        self.mat['K11'] = get_K(matrix=self.mat, params=self.theta, kernel=self.kernel) # using tuned hyper-parameters
        self.mat['K01'] = get_K_off_diag(matrix=self.mat, params=self.theta) # see module_pred.py 
        self.mat['K10'] = self.mat['K01'].T

        self.pred_df = get_xpred(matrix=self.mat)
        self.y_pred = get_ypred(matrix=self.mat)
        
        return self 
    