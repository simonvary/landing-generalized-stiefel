import time
import numpy as np
from scipy.optimize import line_search

from numpy.linalg import norm

from .solver import IterativeSolver
from .manifold_func import *

class RiemannianSteepestDescent(IterativeSolver):
    '''
    https://arxiv.org/pdf/2209.03480.pdf, Sec.5

        min -Tr(X^T A X), s.t. X^T X = Id
    
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def solve(self, A, k, eta, x0=None, tol=1e-10,geodesic = True, k_norm = 2):
        """
        """

        m = A.shape[0]

        if x0 is None:
            x = np.eye(m)[:,:k]
        else:
            x = x0.copy()

        self._start_optlog(extraiterfields = ['gradnorm'])
        
        iter_n = 0
        # Iter_n append
        time0 = time.time()

        while True:
            iter_n = iter_n + 1 
            Ax = A@x
            xtAx = x.T @ Ax
            grad = 2*(-Ax + x@(xtAx))

            if geodesic:
                w = grassmannian_exp(x, -eta*grad)
                if iter_n % k_norm == 0:
                    x, _ = np.linalg.qr(w)
                else:
                    x = w
            else:
                x, _ = np.linalg.qr(x-eta*grad)
            
            gradnorm = norm(grad)
            objective_value = -np.trace(x.T@A@x)
            running_time = time.time() - time0

            if self._logverbosity >= 1:
                self._append_optlog(iter_n, running_time, objective_value, gradnorm=gradnorm, xdist=None)

            stop_reason = self._check_stopping_criterion(
                running_time, iter=iter_n, objective_value=objective_value, stepsize=eta, gradnorm=gradnorm)

            if stop_reason:
                if self._verbosity >= 1:
                    print(stop_reason)
                    print('')
                break
        
        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(iter, -np.trace(x.T@A@x), stop_reason, running_time, stepsize=eta, gradnorm=gradnorm)
            return x, self._optlog

if __name__ == 'main':
    pass