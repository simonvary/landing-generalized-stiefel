import time
import numpy as np

from numpy.linalg import norm

from .solver import IterativeSolver

class SimultaneousIteration(IterativeSolver):
    '''
    Algorithm 28.3. NLA Trefethen & Bau
    also:
    https://netlib.org/utk/people/JackDongarra/etemplates/node98.html
    
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def solve(self, A, k, x0=None, tol=1e-10):
        """
        Arguments:
            - A (np.array) or a scipy.sparse.linalg.LinearOperator
            - x0=None
                Optional parameter. Starting point. If none
                then a starting point will be computed from A.rmatvec(b).
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
            - outlog
        """

        m = A.shape[0]
        diag_ind = np.matrix(np.eye(k, dtype = bool))

        if x0 is None:
            x = np.eye(m)[:,:k]
        else:
            x = x0.copy()
        
        self._start_optlog(extraiterfields = ['gradnorm'])

        iter_n = 0
        time0 = time.time()

        while True:
            iter_n = iter_n + 1

            z = A@x
            x, R = np.linalg.qr(z, mode = 'reduced')   

            gradnorm = norm(R[~diag_ind])
            objective_value = -np.trace(x.T@A@x)
            running_time = time.time() - time0

            if self._logverbosity >= 1:
                self._append_optlog(iter_n, running_time, objective_value, gradnorm=gradnorm, xdist=None)

            stop_reason = self._check_stopping_criterion(
                running_time, iter=iter_n, objective_value=objective_value, gradnorm=gradnorm)

            if stop_reason:
                if self._verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(iter, -np.trace(x.T@A@x), stop_reason, running_time, gradnorm=gradnorm)
            return x, self._optlog

if __name__ == 'main':
    pass