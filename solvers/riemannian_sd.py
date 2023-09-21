import time
import numpy as np
import cupy as cp
from cupy.linalg import norm
from cupy import trace

import scipy


from .solver import IterativeSolver

def symm(x):
    return(0.5*(x + x.T))

class RiemmGeneralizedStiefel(IterativeSolver):
    ''' Riemannian SD (with lineseach) for solving

            min -0.5*Tr(X^T A X), s.t. X^T B X = Id
        
        with retractions based on the paper:
        
        'Cholesky QR-based retraction on the generalized Stiefel manifold, H. Sato and K. Aihara, 2019'
        https://link.springer.com/article/10.1007/s10589-018-0046-7

    '''
    def __init__(self, A, B, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = cp.asarray(A)
        self.B = cp.asarray(B)
        self.n = B.shape[0]
        self.p = p
        self.objective = lambda x: -0.5*cp.trace(x.T@self.A @ x).item()
        self.Id = cp.eye(p)
        self.constraint = lambda x: .25*(cp.linalg.norm(x.T@self.B@x - self.Id, 'fro')**2).item()
        pass
    
    def _retraction(self, B, x, v, check_tol=1e-10):
        a = x + v
        r = cp.linalg.cholesky(a.T @ B @ a)
        return cp.linalg.solve(r, a.T).T

    def _line_search(self, B, x, direction, eta, df0, gamma = 1e-2, tau = 1e-1, max_iter = 25):
        ''' 
        Backtracking line-search
        
        Input:
            x - current iterate
            direction - direction
            eta0 - initial step-size
            df0 : directional derivative at x along d
            gamma : control parameter
            tau : reduction parameter
        '''

        f0 = self.objective(x)
        it = 1
        x_new = self._retraction(B, x, eta*direction)
        f_new = self.objective(x_new)
        while f_new > f0 + gamma*eta*df0:
            # Reduce the step size and look closer
            eta = tau * eta
            x_new = self._retraction(B, x, eta*direction) 
            f_new = self.objective(x_new)
            it = it + 1
            if it == max_iter:
                Warning('No descent direction found!')
                return(x, f0, eta, it)
        if it == 1:
            eta = 2*eta    
        return (x_new, f_new, eta, it)

    def solve(self, eta = 0.5, step_type = 'fixed', x0=None):
        """
        """
        n = self.n
        p = self.p

        if x0 is None:
            x = np.eye(self.n)[:,:p]
        else:
            x = x0.copy()
        x = cp.asarray(x)

        self._start_optlog(extraiterfields = ['gradnorm', 'distance'])
        iter_n = 0

        # Iter_n append
        time0 = time.time()
        while True:
            B = self.B 
            iter_n = iter_n + 1
            egrad = -self.A @x
            egrad_scaled = cp.linalg.solve(B, egrad)
            rgrad = egrad_scaled - x@symm(x.T @ egrad)
            if step_type == 'fixed':
                x = self._retraction(B, x, -eta*rgrad)
                f_new = self.objective(x)
                dist_new = self.constraint(x)
            elif step_type == 'lsearch':
                df0 = cp.inner(-rgrad.flatten(), egrad.flatten()) 
                x, f_new, eta, lsearch_iter = self._line_search(B, x, -rgrad, eta, df0, max_iter=25)
                dist_new = self.constraint(x)
            
            gradnorm = norm(rgrad, 'fro').item()
            
            print('Iteration %d, Objective: %2.2f Distance: %2.2e gradnorm: %2.2e' % (iter_n, f_new, dist_new, gradnorm) )
            running_time = time.time() - time0

            if self._logverbosity >= 1:
                self._append_optlog(iter_n, running_time, f_new, gradnorm=gradnorm, xdist=None, distance = dist_new)

            stop_reason = self._check_stopping_criterion(
                running_time, iter=iter_n, objective_value=f_new, stepsize=eta, gradnorm=gradnorm)

            if stop_reason:
                if self._verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        if self._logverbosity <= 0:
            x = cp.asnumpy(x)
            return x
        else:
            self._stop_optlog(iter, self.objective(x), stop_reason, running_time, stepsize=eta, gradnorm=gradnorm)
            x = cp.asnumpy(x)
            return x, self._optlog

if __name__ == 'main':
    pass