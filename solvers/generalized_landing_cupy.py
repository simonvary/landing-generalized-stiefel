import time
import numpy as np
import cupy as cp
import scipy

from cupy.linalg import norm
from cupy import trace

from .solver import IterativeSolver

def symm(x):
    return(0.5*(x + x.T))

class GeneralizedLanding(IterativeSolver):
    '''
    

        min -Tr(X^T A X), s.t. X^T B X = Id
    
    '''
    def __init__(self,A, B, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = cp.asarray(A)
        self.B = cp.asarray(B)
        self.n = B.shape[0]
        self.p = p
        self.objective = lambda x: -0.5*cp.trace(x.T@self.A @ x).item()
        self.Id = cp.eye(p)
        self.constraint = lambda x: .25*(cp.linalg.norm(x.T@self.B@x - self.Id, 'fro')**2).item()
        pass

    def _line_search(self, B, x, direction, eta, df0, safe_step =1, gamma = 1e-2, tau = 1e-1, max_iter = 25):
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
        x_new = x - eta*direction
        f_new = self.objective(x_new)
        dist_new = self.constraint(x_new)
        while f_new > f0 + gamma*eta*df0 or dist_new > safe_step:
            # Reduce the step size and look closer
            eta = tau * eta
            x_new = x - eta*direction
            f_new = self.objective(x_new)
            dist_new = self.constraint(x_new)
            it = it + 1
            if it == max_iter:
                Warning('No descent direction found!')
                return(x, f0, eta, it)
        if it == 1:
            eta = 2*eta    
        return (x_new, f_new, eta, it)
    
    def _safe_step_size(d, g, lambda_regul, eps_d):
        beta = lambda_regul * d * (d-1)
        alpha = g**2
        sol = (- beta + cp.sqrt(beta**2 - alpha * (d - eps_d)))/alpha
        return sol

    def solve(self, eta, omega, safe_step = None, grad_type = 'precon', step_type = 'fixed', x0=None):
        """
        """

        if x0 is None:
            x = np.eye(self.n)[:,:self.p]
        else:
            x = x0.copy()
        x = cp.asarray(x)

        self._start_optlog(extraiterfields = ['gradnorm', 'distance'])
        iter_n = 0

        # Iter_n append
        time0 = time.time()
        while True:
            iter_n = iter_n + 1
            Bx = self.B@x
            egrad = -self.A@x
            xtBx = x.T@Bx
            normal_direction = Bx@(xtBx - self.Id)
            if grad_type == 'precon':
                relative_gradient = .5*(egrad@(Bx.T@Bx) - Bx@ (egrad.T @ Bx))
            elif grad_type == 'riem':
                egrad_scaled = cp.linalg.solve(self.B, egrad)
                relative_gradient = egrad_scaled - x@symm(x.T @ egrad)
            elif grad_type == 'precon2':
                relative_gradient = .5*(egrad@(x.T@Bx) - x@ (egrad.T @ Bx))
            landing_direction = relative_gradient + omega * normal_direction

            if step_type == 'fixed':
                if safe_step:
                    d = distance
                    g = cp.linalg.norm(landing_direction)
                    max_step = self._safe_step_size(d, g, omega, safe_step)
                    eta = cp.max(eta, max_step)
                x = x - eta*landing_direction
            elif step_type == 'lsearch':
                df0 = cp.inner(-landing_direction.flatten(),egrad.flatten())
                x, _, eta, lsearch_it = self._line_search(self.B, x, -landing_direction, eta, df0, safe_step =safe_step)

            gradnorm = norm(relative_gradient, 'fro').item()
            objective_value = self.objective(x)
            distance = self.constraint(x)
            
            print('Iteration %d, Objective: %2.2f Distance: %2.2e' % (iter_n, objective_value, distance) )
            running_time = time.time() - time0

            if self._logverbosity >= 1:
                self._append_optlog(iter_n, running_time, objective_value, gradnorm=gradnorm, xdist=None, distance = distance)

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
            self._stop_optlog(iter, self.objective(x), stop_reason, running_time, stepsize=eta, gradnorm=gradnorm)
            return x, self._optlog

if __name__ == 'main':
    pass