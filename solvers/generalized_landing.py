import time
import numpy as np
import cupy as cp

from .solver import IterativeSolver

class GeneralizedLanding(IterativeSolver):
    '''
    

        min -Tr(X^T A X), s.t. X^T B X = Id
    
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _line_search(self, A, B, p, x, direction, eta0, epsilon):
        ''' Applies line-search to decrease value of objective and do not leave prescribed distance to the constraint'''

        max_iter = 10

        Id = cp.eye(p)

        objective_fun = lambda x: -cp.trace(x.T@A @ x).item()
        constraint_fun = lambda x: cp.linalg.norm(x.T@B@x - Id, 'fro').item()
        
        objective0 = objective_fun(x)
        constraint0 = constraint_fun(x)
        eta = eta0
        
        it = 1
        x_new = x - eta*direction
        objective_new = objective_fun(x_new)
        constraint_new = constraint_fun(x_new)
        #while ((objective_new >= objective0) or (constraint_new >= constraint0)) and (it <= max_iter):
        while (objective_new >= objective0) and (it <= max_iter):
            it = it + 1
            eta = eta / 10
            x_new = x - eta*direction
            objective_new = objective_fun(x_new)
            constraint_new = constraint_fun(x_new)
        if it == 1: 
            eta = eta*2
        return(eta, x_new)

    def solve(self, A, B, p, eta, lambda_param, epsilon, x0=None, tol=1e-10, use_cupy=True):
        """
        """

        m = A.shape[0]

        if x0 is None:
            x = np.eye(m)[:,:p]
        else:
            x = x0.copy()

        if use_cupy:
            A = cp.asarray(A)
            B = cp.asarray(B)
            x = cp.asarray(x)
            Id = cp.eye(p)
            from cupy.linalg import norm
            from cupy import trace
        else:
            Id = np.eye(p)
            from numpy.linalg import norm
            from numpy import trace

        objective_fun = lambda x: -cp.trace(x.T@A @ x).item()
        constraint_fun = lambda x: cp.linalg.norm(x.T@B@x - Id, 'fro').item()

        self._start_optlog(extraiterfields = ['gradnorm', 'distance'])
        
        iter_n = 0
        # Iter_n append
        time0 = time.time()

        while True:
            iter_n = iter_n + 1
            Bx = B@x
            grad = -A@x
            xtBx = x.T@Bx
            normal_direction = Bx@(xtBx - Id)
            #relative_gradient =  grad - x@ (grad.T@Bx + Bx.T@grad)
            #relative_gradient = grad@xtBx - x@ grad.T @ Bx
            relative_gradient = grad@(Bx.T@ Bx) - Bx@(grad.T @ Bx)
            landing_direction = relative_gradient + lambda_param * normal_direction# relative_gradient + lambda_param * normal_direction
            objective0 = objective_fun(x)
            constraint0 = constraint_fun(x)
            
            #x = x - eta*landing_direction
            eta, x = self._line_search(A, B, p, x, landing_direction, eta, epsilon)
            objective_new = objective_fun(x)
            constraint_new = constraint_fun(x)

            kappa = (constraint0/ objective0) * (objective_new - objective0) / (constraint_new - constraint0)
            lambda_param = 1-kappa
            gradnorm = norm(relative_gradient, 'fro').item()
            objective_value = -trace(x.T@A @ x).item()
            distance = norm(xtBx - Id, 'fro').item()
            
            print(objective_value)
            print(distance)
            print(kappa)
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
            if use_cupy:
                x = cp.asnumpy(x)
            return x
        else:
            self._stop_optlog(iter, -trace(x.T@A@x).item(), stop_reason, running_time, stepsize=eta, gradnorm=gradnorm)
            if use_cupy:
                x = cp.asnumpy(x)
            return x, self._optlog

if __name__ == 'main':
    pass