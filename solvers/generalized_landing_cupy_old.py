import time
import numpy as np
import cupy as cp
import scipy

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
    
    def _safe_step_size(d, g, lambda_regul, eps_d):
        beta = lambda_regul * d * (d-1)
        alpha = g**2
        sol = (- beta + cp.sqrt(beta**2 - alpha * (d - eps_d)))/alpha
        return sol

    def solve(self, A, B, p, eta, lambda_param, safe_step, grad_type = 1, Binv = None, x0=None, tol=1e-10, track_diff=True, use_cupy=True):
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
            Binv = cp.asarray(Binv)
            from cupy.linalg import norm
            from cupy import trace
        else:
            Id = np.eye(p)
            from numpy.linalg import norm
            from numpy import trace

        objective_fun = lambda x: -.5*cp.trace(x.T@A @ x).item()
        constraint_fun = lambda x: .25*(cp.linalg.norm(x.T@B@x - Id, 'fro')**2).item()

        if track_diff:
            self._start_optlog(extraiterfields = ['gradnorm', 'distance', 'df1', 'df2', 'dc1', 'dc2', 'df3'])
        else:
            self._start_optlog(extraiterfields = ['gradnorm', 'distance'])
        iter_n = 0
        # Iter_n append
        time0 = time.time()

        while True:
            iter_n = iter_n + 1
            Bx = B@x
            grad = -A@x
            xtBx = x.T@Bx
            lambda_current = lambda_param
            if grad_type == 1:
                # sk(grad (BX)^T) BX
                normal_direction = Bx@(xtBx - Id)
                relative_gradient = grad@xtBx - x@ grad.T @ Bx
            elif grad_type == 2:
                # sk(grad X^T) BX
                normal_direction = Bx@(xtBx - Id)
                relative_gradient = (grad@x.T - x @ grad.T)@Bx
            elif grad_type == 3:
                normal_direction = Bx@(xtBx - Id)
                relative_gradient = Binv @ grad @ xtBx - x @ grad.T@ x
            elif grad_type == 4:
                relative_gradient = grad @ xtBx - Bx @ grad.T @ x
                normal_direction = x@(xtBx - Id)
                print()
            elif grad_type == 5:
                '''Sylvester equation'''
                normal_direction = Bx@(xtBx - Id)
                rhs = Bx.T @ grad + grad.T@Bx
                lhs = Bx.T @ Bx
                S = cp.asarray(scipy.linalg.solve_continuous_lyapunov(cp.asnumpy(lhs), cp.asnumpy(rhs)) )
                relative_gradient = grad - Bx @ S
            elif grad_type == 6:
                delta = xtBx - Id
                normal_direction = Bx@delta
                relative_gradient = .5*(grad@(Bx.T@Bx) - Bx@ (grad.T @ Bx))

            if track_diff:
                delta = 1e-6
                relative_gradient1 = relative_gradient# (grad @ xtBx - Bx @ grad.T @ x)
                relative_gradient2 = Binv @ grad @ xtBx - x @ grad.T@ x
                relative_gradient3 = .5*(grad@(Bx.T@Bx) - Bx@ (grad.T @ Bx))
                df1 = cp.inner(relative_gradient1.flatten(), grad.flatten())/ (cp.linalg.norm(grad)*cp.linalg.norm(relative_gradient1) ) #(objective_fun(x - delta*relative_gradient1) - objective_fun(x)) / np.linalg.norm(delta*relative_gradient1)
                df2 = cp.inner(relative_gradient2.flatten(), grad.flatten())/ (cp.linalg.norm(grad)*cp.linalg.norm(relative_gradient2) ) # (objective_fun(x - delta*relative_gradient2) - objective_fun(x)) / np.linalg.norm(delta*relative_gradient2)
                df3 = cp.inner(relative_gradient3.flatten(), grad.flatten())/ (cp.linalg.norm(grad)*cp.linalg.norm(relative_gradient3) ) #(objective_fun(x - delta*relative_gradient1) - objective_fun(x)) / np.linalg.norm(delta*relative_gradient1)

                dc1 = (constraint_fun(x - delta*relative_gradient1) - constraint_fun(x)) / np.linalg.norm(delta*relative_gradient1)
                dc2 = (constraint_fun(x - delta*relative_gradient2) - constraint_fun(x)) / np.linalg.norm(delta*relative_gradient2)
                
            #relative_gradient = grad@(Bx.T@ Bx) - Bx@(grad.T @ Bx)
            landing_direction = relative_gradient + lambda_current * normal_direction# relative_gradient + lambda_param * normal_direction
            objective0 = objective_fun(x)
            constraint0 = constraint_fun(x)


            if safe_step:
                d = distance
                g = cp.linalg.norm(landing_direction)
                max_step = self._safe_step_size(d, g, lambda_param, safe_step)
                eta = cp.max(eta, max_step)
            
            x = x - eta*landing_direction
            #eta, x = self._line_search(A, B, p, x, landing_direction, eta, epsilon)
            objective_new = objective_fun(x)
            constraint_new = constraint_fun(x)

            #kappa = (constraint0/ objective0) * (objective_new - objective0) / (constraint_new - constraint0)
            #lambda_param = 1-kappa
            gradnorm = norm(relative_gradient, 'fro').item()
            objective_value = -trace(x.T@A @ x).item()
            distance = norm(xtBx - Id, 'fro').item()
            
            print('Iteration %d, Objective: %2.2f Distance: %2.2e' % (iter_n, objective_value, distance) )
            running_time = time.time() - time0

            if self._logverbosity >= 1:

                if track_diff:
                    self._append_optlog(iter_n, running_time, objective_value, gradnorm=gradnorm, xdist=None, distance = distance, df1 = df1.item(), df2 = df2.item(), dc1 = dc1.item(), dc2=dc2.item(), df3=df3.item())
                else:
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