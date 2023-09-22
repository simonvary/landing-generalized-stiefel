import numpy as np

import torch
import torch.optim.optimizer
import torch.nn.functional as f


def symm(x):
    return 0.5*(x+x.T)

class RiemannianGeneralizedStiefel(torch.optim.Optimizer):
    r"""
    Baseline algorithm of performing a Riemannian Steepest Descent 
    on the generalized Stiefel manifold with evolving matrix B with 
    the same API as :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups. Must contain square orthogonal matrices.
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)
    omega : float (optional)
        the hyperparameter lambda that controls the tradeoff between
        optimization in f and landing speed (default: 1.)
    check_type : bool (optional)
        whether to check that the parameters are all orthogonal matrices

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        omega=0,
        check_type=False,
        eps_regul = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if omega < 0.0:
            raise ValueError(
                "Invalid omega value: {}".format(omega)
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            omega=omega,
            check_type=check_type,
            eps_regul=eps_regul
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a positive momentum and zero dampening"
            )
        
        super().__init__(params, defaults)

    def _retraction(self, B, x, direction):
        a = x + direction
        r = torch.linalg.cholesky(a.T @ B @ a)
        return torch.linalg.solve(r, a.T).T

    def step(self, regul_groups):
        '''
            regul_groups is a list of lists corresponding to param_groups[group]["params"] = regul_groups[group]=
            regul_group has either square or rectangular matrix
        '''
        loss = None
        with torch.no_grad():
            for (param_group, regul_group) in zip(self.param_groups, regul_groups):
                weight_decay = param_group["weight_decay"]
                momentum = param_group["momentum"]
                learning_rate = param_group["lr"]
                eps_regul = param_group["eps_regul"]

                for x, B in zip(param_group["params"], regul_group):
                    grad = x.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "LandingSGD does not support sparse gradients."
                        )
                    state = self.state[x]

                    # State initialization
                    if len(state) == 0: 
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()
                    grad.add_(x, alpha=weight_decay)

                    # Riemannian gradient
                    B = B + eps_regul * torch.eye(B.size(0), device = B.device)
                    grad_scaled = torch.linalg.solve(B, grad)
                    riem_grad = grad_scaled - x@symm(x.T @ grad)
                    # Perform retraction
                    new_x = self._retraction(B, x, -learning_rate*riem_grad)
                    x.copy_(new_x)
        return loss
    
