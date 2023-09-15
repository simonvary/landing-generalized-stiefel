import numpy as np

import torch
import torch.optim.optimizer
import torch.nn.functional as f


def sylvester(A, B, C, X=None):
    ''' From the answer here: 
        https://stackoverflow.com/questions/73713072/solving-sylvester-equations-in-pytorch
    '''
    m = B.shape[-1];
    n = A.shape[-1];
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)
    F = torch.linalg.solve(U, (C + 0j) @ V)
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = U[...,:n,:n] @ Y[...,:n,:m] @ torch.linalg.inv(V)[...,:m,:m]
    return X.real if all(torch.isreal(x.flatten()[0]) 
                for x in [A, B, C]) else X


class LandingGeneralizedStiefel(torch.optim.Optimizer):
    r"""
    Generalized Landing algorithm on the generalized Stiefel manifold with the same API as
    :class:`torch.optim.SGD`.

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
        normalize_columns=False,
        safe_step=0.5,
        grad_type='precon',
        check_type=False
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
            normalize_columns=normalize_columns,
            safe_step=safe_step,
            check_type=check_type,
            grad_type=grad_type
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a positive momentum and zero dampening"
            )
        
        super().__init__(params, defaults)

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
                dampening = param_group["dampening"]
                nesterov = param_group["nesterov"]
                learning_rate = param_group["lr"]
                omega = param_group["omega"]
                grad_type = param_group["grad_type"]

                for x, Bx in zip(param_group["params"], regul_group):
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

                    # If orthogonalization is applied
                    if omega>0:
                        n, p = x.shape
                        Id = torch.eye(p, device=x.device)
                        xtBx = x.T@Bx
                        normal_direction = Bx@(xtBx - Id)
                        if grad_type == 'precon':
                            relative_gradient = 0.5*(grad@(xtBx) - Bx@(grad.T @ Bx))
                        elif grad_type == 'riem':
                            '''The following does not work and returns NaNs'''
                            gradTBx = grad.T@Bx
                            xtB2x = Bx.T@ Bx + 1e-3*torch.eye(Bx.size(1),device = Bx.device)
                            print(xtB2x)
                            print(gradTBx)
                            grad_projected = 2*Bx@sylvester(xtB2x,xtB2x, gradTBx+gradTBx.T)
                            relative_gradient = grad - grad_projected
                        landing_direction = relative_gradient + omega * normal_direction
                        # Take the step with orthogonalization
                        new_x = x - learning_rate * landing_direction
                    else: 
                        # Take the step without orthogonalization
                        new_x = x - learning_rate * grad
                    x.copy_(new_x)
        return loss
    
