import numpy as np

import torch
import torch.optim.optimizer
import torch.nn.functional as f


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
    lambda_regul : float (optional)
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
        stabilize=None,
        lambda_regul=0,
        normalize_columns=False,
        safe_step=0.5,
        check_type=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if lambda_regul < 0.0:
            raise ValueError(
                "Invalid lambda_regul value: {}".format(lambda_regul)
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            lambda_regul=lambda_regul,
            normalize_columns=normalize_columns,
            safe_step=safe_step,
            check_type=check_type,
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
                lambda_regul = param_group["lambda_regul"]
                for point, point_regul in zip(param_group["params"], regul_group):
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "LandingSGD does not support sparse gradients."
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0: 
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()
                    grad.add_(point, alpha=weight_decay)

                    # If orthogonalization is applied
                    if lambda_regul>0:
                        n1, p = point.shape
                        b, n2 = point_regul.shape
                        # we should have n2==n1
                        Id = torch.eye(p, device=point.device)
                        Ax = point_regul @ point
                        xtAx = point.T@Ax
                        normal_direction = Ax@(xtAx - Id)
                        relative_gradient = 0.5*(grad@(Ax.T@Ax) - Ax@(grad.T @ Ax))
                        # Take the step with orthogonalization
                        new_point = point - learning_rate * (relative_gradient + lambda_regul * normal_direction)
                    else: 
                        # Take the step without orthogonalization
                        new_point = point - learning_rate * grad
                    point.copy_(new_point)
        return loss
    
