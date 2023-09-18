"""
A simple example of the Stiefel landing algorithm on PCA problem
"""
from time import time

import matplotlib.pyplot as plt

import numpy as np
import cupy as cp

from solvers import GeneralizedLanding

from scipy.linalg import eigh
import scipy

def run_gevp_experiment(problem_parameters, method_name, method_parameters):

    # Load experiment parameters

    # Generate the problem
    A = generate_gevp_problem(n_samples, n_features, p_subspace, sdev=noise_sdev)
    A = A.to(device)
    objective = lambda x : -.5 * (torch.linalg.norm(A @ x )**2).item() / n_samples
    # Compute the exact solution using SVD
    _, _, vh = torch.linalg.svd(A, full_matrices = False)
    x_star = vh[:p_subspace,:].T
    loss_star = objective(x_star)

    solver = GeneralizedLanding(maxiter = 15000, mingradnorm=1e-16)
    x4, optlog4 = solver.solve(A, B, p, 4, 0.1, None, grad_type = 6, Binv = Binv, x0=x0)

    # Initialization
    if x0 is None:
        x = torch.randn(n_features, p_subspace).to(device) / n_samples
    else:
        x = x0
    if init_project:
        x = stiefel_project(x)
    x = torch.nn.Parameter(x)

    # Prepare optimizer
    if method_name == 'landing':
        optimizer = LandingStiefelSGD((x,), lr=learning_rate, lambda_regul=lambda_regul, safe_step=safe_step)
    elif method_name == 'retraction':
        x = geoopt.ManifoldParameter(x, manifold=geoopt.Stiefel(canonical=False))
        optimizer = geoopt.optim.RiemannianSGD((x,), lr=learning_rate)
    elif method_name == 'regularization':
        optimizer = torch.optim.SGD((x,), lr=learning_rate)
    else:
        raise ValueError('Unrecognized method_name.')
    
    scheduler = scheduler(optimizer)

    # Train
    train_loss = [objective(x)-loss_star]
    time_list = [0]
    stiefel_distances = [stiefel_distance((x,), device).item()]
    for epoch in range(n_epochs):
        time_start = time()
        permutation = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()

            ind_batch = permutation[i:i+batch_size]
            A_batch = A[ind_batch,:]
            loss = -.5 * torch.linalg.norm(A_batch @ x )**2 / A_batch.shape[0]
            if method_name == 'regularization':
                loss += .5*lambda_regul * stiefel_distance((x,), device=device) 
            loss.backward()
            optimizer.step()
        
        if epoch == 0:
            time_list.append(time() - time_start)
        else:
            time_list.append(time_list[-1] + (time() - time_start))
        train_loss.append(objective(x)-loss_star)
        stiefel_distances.append(stiefel_distance((x,), device).item())
        scheduler.step()
    
    return({'train_loss' : train_loss,
            'stiefel_distances' : stiefel_distances,
            'time_list' : time_list
    })

if __name__ == "__main__":

    def scheduler_function(optimizer):
        return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50], gamma=0.1))

    # Test
    problem_parameters = {
        'n_samples' : 10000,
        'n_features': 2000,
        'p_subspace': 1000,
        'noise_sdev': 2*1e-2
    }

    method_parameters = {
        'method_name': 'landing',
        'batch_size': 128,
        'n_epochs': 100,
        'learning_rate': 1e-2,
        'lambda_regul': 1, 
        'safe_step': 0.5, 
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': torch.device('cuda')
    }

    torch.manual_seed(0)
    method_name = method_parameters['method_name']
    out = run_pca_experiment(problem_parameters, method_name, method_parameters)
    
    train_loss, stiefel_distances, time_list = out.values()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    axs[0].semilogy(time_list, train_loss)
    axs[0].set_xlabel('time (sec.)')
    axs[0].set_ylabel('Train loss (objective)')

    axs[1].semilogy(time_list, stiefel_distances)
    axs[1].set_xlabel('time (sec.)')
    axs[1].set_ylabel('Stiefel distance (objective)')

    plt.savefig('plot.pdf')