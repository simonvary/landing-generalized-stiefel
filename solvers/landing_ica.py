import time
import torch

from solvers import LandingGeneralizedStiefel
from solvers import compute_mean_std,loader_to_cov

def sigma(x):
    return(torch.log(torch.cosh(torch.abs(x))))

def sigma_pa(X):
    Y = torch.abs(X)
    return Y + torch.log1p(torch.exp(-2 * Y))

def amari_distance(W, A):
    """
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.
    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix
    A : ndarray, shape (n_features, n_features)
        Input matrix
    Returns
    -------
    d : float
        The Amari distance
    """
    P = W @ A
    def s(r):
        val_max, _ = torch.max(r ** 2, axis=1)
        return torch.sum(torch.sum(r ** 2, axis=1) / val_max - 1)
    return ((s(torch.abs(P)) + s(torch.abs(P.T))) / (2 * P.shape[0])).item()


def LandingICA(loader, mixing_true, p = 10, learning_rate = 1e-3, omega = 1,  n_epochs=10, device = 'cpu', grad_type = 'precon',regul_type='matvec', averaging=False, per_epoch_log=True, lr_milestones=[40,60,80], init_batch_size = 1):
    ''' Takes two iter objects that return matrices that return the same number of batches'''

    A_full = loader.dataset.tensors[0]
    n_samples, n_features = loader.dataset.tensors[0].shape 
    Id =  torch.eye(p,p, device = device)
    AtA_full = A_full.T @ A_full / n_samples
    objective = lambda x: torch.sum(sigma(A_full @ x)).item() / n_samples
    distance = lambda x: torch.linalg.norm(x.T@AtA_full@x-Id).item()

    # initialization based on the first init_batch_size
    n_samples_seen = 0
    AtA = torch.zeros_like(AtA_full)
    iterloader = iter(loader)
    for i in range(init_batch_size):
        A_sample = iterloader.__next__()
        A_sample = A_sample[0].to(device)
        n_samples_batch, n_features = A_sample.shape
        AtA = (AtA*n_samples_seen + A_sample.T @ A_sample)/(n_samples_seen+n_samples_batch)
        n_samples_seen = n_samples_seen + n_samples_batch
    x0,_ = torch.linalg.qr(torch.randn(n_features, p, device=device, dtype=float))
    R = torch.linalg.cholesky(x0.T@ AtA @ x0);
    #R = torch.linalg.cholesky(x0.T@ AtA_full @x0);
    x0 = torch.linalg.solve(R, x0.T).T
    x = torch.nn.Parameter(x0).to(device)

    optimizerICA = LandingGeneralizedStiefel((x,), lr=learning_rate, omega=omega,grad_type=grad_type,regul_type=regul_type)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerICA, milestones=lr_milestones, gamma=0.1)

    out = {'fx': [],
           'distance': [],
           'amari_distance': [],
           'time': [],
           'iteration': []}
    
    def _append_values(epoch, iteration, time_val, x,verb=True):
        objective_val = objective(x)
        distance_val = distance(x)
        amari_distance_val = amari_distance(mixing_true.T, x)
        out['fx'].append(objective_val)
        out['distance'].append(distance_val)
        out['iteration'].append(iteration)
        out['amari_distance'].append(amari_distance_val)
        out['time'].append(time_val)
        if verb:
            print('Epoch/Iter: (%d, %d), Distance: (%2.5f), Amari distance: (%2.5f), Objective: %2.5f' % 
                (epoch, iteration, distance_val, amari_distance_val, objective_val))
    iteration = 0
    _append_values(0, iteration, 0, x, True)

    time0 = time.time()
    for epoch in range(n_epochs):
        
        # If averaging towards the full covariance prepare memory vars
        if averaging and epoch == 0 and regul_type=='matrix':
            n_samples_seen = 0
            AtA = torch.zeros_like(AtA_full, device=device)

        # Start epochs
        for ind, A_sample in enumerate(loader):
            iteration = iteration + 1
            A_sample = A_sample[0].to(device)
            n_batch =  A_sample.size(0)
            if regul_type == 'matrix':
                if averaging and epoch == 0:
                    AtA = (AtA*n_samples_seen + A_sample.T@A_sample)/(n_samples_seen+n_batch)
                    n_samples_seen = n_samples_seen + n_batch
                elif not averaging:
                    AtA = A_sample.T@A_sample/n_batch
                objective_optim = (torch.sum(sigma(A_sample@x)) / n_batch).to(device)
                optimizerICA.zero_grad()
                objective_optim.backward()
                optimizerICA.step(((AtA,),))
            elif regul_type == 'matvec':
                Ax = A_sample @ x
                objective_optim = (torch.sum(sigma(A_sample@x)) / n_batch).to(device)
                optimizerICA.zero_grad()
                objective_optim.backward()
                optimizerICA.step(((Ax/n_batch,),))
            if not per_epoch_log:
                running_time = time.time() - time0
                _append_values(epoch, iteration, running_time, x, True)
        if per_epoch_log:
            running_time = time.time() - time0
            _append_values(epoch, iteration, running_time, x, True)
        scheduler.step()
    return(x.detach(), out)
