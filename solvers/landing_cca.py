import time
import torch

from solvers import LandingGeneralizedStiefel
from solvers import compute_mean_std,loader_to_cov

def LandingCCA(loaderA, loaderB, p = 10, learning_rate = 1e-3, omega = 1,  n_epochs=10, device = 'cpu', eps = 0, grad_type = 'precon',regul_type='matvec', averaging=False, per_epoch_log=True, lr_milestones=[40,60,80]):
    ''' Takes two iter objects that return matrices that return the same number of batches'''

    meanA,_ = compute_mean_std(loaderA)
    meanB,_ = compute_mean_std(loaderB)
    meanA, meanB = meanA.to(device), meanB.to(device)

    covA_full, covB_full, covAB_full = loader_to_cov(loaderA, loaderB, loader_meanA=meanA, loader_meanB=meanB, device=device)

    Id =  torch.eye(p,p, device = device)

    objective = lambda x,y: -torch.trace(x.T@ covAB_full @ y).item()
    distanceA = lambda x: torch.linalg.norm(x.T@covA_full@x-Id).item()
    distanceB = lambda y: torch.linalg.norm(y.T@covB_full@y-Id).item()

    dimA = meanA.size(0)
    dimB = meanB.size(0)

    # initialization based on the first batch   
    batchA, batchB = zip(loaderA, loaderB).__next__()
    batchA, batchB = batchA.to(device), batchB.to(device)
    x0,_ = torch.linalg.qr(torch.randn(dimA, p, device=device))
    Ax0 = batchA @ x0
    R = torch.linalg.cholesky(Ax0.T @ Ax0 / batchA.size(1));
    x0 = torch.linalg.solve(R, x0.T).T

    y0,_ = torch.linalg.qr(torch.randn(dimB, p, device='cuda'))
    By0 = batchB @ y0
    R = torch.linalg.cholesky(By0.T @ By0 / batchB.size(1));
    y0 = torch.linalg.solve(R, y0.T).T

    x = torch.nn.Parameter(x0).to(device)
    y = torch.nn.Parameter(y0).to(device)

    optimizerCCA = LandingGeneralizedStiefel((x,y), lr=learning_rate, omega=omega,grad_type=grad_type,regul_type=regul_type)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerCCA, milestones=lr_milestones, gamma=0.1)

    out = {'fx': [],
           'distanceA': [],
           'distanceB': [],
           'time': [],
           'iteration': []}
    
    def _append_values(epoch, iteration, time_val, x,y,verb=True):
        objective_val = objective(x,y)
        distanceA_val = distanceA(x)
        distanceB_val = distanceB(y)
        out['fx'].append(objective_val)
        out['distanceA'].append(distanceA_val)
        out['distanceB'].append(distanceB_val)
        out['iteration'].append(iteration)
        out['time'].append(time_val)
        if verb:
            print('Epoch/Iter: (%d, %d), Distance: (%2.5f, %2.5f), Objective: %2.5f' % 
                (epoch, iteration, distanceA_val, distanceB_val, objective_val))
    iteration = 0
    _append_values(0, iteration, 0, x, y, True)

    time0 = time.time()
    for epoch in range(n_epochs):
        
        # If averaging towards the full covariance prepare memory vars
        if averaging and epoch == 0 and regul_type=='matrix':
            n_samples_seen = 0
            covA = torch.empty_like(covA_full, device=device)
            covB = torch.empty_like(covB_full, device=device)
            covAB = torch.empty_like(covAB_full, device=device)

        # Start epochs
        for ind, (A, B) in enumerate(zip(loaderA, loaderB)):
            iteration = iteration + ind
            A, B = A.to(device), B.to(device)
            A = A - meanA
            B = B - meanB
            n_batch =  A.size(0)

            if regul_type == 'matrix':
                if averaging and epoch == 0:
                    covA = (covA*n_samples_seen + A.T@A)/(n_samples_seen+n_batch)
                    covB = (covB*n_samples_seen + B.T@B)/(n_samples_seen+n_batch)
                    covAB = (covAB*n_samples_seen + A.T@B)/(n_samples_seen+n_batch)
                    n_samples_seen = n_samples_seen + n_batch
                elif not averaging:
                    covA = A.T@A/n_batch
                    covB = B.T@B/n_batch
                    covAB = A.T@B/n_batch
                objective_optim = -(torch.trace( x.T@covAB@y)).to(device)
                optimizerCCA.zero_grad()
                objective_optim.backward()
                optimizerCCA.step(((covA, covB),))
            elif regul_type == 'matvec':
                Ax = A @ x
                By = B @ y
                objective_optim = -(torch.trace(Ax.T@By) / n_batch).to(device)
                optimizerCCA.zero_grad()
                objective_optim.backward()
                optimizerCCA.step(((A.T@Ax/n_batch, B.T@By/n_batch),))
            if not per_epoch_log:
                running_time = time.time() - time0
                _append_values(epoch, iteration, running_time, x, y, True)
        if per_epoch_log:
            running_time = time.time() - time0
            _append_values(epoch, iteration, running_time, x, y, True)
        scheduler.step()
    return(x.detach(), y.detach(), out)
