from time import time

import torch

from solvers import LandingGeneralizedStiefel
from solvers import compute_mean_std,loader_to_cov

def LandingCCA(loaderA, loaderB, p = 10, learning_rate = 1e-3, omega = 1,  n_epochs=10, device = 'cpu', eps = 0, grad_type = 'precon'):
    ''' Takes two iter objects that return matrices that return the same number of batches'''

    meanA,_ = compute_mean_std(loaderA)
    meanB,_ = compute_mean_std(loaderB)
    meanA, meanB = meanA.to(device), meanB.to(device)

    covA, covB, covAB = loader_to_cov(loaderA, loaderB, loader_meanA=meanA, loader_meanB=meanB, device=device)

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

    print('Dist X: %2.5f' % (torch.linalg.norm( x.T @ covA @ x - torch.eye(p,p, device = device)).item() ))
    print('Dist X: %2.5f' % (torch.linalg.norm( y.T @ covB @ y - torch.eye(p,p, device = device)).item() ))
    
    optimizerCCA = LandingGeneralizedStiefel((x,y), lr=learning_rate, omega=omega,grad_type=grad_type)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerCCA, milestones=[100,125], gamma=0.1)

    out = {'objective': [],
           'distanceX': [],
           'distanceY': [],
           'time': []}

    for epoch in range(n_epochs):
        objective_sum = 0
        for ind, (A, B) in enumerate(zip(loaderA, loaderB)):
            A, B = A.to(device), B.to(device)
            A = A - meanA
            B = B - meanB
            n_batch =  A.size(0)
            Ax = A @ x
            By = B @ y
            objective = -(torch.trace( Ax.T @ By) / n_batch).to(device)
            optimizerCCA.zero_grad()
            objective.backward()
            optimizerCCA.step(((A.T@Ax/n_batch, B.T@By/n_batch),))
        objective_sum = -torch.trace( x.T @ covAB @ y).item()
        
        out['objective'].append(objective_sum)
        out['distanceX'].append(torch.linalg.norm( x.T @ covA @ x - torch.eye(p,p, device = device)).item())
        out['distanceY'].append(torch.linalg.norm(y.T @ covB @ y - torch.eye(p,p, device = device)).item())

        print('Objective: %2.5f' % objective_sum)
        print('Dist X: %2.5f' % (torch.linalg.norm( x.T @ covA @ x - torch.eye(p,p, device = device)).item() ))
        print('Dist Y: %2.5f' % (torch.linalg.norm(y.T @ covB @ y - torch.eye(p,p, device = device)).item()) )
        scheduler.step()
    return(x.detach(), y.detach(), out)
