from time import time

import torch

from solvers import RiemannianGeneralizedStiefel
from solvers import compute_mean_std,loader_to_cov

def RiemannianRollingCCA(loaderA, loaderB, p = 10, learning_rate = 1e-3, n_epochs=10, device = 'cpu', averaging=True, eps_regul=1e-3):
    ''' Takes two iter objects that return matrices that return the same number of batches'''

    meanA,_ = compute_mean_std(loaderA)
    meanB,_ = compute_mean_std(loaderB)
    meanA, meanB = meanA.to(device), meanB.to(device)

    covA_full, covB_full, covAB_full = loader_to_cov(loaderA, loaderB, loader_meanA=meanA, loader_meanB=meanB, device=device)

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
    Id =  torch.eye(p,p, device = device)

    print('Dist X: %2.5f' % (torch.linalg.norm(x.T@covA_full@x-Id).item() ))
    print('Dist X: %2.5f' % (torch.linalg.norm(y.T@covB_full@y-Id).item() ))
    
    optimizerCCA = RiemannianGeneralizedStiefel((x,y), lr=learning_rate,eps_regul=eps_regul)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerCCA, milestones=[100,125], gamma=0.1)

    out = {'objective': [],
           'distanceX': [],
           'distanceY': [],
           'time': []}

    for epoch in range(n_epochs):
        objective_sum = 0
        n_samples_seen = 0
        covA = torch.empty_like(covA_full, device=device)
        covB = torch.empty_like(covB_full, device=device)
        covAB = torch.empty_like(covAB_full, device=device)
        for ind, (A, B) in enumerate(zip(loaderA, loaderB)):
            A, B = A.to(device), B.to(device)
            A = A - meanA
            B = B - meanB
            n_batch =  A.size(0)
            if averaging:
                covA = (covA*n_samples_seen + A.T@A)/(n_samples_seen+n_batch)
                covB = (covB*n_samples_seen + B.T@B)/(n_samples_seen+n_batch)
                covAB = (covAB*n_samples_seen + A.T@B)/(n_samples_seen+n_batch)
                n_samples_seen = n_samples_seen + n_batch
            else:
                covA = A.T@A/n_batch
                covB = B.T@B/n_batch
                covAB = A.T@B/n_batch
            objective = -.5*(torch.trace(x.T@covAB @ y)).to(device)
            optimizerCCA.zero_grad()
            objective.backward()
            optimizerCCA.step(((covA, covB),))
        objective_sum = -.5*(torch.trace( x.T @ covAB_full @ y)).item()
        
        out['objective'].append(objective_sum)
        out['distanceX'].append((torch.linalg.norm(x.T@covA_full@x-Id)**2).item())
        out['distanceY'].append((torch.linalg.norm(y.T@covB_full@y-Id)**2).item())

        print('Objective: %2.5f' % objective_sum)
        print('Dist X: %2.5f' % (torch.linalg.norm(x.T@covA_full@x-Id)**2).item())
        print('Dist Y: %2.5f' % (torch.linalg.norm(y.T@covB_full@y-Id)**2).item() )
        scheduler.step()
    return(x.detach(), y.detach(), out)
