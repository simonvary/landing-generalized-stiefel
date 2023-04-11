from time import time

import torch

from solvers import LandingGeneralizedStiefel

def LandingCCA(dataloader_A, dataloader_B, p = 10, learning_rate = 1e-3, lambda_regul = 1,  n_epochs=10, device = 'cpu'):
    ''' Takes two iter objects that return matrices that return the same number of batches'''

    A = iter(dataloader_A).__next__().to(device)
    B = iter(dataloader_B).__next__().to(device)
    
    n_length = min(len(dataloader_A), len(dataloader_A))

    # initialization with on the first batch   
    _, S_X, X = torch.linalg.svd(A.T @ A, full_matrices=False)
    _, S_Y, Y = torch.linalg.svd(B.T @ B, full_matrices=False)
    
    X = torch.nn.Parameter(X[:p,:].T / S_X[:p]).to(device)
    Y = torch.nn.Parameter(Y[:p,:].T / S_Y[:p]).to(device)

    optimizerCCA = LandingGeneralizedStiefel((X,Y), 
        lr=learning_rate, lambda_regul=lambda_regul)
    
    out = {'objective': [],
           'distanceX': [],
           'distanceY': [],
           'time': []}

    AtA_full = torch.zeros(A.size(1), A.size(1), device = device)
    BtB_full = torch.zeros(B.size(1), B.size(1), device = device)
    AtB_full = torch.zeros(A.size(1), B.size(1), device = device)
    for _, (A, B) in enumerate(zip(dataloader_A, dataloader_B)):
        A, B = A.to(device), B.to(device)
        AtA_full += A.T @ A / A.size(0)
        BtB_full += B.T @ B / B.size(0)
        AtB_full += A.T @ B / A.size(0)
    AtA_full = AtA_full / len(dataloader_A)
    AtB_full = AtB_full / len(dataloader_A)
    BtB_full = BtB_full / len(dataloader_B)

    for epoch in range(n_epochs):
        objective_sum = 0
        for ind, (A, B) in enumerate(zip(dataloader_A, dataloader_B)):
            A, B = A.to(device), B.to(device)
            n_batch =  A.size(0)
            AX = A @ X
            BY = B @ Y
            objective = -(torch.trace( AX.T @ BY) / n_batch).to(device)
            optimizerCCA.zero_grad()
            objective.backward()
            optimizerCCA.step(((A.T@A / n_batch, B.T@B / n_batch),))
        objective_sum = -torch.trace( X.T @ AtB_full @ Y).item()
        
        out['objective'].append(objective_sum)
        out['distanceX'].append(torch.linalg.norm( X.T @ AtA_full @ X - torch.eye(p,p, device = device)).item())
        out['distanceY'].append(torch.linalg.norm(Y.T @ BtB_full @ Y - torch.eye(p,p, device = device)).item())

        print('Objective: %2.5f' % objective_sum)
        print('Dist X: %2.5f' % (torch.linalg.norm( X.T @ AtA_full @ X - torch.eye(p,p, device = device)).item() ))
        print('Dist Y: %2.5f' % (torch.linalg.norm(Y.T @ BtB_full @ Y - torch.eye(p,p, device = device)).item()) )
    return(X, Y, AtA_full, BtB_full, AtB_full, out)
