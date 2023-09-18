from time import time

import numpy as np
import torch

from solvers import LandingGeneralizedStiefel


def CCA(dataloader_A, dataloader_B, p = 10, learning_rate = 1e-3, lambda_regul = 1,  n_epochs=10, device = 'cpu'):
    ''' Takes two iter objects that return matrices that return the same number of batches'''

    A = dataloader_A.__next__().to(device)
    B = dataloader_B.__next__().to(device)
    
    n_length = min(len(dataloader_A), len(dataloader_A))

    # initialization with on the first batch   
    _, S_X, X = torch.linalg.svd(A, full_matrices=False)
    _, S_Y, Y = torch.linalg.svd(B, full_matrices=False)

    X = torch.nn.Parameter(X[:p,:].T / S_X[:p])
    Y = torch.nn.Parameter(Y[:p,:].T / S_Y[:p])

    optimizerCCA = LandingGeneralizedStiefel((X,Y), 
        lr=learning_rate, lambda_regul=lambda_regul)


    for epoch in range(n_epochs):
        objective_function = 0
        dist_X = torch.zeros(p,p, device = device)
        dist_Y = torch.zeros(p,p, device = device)
        for ind, (A, B) in enumerate(zip(dataloader_A, dataloader_B)):

            A, B = A.to(device), B.to(device)
            n_batch =  A.size(0)
            AX = A @ X
            BY = B @ Y
            loss = -torch.trace( AX.T @ BY) / n_batch
    
            optimizerCCA.zero_grad()
            loss.backward()
            optimizerCCA.step(((A, B ),))
            dist_X = dist_X + AX.T @ AX / n_batch
            dist_Y += dist_Y + BY.T @ BY / n_batch
            train_loss = train_loss + loss.item()

        print('Objective: %2.5f' % train_loss)
        print('Dist X: %2.5f' % (torch.linalg.norm(dist_X - torch.eye(p,p, device = device)).item() ))
        print('Dist Y: %2.5f' % (torch.linalg.norm(dist_Y - torch.eye(p,p, device = device)).item()) )
