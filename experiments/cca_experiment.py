import sys, os
from time import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import CCA


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def loader2cov(loaderA, loaderB, sizeA, sizeB, device):
    covA = torch.zeros(sizeA, sizeA, device = device)
    covB = torch.zeros(sizeB, sizeB, device = device)
    covAB = torch.zeros(sizeA, sizeB, device = device)
    for _, (A, B) in enumerate(zip(loaderA, loaderB)):
        A, B = A.to(device), B.to(device)
        covA += A.T @ A / A.size(0)
        covB += B.T @ B / B.size(0)
        covAB += A.T @ B / A.size(0)
    covA = covA / len(loaderA)
    covB = covB / len(loaderB)
    covAB = covAB / len(loaderA)
    return(covA, covB, covAB)

def full_CCA(loaderA, loaderB, sizeA, sizeB, num_components, device):
    covA, covB, covAB = loader2cov(loaderA, loaderB, sizeA, sizeB, device)
    evals_covA, evecs_covA = torch.linalg.eig(covA)
    evals_covB, evecs_covB = torch.linalg.eig(covB)
    covA_isqrt = evecs_covA @ torch.diag(evals_covA**(-0.5)) @ evecs_covA.T
    covB_isqrt = evecs_covB @ torch.diag(evals_covB**(-0.5)) @ evecs_covB.T
    u, s, v = torch.linalg.svd(covA_isqrt @ covAB @ covB_isqrt, full_matrices = False)
    out = []
    out['x'] = u[:,:num_components]
    out['xy_corr'] = s[:num_components]
    out['y'] = v[:,:num_components]
    return(out)

def run_cca_experiment(problem, method_name, method, run_file_name):
    loaderA = problem['loaderA']
    loaderB = problem['loaderB']
    sizeA = problem['sizeA']
    sizeB = problem['sizeB']
    num_components = problem['num_components']

    batch_size = method['batch_size']
    n_epochs = method['n_epochs']
    lambda_regul = method['lambda_regul']
    safe_step = method['safe_step']
    learning_rate = method['learning_rate']
    weight_decay = method['weight_decay']
    init_project = method['init_project']
    scheduler = method['scheduler']
    model = method['model']()
    x0 = method['x0']
    device = method['device']

    if method_name == 'fullCCA':
