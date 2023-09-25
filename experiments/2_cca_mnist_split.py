"""

This is to perform top-k CCA on full dataset of mnist

"""
import sys
sys.path.append("../")

from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from solvers import *
from experiments.utils import *
import pickle


from scipy.linalg import eigh


device = 'cuda'
n_epochs = 5
batch_size = 512

# Load the dataset and prepare its full covariance matrices
loader_A, loader_B = dataset_MNIST(batch_size=batch_size, download=False)
covA, covB, covAB = loader_to_cov(loader_A, loader_B, device = 'cuda')



# p = 5
p = 5
filename = '2_cca_mnist_split_p5.pkl'
results = {}

# Compute true regularized (1e-3, else ill-posed) solution
u_true, s_true, v_true = cca_closed_form(covA, covB, covAB, epsilon=1e-3, verb = True)
u_true = u_true[:,:p]
v_true = v_true[:,:p]
obj_true = -torch.trace(u_true.T@covAB@v_true).item()
results['obj_true'] = obj_true

# Averaged memory matrix
x, y, out  = RiemannianRollingCCA(loader_A, loader_B, p = p, learning_rate = 1e-2,  n_epochs=n_epochs, device = torch.device('cuda'), eps_regul = 1e-10, averaging=True, per_epoch_log=False, lr_milestones=[10])
results['rrsd'] = out

x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = 1e-2, omega = 1,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matrix',per_epoch_log=False, averaging=True, lr_milestones=[4])
results['land_precon_avg'] = out

# Online covariance
x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = 1e-2, omega = 1,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matvec',per_epoch_log=False, averaging=False, lr_milestones=[4])
results['land_precon'] = out

with open('../figures/data/'+filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



# p = 10
p = 10
filename = '2_cca_mnist_split_p10.pkl'
results = {}

# Compute true regularized (1e-3, else ill-posed) solution
u_true, s_true, v_true = cca_closed_form(covA, covB, covAB, epsilon=1e-3, verb = True)
u_true = u_true[:,:p]
v_true = v_true[:,:p]
obj_true = -torch.trace(u_true.T@covAB@v_true).item()
results['obj_true'] = obj_true

# Averaged memory matrix
x, y, out  = RiemannianRollingCCA(loader_A, loader_B, p = p, learning_rate = 1e-2,  n_epochs=n_epochs, device = torch.device('cuda'), eps_regul = 1e-10, averaging=True, per_epoch_log=False, lr_milestones=[n_epochs])
results['rrsd'] = out

x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = 1e-2, omega = 1,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matrix',per_epoch_log=False, averaging=True, lr_milestones=[4])
results['land_precon_avg'] = out

# Online covariance
x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = 1e-2, omega = 1,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matvec',per_epoch_log=False, averaging=False, lr_milestones=[4])
results['land_precon'] = out

with open('../figures/data/'+filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
             