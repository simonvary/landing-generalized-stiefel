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
n_epochs = 8
lr_milestones = [4]
batch_size = 512

# Load the dataset and prepare its full covariance matrices
loader_A, loader_B = dataset_MNIST(batch_size=batch_size, download=False)
covA, covB, covAB = loader_to_cov(loader_A, loader_B, device = 'cuda')

coef = 1
coef_rrsd = 2
coef_land_precon = 1

lr = coef*1e-2
omega = 2#/coef

# p = 5
p = 5
filename = '2_cca_mnist_split_c'+str(coef)
filename = '2_cca_mnist_split_cS'
results = {}

# Compute true regularized (1e-3, else ill-posed) solution
u_true, s_true, v_true = cca_closed_form(covA, covB, covAB, epsilon=1e-3, verb = True)
u_true = u_true[:,:p]
v_true = v_true[:,:p]
obj_true = -torch.trace(u_true.T@covAB@v_true).item()
results['obj_true'] = obj_true



x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = lr, omega = omega,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matrix',per_epoch_log=False, averaging=True, lr_milestones=lr_milestones)
results['land_precon_avg'] = out

x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = lr, omega = omega,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='plam',regul_type='matrix',per_epoch_log=False, averaging=True, lr_milestones=lr_milestones)
results['land_plam_avg'] = out

# Online covariance
lr = coef_land_precon*1e-2
omega = 2/coef_land_precon
x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = lr, omega = omega,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matvec',per_epoch_log=False, averaging=False, lr_milestones=lr_milestones)
results['land_precon'] = out


# Averaged memory matrix
lr = coef_rrsd*1e-2
omega = 2/coef_rrsd
x, y, out  = RiemannianRollingCCA(loader_A, loader_B, p = p, learning_rate = lr,  n_epochs=n_epochs, device = torch.device('cuda'), eps_regul = 1e-10, averaging=True, per_epoch_log=False, lr_milestones=lr_milestones)
results['rrsd'] = out


with open('../figures/data/'+filename+ '_p5.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



# p = 10
# p = 10
# results = {}

# lr = coef*1e-2
# omega = 1 / coef

# # Compute true regularized (1e-3, else ill-posed) solution
# u_true, s_true, v_true = cca_closed_form(covA, covB, covAB, epsilon=1e-3, verb = True)
# u_true = u_true[:,:p]
# v_true = v_true[:,:p]
# obj_true = -torch.trace(u_true.T@covAB@v_true).item()
# results['obj_true'] = obj_true

# # Averaged memory matrix
# x, y, out  = RiemannianRollingCCA(loader_A, loader_B, p = p, learning_rate = lr,  n_epochs=n_epochs, device = torch.device('cuda'), eps_regul = 1e-10, averaging=True, per_epoch_log=False, lr_milestones=[n_epochs])
# results['rrsd'] = out

# x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = lr, omega = omega,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matrix',per_epoch_log=False, averaging=True, lr_milestones=lr_milestones)
# results['land_precon_avg'] = out

# x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = lr, omega = omega,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='plam',regul_type='matrix',per_epoch_log=False, averaging=True, lr_milestones=lr_milestones)
# results['land_plam_avg'] = out

# # Online covariance
# x, y, out  = LandingCCA(loader_A, loader_B, p = p, learning_rate = lr, omega = omega,  n_epochs=n_epochs, device = torch.device('cuda'), grad_type='precon',regul_type='matvec',per_epoch_log=False, averaging=False, lr_milestones=lr_milestones)
# results['land_precon'] = out

# with open('../figures/data/'+filename+'_p10.pkl', 'wb') as handle:
#         pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
             