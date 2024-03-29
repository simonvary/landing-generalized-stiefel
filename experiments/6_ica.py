"""

This is to perform top-k ICA on a synthetic data

"""
import sys
sys.path.append("../")
"""
A simple example of the Stiefel landing algorithm on PCA problem
"""

from time import time

import matplotlib.pyplot as plt

import numpy as np

import torch

from solvers import *
from experiments.utils import *

import pickle


def dataset_simulated_ica(n_samples = 10000, n_features = 10, p = 10, batch_size=100, st_dev = 1, device = 'cpu', random_state=42):
    rng = np.random.RandomState(random_state)
    sources = st_dev*rng.laplace(size=(n_samples, n_features))
    q_haar, _ = np.linalg.qr(rng.randn(p, n_features))
    mixing = q_haar @ q_haar.T
    #X = np.dot(sources, mixing.T)
    #W = np.linalg.pinv(sqrtm(X.T.dot(X) / n_samples))
    X = np.dot(sources, mixing.T)
    #mixing = np.dot(W, mixing)
    X = torch.from_numpy(X).to(device=device)
    mixing = torch.from_numpy(mixing).to(device=device)
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size, shuffle=False)
    return dict(dataloader=dataloader, mixing=mixing)

n_samples = int(1e5)
n_features = 10
p = n_features
batch_size = 1000
data = dataset_simulated_ica(device = 'cuda', n_samples = n_samples, n_features=n_features, p = p, st_dev = 1, batch_size = batch_size)
dataloader = data['dataloader']
mixing = data['mixing']

device = 'cuda'

A = dataloader.dataset.tensors[0]
AtA = A.T @ A / A.shape[0]

evals = torch.linalg.eigvals(AtA)
print(evals[0]/evals[-1])

coef = 4
coef_land_plam_avg = 1
coef_rrsd = 2
per_epoch_log = False
n_epochs = 5

lr_rate = coef*1e-1
omega = 1 /coef

lr_milestones = [3,4]
init_batch_size = 10

filename = '6_ica_c'+str(coef)+'_n'+str(n_features)
filename = '6_ica_cS_n'+str(n_features)

results = {}


x_l, out_l = LandingICA(loader=dataloader, mixing_true=mixing, p=p, grad_type='precon', averaging=False, learning_rate = lr_rate, omega = omega, regul_type='matrix', n_epochs=n_epochs,device=device, lr_milestones=lr_milestones, per_epoch_log=per_epoch_log, init_batch_size = 10)
results['land_precon'] = out_l


x_la, out_la = LandingICA(loader=dataloader, mixing_true=mixing, p=p, grad_type='precon', averaging=True, learning_rate = lr_rate, omega = omega, regul_type='matrix', n_epochs=n_epochs,device=device, lr_milestones=lr_milestones, per_epoch_log=per_epoch_log, init_batch_size = 10)
results['land_precon_avg'] = out_la

lr_rate = coef_rrsd*1e-1
omega = 1 /coef_rrsd
x_r, out_r = RiemannianRollingICA(loader=dataloader, mixing_true=mixing, p=p, averaging=True, learning_rate = lr_rate, n_epochs=n_epochs,device=device, lr_milestones=lr_milestones, per_epoch_log=per_epoch_log,init_batch_size=init_batch_size, eps_regul= 1e-6)
results['rrsd'] = out_r

coef = coef_land_plam_avg
lr_rate = coef_land_plam_avg*1e-1
omega = 1 /coef_land_plam_avg
x_lplam, out_lplam = LandingICA(loader=dataloader, mixing_true=mixing, p=p, grad_type='plam', averaging=True, learning_rate = lr_rate, omega = omega, regul_type='matrix', n_epochs=n_epochs,device=device, lr_milestones=lr_milestones, per_epoch_log=per_epoch_log, init_batch_size = 10)
results['land_plam_avg'] = out_lplam


with open('../figures/data/'+filename+'.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

