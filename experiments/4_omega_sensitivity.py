# To check the sensitivity of the landing method for different choices of omega

"""
Comparison of solving GEVP with RiemannianSD vs Landing
Deterministic experiment using cupy on GPU

"""
from time import time
import random
import pickle

import matplotlib.pyplot as plt
from itertools import product

import numpy as np
import cupy as cp

import sys
sys.path.append("../")

from solvers import GeneralizedLanding
from solvers import RiemmGeneralizedStiefel
from experiments import generate_spd

from scipy.linalg import eigh


seed = 123
cp.random.seed(seed)
np.random.seed(seed)
random.seed(seed)

filename = '4_gevp.pkl'

n = 1000
p = 500

max_time = 120 # in seconds
maxiter = 20000
omega_var_max = .75
eta_var_max = 0.75
n_omega = 4
n_eta = 4
omega_vars = np.linspace(1-omega_var_max, 1+omega_var_max, num=n_omega)
#omega_vars = [1/4, 1/2 , 1, 2, 4]
#eta_vars = [1/4, 1/2 , 1, 2, 4]
eta_vars = np.linspace(1-eta_var_max, 1+eta_var_max, num=n_omega)

cond_number = 1e2

_,_,A = generate_spd(n, type='equidistant', cond_number= cond_number)
_,_,B = generate_spd(n, type = 'exponential', cond_number= cond_number)

# Normalize to have unit Frobenius norm
A = A / (np.linalg.norm(A))
B = B / (np.linalg.norm(B))

# Use scipy to compute the exact solution
eigvals, eigvecs = eigh(A, B, eigvals_only=False, subset_by_index=[n-p, n-1])
obj_true = -0.5*np.trace(eigvecs.T @ A @ eigvecs)
print(obj_true)
print(eigvals[0])

# Same initial guess on the manifold
x0,_ = np.linalg.qr(np.random.randn(n,p))
r = np.linalg.cholesky(x0.T @ B @ x0)
x0 = np.linalg.solve(r, x0.T).T

optlogs = []

for i, (eta_var, omega_var) in enumerate(product(eta_vars, omega_vars)):
       optlogs.append({})
       solver_plam = GeneralizedLanding(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
       x_plam, optlog_plam = solver_plam.solve(0.05*eta_var, 200*omega_var, grad_type='plam', x0=x0, step_type='fixed')
       optlogs[i]['plam'] = optlog_plam
       optlogs[i]['plam_omega'] = 200*omega_var
       optlogs[i]['plam_eta'] = .05*eta_var
       
       solver_land_precon = GeneralizedLanding(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
       x_land_precon, optlog_land_precon = solver_land_precon.solve(100*eta_var, 0.1*omega_var, grad_type='precon', x0=x0, step_type='fixed')
       optlogs[i]['land_precon'] = optlog_land_precon
       optlogs[i]['land_precon_omega'] = 0.1*omega_var
       optlogs[i]['land_precon_eta'] = 100*eta_var

       optlogs[i]['omega_var'] = omega_var
       optlogs[i]['eta_var'] = omega_var


results = {
        'optlogs' : optlogs,
        'obj_true' : obj_true
}

with open('../figures/data/'+filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
