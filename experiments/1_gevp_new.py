"""
Comparison of solving GEVP with RiemannianSD vs Landing
Deterministic experiment using cupy on GPU

"""
from time import time
import random
import pickle

import matplotlib.pyplot as plt

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

filename = '1_gevp_new.pkl'

n = 1000
p = 500

max_time =.25*60 # in seconds
maxiter = 20000

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

solver_land_precon = GeneralizedLanding(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
x_land_precon, optlog_land_precon = solver_land_precon.solve(10000, .1, grad_type='precon', x0=x0, step_type='fixed', eps_d = 1)

solver_land_R = GeneralizedLanding(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
x_land_R, optlog_land_R = solver_land_R.solve(0.001, 10000, grad_type='R', x0=x0, step_type='fixed', eps_d = 1)



solver_plam = GeneralizedLanding(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
x_plam, optlog_plam = solver_plam.solve(.05, 200, grad_type='plam', x0=x0, eps_d = None, step_type='fixed')

solver_rsd = RiemmGeneralizedStiefel(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
x_rsd, optlog_rsd = solver_rsd.solve(eta = .001, step_type = 'fixed', x0=x0)

solver_land_riem = GeneralizedLanding(A, B, p, maxiter = maxiter, mingradnorm=1e-6,maxtime=max_time)
x_land_riem, optlog_land_riem = solver_land_riem.solve(.001, 1, grad_type='riem', x0=x0, step_type='fixed', eps_d = 1)




results = {
        'optlog_land_precon' : optlog_land_precon,
        'optlog_rsd' : optlog_rsd,
        'optlog_land_R' : optlog_land_R,
        'optlog_plam' : optlog_plam,
        'optlog_land_riem' : optlog_land_riem,
        'obj_true' : obj_true
}

with open('../figures/data/'+filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
