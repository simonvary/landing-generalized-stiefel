import numpy as np
import cupy as cp

from time import time
import matplotlib.pyplot as plt


def skew(M):
    return .5 * (M - M.T)
    

def psi(X, G, B):
    BX = B @ X
    M = G @ X.T @ B
    return skew(M) @ BX


def psir(X, G, B):
    BX = B @ X
    M = cp.linalg.inv(B) @ G @ X.T
    return skew(M) @ BX


n = 10
m = n
n_reps = 100


time_dirs = []
n_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 2*2048]
for n in n_list:
    m = n
    times = []
    X = cp.random.randn(n, m)
    G = cp.random.randn(n, m)
    B = cp.random.randn(n, n)
    for i in range(n_reps):
        t0 = time()
        psi(X, G, B)
        times.append(time() - t0)
        t1 = time()
        psir(X, G, B)
        times.append(time() - t1)
    times = np.array(times).reshape(n_reps, 2)
    time_dirs.append(times)

with open('../figures/data/'+filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
