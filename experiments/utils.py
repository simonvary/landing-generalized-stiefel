import numpy as np

def compute_mean_std(loader):
    mean = 0.
    std = 0.
    for batch, _ in loader:
        batch_n = batch.size(0)
        batch = batch.view(batch_n, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return(mean, std)

def generate_spd(n, type='equidistant', cond_number = 1e2):
    Q,_ = np.linalg.qr(np.random.randn(n,n))
    eval_max = 1
    eval_min = 1/cond_number
    if type == 'equidistant':
        evals = np.linspace(eval_max, eval_min, num = n)
    elif type == 'exponential':
        p = n
        evals = np.exp(np.log(1/p)*(1 - np.linspace(n, (n-p+1), num = p)/n))
    elif type == 'wigner':
        evals = None
    else:
        evals = None
    return(evals, Q, Q@np.diag(evals)@Q.T)