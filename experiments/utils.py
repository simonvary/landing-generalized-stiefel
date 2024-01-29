import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from scipy.linalg import sqrtm

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

def amari_distance(W, A):
    """
    Source: 
       https://pierreablin.github.io/ksddescent/auto_examples/plot_ica.html
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.
    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix
    A : ndarray, shape (n_features, n_features)
        Input matrix
    Returns
    -------
    d : float
        The Amari distance
    """
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])

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

def dataset_simulated_ica(n_samples = 10000, n_features = 10, batch_size=128, device = 'cpu', random_state=42):
    rng = np.random.RandomState(random_state)
    sources = rng.laplace(size=(n_samples, n_features))
    mixing = rng.randn(n_features, n_features)
    X = np.dot(sources, mixing.T)
    W = np.linalg.pinv(sqrtm(X.T.dot(X) / n_samples))
    X = np.dot(X, W.T)
    mixing = np.dot(W, mixing)
    data = dict(X=X, mixing=mixing)
    X = torch.from_numpy(X).to(device=device)
    mixing = torch.from_numpy(mixing).to(device=device)
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size, shuffle=False)
    return dict(dataloader=dataloader, mixing=mixing)
    
def dataset_MNIST(batch_size = 256, download=True):
    '''
        Returns two dataloaders of MNIST with prescribed batch_size.
        Each returns different halves of the image, split vertically
        in the middle.
    '''
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_MNIST = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    trainset_MNIST = torchvision.datasets.MNIST(root='../data', download=download, train=True, transform=transform_MNIST)

    def VectorizeLeftImage(batch):
        data = [item[0][:,:,:14].reshape(-1).unsqueeze(0) for item in batch]
        return(torch.cat(data, dim = 0))

    def VectorizeRightImage(batch):
        data = [item[0][:,:,14:].reshape(-1).unsqueeze(0) for item in batch]
        return(torch.cat(data, dim = 0))

    loader_left = torch.utils.data.DataLoader(trainset_MNIST,
                        batch_size=batch_size, shuffle=False, 
                        num_workers=2, collate_fn=VectorizeLeftImage)
    loader_right = torch.utils.data.DataLoader(trainset_MNIST,
                        batch_size=batch_size, shuffle=False, 
                        num_workers=2, collate_fn=VectorizeRightImage)
    return(loader_left, loader_right)