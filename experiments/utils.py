import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

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
    trainset_MNIST = torchvision.datasets.MNIST(root='data', download=download, train=True, transform=transform_MNIST)

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