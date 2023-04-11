"""
A simple example of the Stiefel landing algorithm on PCA problem
"""
from time import time


import numpy as np
import torch


import torchvision
import torchvision.transforms as transforms

from models import VGG16, ResNet18

from solvers import *


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1024

trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def VectorizeImage(batch):
    data = [item[0].view(-1).unsqueeze(0) for item in batch]
    return(torch.cat(data, dim = 0))

loader_A = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=VectorizeImage)


X, Y = LandingCCA(loader_A, loader_A, p = 10, learning_rate = 1e-4, lambda_regul = 1,  n_epochs=5, device = 'cuda')