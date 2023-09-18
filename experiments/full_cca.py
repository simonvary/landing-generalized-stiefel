





# Run full CCA


def full_CCA(loaderA, loaderB, sizeA, sizeB, num_components, device):
    covA, covB, covAB = loader2cov(loaderA, loaderB, sizeA, sizeB, device)
    evals_covA, evecs_covA = torch.linalg.eig(covA)
    evals_covB, evecs_covB = torch.linalg.eig(covB)
    covA_isqrt = evecs_covA @ torch.diag(evals_covA**(-0.5)) @ evecs_covA.T
    covB_isqrt = evecs_covB @ torch.diag(evals_covB**(-0.5)) @ evecs_covB.T
    u, s, v = torch.linalg.svd(covA_isqrt @ covAB @ covB_isqrt, full_matrices = False)
    x_true = u[:,:num_components]
    xy_corr = s[:num_components]
    y_true = v[:,:num_components]