import torch

def positivedef_matrix_sqrt(array):
  """Stable method for computing matrix square roots, supports complex matrices.

  Args:
            array: A numpy 2d array, can be complex valued that is a positive
                   definite symmetric (or hermitian) matrix

  Returns:
            sqrtarray: The matrix square root of array
  """
  w, v = torch.linalg.eigh(array)
  wsqrt = torch.sqrt(w)
  sqrtarray = v @ (torch.diag_embed(wsqrt) @ torch.conj(v).T)
  return sqrtarray


def compute_mean_std(loader):
    mean = 0.
    std = 0.
    for batch in loader:
        batch_n = batch.size(0)
        batch = batch.view(batch_n, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return(mean, std)

def loader_to_cov(loaderA, loaderB, loader_meanA=None, loader_meanB=None, device='cpu'):
    if loader_meanA is None:
        loader_meanA,_ = compute_mean_std(loaderA)
    if loader_meanB is None:
        loader_meanB,_ = compute_mean_std(loaderB)
    loader_meanA, loader_meanB = loader_meanA.to(device), loader_meanB.to(device)
    
    n1 = loader_meanA.size(0)
    n2 = loader_meanB.size(0)

    covA = torch.zeros(n1, n1, device = device)
    covB = torch.zeros(n2, n2, device = device)
    covAB = torch.zeros(n1, n2, device = device)

    for _, (batchA, batchB) in enumerate(zip(loaderA, loaderB)):
        batchA, batchB = batchA.to(device), batchB.to(device)
        batchA = batchA - loader_meanA
        batchB = batchB - loader_meanB
        covA = covA + batchA.T @ batchA
        covB = covB + batchB.T @ batchB
        covAB = covAB + batchA.T @ batchB
    covA = covA / len(loaderA.dataset)
    covB = covB / len(loaderB.dataset)
    covAB = covAB / len(loaderA.dataset)
    return(covA, covB, covAB)


def cca_closed_form(covA, covB, covAB, epsilon=1e-10, verb = True):
    dimA = covA.shape[0]
    dimB = covB.shape[0]

    if dimA == 0 or dimB == 0:
        return ([0, 0, 0], [0, 0, 0], torch.zeros_like(covA), 
                torch.zeros_like(covB))

    if verb:
        print("adding eps to diagonal and taking inverse")

    covA += epsilon * torch.eye(dimA, device = covA.device)
    covB += epsilon * torch.eye(dimB, device = covB.device)   
    covAinv = torch.linalg.pinv(covA)
    covBinv = torch.linalg.pinv(covB)
    
    if verb:
        print("taking square root")

    invsqrtA = positivedef_matrix_sqrt(covAinv)
    invsqrtB = positivedef_matrix_sqrt(covBinv)

    if verb:
        print("dot products...")
    arr = invsqrtA @ covAB @ invsqrtB

    if verb:
        print("trying to take final svd")
    u, s, vh = torch.linalg.svd(arr)

    if verb:
        print("computed everything!")
    return invsqrtA@u, s, invsqrtB@vh.T


def svcca(covA, covB, covAB, p, epsilon=1e-10, verb=False):
    dimA = covA.shape[0]
    dimB = covB.shape[0]

    if dimA == 0 or dimB == 0:
        return ([0, 0, 0], [0, 0, 0], torch.zeros_like(covA), 
                torch.zeros_like(covB))
    
    if verb:
        print("adding eps to diagonal and taking inverse")
    covA += epsilon * torch.eye(dimA, device = covA.device)
    covB += epsilon * torch.eye(dimB, device = covB.device)   

    # Perform SVD
    covA_U, covA_s, _ = torch.linalg.svd(covA, full_matrices=False)
    covB_U, covB_s, _ = torch.linalg.svd(covB, full_matrices=False)

    arr = torch.diag_embed(-torch.sqrt(covA_s[:p])) @ covA_U[:,:p].T @ covAB @ covB_U[:,:p] @ torch.diag_embed(-torch.sqrt(covB_s[:p]))
    
    u, s, vh = torch.linalg.svd(arr)

    covAinvsqrt = positivedef_matrix_sqrt(torch.linalg.pinv(covA))
    covBinvsqrt = positivedef_matrix_sqrt(torch.linalg.pinv(covB))
    u_final = covAinvsqrt@ covA_U[:,:p] @ u
    v_final = covBinvsqrt @ covB_U[:,:p] @ vh.T
    return u_final, s, v_final, covA_U[:,:p]@torch.diag_embed(covA_s[:p]) @ covA_U[:,:p].T 