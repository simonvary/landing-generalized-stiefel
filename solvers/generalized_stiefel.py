import numpy as np

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class GeneralizedStiefel(EuclideanEmbeddedSubmanifold):
    """
    Factory class for the generalized Stiefel manifold.
    Instantiation requires the dimensions n, p to be specified.

    Elements are represented as n x p matrices.
    
    Source: https://github.com/chovine/pymanopt/blob/generalized_stiefel/pymanopt/manifolds/generalized_stiefel.py
    """

    def __init__(self, n, p, B=None):
        self._n = n
        self._p = p
        if B is None:
            self._B = np.identity(n)
        else:
            self._B = B

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError("Need n >= p >= 1. Values supplied were n = %d "
                             "and p = %d." % (n, p))
        name = "Generalized Stiefel manifold St(%d, %d)" % (n, p)
        dimension = int(n * p - p * (p + 1) / 2)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self._p)

    def inner(self, X, eta, zeta):
        # Inner product (Riemannian metric) on the tangent space
        # For the stiefel this is the Frobenius inner product.
        return np.trace(eta.T @ self._B @ zeta)

    def proj(self, X, U):
        BXU = (self._B @ X).T @ U
        return U - X @ _symm(BXU)

    def ehess2rhess(self, X, egrad, ehess, H):
        egraddot = ehess
        Xdot = H

        # Directional derivative of the Riemannian gradient
        egrad_scaledot = np.linalg.inv(self._B) @ egraddot
        rgraddot = egrad_scaledot
        rgraddot -= Xdot @ _symm(X.T @ egrad)
        rgraddot -= X @ _symm(Xdot.T @ egrad)
        rgraddot -= X @ _symm(X.T @ egrad)
        return self.proj(X, rgraddot)

    # Retract to the Stiefel using the qr decomposition of X + G.
    def retr(self, X, G):
        # Calculate the generalized qr decomposition of X + G
        A = X + G
        R = np.linalg.cholesky(A.T @ self._B @ A)
        return A @ np.linalg.inv(R)

    def norm(self, X, eta):
        return np.sqrt(self.inner(X, eta, eta))

    # Generate random Stiefel point using qr of random normally distributed
    # matrix.
    def rand(self):
        X = np.random.randn(self._n, self._p)
        return self._guf(X)

    def randvec(self, X):
        U = np.random.randn(np.random.randn(self._n, self._p))
        U = self.proj(X, U)
        U = U / np.linalg.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def zerovec(self, X):
        return np.zeros((self._n, self._p))

    def _guf(self, X):
        U, _, VH = np.linalg.svd(X, full_matrices=False)
        SS, Q = np.linalg.eig(U.T @ self._B @ U)
        A = Q @ np.linalg.inv(np.diag(np.sqrt(SS))) @ Q.T
        return U @ A @ VH


def _symm(D):
    return (D + D.T)/2