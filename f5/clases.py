import numpy as np
from scipy.sparse import diags, identity, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import cosm
from typing import Tuple

class Fluxonium:

    def __init__(self, EJ: float, EC: float, EL: float, phi_ext: float, dim: int = 30, n_eig: int = 5):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.phi_ext = phi_ext
        self.dim = dim
        self.n_eig = n_eig

        self._adag = self._operador_creacion()
        self._a = self._operador_aniquilacion()

        self._N = self._crear_N()
        self._phi = self._crear_phi()
        self._H = self._crear_H()

        self._eigval, self._eivec = self._diagonalizar()

    def _operador_creacion(self) -> csr_matrix:
        n = np.sqrt(np.arange(1, self.dim))
        return diags(n, 1, format='csr')
    
    def _operador_aniquilacion(self) -> csr_matrix:
        n = np.sqrt(np.arange(1, self.dim))
        return diags(n, -1, format='csr')
    
    def _crear_N(self) -> csr_matrix:
        const_n = (self.EL / (32 * self.EC))**(1/4)
        return 1j * const_n * (self._a - self._adag)
    
    def _crear_phi(self) -> csr_matrix:
        const_phi = (2 * self.EC/ self.EL)**(1/4)
        return const_phi * (self._a + self._adag)
    
    def _crear_H(self) -> csr_matrix:
        N_2 = self._N @ self._N
        phi_2 = self._phi @ self._phi
        H_1 = 4 * self.EC * N_2 + 0.5 * self.EL * phi_2

        id = identity(self.dim, format='csr')

        phi_minus_ext = (self._phi - self.phi_ext * id).toarray()
        H_2 = - self.EJ * cosm(phi_minus_ext)

        return (H_1 + csr_matrix(H_2)).tocsr()
    
    def _diagonalizar(self) -> Tuple[np.ndarray, np.ndarray]:
        eigval, eigvec = eigsh(self._H, k = self.n_eig, which='SA')
        idx = np.argsort(eigval)
        eigval, eigvec = eigval[idx], eigvec[:, idx]

        return eigval, eigvec.T
    
    @property
    def H(self) -> csr_matrix:
        return self._H
    
    @property
    def N(self) -> csr_matrix:
        return self._N
    
    @property
    def phi(self) -> csr_matrix:
        return self._phi
    
    @property
    def eig(self):
        return self._eigval, self._eivec