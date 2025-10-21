import numpy as np
from scipy.sparse import diags, identity, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import cosm
from typing import Tuple

# Agradecimientos a GPT por la integridad de los comentarios

class Fluxonium:
    """
    Representa el Hamiltoniano del circuito superconductivo tipo Fluxonium.

    Esta clase construye los operadores del sistema (N, φ) a partir de los operadores
    de creación y aniquilación, y ensambla el Hamiltoniano total del Fluxonium:

        H = 4 EC N² + ½ EL φ² - EJ cos(φ - φ_ext)

    También calcula los primeros autovalores y autovectores del sistema.

    Parameters
    ----------
    EJ : float
        Energía de Josephson.
    EC : float
        Energía de carga.
    EL : float
        Energía inductiva.
    phi_ext : float
        Flujo externo aplicado (en unidades reducidas, φ₀ = ħ/2e).
    dim : int, optional
        Dimensión de la base numérica (número de estados). Default = 30.
    n_eig : int, optional
        Número de autovalores y autovectores a calcular. Default = 5.

    Attributes
    ----------
    H : csr_matrix
        Hamiltoniano total en forma dispersa.
    N : csr_matrix
        Operador de número reducido.
    phi : csr_matrix
        Operador de fase reducido.
    eig : tuple[np.ndarray, np.ndarray]
        Par (autovalores, autovectores) ordenados por energía creciente.
    """

    def __init__(self, EJ: float, EC: float, EL: float, phi_ext: float, dim: int = 50, n_eig: int = 5):
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
        """
        Construye el operador de creación a† en representación de base armónica.

        Returns
        -------
        csr_matrix
            Matriz dispersa con elementos ⟨n+1|a†|n⟩ = √(n+1).
        """

        n = np.sqrt(np.arange(1, self.dim))
        return diags(n, 1, format='csr')
    
    def _operador_aniquilacion(self) -> csr_matrix:
        """
        Construye el operador de aniquilación a en representación de base armónica.

        Returns
        -------
        csr_matrix
            Matriz dispersa con elementos ⟨n-1|a|n⟩ = √n.
        """

        n = np.sqrt(np.arange(1, self.dim))
        return diags(n, -1, format='csr')
    
    def _crear_N(self) -> csr_matrix:
        """
        Construye el operador de número reducido N.

        Definido como:
            N = i * (EL / (32 EC))^(1/4) * (a† - a)

        Returns
        -------
        csr_matrix
            Matriz dispersa correspondiente al operador N.
        """

        const_n = (self.EL / (32 * self.EC))**(1/4)
        return 1j * const_n * (self._a - self._adag)
    
    def _crear_phi(self) -> csr_matrix:
        """
        Construye el operador de fase reducida φ.

        Definido como:
            φ = (2 EC / EL)^(1/4) * (a† + a)

        Returns
        -------
        csr_matrix
            Matriz dispersa correspondiente al operador φ.
        """

        const_phi = (2 * self.EC/ self.EL)**(1/4)
        return const_phi * (self._a + self._adag)
    
    def _crear_H(self) -> csr_matrix:
        """
        Ensambla el Hamiltoniano total del Fluxonium.

        H = 4 EC N² + ½ EL φ² - EJ cos(φ - φ_ext)

        Returns
        -------
        csr_matrix
            Matriz dispersa correspondiente al Hamiltoniano.
        """

        N_2 = self._N @ self._N
        phi_2 = self._phi @ self._phi
        H_1 = 4 * self.EC * N_2 + 0.5 * self.EL * phi_2

        id = identity(self.dim, format='csr')

        phi_minus_ext = (self._phi - self.phi_ext * id).toarray()
        H_2 = - self.EJ * cosm(phi_minus_ext)

        return (H_1 + csr_matrix(H_2)).tocsr()
    
    def _diagonalizar(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonaliza el Hamiltoniano y obtiene los primeros autovalores y autovectores.

        Usa el método iterativo de Lanczos (`scipy.sparse.linalg.eigsh`).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Autovalores ordenados de menor a mayor.
            - Autovectores normalizados (cada fila corresponde a un autovector).
        """

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