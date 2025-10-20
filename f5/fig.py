import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

'''
En mis intentos de continuar con el código,
dejé pasar un dia y olvidé todo, así que mejor 
hago todo desde 0, a ver si mejoro el orden.

Despues de considerarlo, por las necesidades de las graficas, no me conviene
crear una función que devuelva autovalores y autovectores con entradas array.
Por ello solo lo vectorizo en su interior. SI deseara hacer uno que reciba
arrays, en las figuras (a)-(c) deberia ingresar para un vector de 200 valores
identicos a la razon r, lo cual preservaria la consistencia con la generacion
elemento (r) a elemento (n).
'''

def H_transmon(r, n, m, eig_num=1):
    '''
    r: Es la razón E_J/E_C.
    n: Es el valor del offset de carga.
    m: Es el truncamiento centrado en 0.
    eig_num: Es el numero de autovalores requerido.

    La función H_base devuelve los autovalores,
    autovectores, y la lista de -m a m usada para la matriz.
    '''

    m_diag = np.arange(-m, m+1)
    dim = len(m_diag)
    
    diag_pri = 4*(n - m_diag)**2
    diag_sec = -0.5*r*np.ones(dim - 1)

    H = diags(
        diagonals=[diag_pri, diag_sec, diag_sec],
        offsets=[0, -1, 1],
        format="csr"
    )

    E, V = eigsh(H, k=eig_num, which="SA", return_eigenvectors=True)

    idx = np.argsort(E)
    E = E[idx]
    V = V[:, idx]

    return E, V, m_diag

def H_transmon_offset(r_array, n_limits, E_limits, trunc_num, eigs_num):
    '''
    r_array: Es la lista de r asociados a cada figura, su longitud es el número de gráficas.
    n_limits: Es el intervalo de los valores que tomará el offset, tomará 100 puntos por cada unidad.
    E_limits: Es el intervalo en el eje 'y' que mostrará por cada gráfica.
    trunc_num: Es el número de truncamiento centrado en 0.
    eigs_num: Es el número de autovalores que mostrará en la gráfica.

    Grafica los autovalores de H_transmon para distintos valores del offset.
    '''

    fig_num = len(r_array)
    intervalo = np.linspace(n_limits[0], 
                            n_limits[1], 
                            int(100*(n_limits[1] - n_limits[0])) + 1)
    '''
    Donde 'intervalo' son los valores que tomará n en las figuras.
    '''

    arreglo_eig = np.zeros((fig_num, eigs_num, len(intervalo)))

    for i, r in enumerate(r_array):
        eigenvalues = np.array([H_transmon(r,n,trunc_num,eigs_num)[0] for n in intervalo])
        arreglo_eig[i, :eigs_num, :] = eigenvalues.T

    # --- Graficar ---
    _, axes = plt.subplots(1, fig_num, figsize = (5*fig_num,5))

    for i, ax in enumerate(axes):
        for j in range(eigs_num):
            ax.plot(intervalo, arreglo_eig[i,j,:]/r_array[i], label=f'E_{j}')

        ax.set_xlabel('n (offset de carga)')
        ax.set_ylabel(r'E/E_{J}')
        ax.set_title(f'ratio = {r_array[i]}')
        ax.legend()
        ax.set_ylim(E_limits[i])
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def H_transmon_carga(r_array, n_array, trunc_num, estados_num):
    '''
    r_array: Lista de razones E_J/E_C 
    n_array: Lista de offsets de carga
    trunc_num: Truncamiento centrado en 0
    estados_num: Número de estados a graficar


    '''
    
    fig_num = len(r_array)
    carga_array = np.arange(-trunc_num, trunc_num + 1)
    arreglo_evec = np.zeros((fig_num, estados_num, 2 * trunc_num + 1))

    evectors = [H_transmon(r, n, trunc_num, estados_num)[1].T
                  for r, n in zip(r_array, n_array)]
    arreglo_evec = np.stack(evectors, axis=0)


    # --- Graficar ---
    _, axes = plt.subplots(1, fig_num, figsize=(5 * fig_num, 5))

    for i, ax in enumerate(axes):
        for j in range(estados_num):
            ax.bar(carga_array, arreglo_evec[i, j, :],
                   width=0.6 if j == 0 else 0.4,
                   color="black" if j == 0 else "red",
                   label=r"$|g\rangle$" if j == 0 else r"$|e\rangle$")

        ax.set_title(f"$E_J/E_C = {r_array[i]}$, $n_g={n_array[i]}$")
        ax.set_xlabel("N")
        ax.set_xlim(-trunc_num - 0.2, trunc_num + 0.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel(r"$\psi(N)$")
    plt.tight_layout()
    plt.show()


def H_transmon_phi(r_array, n_array, trunc_num, estados_num):
    '''
    r_array: Lista de razones E_J/E_C 
    n_array: Lista de offsets de carga
    trunc_num: Truncamiento centrado en 0
    estados_num: Número de estados a graficar
    '''

    fig_num = len(r_array)
    phi = np.linspace(-np.pi, np.pi, 400)
    n_vals = np.arange(-trunc_num, trunc_num + 1)

    arreglo_phi = np.zeros((fig_num, estados_num, len(phi)))

    for i, (r, n) in enumerate(zip(r_array, n_array)):
        _, evec, _ = H_transmon(r, n, trunc_num, eig_num=estados_num)

        exp_factor = np.exp(1j * np.outer(n_vals, phi))
        psi = evec.T @ exp_factor / np.sqrt(2 * np.pi) 
        arreglo_phi[i, :, :] = np.abs(psi)**2

    # --- Graficar ---
    _, axes = plt.subplots(1, fig_num, figsize=(5 * fig_num, 5), sharey=True)
    if fig_num == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(phi, arreglo_phi[i, 0, :], color="black", label=r"$|g\rangle$")
        ax.plot(phi, arreglo_phi[i, 1, :], color="red", label=r"$|e\rangle$")
        ax.set_title(f"$E_J/E_C = {r_array[i]}$, $n_g = {n_array[i]}$")
        ax.set_xlabel(r"$\varphi$")
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel(r"$|\psi(\varphi)|^2$")
    plt.tight_layout()
    plt.show()

from scipy.special import genlaguerre
import math as mt

def H_fluxonio(r1, r2, r_phi, dim, n_ext = 0):
    '''
    r1: razon E_J/E_L
    r2: razon E_J/E_C
    r_phi: razon Phi/Phi_0
    dim: la dimension de la matriz, trunca los estados a dim
    n_ext: offset de carga
    '''

    E_L = 1/r1
    E_C = 1/r2
    N_c = (E_L/(2*E_C))**(0.25)
    P_c = (32*E_C/E_L)**(0.25)
    p_ext = 2*np.pi*r_phi

    H = np.zeros((dim, dim), dtype=complex)

    for m in range(dim):
        for n in range(m+1):
            d = m - n
            H[m,n] += (-1/2)*(np.exp(-1j*(P_c**2)/2) 
                              * (1j*P_c)**d 
                              * np.sqrt(mt.factorial(n)/mt.factorial(m))
                              * genlaguerre(n, d)(P_c**2)
                              * (np.exp(-1j*p_ext) + (-1)**d * np.exp(1j*p_ext)))
            if(m == n):
                H[m,n] += 4*(2*n+1)*E_C*N_c**2 + 4*E_C*n_ext**2 + 0.5*(2*n+1)*E_L*P_c**2
            elif(m == n+1):
                H[m,n] += np.sqrt(n+1) * (1j*8*E_C*N_c*n_ext)
            elif(m == n+2):
                H[m,n] += np.sqrt((n+1)*(n+2)) * (-4*E_C*N_c**2 + 0.5*E_L*P_c**2)

            if(m != n):
                H[n,m] = H[m,n].conjugate()

    return H

def H_fluxiono_phi(r1, r2, dim, n_ext=0):
    '''
    r1: razon E_J/E_L
    r2: razon E_J/E_C
    dim: la dimension de la matriz, trunca los estados a dim
    n_ext: offset de carga
    '''

    x = np.linspace(-1, 1, 200)
    autovalores = np.zeros((5, len(x)))

    for i, r_phi in enumerate(x):
        H = H_fluxonio(r1, r2, r_phi, dim, n_ext)
        eigvals, _ = np.linalg.eigh(H)
        autovalores[:, i] = np.sort(eigvals.real)[:5]

    plt.figure(figsize=(8, 6))
    for n in range(5):
        plt.plot(x, autovalores[n])
    plt.xlabel(r'$\Phi/\Phi_0$')
    plt.ylabel(r'$E/E_J$')
    plt.title(fr'$E_J/E_L = {r1:.1f},\; E_J/E_C = {r2:.1f}$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
            
