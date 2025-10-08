import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

'''
En mis intentos de continuar con el código,
dejé pasar un dia y olvidé todo, así que mejor 
hago todo desde 0, a ver si mejoro elñ orden.
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

    La función H_transmon_offset graficará los autovalores de H_transmon para distintos valores del offset.
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