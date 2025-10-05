import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def H_island_eval(r_array, Nmax):
    num_points=200
    num_eigs=4
    dim = 2*Nmax + 1
    n_vals = np.arange(-Nmax, Nmax+1)

    n_ext_array = np.linspace(-1.0, 1.0, num_points)
    eigenvalues = np.zeros((num_points, len(r_array), num_eigs))

    diag_all = 4 * (n_vals[np.newaxis, :] - n_ext_array[:, np.newaxis])**2

    for j, r in enumerate(r_array):
        sdiag = -r/2 * np.ones(dim-1)

        for i, diag in enumerate(diag_all):
            H = diags(
                diagonals=[diag, sdiag, sdiag],
                offsets=[0, 1, -1],
                format="csr"
            )
            E = eigsh(H, k=num_eigs, which="SA", return_eigenvectors=False)
            eigenvalues[i, j, :] = np.sort(E)

    return eigenvalues, n_ext_array



def plot_H_island(r_array, Nmax, ymin_list, ymax_list):
    eigenvalues, n_ext_array = H_island_eval(r_array, Nmax)

    num_r = len(r_array)
    num_eigs = eigenvalues.shape[2]

    _, axs = plt.subplots(1, num_r, figsize=(5*num_r, 4), sharey=False)

    for j, (ax, r, ymin, ymax) in enumerate(zip(axs, r_array, ymin_list, ymax_list)):
        for k in range(num_eigs):
            ax.plot(n_ext_array, eigenvalues[:, j, k]/r, label=f"$E_{k}$")

        ax.set_xlabel(r"$N_{ext}$")
        ax.set_ylabel(r"$E/E_J$")
        ax.set_title(f"$E_J/E_C = {r}$")
        ax.grid(True)
        #ax.legend()
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()
