import numpy as np
import scqubits as scq
import matplotlib.pyplot as plt

class TransmonMap:
    """
    Calcula los elementos de matriz entre g y e para los operadores N y cos(phi)
    para rangos de ng y EJ/EC.
    """

    def __init__(self, EC, ncut, ng_values, ratio_values):
        self.EC = EC
        self.ncut = ncut
        self.ng_values = np.array(ng_values)
        self.ratio_values = np.array(ratio_values)

        self.data = {
            "N_map": None,
            "cos_map": None,
            "extrema": {}
        }

    # Calcula los mapas
    def compute_maps(self):
        N_ng, N_r = len(self.ng_values), len(self.ratio_values)

        N_map = np.zeros((N_ng, N_r))
        cos_map = np.zeros((N_ng, N_r))

        for i, ng in enumerate(self.ng_values):
            for j, R in enumerate(self.ratio_values):
                EJ = R * self.EC
                tr = scq.Transmon(EJ=EJ, EC=self.EC, ng=ng, ncut=self.ncut)
                evals, evecs = tr.eigensys()

                N_map[i, j] = abs(tr.n_operator((evals, evecs))[0, 1])
                cos_map[i, j] = abs(tr.cos_phi_operator((evals, evecs))[0, 1])

        self.data["N_map"] = N_map
        self.data["cos_map"] = cos_map

        self.data["extrema"] = {
            "N_min_ratio": self._find_extrema(N_map, mode="min"),
            "N_max_ratio": self._find_extrema(N_map, mode="max"),
            "cos_min_ratio": self._find_extrema(cos_map, mode="min"),
            "cos_max_ratio": self._find_extrema(cos_map, mode="max"),
        }

    # Matriz con NaN excepto en los mínimos o máximos por columna
    def _find_extrema(self, M, mode):
        if mode == "min":
            target = np.nanmin(M, axis=0, keepdims=True)
        else:
            target = np.nanmax(M, axis=0, keepdims=True)

        mask = (M == target)
        out = np.full_like(M, np.nan)
        out[mask] = M[mask]
        return out
    
    # Devuelve el mínimo o máximo global, con ng y ratio asociados
    def get_global_extremum(self, key="N_map", mode="min"):
        M = self.data[key]
        base = key.split("_")[0]

        if mode == "min":
            ext_map = self.data["extrema"][f"{base}_min_ratio"]
        else:
            ext_map = self.data["extrema"][f"{base}_max_ratio"]

        vals = ext_map[~np.isnan(ext_map)]
        if len(vals) == 0:
            return None

        if mode == "min":
            target = np.min(vals)
        else:
            target = np.max(vals)

        i, j = np.where(ext_map == target)
        i, j = i[0], j[0]

        return {
            "value": target,
            "ng": self.ng_values[i],
            "ratio": self.ratio_values[j],
            "indices": (i, j)
        }


    # Grafica el mapa de calor con los extremos
    def plot_map(self, key="N_map", show_extrema=True):
        M = self.data[key]

        plt.figure(figsize=(7,6))
        plt.imshow(
            M, aspect="auto", origin="lower",
            extent=[self.ratio_values[0], self.ratio_values[-1],
                    self.ng_values[0], self.ng_values[-1]]
        )
        plt.colorbar()
        plt.xlabel("EJ/EC")
        plt.ylabel("ng")
        plt.title(key)

        if show_extrema:
            self._plot_extrema(key)

        plt.show()

    # Solo grafica mínimos y máximos con fondo blanco
    def _plot_extrema(self, key="N_map"):
        base = key.split("_")[0]
        ext = self.data["extrema"]

        xs_grid = np.repeat(self.ratio_values[np.newaxis, :], len(self.ng_values), axis=0)
        ys_grid = np.repeat(self.ng_values[:, np.newaxis], len(self.ratio_values), axis=1)

        # lima = mínimos
        Emin = ext[f"{base}_min_ratio"]
        mask = ~np.isnan(Emin)
        if np.any(mask):
            plt.scatter(xs_grid[mask], ys_grid[mask], s=10, color="lime", label="min")

        # rojo = máximos
        Emax = ext[f"{base}_max_ratio"]
        mask = ~np.isnan(Emax)
        if np.any(mask):
            plt.scatter(xs_grid[mask], ys_grid[mask], s=10, color="red", label="max")

        plt.legend(loc="upper right")

    # Corte a ratio fijo
    def plot_cut_ratio(self, R_index, key="N_map"):
        M = self.data[key]
        plt.figure(figsize=(6,4))
        plt.plot(self.ng_values, M[:, R_index])
        plt.xlabel("ng")
        plt.ylabel(key)
        plt.grid(True)
        plt.show()
