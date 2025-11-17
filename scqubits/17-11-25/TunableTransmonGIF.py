import numpy as np
import matplotlib.pyplot as plt
import imageio
import scqubits as scq


class TunableTransmonGIF:

    def __init__(self, EC, ncut, ng_values, ratio_values, d=0.0):
        self.EC = EC
        self.ncut = ncut
        self.ng_values = np.array(ng_values)
        self.ratio_values = np.array(ratio_values)
        self.d = d

    def compute_map(self, phi):
        N_ng = len(self.ng_values)
        N_r = len(self.ratio_values)

        N_map = np.zeros((N_ng, N_r))
        cos_map = np.zeros((N_ng, N_r))

        for i, ng in enumerate(self.ng_values):
            for j, R in enumerate(self.ratio_values):
                EJmax = R * self.EC

                tr = scq.TunableTransmon(
                    EJmax=EJmax,
                    EC=self.EC,
                    d=self.d,
                    flux=phi,
                    ng=ng,
                    ncut=self.ncut
                )

                evals, evecs = tr.eigensys()
                N_map[i, j]  = abs(tr.n_operator((evals, evecs))[0, 1])
                cos_map[i, j] = abs(tr.cos_phi_operator((evals, evecs))[0, 1])

        return N_map, cos_map

    def _extrema_mask(self, M, mode):
        if mode == "min":
            col = np.min(M, axis=0, keepdims=True)
        else:
            col = np.max(M, axis=0, keepdims=True)

        mask = (M == col)
        return mask
    
    def _plot_extrema_on_axes(self, ax, M):
        X = np.repeat(self.ratio_values[np.newaxis, :], len(self.ng_values), axis=0)
        Y = np.repeat(self.ng_values[:, np.newaxis], len(self.ratio_values), axis=1)

        mask_min = self._extrema_mask(M, "min")
        mask_max = self._extrema_mask(M, "max")

        if np.any(mask_min):
            ax.scatter(X[mask_min], Y[mask_min], s=8, c="lime", label="min")

        if np.any(mask_max):
            ax.scatter(X[mask_max], Y[mask_max], s=8, c="red", label="max")

        ax.legend(loc="upper right", fontsize=7)

    def _make_frame(self, M, phi, show_extrema):
        fig, ax = plt.subplots(figsize=(6,5), dpi=100)

        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[self.ratio_values[0], self.ratio_values[-1],
                    self.ng_values[0], self.ng_values[-1]]
        )
        ax.set_title(f"phi = {phi:.3f}")
        ax.set_xlabel("EJ/EC")
        ax.set_ylabel("ng")

        if show_extrema:
            self._plot_extrema_on_axes(ax, M)

        fig.canvas.draw()

        rgba = np.asarray(fig.canvas.buffer_rgba())      
        rgb = rgba[:, :, :3].copy()                      

        plt.close(fig)
        return rgb


    def generate_gifs(self, phi_values, key="N_map", duration_seconds=5):
        import os
        cwd = os.getcwd()

        def make_one(show_extrema):
            frames = []
            n_frames = len(phi_values)
            fps = n_frames / duration_seconds
            for phi in phi_values:
                N_map, cos_map = self.compute_map(phi)
                M = N_map if key == "N_map" else cos_map
                frame = self._make_frame(M, phi, show_extrema)
                frames.append(frame)
            filename = f"tunabletransmon_{key}_{'with' if show_extrema else 'no'}extrema.gif"
            path = os.path.join(cwd, filename)
            imageio.v3.imwrite(path, frames, format="GIF", fps=fps)
            return path

        path1 = make_one(True)
        path2 = make_one(False)
        return path1, path2