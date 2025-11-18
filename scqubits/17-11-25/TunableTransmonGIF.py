import numpy as np
import matplotlib.pyplot as plt
import imageio
import scqubits as scq
import os

class TunableTransmonGIF:

    def __init__(self, EC, ncut, ng_values, ratio_values, phi_values, d=0.0):
        self.EC = EC
        self.ncut = ncut
        self.ng_values = np.array(ng_values)
        self.ratio_values = np.array(ratio_values)
        self.phi_values = np.array(phi_values)
        self.d = d
        
        self._compute_all_maps()

    def _compute_all_maps(self):
        import time
        
        N_ng = len(self.ng_values)
        N_r = len(self.ratio_values)
        N_phi = len(self.phi_values)

        self.N_maps = np.zeros((N_phi, N_ng, N_r))
        self.cos_phi_maps = np.zeros((N_phi, N_ng, N_r))

        total_qubits = N_phi * N_ng * N_r
        qubit_count = 0
        start_time = time.time()

        width_total = len(str(total_qubits))
        width_phi = len(str(N_phi))
        width_ng = len(str(N_ng))
        width_r = len(str(N_r))

        for k, phi in enumerate(self.phi_values):
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
                    self.N_maps[k, i, j] = abs(tr.n_operator((evals, evecs))[0, 1])
                    self.cos_phi_maps[k, i, j] = abs(tr.cos_phi_operator((evals, evecs))[0, 1])

                    qubit_count += 1

                    if qubit_count % 20 == 0 or qubit_count == total_qubits:
                        elapsed = time.time() - start_time
                        percent = (qubit_count / total_qubits) * 100
                        avg_time_per_qubit = elapsed / qubit_count
                        remaining_qubits = total_qubits - qubit_count
                        time_remaining = avg_time_per_qubit * remaining_qubits
                        time_total = elapsed + time_remaining

                        elapsed_h = int(elapsed // 3600)
                        elapsed_m = int((elapsed % 3600) // 60)
                        elapsed_s = int(elapsed % 60)

                        remaining_h = int(time_remaining // 3600)
                        remaining_m = int((time_remaining % 3600) // 60)
                        remaining_s = int(time_remaining % 60)

                        total_h = int(time_total // 3600)
                        total_m = int((time_total % 3600) // 60)
                        total_s = int(time_total % 60)

                        progress_str = (f"\rProgreso: [{percent:.2f}%][{qubit_count:0{width_total}d}/{total_qubits}]"
                                      f"[{k+1:0{width_phi}d}/{N_phi} phi][{i+1:0{width_ng}d}/{N_ng} ng][{j+1:0{width_r}d}/{N_r} EJ/EC] | "
                                      f"Transcurrido: {elapsed_h}h {elapsed_m:02d}m {elapsed_s}s | "
                                      f"Restante: {remaining_h}h {remaining_m:02d}m {remaining_s}s | "
                                      f"Total: {total_h}h {total_m:02d}m {total_s}s")

                        print(progress_str, end='', flush=True)

        print()

    def _compute_extrema(self, maps):
        N_phi = maps.shape[0]
        N_ng = maps.shape[1]
        N_r = maps.shape[2]

        extrema = {
            'min': np.zeros((N_phi, N_ng, N_r), dtype=bool),
            'max': np.zeros((N_phi, N_ng, N_r), dtype=bool),
        }

        for k in range(N_phi):
            M = maps[k]

            col_min = np.min(M, axis=0, keepdims=True)
            col_max = np.max(M, axis=0, keepdims=True)
            extrema['min'][k] = (M == col_min)
            extrema['max'][k] = (M == col_max)

        return extrema

    def _plot_extrema_on_axes(self, ax, M, extrema_frame):
        X = np.repeat(self.ratio_values[np.newaxis, :], len(self.ng_values), axis=0)
        Y = np.repeat(self.ng_values[:, np.newaxis], len(self.ratio_values), axis=1)

        mask_min = extrema_frame['min']
        mask_max = extrema_frame['max']

        if np.any(mask_min):
            ax.scatter(X[mask_min], Y[mask_min], s=8, c="lime", label="min")

        if np.any(mask_max):
            ax.scatter(X[mask_max], Y[mask_max], s=8, c="red", label="max")

        ax.legend(loc="upper right", fontsize=7)

    def _make_frame(self, M, phi, operator_name, extrema_frame, show_extrema):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)

        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[self.ratio_values[0], self.ratio_values[-1],
                    self.ng_values[0], self.ng_values[-1]]
        )

        fig.colorbar(im, ax=ax)

        fig.suptitle(f"{operator_name}, phi_ext={phi:.3f}", fontsize=12)

        ax.set_xlabel("EJ/EC")
        ax.set_ylabel("ng")

        if show_extrema:
            self._plot_extrema_on_axes(ax, M, extrema_frame)

        fig.canvas.draw()

        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()

        plt.close(fig)
        return rgb

    def _generate_gifs_generic(self, maps, operator_name, duration_seconds=5):
        
        cwd = os.getcwd()
        output_dir = os.path.join(cwd, "tuntr")
        os.makedirs(output_dir, exist_ok=True)

        extrema = self._compute_extrema(maps)

        frames_no_extrema = []
        frames_with_extrema = []

        n_frames = len(self.phi_values)
        fps = n_frames / duration_seconds

        for k, phi in enumerate(self.phi_values):
            M = maps[k]
            extrema_frame = {key: extrema[key][k] for key in extrema}

            frame_no = self._make_frame(M, phi, operator_name, extrema_frame, False)
            frames_no_extrema.append(frame_no)

            frame_with = self._make_frame(M, phi, operator_name, extrema_frame, True)
            frames_with_extrema.append(frame_with)

        filename_no = f"tunabletransmon_{operator_name}_no_extrema.gif"
        filename_with = f"tunabletransmon_{operator_name}_with_extrema.gif"

        path_no = os.path.join(output_dir, filename_no)
        path_with = os.path.join(output_dir, filename_with)

        imageio.v3.imwrite(path_no, frames_no_extrema, format="GIF", fps=fps)
        imageio.v3.imwrite(path_with, frames_with_extrema, format="GIF", fps=fps)

        return path_no, path_with

    def generate_gifs_N(self, duration_seconds=5):
        return self._generate_gifs_generic(self.N_maps, "N", duration_seconds)

    def generate_gifs_cos_phi(self, duration_seconds=5):
        return self._generate_gifs_generic(self.cos_phi_maps, "cos_phi", duration_seconds)