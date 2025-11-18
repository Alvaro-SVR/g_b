import numpy as np
import matplotlib.pyplot as plt
import imageio
import scqubits as scq
import os

class FluxoniumGIF:

    def __init__(self, EC, cutoff, EJ_EC_values, EL_EJ_values, phi_values):
        self.EC = EC
        self.cutoff = cutoff
        self.EJ_EC_values = np.array(EJ_EC_values)
        self.EL_EJ_values = np.array(EL_EJ_values)
        self.phi_values = np.array(phi_values)
        
        self._compute_all_maps()

    def _compute_all_maps(self):
        import time
        
        N_y = len(self.EL_EJ_values)
        N_x = len(self.EJ_EC_values)
        N_phi = len(self.phi_values)

        self.N_maps = np.zeros((N_phi, N_y, N_x))
        self.phi_maps = np.zeros((N_phi, N_y, N_x))
        self.cos_phi_maps = np.zeros((N_phi, N_y, N_x))

        total_qubits = N_phi * N_y * N_x
        qubit_count = 0
        start_time = time.time()

        for k, phi in enumerate(self.phi_values):
            for i, EL_EJ in enumerate(self.EL_EJ_values):
                for j, EJ_EC in enumerate(self.EJ_EC_values):
                    EJ = EJ_EC * self.EC
                    EL = EL_EJ * EJ

                    qubit = scq.Fluxonium(
                        EJ=EJ,
                        EC=self.EC,
                        EL=EL,
                        flux=phi,
                        cutoff=self.cutoff
                    )

                    evals, evecs = qubit.eigensys()
                    self.N_maps[k, i, j] = abs(qubit.n_operator((evals, evecs))[0, 1])
                    self.phi_maps[k, i, j] = abs(qubit.phi_operator((evals, evecs))[0, 1])
                    self.cos_phi_maps[k, i, j] = abs(qubit.cos_phi_operator(energy_esys=(evals, evecs))[0, 1])

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

                        width_total = len(str(total_qubits))
                        width_phi = len(str(N_phi))
                        width_y = len(str(N_y))
                        width_x = len(str(N_x))
                        
                        progress_str = (f"\rProgreso: [{percent:.2f}%][{qubit_count:0{width_total}d}/{total_qubits}]"
                                      f"[{k+1:0{width_phi}d}/{N_phi} phi][{i+1:0{width_y}d}/{N_y} EL/EJ][{j+1:0{width_x}d}/{N_x} EJ/EC] | "
                                      f"Transcurrido: {elapsed_h}h {elapsed_m:02d}m {elapsed_s:02d}s | "
                                      f"Restante: {remaining_h}h {remaining_m:02d}m {remaining_s:02d}s | "
                                      f"Total: {total_h}h {total_m:02d}m {total_s:02d}s")
                        
                        print(progress_str, end='', flush=True)

        print()

    def _compute_extrema(self, maps):
        N_phi = maps.shape[0]
        N_y = maps.shape[1]
        N_x = maps.shape[2]

        extrema = {
            'EJ_EC_min': np.zeros((N_phi, N_y, N_x), dtype=bool),
            'EJ_EC_max': np.zeros((N_phi, N_y, N_x), dtype=bool),
            'EL_EJ_min': np.zeros((N_phi, N_y, N_x), dtype=bool),
            'EL_EJ_max': np.zeros((N_phi, N_y, N_x), dtype=bool),
        }

        for k in range(N_phi):
            M = maps[k]

            col_min = np.min(M, axis=0, keepdims=True)
            col_max = np.max(M, axis=0, keepdims=True)
            extrema['EJ_EC_min'][k] = (M == col_min)
            extrema['EJ_EC_max'][k] = (M == col_max)

            row_min = np.min(M, axis=1, keepdims=True)
            row_max = np.max(M, axis=1, keepdims=True)
            extrema['EL_EJ_min'][k] = (M == row_min)
            extrema['EL_EJ_max'][k] = (M == row_max)

        return extrema

    def _plot_extrema_on_axes(self, ax, M, extrema_frame, show_EJ_EC, show_EL_EJ):
        X = np.repeat(self.EJ_EC_values[np.newaxis, :], len(self.EL_EJ_values), axis=0)
        Y = np.repeat(self.EL_EJ_values[:, np.newaxis], len(self.EJ_EC_values), axis=1)

        if show_EJ_EC:
            mask_min = extrema_frame['EJ_EC_min']
            mask_max = extrema_frame['EJ_EC_max']

            if np.any(mask_min):
                ax.scatter(X[mask_min], Y[mask_min], s=8, c="lime", label="min (EJ/EC)")

            if np.any(mask_max):
                ax.scatter(X[mask_max], Y[mask_max], s=8, c="red", label="max (EJ/EC)")

        if show_EL_EJ:
            mask_min = extrema_frame['EL_EJ_min']
            mask_max = extrema_frame['EL_EJ_max']

            if np.any(mask_min):
                ax.scatter(X[mask_min], Y[mask_min], s=8, c="white", edgecolors='black', linewidths=0.5, label="min (EL/EJ)")

            if np.any(mask_max):
                ax.scatter(X[mask_max], Y[mask_max], s=8, c="black", label="max (EL/EJ)")

        if show_EJ_EC or show_EL_EJ:
            ax.legend(loc="upper right", fontsize=7)

    def _make_frame(self, M, phi, operator_name, extrema_frame, show_EJ_EC, show_EL_EJ, log_x, log_y):
        from matplotlib.ticker import LogLocator
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)

        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[self.EJ_EC_values[0], self.EJ_EC_values[-1],
                    self.EL_EJ_values[0], self.EL_EJ_values[-1]]
        )

        fig.colorbar(im, ax=ax)

        fig.suptitle(f"{operator_name}, phi_ext={phi:.3f}", fontsize=12)

        ax.set_xlabel("EJ/EC")
        ax.set_ylabel("EL/EJ")

        if log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=20))
        if log_y:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=20))

        if show_EJ_EC or show_EL_EJ:
            self._plot_extrema_on_axes(ax, M, extrema_frame, show_EJ_EC, show_EL_EJ)

        fig.canvas.draw()

        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()

        plt.close(fig)
        return rgb

    def _generate_gifs_generic(self, maps, operator_name, show_EJ_EC=False, show_EL_EJ=False, 
                               log_x=False, log_y=False, duration_seconds=5):
        
        cwd = os.getcwd()
        output_dir = os.path.join(cwd, "flux")
        os.makedirs(output_dir, exist_ok=True)

        extrema = self._compute_extrema(maps)

        frames_no_extrema = []
        frames_with_extrema = []

        n_frames = len(self.phi_values)
        fps = n_frames / duration_seconds

        for k, phi in enumerate(self.phi_values):
            M = maps[k]
            extrema_frame = {key: extrema[key][k] for key in extrema}

            frame_no = self._make_frame(M, phi, operator_name, extrema_frame, False, False, log_x, log_y)
            frames_no_extrema.append(frame_no)

            frame_with = self._make_frame(M, phi, operator_name, extrema_frame, show_EJ_EC, show_EL_EJ, log_x, log_y)
            frames_with_extrema.append(frame_with)

        log_suffix = ""
        if log_x and log_y:
            log_suffix = "_logxy"
        elif log_x:
            log_suffix = "_logx"
        elif log_y:
            log_suffix = "_logy"

        extrema_str = ""
        if show_EJ_EC and show_EL_EJ:
            extrema_str = "both"
        elif show_EJ_EC:
            extrema_str = "EJ_EC"
        elif show_EL_EJ:
            extrema_str = "EL_EJ"

        filename_no = f"fluxonium_{operator_name}_no_extrema{log_suffix}.gif"
        filename_with = f"fluxonium_{operator_name}_extrema_{extrema_str}{log_suffix}.gif"

        path_no = os.path.join(output_dir, filename_no)
        path_with = os.path.join(output_dir, filename_with)

        imageio.v3.imwrite(path_no, frames_no_extrema, format="GIF", fps=fps)
        imageio.v3.imwrite(path_with, frames_with_extrema, format="GIF", fps=fps)

        return path_no, path_with

    def generate_gifs_N(self, show_EJ_EC=False, show_EL_EJ=False, log_x=False, log_y=False, duration_seconds=5):
        return self._generate_gifs_generic(self.N_maps, "N", show_EJ_EC, show_EL_EJ, log_x, log_y, duration_seconds)

    def generate_gifs_phi(self, show_EJ_EC=False, show_EL_EJ=False, log_x=False, log_y=False, duration_seconds=5):
        return self._generate_gifs_generic(self.phi_maps, "phi", show_EJ_EC, show_EL_EJ, log_x, log_y, duration_seconds)

    def generate_gifs_cos_phi(self, show_EJ_EC=False, show_EL_EJ=False, log_x=False, log_y=False, duration_seconds=5):
        return self._generate_gifs_generic(self.cos_phi_maps, "cos_phi", show_EJ_EC, show_EL_EJ, log_x, log_y, duration_seconds)