import os
import numpy as np
import matplotlib.pyplot as plt


def plot_signal_and_modes(x, sfreq, modes, method, ch, output_dir, duration=None, max_points=1000):
    """
    Plot original signal, spectrum, and decomposed modes.
    """
    if duration is not None:
        n_samples_to_plot = int(sfreq * duration)
        x = x[:n_samples_to_plot]
        modes = modes[:, :n_samples_to_plot]
    else:
        n_samples_to_plot = len(x)

    step = max(1, int(n_samples_to_plot / max_points)) if max_points else 1
    x_ds = x[::step]
    modes_ds = modes[:, ::step]

    t = np.linspace(0, len(x_ds) / sfreq, len(x_ds))
    assert len(t) == len(x_ds), f"Mismatch in time ({len(t)}) and signal ({len(x_ds)}) length"

    method_fig_dir = os.path.join(output_dir, method, "figures")
    os.makedirs(method_fig_dir, exist_ok=True)

    Nmode = modes.shape[0]
    ncols = 2
    nrows = 2 + int(np.ceil(Nmode / ncols))

    plt.figure(figsize=(10, 2.5 * nrows))

    # Original Signal
    plt.subplot(nrows, ncols, 1)
    plt.plot(t, x_ds, color='k')
    plt.title(f'Original Signal - Ch{ch}')
    plt.xlabel('Time (s)')

    # Spectrum
    plt.subplot(nrows, ncols, 2)
    f_fft = np.fft.fft(x)
    f_fft = np.abs(f_fft[:len(f_fft)//2])
    f_freq = np.fft.fftfreq(len(x), 1/sfreq)[:len(x)//2]
    plt.plot(f_freq, f_fft, color='k')
    plt.title(f'Spectrum - Ch{ch}')
    plt.xlabel('Frequency (Hz)')

    # Modes
    for i in range(Nmode):
        plt.subplot(nrows, ncols, 3 + i)
        plt.plot(t, modes_ds[i], color='k')
        plt.title(f'{method} Mode {i}')
        plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(method_fig_dir, f'channel_{ch}_{method}_overview.png'))
    plt.close()


def plot_mvmd_grid(signal_data, mvmd_result, sfreq, output_path, downsample=True, max_points=5000):
    """
    Plot multivariate VMD results across channels and modes.
    """
    K, T, C = mvmd_result.shape

    if downsample and T > max_points:
        step = T // max_points
        signal_data_ds = signal_data[::step, :]
        mvmd_result_ds = mvmd_result[:, ::step, :]
        T_ds = signal_data_ds.shape[0]
        t = np.linspace(0, T / sfreq, T_ds)
    else:
        signal_data_ds = signal_data
        mvmd_result_ds = mvmd_result
        t = np.linspace(0, T / sfreq, T)

    fig, axs = plt.subplots(nrows=K + 1, ncols=C, figsize=(4 * C, 2.5 * (K + 1)), sharex=True)

    for ch in range(C):
        axs[0, ch].plot(t, signal_data_ds[:, ch], color='k')
        axs[0, ch].set_title(f"Original Ch{ch}")
        axs[0, ch].set_ylabel("Amplitude")

    for mode_idx in range(K):
        for ch in range(C):
            axs[mode_idx + 1, ch].plot(t, mvmd_result_ds[mode_idx, :, ch], color='k')
            axs[mode_idx + 1, ch].set_title(f"Mode {mode_idx} - Ch{ch}")
            axs[mode_idx + 1, ch].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
