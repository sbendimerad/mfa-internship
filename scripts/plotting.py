import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

def compute_spectrum(signal, sfreq, method='fft'):
    """
    Compute frequency and spectrum amplitude based on chosen method.
    """
    if method == 'fft':
        f = np.fft.fftfreq(len(signal), d=1/sfreq)[:len(signal)//2]
        S = np.abs(np.fft.fft(signal))[:len(signal)//2]
    elif method == 'psd':
        f, S = welch(signal, fs=sfreq, nperseg=1024)
    elif method == 'windowed_fft':
        window = windows.gaussian(len(signal), std=len(signal)/8)
        signal_win = signal * window
        f = np.fft.fftfreq(len(signal_win), d=1/sfreq)[:len(signal_win)//2]
        S = np.abs(np.fft.fft(signal_win))[:len(signal_win)//2]
    else:
        raise ValueError(f"Unknown spectrum method: {method}")
    return f, S

def plot_signal_and_modes(x, sfreq, modes, method, ch, output_dir, duration=None, max_points=1000, spectrum_method='fft'):
    """
    Plot original signal and PSD, then each mode and its PSD.
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
    
    t = np.arange(len(x_ds)) * step / sfreq
    assert len(t) == len(x_ds), f"Mismatch in time ({len(t)}) and signal ({len(x_ds)}) length"

    method_fig_dir = os.path.join(output_dir, method, "figures")
    os.makedirs(method_fig_dir, exist_ok=True)

    Nmode = modes.shape[0]
    ncols = 2
    nrows = 1 + Nmode  # 1 row for signal + 1 for each mode

    plt.figure(figsize=(10, 2.5 * nrows))

    # Plot Original Signal Time Series
    plt.subplot(nrows, ncols, 1)
    plt.plot(t, x_ds, color='k')
    plt.title(f'Original Signal - Ch{ch}')
    plt.xlabel('Time (s)')

    plt.subplot(nrows, ncols, 2)
    f_fft, S_fft = compute_spectrum(x, sfreq, spectrum_method)
    plt.loglog(f_fft, S_fft, color='k')
    plt.title(f'Spectrum ({spectrum_method}) - Ch{ch}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    # Plot each Mode + its Spectrum
    for i in range(Nmode):
        # Mode Time Series
        plt.subplot(nrows, ncols, 2*i + 3)  # row i+2 col 1
        plt.plot(t, modes_ds[i], color='k')
        plt.title(f'{method} Mode {i} - Time')
        plt.xlabel('Time (s)')

        # Mode Spectrum (using helper)
        plt.subplot(nrows, ncols, 2*i + 4)  # row i+2 col 2

        f_mode, S_mode = compute_spectrum(modes[i], sfreq, spectrum_method)
        plt.loglog(f_mode, S_mode, color='k')
        plt.title(f'{method} Mode {i} - Spectrum ({spectrum_method})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

    # Save figure
    plot_path = os.path.join(method_fig_dir, f"channel_{ch}_{method}_{spectrum_method}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
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