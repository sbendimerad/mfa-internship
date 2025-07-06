import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, iirnotch, filtfilt

def double_notch(signal, fs, freqs=[50, 100], Q=30):
    for f in freqs:
        b, a = iirnotch(f, Q, fs)
        signal = filtfilt(b, a, signal)
    return signal


def compute_spectrum(signal, sfreq, method='psd', return_db=True, notch=True):
    """
    Compute power spectrum of the signal using FFT or Welch.
    Optionally applies notch filters at 50 Hz and 100 Hz.
    """
    # Optional: apply notch filters to clean the signal
    if notch:
        signal = double_notch(signal, fs=sfreq, freqs=[50, 100], Q=30)

    # === Spectrum estimation ===
    if method == 'fft':
        f = np.fft.fftfreq(len(signal), d=1/sfreq)[:len(signal)//2]
        S = np.abs(np.fft.fft(signal))[:len(signal)//2] ** 2
    elif method == 'psd':
        f, S = welch(signal, fs=sfreq, nperseg=1024)


    # === Convert to dB ===
    if return_db:
        S = 10 * np.log10(S + 1e-30)

    return f, S


def plot_signal_and_modes(x, sfreq, modes, method, ch, output_dir, duration=None, max_points=1000, spectrum_method='psd'):
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
    eps = 1e-30  # To avoid log(0)

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

    # === Original Signal - Spectrum (Notched + Masked) ===
    plt.subplot(nrows, ncols, 2)
    x_clean = double_notch(x, sfreq)
    freqs_x, psd_x = welch(x_clean, fs=sfreq, nperseg=1024)
    mask = ~((freqs_x > 45) & (freqs_x < 55)) & ~((freqs_x > 95) & (freqs_x < 105))
    freqs_masked = freqs_x[mask]
    psd_masked = psd_x[mask]
    psd_db = 10 * np.log10(psd_masked + eps)
    plt.semilogx(freqs_masked, psd_db, color='k')
    plt.title(f'Spectrum ({spectrum_method}) - Ch{ch}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')

    # Plot each Mode + its Spectrum
    for i in range(Nmode):
        # --- Time Domain Plot ---
        plt.subplot(nrows, ncols, 2*i + 3)
        plt.plot(t, modes_ds[i], color='k')
        plt.title(f'{method} Mode {i} - Time')
        plt.xlabel('Time (s)')

        # --- Frequency Domain Plot (Notch + Mask) ---
        plt.subplot(nrows, ncols, 2*i + 4)

        # Apply notch filter at 50 Hz and 100 Hz
        cleaned_mode = double_notch(modes[i], fs=sfreq, freqs=[50, 100], Q=30)

        # Compute Welch PSD
        freqs, psd = welch(cleaned_mode, fs=sfreq, nperseg=1024)

        # Mask out 50±5 Hz and 100±5 Hz
        mask = ~((freqs > 45) & (freqs < 55)) & ~((freqs > 95) & (freqs < 105))
        freqs_masked = freqs[mask]
        psd_masked = psd[mask]

        # Convert to dB
        eps = 1e-30
        psd_db = 10 * np.log10(psd_masked + eps)

        # Plot the clean, masked spectrum
        plt.semilogx(freqs_masked, psd_db, color='k')
        plt.title(f'{method} Mode {i} - Spectrum (masked)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')

        # === Save the figure ===
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


    