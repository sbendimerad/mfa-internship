import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import seaborn as sns
from scipy.signal import welch, iirnotch, filtfilt
from scipy.cluster.hierarchy import linkage, leaves_list


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
    plot_path = os.path.join(method_fig_dir, f"channel_{ch}_{method}_{spectrum_method}_std.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_mvmd_grid(
    original,
    modes,
    omega,
    fs,
    selected_channels,
    max_points=1000,
    duration=None,
    standardize_time=True,
    standardize_psd=False,
    notch=True,
    eps=1e-10,
    log_psd=True,
    log_freq=True
):
    n_modes, n_samples, n_channels = modes.shape

    if duration:
        n_samples = min(n_samples, int(duration * fs))
        original = original[:, :n_samples]
        modes = modes[:, :n_samples, :]

    step = max(1, int(n_samples / max_points)) if max_points else 1
    t = np.arange(0, n_samples, step) / fs

    ncols = len(selected_channels) + 1  # +1 for PSD
    nrows = n_modes + 1  # +1 for original

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.2 * nrows), sharex=False)
    # Force shape (nrows, ncols) — always 2D array
    if nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    axes = np.array(axes).reshape(nrows, ncols)    
    mode_colors = sns.color_palette("husl", n_modes)

    # === COLUMN TITLES ===
    for j in range(len(selected_channels)):
        axes[0, j].set_title(f"Channel {j+1}", fontsize=11)
    axes[0, -1].set_title("Mean PSD", fontsize=11)

    # === ROW LABELS ===
    axes[0, 0].set_ylabel("Original", fontsize=11)
    for i in range(n_modes):
        axes[i + 1, 0].set_ylabel(f"Mode {i+1}", fontsize=11)

    # === ORIGINAL SIGNALS ===
    for j, ch in enumerate(selected_channels):
        x = original[ch]
        x_ds = x[::step]
        axes[0, j].plot(t, x_ds, color='black')
        axes[0, j].set_xlabel("Time (s)")
        axes[0, j].grid(True)
    axes[0, -1].axis('off')  # last col for original row = PSD placeholder

    # === MODES + PSD ===
    for i in range(n_modes):
        all_psd = []
        mode_color = mode_colors[i]

        for j, ch in enumerate(selected_channels):
            y = modes[i, :n_samples, ch]
            y_ds = y[::step]

            if standardize_time:
                y_ds = (y_ds - np.mean(y_ds)) / (np.std(y_ds) + eps)

            axes[i+1, j].plot(t, y_ds, color=mode_color)
            axes[i+1, j].set_xlabel("Time (s)")
            axes[i+1, j].grid(True)

            # === PSD Prep ===
            y_psd = y.copy()
            if notch:
                y_psd = double_notch(y_psd, fs)
            if standardize_psd:
                y_psd = (y_psd - np.mean(y_psd)) / (np.std(y_psd) + eps)

            f, psd = welch(y_psd, fs=fs, nperseg=fs * 2)
            mask = ~((f > 45) & (f < 55)) & ~((f > 95) & (f < 105))
            all_psd.append(psd[mask])

        # === PSD Plot ===
        all_psd = np.array(all_psd) + eps
        mean_psd = np.mean(all_psd, axis=0)
        std_psd = np.std(all_psd, axis=0)
        f_masked = f[mask]

        if log_psd:
            mean_val = 10 * np.log10(mean_psd)
            std_val = 10 * np.log10(mean_psd + std_psd) - mean_val
            ylabel = "Power (dB)"
        else:
            mean_val = mean_psd
            std_val = std_psd
            ylabel = "Power"
            mean_val = np.clip(mean_val, a_min=eps, a_max=None)

        ax = axes[i+1, -1]
        ax.plot(f_masked, mean_val, color=mode_color)
        ax.fill_between(f_masked, mean_val - std_val, mean_val + std_val, color=mode_color, alpha=0.3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        if log_freq:
            ax.set_xscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle("MVMD Decomposition per Channel + Mean PSD", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.3, hspace=0.5)  # tweak as needed
    plt.show()


def plot_mode_psds_multi_channel_modes_first(
    modes, 
    sfreq, 
    channel_indices=None, 
    standardize_time=False,
    standardize_psd=False,
    notch=True,
    log_scale=False,
    db_scale=True,     
    eps=1e-12
):
    """
    Plot PSDs (in dB or linear) for each mode across selected channels.

    Parameters:
    - modes: np.ndarray (n_modes, n_samples, n_channels)
    - sfreq: float, sampling frequency
    - channel_indices: list of channel indices (default: all)
    - standardize_time: z-score time series
    - standardize_psd: z-score PSD across channels
    - notch: mask 50/100Hz bands
    - log_scale: apply log scale to X-axis
    - db_scale: if True, use dB (10*log10). Else linear scale
    - eps: small number to avoid log(0)
    """
    n_modes, n_samples, n_channels = modes.shape

    if channel_indices is None:
        channel_indices = list(range(n_channels))

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4))  # No sharey
    if n_modes == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for mode_idx in range(n_modes):
        ax = axes[mode_idx]
        psds_list = []

        for i, ch in enumerate(channel_indices):
            signal = modes[mode_idx, :, ch]

            if standardize_time:
                signal = (signal - np.mean(signal)) / (np.std(signal) + eps)

            freqs, psd = welch(signal, fs=sfreq, nperseg=min(1024, n_samples))

            if notch:
                mask = ~((freqs > 45) & (freqs < 55)) & ~((freqs > 95) & (freqs < 105))
                freqs = freqs[mask]
                psd = psd[mask]

            if db_scale:
                psd_to_plot = 10 * np.log10(psd + eps)
            else:
                psd_to_plot = psd

            psds_list.append(psd_to_plot)

            ch_color = colors[i % len(colors)]
            ax.plot(freqs, psd_to_plot, color=ch_color, alpha=1.0, label=f"Ch {ch}")

        psds_array = np.array(psds_list)

        if standardize_psd:
            psds_array = (psds_array - np.mean(psds_array, axis=0)) / (np.std(psds_array, axis=0) + eps)

        ax.set_title(f"Mode {mode_idx + 1}")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both", ls="--", alpha=0.5)

        if log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))

        if mode_idx == 0:
            ax.set_ylabel("Power (dB)" if db_scale else "Power (Linear)")

    # Legend
    handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], label=f"Ch {ch}") 
               for i, ch in enumerate(channel_indices)]

    fig.legend(handles=handles, loc='lower center', ncol=len(channel_indices), title="Channels", frameon=False)

    ch_str = ", ".join(str(ch) for ch in channel_indices)
    fig.suptitle(f"PSD per Mode — Channels {ch_str}", fontsize=15, weight='bold')

    plt.tight_layout(rect=[0, 0.12, 1, 0.92])
    plt.show()


def mvmd_plot_correlation_matrix_enveloppe(corr, mode_idx, figsize=(8, 6), cmap="coolwarm", vmin=-1, vmax=1):
    """
    Plot a correlation matrix using seaborn heatmap.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix (channels x channels)
    mode_idx : int
        Mode index (for title)
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    vmin : float
        Minimum value for color scale
    vmax : float
        Maximum value for color scale
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap=cmap, center=0, vmin=vmin, vmax=vmax,
                square=True, xticklabels=False, yticklabels=False)
    plt.title(f"Envelope Correlation Matrix - Mode {mode_idx}")
    plt.tight_layout()
    plt.show()


    """
    Plot reordered correlation matrices in a 1-row grid.
    """
    n_modes = len(mode_indices)
    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4))

    if n_modes == 1:
        axes = [axes]  # Ensure iterable

    for i, mode_idx in enumerate(mode_indices):
        corr = corr_dict[mode_idx]
        reordered = reorder_matrix_hierarchical(corr)

        ax = axes[i]
        sns.heatmap(
            reordered,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False,
            cbar=(i == n_modes - 1),  # Show colorbar only once
            square=True,
        )
        ax.set_title(f"Mode {mode_idx}", fontsize=14)

    plt.suptitle("Envelope Correlation Matrices", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

def mvmd_plot_correlation_matrix_enveloppe_reordered(corr_dict, mode_indices=[0, 1, 2, 3, 4], cmap="coolwarm", vmin=-1, vmax=1):
    """
    Plot reordered correlation matrices in a 1-row grid.
    """
    n_modes = len(mode_indices)
    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4))

    if n_modes == 1:
        axes = [axes]  # Ensure iterable

    for i, mode_idx in enumerate(mode_indices):
        corr = corr_dict[mode_idx]
        reordered = mvmd_reorder_matrix_hierarchical(corr)

        ax = axes[i]
        sns.heatmap(
            reordered,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False,
            cbar=(i == n_modes - 1),  # Show colorbar only once
            square=True,
        )
        ax.set_title(f"Mode {mode_idx}", fontsize=14)

    plt.suptitle("Envelope Correlation Matrices", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


def mvmd_reorder_matrix_hierarchical(matrix):
    """
    Reorder a correlation matrix based on hierarchical clustering.
    """
    linkage_result = linkage(matrix, method='ward')
    order = leaves_list(linkage_result)
    reordered = matrix[np.ix_(order, order)]
    return reordered

def mvmd_plot_mean_mode_psds(
    modes,                  # ndarray, shape (n_modes, n_samples, n_channels)
    sfreq,                  # float, sampling frequency in Hz
    omega=None,             # optional ndarray of mode peak frequencies, shape (1, n_modes)
    nperseg=1024,           # int, window length for Welch PSD
    standardize=False,      # bool, whether to z-score the signal before PSD
    notch=True,             # bool, whether to apply 50 Hz and 100 Hz notch filters
    scale='db',             # str, 'db' for decibel scale or 'linear' for raw power
    layout='horizontal',    # str, 'horizontal' or 'vertical' layout of subplots
    suptitle="Mean PSD per Mode"  # str, title for the entire figure
):
    """
    Plot mean PSD for each mode across all channels.

    Parameters
    ----------
    modes : ndarray of shape (n_modes, n_samples, n_channels)
        The decomposed modes to analyze.

    sfreq : float
        Sampling frequency of the signal in Hz.

    omega : ndarray of shape (1, n_modes), optional
        Peak frequency of each mode, used for display in plot titles.

    nperseg : int, default=1024
        Segment length used in Welch's method for PSD estimation.

    standardize : bool, default=False
        If True, apply z-score normalization to each mode before PSD.

    notch : bool, default=True
        If True, apply notch filters at 50 Hz and 100 Hz.

    scale : {'db', 'linear'}, default='db'
        Whether to plot the power in decibels or linear scale.

    layout : {'horizontal', 'vertical'}, default='horizontal'
        Layout of subplots — one row or one column.

    suptitle : str, default="Mean PSD per Mode"
        Title for the entire figure.

    Returns
    -------
    None
        Displays a matplotlib figure with the PSDs.
    """
    n_modes, _, n_channels = modes.shape
    eps = 1e-30  # internal stability constant

    if layout == 'vertical':
        fig, axes = plt.subplots(n_modes, 1, figsize=(6, 4 * n_modes), sharex=True)
    else:
        fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4))  # no sharey to allow different y-scales

    if n_modes == 1:
        axes = [axes]  # ensure iterable

    for i in range(n_modes):
        ax = axes[i]
        all_psd = []

        for ch in range(n_channels):
            signal = modes[i, :, ch]

            # === Preprocessing ===
            if notch:
                signal = double_notch(signal, sfreq)
            if standardize:
                signal = (signal - np.mean(signal)) / (np.std(signal) + eps)

            # === Compute PSD ===
            freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)

            # Remove 50 Hz and 100 Hz noise
            mask = ~((freqs > 45) & (freqs < 55)) & ~((freqs > 95) & (freqs < 105))
            psd_masked = psd[mask]
            all_psd.append(psd_masked)

        all_psd = np.array(all_psd) + eps
        mean_psd = np.mean(all_psd, axis=0)
        std_psd = np.std(all_psd, axis=0)
        freqs_masked = freqs[mask]

        # === Choose scale ===
        if scale == 'db':
            mean_y = 10 * np.log10(mean_psd)
            std_y = 10 * np.log10(mean_psd + std_psd) - mean_y
            ylabel = "Power (dB)"
        else:
            mean_y = mean_psd
            std_y = std_psd
            ylabel = "Power (linear)"

        # === Plot ===
        ax.plot(freqs_masked, mean_y, color="blue", label="Mean PSD")
        ax.fill_between(freqs_masked, mean_y - std_y, mean_y + std_y, color="skyblue", alpha=0.4)

        # === Title per mode ===
        if omega is not None:
            peak_freq = omega[0, i].real
            ax.set_title(f"Mode {i} — Peak @ {peak_freq:.1f} Hz")
        else:
            ax.set_title(f"Mode {i}")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_xscale("log")
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.set_ylabel(ylabel)

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
