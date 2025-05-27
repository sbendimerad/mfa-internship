import time 
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import periodogram
from scipy.stats import entropy, kurtosis, skew
from PyEMD import EMD, EEMD, CEEMDAN
import ewtpy
from sktime.libs.vmdpy import VMD
from sktime.transformations.series.vmd import VmdTransformer
from pandas import Series


def extract_mode_features(mode, Fs):
    """Compute features from a single mode."""
    features = []
    labels = []

    mode = np.asarray(mode).flatten()

    # AM (mean absolute value)
    AM = np.mean(np.abs(mode))
    features.append(AM)
    labels.append("AM")

    # BM (band energy)
    BM = np.sum(mode ** 2)
    features.append(BM)
    labels.append("BM")

    # Shannon entropy
    hist, _ = np.histogram(mode, bins=100, density=True)
    hist += 1e-12  # Avoid log(0)
    ent_val = entropy(hist)
    features.append(ent_val)
    labels.append("ent")

    # Spectral features
    f, Pxx = periodogram(mode, fs=Fs)
    Pxx = np.nan_to_num(Pxx)
    f = np.nan_to_num(f)

    # Spectral centroid
    centroid = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0.0
    features.append(centroid)
    labels.append("Cent")

    # Spectral power
    power = np.sum(Pxx)
    features.append(power)
    labels.append("Pfreq")

    # Peak frequency
    peak_freq = f[np.argmax(Pxx)] if len(Pxx) > 0 else 0.0
    features.append(peak_freq)
    labels.append("Ppeak")

    # Skewness
    skewness = skew(mode)
    features.append(skewness)
    labels.append("skew")

    # Kurtosis
    kurt = kurtosis(mode)
    features.append(kurt)
    labels.append("kurt")

    # Hjorth mobility
    dx = np.diff(mode)
    var_mode = np.var(mode)
    var_dx = np.var(dx)
    Hmob = np.sqrt(var_dx / var_mode) if var_mode > 0 else 0.0
    features.append(Hmob)
    labels.append("Hmob")

    # Hjorth complexity
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    Hcomp = np.sqrt(var_ddx / var_dx) / Hmob if var_dx > 0 and Hmob > 0 else 0.0
    features.append(Hcomp)
    labels.append("Hcomp")

    return features, labels

def plot_signal_and_modes(x, sfreq, modes, method, ch, output_dir, duration=None, max_points=None):
    """Plot original signal, spectrum, and decomposed modes with optional duration and point limitation."""

    # Handle duration limitation
    if duration is not None:
        n_samples_to_plot = int(sfreq * duration)
        x = x[:n_samples_to_plot]
        modes = modes[:, :n_samples_to_plot]
    else:
        n_samples_to_plot = len(x)

    # Handle downsampling
    step = max(1, int(n_samples_to_plot / max_points)) if max_points else 1
    x_ds = x[::step]
    modes_ds = modes[:, ::step]

    # Fix time vector length to match downsampled signal
    t = np.linspace(0, len(x_ds) / sfreq, len(x_ds))

    # Safety check
    assert len(t) == len(x_ds), f"Mismatched time ({len(t)}) and signal ({len(x_ds)}) lengths"

    # Prepare saving directory
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

    # Spectrum (no downsampling)
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

    K, T, C = mvmd_result.shape

    # Downsample if needed
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

    # Plotting
    fig, axs = plt.subplots(nrows=K + 1, ncols=C, figsize=(4 * C, 2.5 * (K + 1)), sharex=True)

    # Top row: original signals
    for ch in range(C):
        axs[0, ch].plot(t, signal_data_ds[:, ch], color='k')
        axs[0, ch].set_title(f"Original Ch{ch}")
        axs[0, ch].set_ylabel("Amplitude")

    # MVMD modes
    for mode_idx in range(K):
        for ch in range(C):
            axs[mode_idx + 1, ch].plot(t, mvmd_result_ds[mode_idx, :, ch], color='k')
            axs[mode_idx + 1, ch].set_title(f"Mode {mode_idx} - Ch{ch}")
            axs[mode_idx + 1, ch].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_all_decompositions(signal_in, Fs, Nmodes,
                           methods=None,
                           eemd_trials=100,
                           vmd_alpha=200,
                           vmd_tau=0,
                           vmd_DC=0,
                           vmd_init=1,
                           vmd_tol=1e-7,
                           return_modes=False):
    """
    Run multiple decomposition methods on a signal with optional parameterization.

    Parameters:
    - signal_in: Input 1D numpy array.
    - Fs: Sampling frequency in Hz.
    - Nmodes: Number of modes to extract.
    - methods: List of methods to apply.
    - eemd_trials: Number of EEMD trials.
    - vmd_alpha, vmd_tau, vmd_DC, vmd_init, vmd_tol: VMD parameters.
    - return_modes: If True, returns the decomposed modes.

    Returns:
    - results: Dictionary of extracted features.
    - modes_out: Dictionary of modes (if return_modes=True).
    """
    if methods is None:
        methods = ['EMD', 'EEMD', 'CEEMDAN', 'EWT', 'VMD', 'MVMD']

    results = {}
    modes_out = {}

    for method in methods:
        try:
            start_time = time.time()

            if method == 'EMD':
                emd = EMD()
                IMFs = emd.emd(signal_in,max_imf=8)

            elif method == 'EEMD':
                eemd = EEMD()
                eemd.trials = eemd_trials
                IMFs = eemd(signal_in, max_imf=Nmodes)

            elif method == 'CEEMDAN':
                ceemdan = CEEMDAN()
                IMFs = ceemdan(signal_in, max_imf=Nmodes)

            elif method == 'EWT':
                ewt, _, _ = ewtpy.EWT1D(signal_in, N=Nmodes)
                IMFs = ewt.T

            elif method == 'VMD':
                IMFs, _, _ = VMD(signal_in,
                                 vmd_alpha,
                                 vmd_tau,
                                 Nmodes,
                                 vmd_DC,
                                 vmd_init,
                                 vmd_tol)

            elif method == 'VMDtransformer':
                vmd_transformer = VmdTransformer(K=None, kMax=8, energy_loss_coefficient=0.0001)
                IMFs = vmd_transformer.fit_transform(Series(signal_in)).to_numpy().T
                if IMFs.ndim == 1:
                    IMFs = IMFs.reshape(1, -1)

            else:
                continue

            exec_time = time.time() - start_time
            modes_out[method] = IMFs

            all_feats = []
            all_labels = []

            for idx, mode in enumerate(IMFs):
                mode = np.ravel(mode)
                feats, labels = extract_mode_features(mode, Fs)
                all_feats.extend(feats)
                all_labels.extend([f"{label}{idx}" for label in labels])

            all_feats.append(exec_time)
            all_labels.append("decTime")
            all_feats.append(len(IMFs))
            all_labels.append("n_modes")

            results[method] = {"labels": all_labels, "values": all_feats}

        except Exception as e:
            print(f"❌ Error processing {method}: {e}")
            continue

    return (results, modes_out) if return_modes else results
