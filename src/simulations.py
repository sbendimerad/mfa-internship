import os
import numpy as np
import pandas as pd
from pymultifracs.simul import mrw_cumul, fbm
import mne
from sktime.libs.vmdpy import VMD
from sklearn.metrics import mean_absolute_error
from scipy.signal import periodogram , hilbert, welch
from scipy.stats import entropy, kurtosis, skew 


def generate_synthetic_neural_signal(fs=250, n_samples=82500, frequencies=[6, 10, 30, 80], 
                                     c1=0.0, c2=0.0, H=0.98,
                                     add_fbm_noise=False, add_powerline=False):
    """
    Generate synthetic signal with optional MRW modulation and FBM noise.
    """
    t = np.arange(n_samples) / fs
    L = n_samples + 1  # For MRW

    modulated_components = []

    for f in frequencies:
        # Modulator
        if c1 != 0.0 or c2 != 0.0:
            mod = mrw_cumul(L, c1, c2, L).flatten()
            mod = np.diff(mod)
            mod = mne.filter.filter_data(mod, fs, f / 4, None)
            mod = np.abs(mod)
        else:
            mod = np.ones(n_samples)  # No modulation

        # Oscillation
        osc = np.sin(2 * np.pi * f * t)
        modulated = mod * osc
        modulated_components.append(modulated)

    signal = np.sum(modulated_components, axis=0)

    # FBM noise
    if add_fbm_noise:
        fbm_raw = fbm(shape=(n_samples + 1, 1), H=H)
        fbm_noise = np.diff(fbm_raw.squeeze())
        fbm_noise = fbm_noise / np.std(fbm_noise) * np.std(signal)
        signal += fbm_noise

    # Powerline (optional)
    if add_powerline:
        signal += 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)

    return {
        'time': t,
        'final_signal': signal,
        'modulator': mod
    }

def summarize_decomposition_results(signal_list, signal_names, sfreq,
                                     base_dir='results/synthetic',
                                     methods_to_check=None,
                                     n_modes_to_use=None):
    """
    Load modes from disk, extract features, and compute reconstruction errors.

    n_modes_to_use: int or None
        Number of modes to consider for reconstruction and feature extraction.
        If None, use all modes.
    """
    if methods_to_check is None:
        methods_to_check = ["EMD", "VMD", "VMDtransformer"]

    records = []

    for method in methods_to_check:
        for signal_name, original_signal in zip(signal_names, signal_list):
            mode_file = os.path.join(base_dir, signal_name, method, "modes", f"{signal_name}_modes.npy")
            if not os.path.exists(mode_file):
                continue

            modes = np.load(mode_file)
            n_modes_extracted = modes.shape[0]

            # Use n_modes_to_use or all modes if None
            modes_for_error = modes[:n_modes_to_use] if n_modes_to_use else modes

            mse_full = compute_reconstruction_error(original_signal, modes)
            mse_first = compute_reconstruction_error(original_signal, modes_for_error)

            # Limit modes to first n_modes_to_use for feature extraction as well
            modes_for_features = modes[:n_modes_to_use] if n_modes_to_use else modes

            for idx, mode in enumerate(modes_for_features):
                feats, labels = extract_mode_features(mode, sfreq)
                feat_dict = dict(zip(labels, feats))

                record = {
                    "Method": method,
                    "Signal": signal_name,
                    "Mode Index": idx,
                    "Peak Frequency (Hz)": feat_dict.get("Ppeak", np.nan),
                    "Reconstruction MSE (Full)": mse_full,
                    "Reconstruction MSE (First N Modes)": mse_first,
                    "Number of Extracted Modes": n_modes_extracted,
                    # Additional features
                    "AM": feat_dict.get("AM", np.nan),
                    "BM": feat_dict.get("BM", np.nan),
                    "Entropy": feat_dict.get("ent", np.nan),
                    "Centroid": feat_dict.get("Cent", np.nan),
                    "Power Spectrum Sum": feat_dict.get("Pfreq", np.nan),
                    "Skewness": feat_dict.get("skew", np.nan),
                    "Kurtosis": feat_dict.get("kurt", np.nan),
                    "Hjorth Mobility": feat_dict.get("Hmob", np.nan),
                    "Hjorth Complexity": feat_dict.get("Hcomp", np.nan)
                }
                records.append(record)

    return pd.DataFrame(records)


    K = len(modes)
    soi = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            soi[i, j] = spectral_overlap_index(modes[i], modes[j], fs)
    mean_soi = np.mean(soi[np.triu_indices(K, k=1)])
    return soi, mean_soi

def peak_frequency(mode, fs):
    f, Pxx = periodogram(mode, fs=fs)
    if len(Pxx) == 0:
        return 0
    return f[np.argmax(Pxx)]

def test_vmd_params(signal, fs, param_grid, expected_freqs, tolerance=2.0):
    records = []
    spectra = {}
    for params in param_grid:
        Nm = params['Nmodes']
        alpha = params['alpha']
        tau = params['tau']
        DC = params['DC']
        init = params['init']
        tol = params['tol']
        IMFs, _, _ = VMD(signal, alpha, tau, Nm, DC, init, tol)
        reconstructed = np.sum(IMFs, axis=0)
        mae = mean_absolute_error(signal, reconstructed)
        _, mean_soi = compute_soi_matrix(IMFs, fs)
        peak_freqs = [peak_frequency(mode, fs) for mode in IMFs]
        all_covered = all(any(abs(pf - ef) <= tolerance for pf in peak_freqs) for ef in expected_freqs)
        record = {
            'Nmodes': Nm, 'alpha': alpha, 'tau': tau, 'DC': DC, 'init': init, 'tol': tol,
            'peak_frequencies': ', '.join(f"{pf:.2f}" for pf in peak_freqs),
            'all_covered': all_covered, 'mae': mae, 'mean_soi': mean_soi
        }
        records.append(record)
        key = (Nm, alpha, tau, DC, init, tol)
        spectra[key] = IMFs

    df = pd.DataFrame(records)
    return df, spectra

def extract_mode_features_01(mode, Fs):
    mode = np.asarray(mode).flatten()
    hilb = hilbert(mode)
    A = np.abs(hilb)
    A = A[150:-150]  # remove borders
    phase = np.unwrap(np.angle(hilb[150:-150]))
    inst_freq = np.diff(phase) * Fs / (2 * np.pi)
    E = np.sum(np.abs(hilb[150:-150]) ** 2) / len(A)
    CW = np.sum(np.diff(phase) * Fs * A[:-1]**2) / (2 * np.pi * E)
    AM = np.sqrt(np.sum((np.diff(A) * Fs) ** 2)) / E
    BM = np.sqrt(np.sum(((inst_freq - CW) ** 2) * A[:-1]**2) / E)

    # Welch PSD
    f, Pxx = welch(mode, fs=Fs, nperseg=1024, noverlap=int(0.85*1024))
    Pxx /= np.sum(Pxx)
    ent = -np.sum(Pxx * np.log2(Pxx + 1e-12))  # spectral entropy
    Spow = np.mean(Pxx**2)
    Cent = np.sum(f * Pxx)
    Ppeak = np.max(Pxx)
    Pxx_norm = Pxx / np.sum(Pxx)
    Pfreq = f[np.argmax(Pxx_norm)]

    skew_val = skew(mode)
    kurt_val = kurtosis(mode)

    dx = np.diff(mode)
    ddx = np.diff(dx)
    Hmob = np.sqrt(np.var(dx) / np.var(mode)) if np.var(mode) > 0 else 0.0
    Hcomp = (np.sqrt(np.var(ddx) / np.var(dx)) / Hmob) if Hmob > 0 else 0.0

    features = [AM, BM, ent, Spow, Cent, Ppeak, Pfreq, skew_val, kurt_val, Hmob, Hcomp]
    labels = ["AM", "BM", "ent", "pow", "Cent", "Ppeak", "Pfreq", "skew", "kurt", "Hmob", "Hcomp"]

    return features, labels

def extract_mode_features(mode, Fs):
    """Compute features from a single mode."""
    features, labels = [], []
    mode = np.asarray(mode).flatten()

    features.append(np.mean(np.abs(mode)))
    labels.append("AM")

    features.append(np.sum(mode ** 2))
    labels.append("BM")

    hist, _ = np.histogram(mode, bins=100, density=True)
    hist += 1e-12
    features.append(entropy(hist))
    labels.append("ent")

    f, Pxx = periodogram(mode, fs=Fs)
    Pxx, f = np.nan_to_num(Pxx), np.nan_to_num(f)

    centroid = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0.0
    features.append(centroid)
    labels.append("Cent")

    features.append(np.sum(Pxx))
    labels.append("Pfreq")

    features.append(f[np.argmax(Pxx)] if len(Pxx) > 0 else 0.0)
    labels.append("Ppeak")

    features.append(skew(mode))
    labels.append("skew")

    features.append(kurtosis(mode))
    labels.append("kurt")

    dx = np.diff(mode)
    var_mode, var_dx = np.var(mode), np.var(dx)
    Hmob = np.sqrt(var_dx / var_mode) if var_mode > 0 else 0.0
    features.append(Hmob)
    labels.append("Hmob")

    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    Hcomp = np.sqrt(var_ddx / var_dx) / Hmob if var_dx > 0 and Hmob > 0 else 0.0
    features.append(Hcomp)
    labels.append("Hcomp")

    return features, labels

def compute_features_from_modes_and_save(modes_path, output_features_path, Fs):
    """
    Load saved modes from a .npy file,
    compute features for each mode,
    save all features as CSV.

    Parameters:
        modes_path (str): Path to .npy file with modes (shape: n_modes x n_samples).
        output_features_path (str): Path to save features CSV.
        Fs (float): Sampling frequency.

    Returns:
        None
    """
    modes = np.load(modes_path)
    all_features = []
    all_labels = []

    for idx, mode in enumerate(modes):
        feats, labels = extract_mode_features_01(mode, Fs)
        all_features.extend(feats)
        all_labels.extend([f"{label}{idx}" for label in labels])

    df = pd.DataFrame([all_features], columns=all_labels)
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    df.to_csv(output_features_path, index=False)
    print(f"Features saved to {output_features_path}")
