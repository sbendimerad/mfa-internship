import os
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import entropy, kurtosis, skew 
from scipy.signal import hilbert, welch


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
    Pfreq = f[np.argmax(Pxx)]

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
