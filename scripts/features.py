import numpy as np
from scipy.signal import periodogram
from scipy.stats import entropy, kurtosis, skew

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
