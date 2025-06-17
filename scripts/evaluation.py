import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from features import extract_mode_features


def compute_reconstruction_error(original_signal, modes, n_modes_to_use=None):
    """
    Compute normalized MSE between the original signal and reconstructed signal from modes.
    """
    modes_to_sum = modes[:n_modes_to_use] if n_modes_to_use else modes
    reconstructed = np.sum(modes_to_sum, axis=0)
    mse = np.mean((original_signal - reconstructed) ** 2)
    var = np.var(original_signal)
    return mse / var if var > 0 else np.nan



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
