import os
import numpy as np
import pandas as pd
import pywt

def extract_envelopes(base_dir, method='VMD', sfreq=250, features_df_path=None):
    """
    Extracts and saves the envelope for each mode using CWT from Morlet wavelet.
    
    Parameters:
    - base_dir: path to your base folder with meg_channel_ folders
    - method: decomposition method (e.g. 'VMD', 'EMD', etc.)
    - sfreq: sampling frequency
    - features_df_path: path to CSV file containing mode peak frequencies
    """
    # Load peak frequency info
    features_df = pd.read_csv(features_df_path) if isinstance(features_df_path, str) else features_df

    for ch_folder in os.listdir(base_dir):
        if not ch_folder.startswith("meg_channel_"):
            continue

        ch_path = os.path.join(base_dir, ch_folder, method)
        modes_file = os.path.join(ch_path, "modes", f"{ch_folder}_modes.npy")
        envelope_dir = os.path.join(ch_path, "envelopes")
        os.makedirs(envelope_dir, exist_ok=True)

        modes = np.load(modes_file)
        n_modes, n_samples = modes.shape

        # Filter the feature df for current channel
        ch_idx = int(ch_folder.replace("meg_channel_", ""))
        ch_features = features_df[features_df['channel'] == ch_idx].reset_index(drop=True)

        print(f"Processing channel {ch_idx} with {n_modes} modes...")

        envelopes = np.zeros_like(modes)
        wt = pywt.ContinuousWavelet('cmor5.0-1.0')

        for m in range(n_modes):
            signal = modes[m]
            peak_freq = ch_features.loc[m, 'Pfreq']
            scale = pywt.frequency2scale(wt, peak_freq) * sfreq
            coeffs, _ = pywt.cwt(signal, scale, wt)
            envelopes[m] = np.abs(coeffs[0])

        # Save envelopes
        save_path = os.path.join(envelope_dir, f"{ch_folder}_envelopes.npy")
        np.save(save_path, envelopes)
        print(f"Saved envelopes to {save_path}")
