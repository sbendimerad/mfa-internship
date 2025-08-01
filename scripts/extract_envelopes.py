import os
import numpy as np
import pandas as pd
import seaborn as sns
import pywt
import traceback


def extract_envelopes_modes(base_dir, method='VMD', sfreq=250, features_df_path=None, overwrite=False):
    """
    Extracts and saves the envelope for each mode using CWT from Morlet wavelet.

    Parameters:
    - base_dir: path to your base folder with meg_channel_ folders
    - method: decomposition method (e.g. 'VMD', 'EMD', etc.)
    - sfreq: sampling frequency
    - features_df_path: path to CSV file or DataFrame containing mode peak frequencies
    - overwrite: if False, will skip channels that already have envelope files
    """
    # Load peak frequency info
    features_df = pd.read_csv(features_df_path) if isinstance(features_df_path, str) else features_df

    # Sort folder names numerically
    all_channels = sorted(
        [f for f in os.listdir(base_dir) if f.startswith("meg_channel_")],
        key=lambda x: int(x.replace("meg_channel_", ""))
    )

    for ch_folder in all_channels:
        try:
            ch_idx = int(ch_folder.replace("meg_channel_", ""))
            ch_path = os.path.join(base_dir, ch_folder, method)
            modes_file = os.path.join(ch_path, "modes", f"{ch_folder}_modes.npy")
            envelope_dir = os.path.join(ch_path, "envelopes")
            os.makedirs(envelope_dir, exist_ok=True)
            save_path = os.path.join(envelope_dir, f"{ch_folder}_envelopes.npy")

            if not os.path.exists(modes_file):
                print(f"⚠️ Modes file not found: {modes_file}. Skipping.")
                continue

            if os.path.exists(save_path) and not overwrite:
                print(f"⏩ Envelope already exists for channel {ch_idx}. Skipping.")
                continue

            modes = np.load(modes_file)
            n_modes, n_samples = modes.shape

            ch_features = features_df[features_df['channel'] == ch_idx].reset_index(drop=True)
            if ch_features.shape[0] != n_modes:
                print(f"⚠️ Mismatch: Channel {ch_idx} has {n_modes} modes but {ch_features.shape[0]} rows in features_df. Skipping.")
                continue

            print(f"🔍 Processing channel {ch_idx} with {n_modes} modes...")

            envelopes = np.zeros_like(modes)
            wt = pywt.ContinuousWavelet('cmor5.0-1.0')

            for m in range(n_modes):
                signal = modes[m]
                peak_freq = ch_features.loc[m, 'Pfreq']
                # sampling_period = 1.0 / sfreq
                scale = pywt.frequency2scale(wt, peak_freq) * sfreq
                #scale = pywt.frequency2scale(wt, peak_freq, sampling_period=sampling_period)
                coeffs, _ = pywt.cwt(signal, scale, wt)
                envelopes[m] = np.abs(coeffs[0])

            np.save(save_path, envelopes)
            print(f"✅ Saved envelopes to {save_path}")

        except Exception as e:
            print(f"❌ Error processing channel {ch_folder}: {e}")
            traceback.print_exc()


def extract_mvmd_envelopes(u, omega, save_dir, sfreq=250, wavelet_name='cmor5.0-1.0'):
    """
    Extracts and saves envelopes for MVMD modes using CWT based on omega peak frequencies.

    Parameters:
    - u: np.ndarray of shape (n_modes, n_samples, n_channels)
    - omega: torch.Tensor or np.ndarray of shape (1, n_modes), already multiplied by fs
    - save_dir: directory to save each channel's envelopes
    - sfreq: sampling frequency in Hz
    - wavelet_name: name of the complex Morlet wavelet for CWT
    """

    os.makedirs(save_dir, exist_ok=True)

    n_modes, n_samples, n_channels = u.shape

    # Ensure omega is NumPy and real-valued
    if hasattr(omega, 'detach'):
        omega = omega.detach().cpu().numpy()
    if np.iscomplexobj(omega):
        omega = omega.real
    omega = omega.squeeze()  # Shape: (n_modes,)
    
    print(f"✅ Omega (Hz): {omega} | Shape: {omega.shape}")

    # Define wavelet
    wavelet = pywt.ContinuousWavelet(wavelet_name)

    for ch in range(n_channels):
        try:
            print(f"🔍 Processing channel {ch}...")

            envelopes = np.zeros((n_modes, n_samples))

            for m in range(n_modes):
                signal = u[m, :, ch]
                peak_freq = omega[m]

                # Compute scale
                scale = pywt.frequency2scale(wavelet, peak_freq) * sfreq

                # Perform CWT
                coeffs, _ = pywt.cwt(signal, [scale], wavelet)
                envelopes[m] = np.abs(coeffs[0])

            # Save
            save_path = os.path.join(save_dir, f"meg_channel_{ch}_envelopes.npy")
            np.save(save_path, envelopes)
            print(f"✅ Saved envelopes for channel {ch} to {save_path}")

        except Exception as e:
            print(f"❌ Error processing channel {ch}: {e}")
            traceback.print_exc()
