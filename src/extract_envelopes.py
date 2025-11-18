import os
import numpy as np
import pandas as pd
import pywt
import traceback


def extract_envelopes_modes(
    base_dir,
    method='VMD',
    sfreq=250,
    features_df_path=None,
    overwrite=False
):
    """
    Extracts and saves the envelope for each mode using CWT from Morlet wavelet,
    using flat folder structure (e.g. base_dir/modes/, base_dir/envelopes/).
    """
    import pywt
    import traceback

    # Load peak frequency info
    features_df = pd.read_csv(features_df_path) if isinstance(features_df_path, str) else features_df

    modes_dir = os.path.join(base_dir, "modes")
    envelopes_dir = os.path.join(base_dir, "envelopes")
    os.makedirs(envelopes_dir, exist_ok=True)

    all_channels = sorted(features_df["channel"].unique())

    for ch_idx in all_channels:
        try:
            name = f"meg_channel_{ch_idx}"
            modes_file = os.path.join(modes_dir, f"{name}_{method}_modes.npy")
            envelope_file = os.path.join(envelopes_dir, f"{name}_{method}_envelopes.npy")

            if not os.path.exists(modes_file):
                print(f"⚠️ Modes file not found: {modes_file}. Skipping.")
                continue

            if os.path.exists(envelope_file) and not overwrite:
                print(f"⏩ Envelope already exists for {name}. Skipping.")
                continue

            modes = np.load(modes_file)
            n_modes, n_samples = modes.shape

            ch_features = features_df[features_df["channel"] == ch_idx].reset_index(drop=True)
            if ch_features.shape[0] != n_modes:
                print(f"⚠️ Mismatch: Channel {ch_idx} has {n_modes} modes but {ch_features.shape[0]} rows in features_df. Skipping.")
                continue

            print(f"🔍 Processing {name} with {n_modes} modes...")

            envelopes = np.zeros_like(modes)
            wt = pywt.ContinuousWavelet('cmor5.0-1.0')

            for m in range(n_modes):
                signal = modes[m]
                peak_freq = ch_features.loc[m, 'Pfreq']
                scale = pywt.frequency2scale(wt, peak_freq) * sfreq
                coeffs, _ = pywt.cwt(signal, [scale], wt)
                envelopes[m] = np.abs(coeffs[0])

            np.save(envelope_file, envelopes)
            print(f"✅ Saved envelopes to {envelope_file}")

        except Exception as e:
            print(f"❌ Error processing channel {ch_idx}: {e}")
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


def compute_envelope_corr(envelopes):
    """Compute correlation matrix from list of 1D envelopes."""
    matrix = np.column_stack(envelopes)  # shape: (T, N)
    corr = np.corrcoef(matrix.T)         # shape: (N, N)
    return corr