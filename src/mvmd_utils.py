import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import iirnotch, filtfilt



# === Notch filter ===
def double_notch(signal, fs):
    """Apply notch filters at 50 Hz and 100 Hz."""
    # First notch at 50 Hz
    b1, a1 = iirnotch(w0=50, Q=30, fs=fs)
    signal = filtfilt(b1, a1, signal)

    # Second notch at 100 Hz
    b2, a2 = iirnotch(w0=100, Q=30, fs=fs)
    signal = filtfilt(b2, a2, signal)

    return signal


def compute_mode_corr_matrix_across_channels(u):
    """
    Compute the average correlation matrix between modes across all channels.

    Parameters:
    - u: ndarray of shape (n_modes, n_samples, n_channels)

    Returns:
    - corr_matrix: average correlation matrix of shape (n_modes, n_modes)
    """
    n_modes, n_samples, n_channels = u.shape
    corr_matrices = []

    for ch in range(n_channels):
        data = u[:, :, ch]  # shape: (n_modes, n_samples)
        corr = np.corrcoef(data)
        corr_matrices.append(corr)

    # Average across channels
    corr_matrix = np.mean(corr_matrices, axis=0)
    return corr_matrix


def plot_mode_corr_matrix(
    corr_matrix,
    title="Average Mode Correlation Across Channels",
    cmap="coolwarm",
    figsize=(7, 6),
    mode_names=None,
    fontsize=12,
):
    """
    Plot a clean Seaborn heatmap of the mode correlation matrix.

    Parameters:
    - corr_matrix: square (n_modes x n_modes) matrix
    - title: title of the plot
    - cmap: color palette
    - figsize: figure size
    - mode_names: optional list of labels (e.g., ["Delta", "Theta", ...])
    - fontsize: font size for labels and annotations
    """
    n_modes = corr_matrix.shape[0]
    if mode_names is None:
        mode_names = [f"Mode {i+1}" for i in range(n_modes)]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        xticklabels=mode_names,
        yticklabels=mode_names,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"label": "Correlation"},
        vmin=0, vmax=1
    )

    ax.set_title(title, fontsize=fontsize + 2, weight='bold', pad=20)
    ax.set_xlabel("Mode Indexes", fontsize=fontsize)
    ax.set_ylabel("Mode Indexes", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    plt.show()


def load_all_envelopes(envelopes_dir, n_channels=306):
    """
    Load all envelope .npy files and stack into a single array:
    shape = (n_modes, n_samples, n_channels)
    """
    all_env = []
    for ch in range(n_channels):
        path = os.path.join(envelopes_dir, f"meg_channel_{ch}_envelopes.npy")
        if not os.path.exists(path):
            print(f"⚠️ Missing file: {path}")
            continue
        env = np.load(path)  # shape: (n_modes, n_samples)
        all_env.append(env[..., np.newaxis])  # shape: (n_modes, n_samples, 1)

    # Stack along last axis (channels)
    stacked_env = np.concatenate(all_env, axis=-1)
    return stacked_env


def compute_envelope_correlation_per_mode(envelopes):
    """
    Compute correlation matrices for each mode across all channels.

    Parameters
    ----------
    envelopes : np.ndarray
        Array of shape (n_modes, n_timepoints, n_channels)

    Returns
    -------
    dict
        Dictionary with keys = mode index, values = (306 x 306) correlation matrix
    """
    n_modes, n_timepoints, n_channels = envelopes.shape
    corr_dict = {}

    for m in range(n_modes):
        # Transpose to shape (n_channels, n_timepoints)
        data = envelopes[m].T
        corr = np.corrcoef(data)
        corr_dict[m] = corr

    return corr_dict


def build_modes_df_mvmd(u_path, envelope_dir, raw_path=None):
    """
    Build modes_df for MVMD from decomposed signals and saved envelopes.

    Parameters:
    - u_path: path to .npz file containing MVMD result (`u`)
    - envelope_dir: path to folder containing meg_channel_{i}_envelopes.npy
    - raw_path: optional path to raw MEG file (to extract channel names)

    Returns:
    - modes_df: DataFrame with columns: channel, channel_name, mode_idx, signal, envelope
    """
    # === Load MVMD results ===
    data = np.load(u_path)
    u = data['u']  # Shape: (n_modes, n_samples, n_channels)
    n_modes, n_samples, n_channels = u.shape

    # === Load channel names ===
    channel_names = [f"MEG {i:03d}" for i in range(n_channels)]
    if raw_path is not None:
        raw = mne.io.read_raw_fif(raw_path, preload=False)
        channel_names = raw.info['ch_names']

    # === Build DataFrame ===
    rows = []
    for ch in range(n_channels):
        envelope_path = os.path.join(envelope_dir, f"meg_channel_{ch}_envelopes.npy")
        if not os.path.exists(envelope_path):
            print(f"⚠️ Missing envelope for channel {ch}. Skipping.")
            continue
        envelopes = np.load(envelope_path)  # Shape: (n_modes, n_samples)

        for m in range(n_modes):
            signal = u[m, :, ch]
            envelope = envelopes[m]
            rows.append({
                "channel": ch,
                "channel_name": channel_names[ch],
                "mode_idx": m,
                "signal": signal,
                "envelope": envelope
            })

    modes_df = pd.DataFrame(rows)
    return modes_df

