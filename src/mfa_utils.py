import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymultifracs.wavelet import wavelet_analysis
from pymultifracs.mf_analysis import mfa
from pymultifracs.utils import build_q_log


def compute_mfa(signal, scaling_ranges, q_vals):
    WT = wavelet_analysis(signal, wt_name='db3')
    WTpL = WT.get_leaders(p_exp=2)
    WTpL = WTpL.auto_integrate(scaling_ranges)
    pwt = mfa(WTpL, scaling_ranges, weighted='Nj', q=q_vals)
    return pwt

def plot_mfa(pwt, ch_label=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    pwt.structure.plot_scaling()
    plt.title(f"ζ(q) for {ch_label}")
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 4))
    pwt.cumulants.plot()
    plt.title(f"Cumulants for {ch_label}")
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 4))
    pwt.spectrum.plot()
    plt.title(f"D(h) Spectrum for {ch_label}")
    plt.grid()
    plt.show()

def plot_psd(signal, fs, title="PSD"):
    from scipy.signal import periodogram
    import matplotlib.pyplot as plt
    
    f, Pxx = periodogram(signal, fs)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density")
    plt.grid()
    plt.show()

def analyze_signal_and_modes(signal, modes, scaling_ranges, q_vals, signal_name):    
    records = []

    # Compute MFA on original signal
    pwt_signal = compute_mfa(signal, scaling_ranges, q_vals)
    plot_mfa(pwt_signal, ch_label=f"{signal_name} (Original)")

    # Store cumulants summary for original signal
    records.append({
        "Signal": signal_name,
        "Mode": "Original",
        "Log-cumulant 1": pwt_signal.cumulants.data[0],
        "Log-cumulant 2": pwt_signal.cumulants.data[1],
        #"Log-cumulant 3": pwt_signal.cumulants.data[2],
    })

    # Process each mode
    for i, mode in enumerate(modes):
        pwt_mode = compute_mfa(mode, scaling_ranges, q_vals)
        plot_mfa(pwt_mode, ch_label=f"{signal_name} Mode {i}")

        records.append({
            "Signal": signal_name,
            "Mode": f"Mode {i}",
            "Log-cumulant 1": pwt_mode.cumulants.data[0],
            "Log-cumulant 2": pwt_mode.cumulants.data[1],
            #"Log-cumulant 3": pwt_mode.cumulants.data[2],
        })

    # Return dataframe summary
    return pd.DataFrame(records)

def mfa_on_envelope_centroids(modes_df, scaling_ranges, q_vals, output_base, group_by="kmeans_cluster"):
    """
    Compute MFA on centroids of envelopes grouped by a label (e.g., cluster or mode index),
    and display physiological band names instead of raw IDs.

    Parameters:
    - modes_df: DataFrame containing envelopes
    - scaling_ranges: scales for MFA integration
    - q_vals: range of q values
    - output_base: base output path
    - group_by: column to group signals by, e.g., "kmeans_cluster" or "mode_idx"
    """
    summary_records = []

    # Optional: Mapping from cluster ID to physiological band name


    cluster_name_map = {
        0: "High-Freq Noise",
        1: "Beta",
        2: "Low-Freq Noise",
        3: "High Gamma",
        4: "Alpha",
        5: "Low Gamma",
    }

    if group_by not in modes_df.columns:
        raise ValueError(f"❌ The column '{group_by}' does not exist in modes_df.")

    for group_id in sorted(modes_df[group_by].unique()):
        print(f"\n📊 Processing {group_by} = {group_id}")

        # Label for plots and table
        if group_by == "kmeans_cluster":
            label = cluster_name_map.get(group_id, f"Cluster {group_id}")
        else:
            label = f"{group_by} {group_id}"

        # Extract envelopes in this group
        group_signals = modes_df[modes_df[group_by] == group_id]["envelope"].tolist()

        try:
            centroid = np.mean(np.stack(group_signals), axis=0)
        except Exception as e:
            print(f"❌ Failed to compute centroid for {group_by} = {group_id}: {e}")
            continue

        try:
            pwt_centroid = compute_mfa(centroid, scaling_ranges, q_vals)
            plot_mfa(pwt_centroid, ch_label=f"{label} — Centroid Envelope")

            summary_records.append({
                "Cluster": label,
                "Mode": "Centroid",
                "Log-cumulant 1": pwt_centroid.cumulants.values[0],
                "Log-cumulant 2": pwt_centroid.cumulants.values[1]
            })

        except Exception as e:
            print(f"❌ MFA failed for {label}: {e}")
            continue

    summary_df = pd.DataFrame(summary_records)
    return summary_df
