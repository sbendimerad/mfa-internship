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
        "Log-cumulant 3": pwt_signal.cumulants.data[2],
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
            "Log-cumulant 3": pwt_mode.cumulants.data[2],
        })

    # Return dataframe summary
    return pd.DataFrame(records)
