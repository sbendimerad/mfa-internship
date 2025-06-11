import os
import numpy as np
import matplotlib.pyplot as plt

from pymultifracs.wavelet import wavelet_analysis
from pymultifracs.mf_analysis import mfa
from pymultifracs.utils import build_q_log


def apply_mfa(signal, scaling_ranges, q_vals, ch_label=""):
    """Apply MFA to a 1D signal and plot results."""
    print(f"\n Applying MFA to {ch_label} — Length: {len(signal)}")

    WT = wavelet_analysis(signal, wt_name='db3')
    WTpL = WT.get_leaders(p_exp=2)
    WTpL = WTpL.auto_integrate(scaling_ranges)
    pwt = mfa(WTpL, scaling_ranges, weighted='Nj', q=q_vals)

    # ζ(q)
    plt.figure(figsize=(8, 4))
    pwt.structure.plot_scaling()
    plt.title(f"ζ(q) for {ch_label}")
    plt.grid()
    plt.show()

    # Cumulants
    plt.figure(figsize=(8, 4))
    pwt.cumulants.plot()
    plt.title(f"Cumulants for {ch_label}")
    plt.grid()
    plt.show()

    # D(h)
    plt.figure(figsize=(8, 4))
    pwt.spectrum.plot()
    plt.title(f"D(h) Spectrum for {ch_label}")
    plt.grid()
    plt.show()

    return pwt


def apply_mfa(signal, scaling_ranges, q_vals, ch_label=""):
    """Apply MFA to a 1D signal and plot results."""
    print(f"\n Applying MFA to {ch_label} — Length: {len(signal)}")

    WT = wavelet_analysis(signal, wt_name='db3')
    WTpL = WT.get_leaders(p_exp=2)
    WTpL = WTpL.auto_integrate(scaling_ranges)
    pwt = mfa(WTpL, scaling_ranges, weighted='Nj', q=q_vals)

    # ζ(q)
    plt.figure(figsize=(8, 4))
    pwt.structure.plot_scaling()
    plt.title(f"ζ(q) for {ch_label}")
    plt.grid()
    plt.show()

    # Cumulants
    plt.figure(figsize=(8, 4))
    pwt.cumulants.plot()
    plt.title(f"Cumulants for {ch_label}")
    plt.grid()
    plt.show()

    # D(h)
    plt.figure(figsize=(8, 4))
    pwt.spectrum.plot()
    plt.title(f"D(h) Spectrum for {ch_label}")
    plt.grid()
    plt.show()

    return pwt


def load_modes_and_apply_mfa(modes_filename, base_path, methods, scaling_ranges, q_vals):
    """
    Load saved modes and apply MFA to each.
    """
    for method in methods:
        file_path = os.path.join(base_path, method, "modes", modes_filename)
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        modes = np.load(file_path)
        print(f"\n Loaded {method} modes — shape: {modes.shape}")

        for idx, mode in enumerate(modes):
            label = f"{method} | Mode {idx}"
            apply_mfa(mode, scaling_ranges, q_vals, ch_label=label)



