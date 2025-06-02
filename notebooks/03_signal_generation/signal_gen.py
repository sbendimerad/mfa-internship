import numpy as np
from pymultifracs.simul import mrw_cumul, fbm
import mne


def generate_synthetic_neural_signal(fs=250, n_samples=82500, frequencies=[6, 10, 30, 80], 
                                     c1=0.0, c2=0.0, H=0.98,
                                     add_fbm_noise=False, add_powerline=False):
    """
    Generate synthetic signal with optional MRW modulation and FBM noise.
    """
    t = np.arange(n_samples) / fs
    L = n_samples + 1  # For MRW

    modulated_components = []

    for f in frequencies:
        # Modulator
        if c1 != 0.0 or c2 != 0.0:
            mod = mrw_cumul(L, c1, c2, L).flatten()
            mod = np.diff(mod)
            mod = mne.filter.filter_data(mod, fs, f / 4, None)
            mod = np.abs(mod)
        else:
            mod = np.ones(n_samples)  # No modulation

        # Oscillation
        osc = np.sin(2 * np.pi * f * t)
        modulated = mod * osc
        modulated_components.append(modulated)

    signal = np.sum(modulated_components, axis=0)

    # FBM noise
    if add_fbm_noise:
        fbm_raw = fbm(shape=(n_samples + 1, 1), H=H)
        fbm_noise = np.diff(fbm_raw.squeeze())
        fbm_noise = fbm_noise / np.std(fbm_noise) * np.std(signal)
        signal += fbm_noise

    # Powerline (optional)
    if add_powerline:
        signal += 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)

    return {
        'time': t,
        'final_signal': signal,
        'test': mod
    }
