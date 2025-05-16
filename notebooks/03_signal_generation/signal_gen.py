import numpy as np

def generate_synthetic_neural_signal(mode="basic", 
                                     fs=250, 
                                     n_samples=82500, 
                                     frequencies=[6, 10, 30, 80],
                                     c1=None, c2=None, L=82500,
                                     H=None, lam=None,
                                     add_one_over_f=True,
                                     one_over_f_slope=-1.0,
                                     add_powerline=True):
    """
    Generate synthetic neural signals for validating VMD and MFA.
    
    Parameters:
        mode (str): 'basic' for clean oscillations, 'realistic' for MRW-modulated oscillations.
        fs (int): Sampling frequency in Hz.
        n_samples (int): Number of samples to generate.
        frequencies (list): List of frequencies to simulate in Hz.
        c1, c2, L (float): MRW cumulants and scale (used if provided).
        H, lam (float): MRW manual parameters (used if c1/c2 not provided).
        add_one_over_f (bool): Add 1/f noise (default True).
        one_over_f_slope (float): Slope of 1/f noise (default -1.0).
        add_powerline (bool): Add 50Hz and 100Hz artifacts (default True).

    Returns:
        dict: Dictionary with all signal components and the final signal.
    """
    t = np.arange(n_samples) / fs

    # Base Oscillations (6Hz, 10Hz, etc.)
    base_oscillations = [np.sin(2 * np.pi * f * t) for f in frequencies]
    mixed_oscillation = np.sum(base_oscillations, axis=0)

    # --- Realistic Mode: MRW Modulation ---
    if mode == "realistic":
        mrw_signal = None
        try:
            from pymultifracs.simul import mrw, mrw_cumul
        except ImportError:
            raise ImportError("pymultifracs.simul not found. Make sure pymultifracs is installed.")
        
        if c1 is not None and c2 is not None and L is not None:
            print("Generating MRW with c1, c2, L using mrw_cumul()")
            mrw_signal = mrw_cumul(n_samples, c1, c2, L).flatten()
        elif H is not None and lam is not None and L is not None:
            print("Generating MRW with H, lam, L using mrw()")
            mrw_signal = mrw(shape=(n_samples, 1), H=H, lam=lam, L=L).flatten()
        else:
            raise ValueError("Realistic mode requires either (c1, c2, L) or (H, lam, L) to be provided.")
        
        mrw_modulated = mrw_signal * mixed_oscillation
    else:
        mrw_modulated = mixed_oscillation

    # --- Add 1/f Noise ---
    one_over_f_noise = np.zeros_like(mrw_modulated)
    if add_one_over_f:
        f = np.fft.rfftfreq(n_samples, 1 / fs)
        amplitude_spectrum = 1 / np.maximum(f**(-one_over_f_slope), 1e-6)
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(f)))
        noise_spectrum = amplitude_spectrum * phase
        one_over_f_noise = np.fft.irfft(noise_spectrum, n=n_samples)
        one_over_f_noise /= np.max(np.abs(one_over_f_noise))

    # --- Add Powerline Artifacts (50Hz and 100Hz) ---
    powerline_artifacts = np.zeros_like(mrw_modulated)
    if add_powerline:
        powerline_artifacts += 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)

    # --- Final Signal Composition ---
    final_signal = mrw_modulated + one_over_f_noise + powerline_artifacts

    # --- Return Components ---
    return {
        'time': t,
        'base_oscillations': base_oscillations,
        'mixed_oscillation': mixed_oscillation,
        'mrw_modulated': mrw_modulated,
        'one_over_f_noise': one_over_f_noise,
        'powerline_artifacts': powerline_artifacts,
        'final_signal': final_signal
    }
