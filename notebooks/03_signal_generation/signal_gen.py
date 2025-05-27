import numpy as np
from pymultifracs.simul import mrw, mrw_cumul


def generate_synthetic_neural_signal(fs=250, n_samples=82500, frequencies=[6, 10, 30, 80], 
                                        c1=0.7, c2=-0.04, add_one_over_f=True, add_powerline=True, 
                                        one_over_f_slope=-1.0, noise_power_factor=1e5):
    t = np.arange(n_samples) / fs
    modulated_components = []
    L = n_samples + 1  # For diff
    for f in frequencies:
        # Generate MRW, differentiate, normalize
        mod = mrw_cumul(L, c1, c2, L).flatten()
        mod = np.diff(mod)
        mod -= np.min(mod)
        # Oscillation × modulator
        osc = np.sin(2 * np.pi * f * t)
        modulated = mod * osc
        modulated_components.append(modulated)
    
    # Sum all modulated components
    mrw_modulated = np.sum(modulated_components, axis=0)
    
    # 1/f noise
    one_over_f_noise = np.zeros_like(mrw_modulated)
    if add_one_over_f:
        f = np.fft.rfftfreq(n_samples, 1 / fs)
        amplitude = 1 / np.maximum(f ** (-one_over_f_slope), 1e-6)
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(f)))
        spectrum = amplitude * phase
        one_over_f_noise = np.fft.irfft(spectrum, n=n_samples)
        one_over_f_noise /= np.max(np.abs(one_over_f_noise))
        one_over_f_noise *= noise_power_factor
    
    # Powerline artifacts
    powerline_artifacts = np.zeros_like(mrw_modulated)
    if add_powerline:
        powerline_artifacts += 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)

    final_signal = mrw_modulated + one_over_f_noise + powerline_artifacts

    return {
        'time': t,
        'modulated_components': modulated_components,
        'mrw_modulated': mrw_modulated,
        'one_over_f_noise': one_over_f_noise,
        'powerline_artifacts': powerline_artifacts,
        'final_signal': final_signal
    }
