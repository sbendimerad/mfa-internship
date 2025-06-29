import sys
import os
import numpy as np
import pandas as pd
import time
import mne
import torch
from pymultifracs.simul import mrw_cumul, fbm

# Add MVMD script directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'scripts', 'mvmd')))

# Import different MVMD versions
from mvmd_optimized import mvmd as mvmd_v1
from mvmd_original import mvmd as mvmd_v2
from mvmd_original2 import mvmd as mvmd_v3


# === 1. Generate synthetic multichannel signal ===
def generate_signal(n_channels=3, fs=250, n_samples=82500, as_tensor=False):
    t = np.arange(n_samples) / fs
    L = n_samples + 1
    freqs = [6, 10, 30, 80]
    signals = []

    for ch in range(n_channels):
        if ch == 0:
            c1, c2, add_fbm = 0.0, 0.0, False
        elif ch == 1:
            c1, c2, add_fbm = 0.05, 0.02, False
        elif ch == 2:
            c1, c2, add_fbm = 0.05, 0.02, True
        else:
            c1 = 0.03 + 0.01 * (ch % 3)
            c2 = 0.015 + 0.005 * (ch % 2)
            add_fbm = ch % 2 == 0

        components = []
        for f in freqs:
            mod = mrw_cumul(L, c1, c2, L).flatten() if (c1 or c2) else np.ones(L)
            mod = np.diff(mod)
            mod = mne.filter.filter_data(mod, fs, f / 4, None)
            mod = np.abs(mod)
            osc = np.sin(2 * np.pi * f * t)
            components.append(mod * osc)

        signal = np.sum(components, axis=0)

        if add_fbm:
            fbm_noise = np.diff(fbm((n_samples + 1, 1), H=0.98).squeeze())
            fbm_noise = fbm_noise / np.std(fbm_noise) * np.std(signal)
            signal += fbm_noise

        signals.append(signal)

    data = np.stack(signals)
    return torch.tensor(data, dtype=torch.float32) if as_tensor else data


# === 2. MSE computation ===
def compute_mse(original, modes, is_tensor=False):
    if is_tensor:
        modes = modes.detach().cpu().numpy()
        original = original.detach().cpu().numpy()

    # === Shape normalization to (K, N, C) ===
    if modes.shape == original.shape:
        raise ValueError("Modes and original have the same shape; this can't be right")

    if modes.shape[0] <= 10:
        if modes.shape[1] == original.shape[1] and modes.shape[2] == original.shape[0]:
            # (K, N, C): already in correct format
            pass
        elif modes.shape[1] == original.shape[0] and modes.shape[2] == original.shape[1]:
            # (K, C, N) → (K, N, C)
            modes = np.transpose(modes, (0, 2, 1))
        else:
            raise ValueError(f"Unrecognized modes shape (suspected K-first): {modes.shape}")
    elif modes.shape[0] == original.shape[0] and modes.shape[1] <= 10:
        # (C, K, N) → (K, N, C)
        modes = np.transpose(modes, (1, 2, 0))
    else:
        raise ValueError(f"Unrecognized modes shape: {modes.shape}")


    K, N, C = modes.shape
    total_mse = 0.0

    for c in range(C):
        reconstructed = np.sum(modes[:, :, c], axis=0)  # sum over modes
        mse = np.mean((original[c, :] - reconstructed) ** 2)
        total_mse += mse

    return total_mse / C



# === 3. Run one MVMD method and time it ===
def run_mvmd(signal, mvmd_func, name, is_tensor=False):
    print(f"[DEBUG] Running {name} on signal shape {signal.shape} | dtype: {type(signal)}")

    start = time.time()
    result = mvmd_func(signal, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-7, max_N=5)
    u, u_hat, omega = result  # now you can unpack it
    if name == "MVMD_v3":
        u = np.transpose(u, (0, 2, 1))

    print(f"[DEBUG] modes shape after mvmd_func: {u.shape}")
    print(f"[DEBUG] original shape: {signal.shape}")


    duration = time.time() - start

    mse = compute_mse(signal, u, is_tensor=is_tensor)

    return {
        "Method": name,
        "MSE": round(mse, 6),
        "Execution Time (s)": round(duration, 2),
        "Modes shape": u.shape
    }


# === 4. Wrappers if needed ===
def mvmd_v1_wrapper(signal, **kwargs):
    return mvmd_v1(signal, **kwargs)

def mvmd_v2_wrapper(signal, **kwargs):
    return mvmd_v2(signal, **kwargs)

def mvmd_v3_wrapper(signal, **kwargs):
    return mvmd_v3(
        signal=signal,
        k=kwargs.get('K', 5),
        alpha=kwargs.get('alpha', 2000),
        init=kwargs.get('init', 0),
        tau=kwargs.get('tau', 0),
        tol=kwargs.get('tol', 1e-3),
        verbose=kwargs.get('verbose', False)
    )

# === 5. Main benchmark ===
def main():
    signal_np = generate_signal(as_tensor=False)
    signal_tensor = torch.tensor(signal_np, dtype=torch.float32)

    methods = [
        ("MVMD_v1", mvmd_v1_wrapper, True),
        ("MVMD_v2", mvmd_v2_wrapper, True),
        ("MVMD_v3", mvmd_v3_wrapper, False),
    ]

    results = []
    for name, method, use_tensor in methods:
        signal = signal_tensor if use_tensor else signal_np
        res = run_mvmd(signal, method, name, is_tensor=use_tensor)
        results.append(res)

    df = pd.DataFrame(results)
    print("\nPerformance Summary:")
    print(df)
    return df


if __name__ == "__main__":
    main()
