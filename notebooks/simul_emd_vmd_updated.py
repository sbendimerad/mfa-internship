import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# Import des méthodes de décomposition
from PyEMD import EMD  # Pour EMD
from sktime.libs.vmdpy import VMD  # Pour VMD
import torch

# ---------------------------------------------------------
# Fonctions d'aide (Helpers Functions)
# ---------------------------------------------------------

def mvmd_torch(signal, alpha, tau, K, DC, init, tol, max_N):
    """
    Multivariate Variational Mode Decomposition (MVMD) implémentée avec PyTorch.

    Paramètres:
        signal (torch.Tensor): Signal multi-canal d'entrée, forme (C, T).
        alpha (float): Paramètre de bande passante.
        tau (float): Pas de temps de la montée duale.
        K (int): Nombre de modes à extraire.
        DC (bool): Inclure la composante DC.
        init (int): Méthode d'initialisation (0: zéros, 1: uniforme, 2: aléatoire).
        tol (float): Tolérance pour la convergence.
        max_N (int): Nombre maximum d'itérations.

    Retours:
        u_real (torch.Tensor): Modes extraits, forme (K, T_réduit, C).
        u_hat (torch.Tensor): Spectres finis des modes.
        omega (torch.Tensor): Fréquences centrales finales.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal = signal.to(device)

    C, T = signal.shape

    # --- 1. Extension par miroir (Mirror extension) ---
    # FIX #6: Use T - T//2 for the right half to handle odd T correctly.
    f_mirror = torch.zeros(C, 2 * T, device=device)
    f_mirror[:, 0:T // 2] = torch.flip(signal[:, 0:T // 2], dims=[-1])
    f_mirror[:, T // 2:3 * T // 2] = signal
    f_mirror[:, 3 * T // 2:2 * T] = torch.flip(signal[:, T - T // 2:], dims=[-1])
    f = f_mirror

    T_mirror = float(f.shape[1])
    T_size = int(T_mirror)

    # FIX #3: Keep freqs as float32 — no cfloat cast needed.
    # Explicitly set dtype=torch.float32 AND device to avoid cross-device mismatch
    # on some PyTorch versions where fftfreq doesn't inherit the current device.
    freqs = torch.fft.fftshift(
        torch.fft.fftfreq(T_size, d=1.0 / T_size)
    ).to(dtype=torch.float32, device=device)

    # --- 2. Initialisation ---
    # FIX #5: Alpha should be real (float32), not complex.
    Alpha = alpha * torch.ones(K, dtype=torch.float32, device=device)

    f_hat = torch.fft.fftshift(torch.fft.fft(f), dim=1)
    f_hat_plus = f_hat.clone()
    f_hat_plus[:, 0:T_size // 2] = 0

    u_hat_prev = torch.zeros((T_size, K, C), dtype=torch.cfloat, device=device)

    # FIX #3: omega_prev should be real (float32), not cfloat.
    omega_prev = torch.zeros(K, dtype=torch.float32, device=device)

    if init == 1:
        for i in range(K):
            omega_prev[i] = (0.5 / K) * i * (2 * torch.pi) / T_mirror
    else:
        omega_prev = torch.linspace(0.01, 0.49, K, device=device) * (2 * torch.pi) / T_mirror

    if DC:
        omega_prev[0] = 0

    lamda_hat = torch.zeros((T_size, C), dtype=torch.cfloat, device=device)
    uDiff = tol + 2.2204e-16  # scalar sentinel to enter the loop
    n = 0

    # --- 3. Décomposition itérative (ADMM) ---
    while (uDiff.item() if isinstance(uDiff, torch.Tensor) else uDiff) > tol and n < max_N:
        u_hat_curr = u_hat_prev.clone()
        omega_curr = omega_prev.clone()
        sum_uk = torch.sum(u_hat_prev, dim=1)

        for k in range(K):
            omega_k = omega_prev[k]
            # freqs and omega_k are both real, so denominator stays real.
            den_k = 1 + Alpha[k] * (freqs - omega_k) ** 2

            sum_of_other_modes = sum_uk - u_hat_prev[:, k, :]
            num_k = f_hat_plus.permute(1, 0) - sum_of_other_modes - lamda_hat / 2

            u_hat_curr[:, k, :] = num_k / den_k.unsqueeze(1)

            if k < K - 1:
                sum_uk = sum_uk - u_hat_prev[:, k, :] + u_hat_curr[:, k, :]

            if not (DC and k == 0):
                abs_u_sq = torch.square(torch.abs(u_hat_curr[:, k, :]))
                numerator = torch.sum(freqs.unsqueeze(1) * abs_u_sq)
                denominator = torch.sum(abs_u_sq)
                if denominator.item() != 0:
                    omega_curr[k] = numerator / denominator

        sum_u_final = torch.sum(u_hat_curr, dim=1)
        lamda_hat = lamda_hat + tau * (sum_u_final - f_hat_plus.permute(1, 0))

        uDiff = torch.tensor(0.0, device=device)
        for i in range(K):
            delta = u_hat_curr[:, i, :] - u_hat_prev[:, i, :]
            uDiff += (delta * delta.conj()).real.sum() / T_size

        u_hat_prev = u_hat_curr
        omega_prev = omega_curr
        n += 1

    print(f"MVMD convergée en {n} itérations.")
    uDiff_val = uDiff.item() if isinstance(uDiff, torch.Tensor) else uDiff

    omega = omega_prev.unsqueeze(0)
    u_hat = u_hat_prev

    # Symétrisation
    u_hat_sym = torch.zeros_like(u_hat)
    u_hat_sym[T_size // 2:, :, :] = u_hat[T_size // 2:, :, :]
    pos_idx = list(range(T_size - 1, T_size // 2, -1))
    neg_idx = list(range(1, T_size // 2))

    if len(pos_idx) == len(neg_idx):
        u_hat_sym[neg_idx, :, :] = torch.conj(u_hat[pos_idx, :, :])

    u = torch.zeros((K, T_size, C), dtype=torch.cfloat, device=device)
    for k in range(K):
        for c in range(C):
            u[k, :, c] = torch.fft.ifft(torch.fft.ifftshift(u_hat_sym[:, k, c]))

    # Suppression de l'extension miroir
    u_real = u[:, T_size // 4:3 * T_size // 4, :].real
    return u_real, u_hat, omega


def plot_mvmd_results(u_real, original_signal, Fs, channel_names=None):
    K, T, C = u_real.shape
    if channel_names is None:
        channel_names = [f"Channel {i + 1}" for i in range(C)]
    t = np.arange(T) / Fs
    colors = ["#ff9999", "#c4b000", "#2ca25f", "#1f78b4", "#a876e3"]

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("MVMD Decomposition per Channel + Mean PSD", fontsize=16)
    rows, cols = K + 1, C + 1
    gs = fig.add_gridspec(rows, cols, wspace=0.25, hspace=0.35)

    for c in range(C):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(t, original_signal[c], color="black")
        ax.set_title(channel_names[c])
        ax.grid(True)
        if c == 0:
            ax.set_ylabel("Original", fontsize=12, fontweight="bold")

    for k in range(K):
        for c in range(C):
            ax = fig.add_subplot(gs[k + 1, c])
            ax.plot(t, u_real[k, :, c], color=colors[k % len(colors)])
            ax.grid(True)
            if c == 0:
                ax.set_ylabel(f"Mode {k + 1}", fontsize=12, fontweight="bold")

    for k in range(K):
        ax = fig.add_subplot(gs[k + 1, C])
        psds = []
        for c in range(C):
            f, Pxx = welch(u_real[k, :, c], Fs, nperseg=1024)
            psds.append(Pxx)
        mean_psd = np.mean(psds, axis=0)
        ax.semilogx(f, mean_psd, color=colors[k % len(colors)])
        ax.set_title("PSD")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ---------------------------------------------------------
# FIX #1: All sections are now consistently indented inside
#          the if __name__ == "__main__": block.
# ---------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------
    # 1. Configuration d'un Signal Synthétique pour EMD et VMD
    # -------------------------------------------------------

    # --- Paramètres ---
    Fs = 500          # Fréquence d'échantillonnage (Hz)
    t_end = 2         # Durée en secondes
    N = Fs * t_end    # Nombre total d'échantillons
    t = np.linspace(0, t_end, N, endpoint=False)

    # --- Composantes ---
    # 1. Composante basse fréquence (équivalent bande Alpha)
    f1 = 5
    s1 = 2 * np.sin(2 * np.pi * f1 * t)

    # 2. Composante haute fréquence (équivalent bande Gamma)
    f2 = 30
    s2 = 1.5 * np.sin(2 * np.pi * f2 * t)

    # 3. Bruit blanc
    noise = 0.5 * np.random.randn(N)

    # --- Signal Composite ---
    signal = s1 + s2 + noise

    print("Signal synthétique généré. Prêt pour la décomposition.")

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title("Signal Synthétique (Original)")
    plt.xlabel("Temps (s)")
    plt.show()

    # -------------------------------------------------------
    # 2. Empirical Mode Decomposition (EMD)
    # -------------------------------------------------------
    # EMD iteratively decomposes the signal into Intrinsic Mode Functions (IMFs).
    # EMD is generally sensitive to noise and lacks the mathematical foundation of VMD.
    # Key Parameter: max_imf (maximum number of IMFs to extract).

    emd = EMD()
    IMFs = emd.emd(signal, max_imf=10)

    print(f"EMD extracted {IMFs.shape[0]} Intrinsic Mode Functions (IMFs).")

    fig, axes = plt.subplots(IMFs.shape[0] + 1, 1, figsize=(10, 2 * IMFs.shape[0]))
    fig.suptitle('EMD Decomposition Results', fontsize=14)

    axes[0].plot(t, signal)
    axes[0].set_title('Original Signal')
    axes[0].set_xlim(t[0], t[-1])
    axes[0].grid(True)

    for n, imf in enumerate(IMFs):
        axes[n + 1].plot(t, imf, 'g')
        axes[n + 1].set_title(f"IMF {n + 1}")
        axes[n + 1].set_xlim(t[0], t[-1])
        axes[n + 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # -------------------------------------------------------
    # 3. Variational Mode Decomposition (VMD)
    # -------------------------------------------------------
    # VMD decomposes the signal by solving an optimization problem.
    # It requires a priori setting of the number of modes K.
    #
    # Key Parameters:
    #   alpha : Bandwidth limit. Larger = narrower modes. Typical range: 500–3000.
    #   tau   : Dual ascent time step (0 for quadratic penalty / noise-robustness).
    #   K     : Number of modes. Here K=3 (2 components + noise residual).

    vmd_alpha = 2000
    vmd_tau   = 0
    vmd_K     = 3
    vmd_DC    = 0
    vmd_init  = 1
    vmd_tol   = 1e-7

    modes, omega_vmd, _ = VMD(signal, vmd_alpha, vmd_tau, vmd_K, vmd_DC, vmd_init, vmd_tol)

    print(f"VMD extracted {modes.shape[0]} modes.")

    # FIX #2: omega from sktime's VMD has shape (K, iterations).
    #         Use the last iteration value for the final center frequency estimate,
    #         rather than taking the mean across all iterations.
    omega_hz = omega_vmd[:, -1] * Fs / (2 * np.pi)
    print(f"Estimated Central Frequencies (in Hz): {omega_hz}")

    fig, axes = plt.subplots(modes.shape[0] + 1, 1, figsize=(10, 2 * modes.shape[0]))
    fig.suptitle(f'VMD Decomposition Results (K={vmd_K}, alpha={vmd_alpha})', fontsize=14)

    axes[0].plot(t, signal)
    axes[0].set_title('Original Signal')
    axes[0].set_xlim(t[0], t[-1])
    axes[0].grid(True)

    for n, mode in enumerate(modes):
        axes[n + 1].plot(t, mode.real, 'r')
        axes[n + 1].set_title(f"Mode {n + 1} (Center Freq: {omega_hz[n]:.2f} Hz)")
        axes[n + 1].set_xlim(t[0], t[-1])
        axes[n + 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # -------------------------------------------------------
    # 4. Setup a Synthetic Signal for MVMD
    # -------------------------------------------------------

    T_mvmd = 2000
    C = 2
    t_mvmd = np.linspace(0, 1, T_mvmd)

    # FIX #4: Define the effective sampling rate for the MVMD signal separately.
    #         t = linspace(0, 1, T_mvmd) → effective Fs = T_mvmd = 2000 Hz, not 500.
    Fs_mvmd = T_mvmd

    signal_np = np.stack([
        np.sin(2 * np.pi * 10 * t_mvmd) + 0.3 * np.random.randn(T_mvmd),
        np.sin(2 * np.pi * 15 * t_mvmd) + 0.3 * np.random.randn(T_mvmd)
    ])

    signal_torch = torch.tensor(signal_np, dtype=torch.float32)

    plt.figure(figsize=(12, 5))
    for c in range(C):
        plt.plot(t_mvmd, signal_np[c], label=f'Channel {c + 1}')
    plt.title('Synthetic Multichannel Signal (10 Hz and 15 Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 5. Multivariate Variational Mode Decomposition (MVMD)
    # -------------------------------------------------------
    # MVMD extends VMD to multichannel signals, decomposing all channels jointly
    # while enforcing a shared central frequency per mode. This improves stability
    # and reduces mode misalignment across channels.
    #
    # Key Parameters:
    #   alpha : Bandwidth constraint (same role as in VMD).
    #   tau   : Dual ascent step controlling convergence. Typically 0.
    #   K     : Number of modes (e.g., 2 components + noise → K=3).
    #   DC    : Include/exclude zero-frequency mode (False = no DC).
    #   init  : Center frequency init (0: zeros, 1: uniform, 2: random).
    #   tol   : Convergence tolerance.
    #
    # This example uses the custom mvmd_torch function (defined above) which
    # leverages PyTorch for efficient multichannel optimization.

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU acceleration.")
    else:
        print("CUDA not available. Using CPU.")

    u_real, u_hat, omega_mvmd = mvmd_torch(
        signal_torch,
        alpha=2000,
        tau=0.0,
        K=3,
        DC=True,
        init=1,
        tol=1e-7,
        max_N=200
    )

    u_real_np = u_real.detach().cpu().numpy()    # (K, T_reduced, C)
    omega_np  = omega_mvmd.detach().cpu().numpy()  # (1, K) — real after FIX #3

    # FIX #4: Use Fs_mvmd (= 2000) instead of Fs (= 500) for correct Hz conversion.
    omega_rad = np.real(omega_np).squeeze()
    omega_hz_mvmd = omega_rad * Fs_mvmd / (2 * np.pi)
    print(f"MVMD Estimated Central Frequencies (Hz): {omega_hz_mvmd}")

    plot_mvmd_results(
        u_real_np,
        signal_np[:, :u_real_np.shape[1]],
        Fs_mvmd,
        channel_names=["Channel 1", "Channel 2"]
    )
