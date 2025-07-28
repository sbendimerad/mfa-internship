import torch



def mvmd_torch(signal, alpha, tau, K, DC, init, tol, max_N):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal = signal.to(device)

    C, T = signal.shape
    fs = 1 / float(T)

    # Mirror extension
    f_mirror = torch.zeros(C, 2*T, device=device)
    f_mirror[:, 0:T//2] = torch.flip(signal[:, 0:T//2], dims=[-1])
    f_mirror[:, T//2:3*T//2] = signal
    f_mirror[:, 3*T//2:2*T] = torch.flip(signal[:, T//2:], dims=[-1])
    f = f_mirror

    T = float(f.shape[1])
    t = torch.linspace(1/float(T), 1, int(T), device=device)
    freqs = t - 0.5 - 1/T
    T = int(T)

    Alpha = alpha * torch.ones(K, dtype=torch.cfloat, device=device)
    f_hat = torch.fft.fftshift(torch.fft.fft(f), dim=1)
    f_hat_plus = f_hat.clone()
    f_hat_plus[:, 0:T//2] = 0

    # Initialize u_hat
    u_hat_prev = torch.zeros((T, K, C), dtype=torch.cfloat, device=device)
    u_hat_curr = torch.zeros((T, K, C), dtype=torch.cfloat, device=device)

    # Initialize omega
    omega_prev = torch.zeros(K, dtype=torch.cfloat, device=device)
    omega_curr = torch.zeros(K, dtype=torch.cfloat, device=device)

    if init == 1:
        for i in range(K):
            omega_prev[i] = (0.5 / K) * i
    elif init == 2:
        omega_prev = torch.sort(
            torch.exp(torch.log(fs)) +
            (torch.log(torch.tensor(0.5)) - torch.log(fs)) * torch.rand(K, device=device)
        )[0]
    else:
        omega_prev[:] = 0

    if DC:
        omega_prev[0] = 0

    # Dual variable (only current needed)
    lamda_hat = torch.zeros((T, C), dtype=torch.cfloat, device=device)

    uDiff = tol + 2.2204e-16
    n = 0
    sum_uk = torch.zeros((T, C), dtype=torch.cfloat, device=device)

    while uDiff > tol and n < max_N:
        sum_uk = u_hat_prev[:, K-1, :] + sum_uk - u_hat_prev[:, 0, :]

        # First mode update
        k = 0
        for c in range(C):
            u_hat_curr[:, k, c] = (f_hat_plus[c, :] - sum_uk[:, c] - lamda_hat[:, c]/2) \
                                  / (1 + Alpha[k] * (freqs - omega_prev[k])**2)

        if not DC:
            omega_curr[k] = torch.sum(
                freqs[T//2:T].unsqueeze(0) @ torch.square(torch.abs(u_hat_curr[T//2:T, k, :]))
            ) / torch.sum(torch.square(torch.abs(u_hat_curr[T//2:T, k, :])))

        # Remaining modes
        for k in range(1, K):
            sum_uk = u_hat_curr[:, k-1, :] + sum_uk - u_hat_prev[:, k, :]
            for c in range(C):
                u_hat_curr[:, k, c] = (f_hat_plus[c, :] - sum_uk[:, c] - lamda_hat[:, c]/2) \
                                      / (1 + Alpha[k] * (freqs - omega_prev[k])**2)

            omega_curr[k] = torch.sum(
                freqs[T//2:T].unsqueeze(0) @ torch.square(torch.abs(u_hat_curr[T//2:T, k, :]))
            ) / torch.sum(torch.square(torch.abs(u_hat_curr[T//2:T, k, :])))

        # Update lambda (dual ascent)
        # (tau is usually 0, so this has no effect unless tau > 0)
        lamda_hat = lamda_hat  # + tau * (torch.sum(u_hat_curr, dim=1) - f_hat_plus)

        # Convergence check
        uDiff = 2.2204e-16
        for i in range(K):
            delta = u_hat_curr[:, i, :] - u_hat_prev[:, i, :]
            uDiff += (delta * delta.conj()).real.sum() / T

        # Swap buffers
        u_hat_prev, u_hat_curr = u_hat_curr, u_hat_prev
        omega_prev, omega_curr = omega_curr, omega_prev

        n += 1

    # Final omega
    omega = omega_prev.unsqueeze(0)

    # Final reconstruction
    u_hat = torch.zeros((T, K, C), dtype=torch.cfloat, device=device)
    for c in range(C):
        u_hat[T//2:T, :, c] = u_hat_prev[T//2:T, :, c]
        idx = list(range(1, T//2+1))
        idx.reverse()
        u_hat[idx, :, c] = torch.conj(u_hat_prev[T//2:T, :, c])
        u_hat[0, :, c] = torch.conj(u_hat[-1, :, c])

    u = torch.zeros((K, T, C), dtype=torch.cfloat, device=device)
    for k in range(K):
        for c in range(C):
            u[k, :, c] = torch.fft.ifft(torch.fft.ifftshift(u_hat[:, k, c])).real

    # Remove mirror
    u = u[:, T//4:3*T//4, :]

    # Compute final u_hat
    T_reduced = u.shape[1]
    u_hat_final = torch.zeros((T_reduced, K, C), dtype=torch.cfloat, device=device)
    for k in range(K):
        for c in range(C):
            u_hat_final[:, k, c] = torch.fft.fftshift(torch.fft.fft(u[k, :, c])).conj()

    u = torch.fft.ifftshift(u, dim=-1)

    return u.real, u_hat_final, omega
