import torch

def mirror_signal(signal: torch.Tensor) -> torch.Tensor:
    C, T = signal.shape
    mirrored = torch.zeros(C, 2 * T)
    mirrored[:, 0:T//2] = torch.flip(signal[:, 0:T//2], dims=[-1])
    mirrored[:, T//2:3*T//2] = signal
    mirrored[:, 3*T//2:] = torch.flip(signal[:, T//2:], dims=[-1])
    return mirrored


def initialize_omega(init: int, K: int, DC: bool, fs: float) -> torch.Tensor:
    omega_init = torch.zeros(K, dtype=torch.cfloat)
    if init == 1:
        for i in range(K):
            omega_init[i] = (0.5 / K) * i
    elif init == 2:
        omega_init = torch.sort(
            torch.log(fs) + (torch.log(torch.tensor(0.5)) - torch.log(fs)) * torch.rand(K)
        ).values
    if DC:
        omega_init[0] = 0
    return omega_init


def mvmd(signal, alpha, tau, K, DC, init, tol, max_N):
    assert signal.ndim == 2, "Input signal must be 2D (channels x time)"
    C, T_orig = signal.shape
    fs = 1 / float(T_orig)

    f = mirror_signal(signal)
    T = float(f.shape[1])
    t = torch.linspace(1 / T, 1, int(T))
    freqs = t - 0.5 - 1 / T
    N = max_N

    Alpha = alpha * torch.ones(K, dtype=torch.cfloat)
    f_hat = torch.fft.fftshift(torch.fft.fft(f))
    f_hat_plus = f_hat.clone()
    f_hat_plus[:, 0:int(T // 2)] = 0

    u_hat_plus = torch.zeros((N, len(freqs), K, C), dtype=torch.cfloat)
    omega_plus = torch.zeros((N, K), dtype=torch.cfloat)
    omega_plus[0, :] = initialize_omega(init, K, DC, fs)

    lamda_hat = torch.zeros((N, len(freqs), C), dtype=torch.cfloat)

    uDiff = tol + 1e-15
    n = 1
    sum_uk = torch.zeros((len(freqs), C))
    T = int(T)

    while uDiff > tol and n < N:
        k = 1
        sum_uk = u_hat_plus[n - 1, :, K - 1, :] + sum_uk - u_hat_plus[n - 1, :, 0, :]

        for c in range(C):
            u_hat_plus[n, :, k - 1, c] = (
                f_hat_plus[c, :] - sum_uk[:, c] - lamda_hat[n - 1, :, c] / 2
            ) / (1 + Alpha[k - 1] * (freqs - omega_plus[n - 1, k - 1]) ** 2)

        if not DC:
            power = torch.square(torch.abs(u_hat_plus[n, T // 2:T, k - 1, :]))
            omega_plus[n, k - 1] = torch.sum(
                freqs[T // 2:T].unsqueeze(1) * power
            ) / torch.sum(power)

        for k in range(2, K + 1):
            sum_uk = u_hat_plus[n, :, k - 2, :] + sum_uk - u_hat_plus[n - 1, :, k - 1, :]

            for c in range(C):
                u_hat_plus[n, :, k - 1, c] = (
                    f_hat_plus[c, :] - sum_uk[:, c] - lamda_hat[n - 1, :, c] / 2
                ) / (1 + Alpha[k - 1] * (freqs - omega_plus[n - 1, k - 1]) ** 2)

            power = torch.square(torch.abs(u_hat_plus[n, T // 2:T, k - 1, :]))
            omega_plus[n, k - 1] = torch.sum(
                freqs[T // 2:T].unsqueeze(1) * power
            ) / torch.sum(power)

        lamda_hat[n, :, :] = lamda_hat[n - 1, :, :]  # tau=0

        n += 1
        uDiff = 1e-15
        for i in range(1, K + 1):
            diff = u_hat_plus[n - 1, :, i - 1, :] - u_hat_plus[n - 2, :, i - 1, :]
            uDiff += torch.sum(torch.abs(torch.sum(diff * diff.conj(), dim=0)))

    N = min(N, n)
    omega = omega_plus[0:N, :]

    u_hat = torch.zeros((T, K, C), dtype=torch.cfloat)
    for c in range(C):
        u_hat[T // 2:T, :, c] = torch.squeeze(u_hat_plus[N - 1, T // 2:T, :, c])
        second_index = list(range(1, T // 2 + 1))
        second_index.reverse()
        u_hat[second_index, :, c] = torch.squeeze(torch.conj(u_hat_plus[N - 1, T // 2:T, :, c]))
        u_hat[0, :, c] = torch.conj(u_hat[-1, :, c])

    u = torch.zeros((K, len(t), C), dtype=torch.cfloat)
    for k in range(K):
        for c in range(C):
            u[k, :, c] = torch.fft.ifft(torch.fft.ifftshift(u_hat[:, k, c])).real

    u = u[:, T // 4:3 * T // 4, :]
    u_hat = torch.zeros((T // 2, K, C), dtype=torch.cfloat)
    for k in range(K):
        for c in range(C):
            u_hat[:, k, c] = torch.fft.fftshift(torch.fft.fft(u[k, :, c])).conj()

    u = torch.fft.ifftshift(u, dim=-1)

    return u.real, u_hat, omega
