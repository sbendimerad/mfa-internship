import torch

def mvmd(signal, alpha, tau, K, DC, init, tol, max_N):
    # ---------------------
    #  signal  - the time domain signal (1D) to be decomposed
    #  alpha   - the balancing parameter of the data-fidelity constraint
    #  tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    #  K       - the number of modes to be recovered
    #  DC      - true if the first mode is put and kept at DC (0-freq)
    #  init    - 0 = all omegas start at 0
    #                     1 = all omegas start uniformly distributed
    #                     2 = all omegas initialized randomly
    #  tol     - tolerance of convergence criterion; typically around 1e-6
    #
    #  Output:
    #  -------
    #  u       - the collection of decomposed modes
    #  u_hat   - spectra of the modes
    #  omega   - estimated mode center-frequencies
    #

    # import numpy as np
    # import math
    # import matplotlib.pyplot as plt

    # Period and sampling frequency of input signal
    C, T = signal.shape # T:length of signal C:  channel number
    fs = 1 / float(T)

    # extend the signal by mirroring
    # T = save_T
    # print(T)
    f_mirror = torch.zeros(C, 2*T)
    #print(f_mirror)
    f_mirror[:,0:T//2] = torch.flip(signal[:,0:T//2], dims=[-1]) 
    # print(f_mirror)
    f_mirror[:,T//2:3*T//2] = signal
    # print(f_mirror)
    f_mirror[:,3*T//2:2*T] = torch.flip(signal[:,T//2:], dims=[-1])
    # print(f_mirror)
    f = f_mirror


    # Time Domain 0 to T (of mirrored signal)
    T = float(f.shape[1])
    # print(T)
    t = torch.linspace(1/float(T), 1, int(T))
    # print(t)

    # Spectral Domain discretization
    freqs = t - 0.5 - 1/T

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = max_N

    # For future generalizations: individual alpha for each mode
    Alpha = alpha * torch.ones(K, dtype=torch.cfloat)

    # Construct and center f_hat
    f_hat = torch.fft.fftshift(torch.fft.fft(f))
    f_hat_plus = f_hat
    f_hat_plus[:, 0:int(int(T)/2)] = 0

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = torch.zeros((N, len(freqs), K, C), dtype=torch.cfloat)

    # Initialization of omega_k
    omega_plus = torch.zeros((N, K), dtype=torch.cfloat)
                        
    if (init == 1):
        for i in range(1, K+1):
            omega_plus[0,i-1] = (0.5/K)*(i-1)
    elif (init==2):
        omega_plus[0,:] = torch.sort(torch.exp(torch.log(fs)) +
        (torch.log(0.5) - torch.log(fs)) * torch.random.rand(1, K))
    else:
        omega_plus[0,:] = 0

    if (DC):
        omega_plus[0,0] = 0


    # start with empty dual variables
    lamda_hat = torch.zeros((N, len(freqs), C), dtype=torch.cfloat)

    # other inits
    uDiff = tol+2.2204e-16 #updata step
    n = 1 #loop counter
    sum_uk = torch.zeros((len(freqs), C)) #accumulator

    T = int(T)

    # ----------- Main loop for iterative updates

    while uDiff > tol and n < N:
        # update first mode accumulator
        k = 1
        sum_uk = u_hat_plus[n-1,:,K-1,:] + sum_uk - u_hat_plus[n-1,:,0,:]

        #update spectrum of first mode through Wiener filter of residuals
        for c in range(C):
            u_hat_plus[n,:,k-1,c] = (f_hat_plus[c,:] - sum_uk[:,c] - 
            lamda_hat[n-1,:,c]/2) \
        / (1 + Alpha[k-1] * torch.square(freqs - omega_plus[n-1,k-1]))
   
        #update first omega if not held at 0
        if DC == False:
            omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0), 
                            torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
            / torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))


        for k in range(2, K+1):

            #accumulator
            sum_uk = u_hat_plus[n,:,k-2,:] + sum_uk - u_hat_plus[n-1,:,k-1,:]

            #mode spectrum
            for c in range(C):
                u_hat_plus[n,:,k-1,c] = (f_hat_plus[c,:] - sum_uk[:,c] - 
            lamda_hat[n-1,:,c]/2) \
            / (1 + Alpha[k-1] * torch.square(freqs-omega_plus[n-1,k-1]))
    #         print('u_hat_plus'+str(k))
    #         print(u_hat_plus[n,:,k-1])
            
            #center frequencies
            omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0),
                torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
                /  torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T:,k-1,:])))

        #Dual ascent
    #     print(u_hat_plus.shape) tau一般是0，这里不用管
        lamda_hat[n,:,:] = lamda_hat[n-1,:,:] # + tau * (torch.sum(u_hat_plus[n,:,:,:], dim=1)
                       #  - f_hat_plus)
    #     print('lamda_hat'+str(n))
    #     print(lamda_hat[n,:])

        #loop counter
        n = n + 1

        #converged yet?
        uDiff = 2.2204e-16

        for i in range(1, K+1):
            uDiff=uDiff+1 / float(T) * torch.mm(u_hat_plus[n-1,:,i-1,:] - u_hat_plus[n-2,:,i-1,:], 
                                                ((u_hat_plus[n-1,:,i-1,:]-u_hat_plus[n-2,:,i-1,:]).conj()).conj().T)
            
        uDiff = torch.sum(torch.abs(uDiff))

        
    # ------ Postprocessing and cleanup

    # discard empty space if converged early

    N = min(N, n)
    omega = omega_plus[0:N,:]

    # Signal reconstruction
    u_hat = torch.zeros((T,K,C), dtype=torch.cfloat)
    for c in range(C):
        u_hat[T//2:T,:,c] = torch.squeeze(u_hat_plus[N-1,T//2:T,:,c])
        # print('u_hat')
        # print(u_hat.shape)
        # print(u_hat)
        second_index = list(range(1,T//2+1))
        second_index.reverse()
        u_hat[second_index,:,c] = torch.squeeze(torch.conj(u_hat_plus[N-1,T//2:T,:,c]))
        u_hat[0,:,c] = torch.conj(u_hat[-1,:,c])
    # print('u_hat')
    # print(u_hat)
    u = torch.zeros((K,len(t),C), dtype=torch.cfloat)

    for k in range(1, K+1):
        for c in range(C):
            u[k-1,:,c]  = (torch.fft.ifft(torch.fft.ifftshift(u_hat[:,k-1,c]))).real


    # remove mirror part 
    u = u[:,T//4:3*T//4,:]

    # print(u_hat.shape)
    #recompute spectrum
    u_hat = torch.zeros((T//2,K,C), dtype=torch.cfloat)

    for k in range(1, K+1):
        for c in range(C):
            u_hat[:,k-1,c] = torch.fft.fftshift(torch.fft.fft(u[k-1,:,c])).conj()
    
    # ifftshift 
    u = torch.fft.ifftshift(u, dim=-1)
            
    
        
    return (u.real, u_hat, omega)

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
