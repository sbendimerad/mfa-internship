
import numpy as np
import numba 


def VMD(f, alpha, tau, K, DC, init, tol):
    """
    Advanced Optimized Variational Mode Decomposition
    
    Parameters:
    -----------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent (pick 0 for noise-slack)
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
              1 = all omegas start uniformly distributed
              2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Returns:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """
    # Convert input to numpy array and ensure even length
    f = np.asarray(f, dtype=float)
    if len(f) % 2:
        f = f[:-1]

    # Mirror signal to create cyclical continuity - optimized version
    N = len(f)
    half_N = N // 2
    fMirr = np.empty(2*N, dtype=float)
    
    # Fast mirroring with direct assignment
    fMirr[:half_N] = f[half_N-1::-1]
    fMirr[half_N:half_N+N] = f
    fMirr[half_N+N:] = f[-1:-half_N-1:-1]
    
    # Time and frequency domain setup
    T = len(fMirr)
    T_half = T // 2
    freqs = np.linspace(0, 1, T, endpoint=False) - 0.5 - (1/T)
    
    # Pre-compute FFT
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T_half] = 0
    
    # Init omega with proper method
    omega = np.zeros(K, dtype=float)
    if init == 1:
        # Uniform distribution
        omega = np.linspace(0, 0.5, K, endpoint=False)
    elif init == 2:
        # Random initialization
        fs = 1./N
        omega = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs)) * np.random.rand(K)))
    
    # Force DC
    if DC:
        omega[0] = 0
    
    # Maximum iterations
    Niter = 500
    
    # Core VMD optimization loop - pure NumPy implementation
    u_hat_plus = np.zeros((T, K), dtype=complex)
    prev_u_hat_plus = np.zeros((T, K), dtype=complex)
    lambda_hat = np.zeros(T, dtype=complex)
    omega_history = np.zeros((Niter, K))
    omega_history[0] = omega.copy()
    
    n = 0
    uDiff = tol + 1e-16
    
    # Main optimization loop
    while (uDiff > tol and n < Niter - 1):
        # Save previous iteration
        prev_u_hat_plus = u_hat_plus.copy()
        
        # Mode accumulator - optimized to avoid redundant calculations
        # First, we calculate full sum only once
        sum_all_uk = np.sum(u_hat_plus, axis=1)
        sum_uk = sum_all_uk - u_hat_plus[:, 0]
        
        # Update first mode separately
        u_hat_plus[:, 0] = (f_hat_plus - sum_uk - lambda_hat/2) / (1 + alpha * (freqs - omega[0])**2)
        
        # Update first omega if not DC
        if not DC:
            # Use vectorized operations
            weights = np.abs(u_hat_plus[T_half:, 0])**2
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                omega[0] = np.sum(freqs[T_half:] * weights) / sum_weights
        
        # Update other modes
        for k in range(1, K):
            # Update accumulator efficiently
            sum_uk = sum_uk + u_hat_plus[:, k-1] - u_hat_plus[:, k]
            
            # Update mode spectrum 
            u_hat_plus[:, k] = (f_hat_plus - sum_uk - lambda_hat/2) / (1 + alpha * (freqs - omega[k])**2)
            
            # Update center frequency
            weights = np.abs(u_hat_plus[T_half:, k])**2
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                omega[k] = np.sum(freqs[T_half:] * weights) / sum_weights
        
        # Store omega history
        omega_history[n+1] = omega.copy()
        
        # Dual ascent - recalculate sum with updated modes
        sum_modes = np.sum(u_hat_plus, axis=1)
        lambda_hat = lambda_hat + tau * (sum_modes - f_hat_plus)
        
        # Check convergence with optimized calculation
        uDiff = np.sum(np.abs(u_hat_plus - prev_u_hat_plus)**2) / T
        
        n += 1
    
    # Signal reconstruction with optimized conjugate symmetry
    u_hat = np.zeros((T, K), dtype=complex)
    
    # Only copy the positive frequencies, then efficiently handle conjugate symmetry
    u_hat[T_half:, :] = u_hat_plus[T_half:, :]
    
    # Handle symmetry efficiently with slicing operations
    for k in range(K):
        u_hat[1:T_half, k] = np.conj(np.flip(u_hat_plus[T_half+1:, k]))
    u_hat[0, :] = np.conj(u_hat[-1, :])
    
    # IFFT to time domain with efficient processing
    u = np.zeros((K, T))
    for k in range(K):
        u[k] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    
    # Remove mirror part
    u = u[:, T//4:3*T//4]
    
    # Fast spectrum computation for output
    u_hat_out = np.zeros((u.shape[1], K), dtype=complex)
    for k in range(K):
        u_hat_out[:, k] = np.fft.fftshift(np.fft.fft(u[k]))
    
    return u, u_hat_out, omega_history[:n+1]

    """VMD implementation with Numba-accelerated components"""
    try:
        from numba import jit
        
        @jit(nopython=True)
        def update_omega(freqs, u_hat_plus, k, T_half):
            """Compute updated omega value"""
            weights = np.abs(u_hat_plus[T_half:, k])**2
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                return np.sum(freqs[T_half:] * weights) / sum_weights
            return 0.0
        
        @jit(nopython=True)
        def compute_uDiff(u_hat_plus, prev_u_hat_plus, T):
            """Compute convergence criterion"""
            return np.sum(np.abs(u_hat_plus - prev_u_hat_plus)**2) / T
        
        has_numba = True
    except ImportError:
        # If Numba isn't available, use regular functions
        def update_omega(freqs, u_hat_plus, k, T_half):
            weights = np.abs(u_hat_plus[T_half:, k])**2
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                return np.sum(freqs[T_half:] * weights) / sum_weights
            return 0.0
            
        def compute_uDiff(u_hat_plus, prev_u_hat_plus, T):
            return np.sum(np.abs(u_hat_plus - prev_u_hat_plus)**2) / T
            
        has_numba = False
        print("Warning: Numba not available. Using non-JIT functions.")
        
    # Rest of implementation is the same as VMD_opt...
    # [Code would be duplicated here]
    
    return VMD_opt(f, alpha, tau, K, DC, init, tol)  # For now, just call the standard version