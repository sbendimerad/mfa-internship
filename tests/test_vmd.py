import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sktime.libs.vmdpy import VMD as VMD_origin
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'src', 'vmd')))
from vmd_optimized import VMD as VMD_opt


def generate_test_signal(N=1000):
    """Generate a test signal with multiple components"""
    t = np.linspace(0, 1, N)
    
    # Component 1: Low frequency sine
    f1 = 5
    comp1 = np.sin(2 * np.pi * f1 * t)
    
    # Component 2: Medium frequency sine
    f2 = 15
    comp2 = 0.5 * np.sin(2 * np.pi * f2 * t)
    
    # Component 3: High frequency sine
    f3 = 50
    comp3 = 0.25 * np.sin(2 * np.pi * f3 * t)
    
    # Add some noise
    noise = 0.05 * np.random.randn(N)
    
    # Combine signals
    signal = comp1 + comp2 + comp3 + noise
    
    return signal, t, [comp1, comp2, comp3]

def run_benchmark(signal_sizes=[1000, 10000,300000]):
    """Run benchmark comparing original and optimized VMD implementations"""
    # Parameters for VMD
    alpha = 2000    # moderate bandwidth constraint
    tau = 0         # noise-tolerance (no strict fidelity enforcement)
    K = 3           # 3 modes
    DC = 0          # no DC part imposed
    init = 1        # initialize omegas uniformly
    tol = 1e-7      # tolerance
    
    # For storing results
    orig_times = []
    opt_times = []
    
    print("VMD Benchmark")
    print("=============")
    print(f"{'Signal Size':<15}{'Original (s)':<15}{'Optimized (s)':<15}{'Speedup':<15}")
    print("-" * 60)
    
    for size in signal_sizes:
        # Generate test signal
        signal, _, _ = generate_test_signal(size)
        
        # Time original implementation
        start = time.time()
        u_orig, u_hat_orig, omega_orig = VMD_origin(signal, alpha, tau, K, DC, init, tol)

        orig_time = time.time() - start
        orig_times.append(orig_time)
        
        # Time optimized implementation
        start = time.time()
        u_opt, u_hat_opt, omega_opt = VMD_opt(signal, alpha, tau, K, DC, init, tol)

        opt_time = time.time() - start
        opt_times.append(opt_time)
        
        # Calculate speedup
        speedup = orig_time / opt_time
        
        # Check if results are similar
        error = np.mean([np.mean(np.abs(u_orig[k] - u_opt[k])) for k in range(K)])
        result = "PASSED" if error < 1e-5 else f"FAILED (error: {error:.2e})"
        
        print(f"{size:<15}--{orig_time:.6f}--{opt_time:.6f}--{speedup:.2f}x        {result}")
    
    return signal_sizes, orig_times, opt_times, u_orig, u_hat_orig, omega_orig, u_opt, u_hat_opt, omega_opt

def plot_benchmark_results(sizes, orig_times, opt_times):
    """Plot benchmark results"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, orig_times, 'o-', label='Original VMD')
    plt.plot(sizes, opt_times, 's-', label='Optimized VMD')
    plt.xlabel('Signal Size')
    plt.ylabel('Time (seconds)')
    plt.title('VMD Performance Comparison')
    plt.grid(True)
    plt.legend()
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('vmd_benchmark.png')
    plt.show()


# Run benchmark
sizes, orig_times, opt_times,  u_orig, u_hat_orig, omega_orig, u_opt, u_hat_opt, omega_opt = run_benchmark()
# Plot performance comparison
plot_benchmark_results(sizes, orig_times, opt_times)