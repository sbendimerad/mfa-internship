import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'vmd')))

from vmd_original import VMD as VMD
from vmd_optimized import VMD as VMD_opt

def generate_test_signal(N=1000):
    t = np.linspace(0, 1, N)
    f1, f2, f3 = 5, 15, 50
    comp1 = np.sin(2 * np.pi * f1 * t)
    comp2 = 0.5 * np.sin(2 * np.pi * f2 * t)
    comp3 = 0.25 * np.sin(2 * np.pi * f3 * t)
    noise = 0.05 * np.random.randn(N)
    signal = comp1 + comp2 + comp3 + noise
    return signal, t, [comp1, comp2, comp3]

def run_benchmark(signal_sizes=[1000, 10000, 100000]):
    alpha, tau, K, DC, init, tol = 2000, 0, 3, 0, 1, 1e-7
    results = []

    print("VMD Benchmark")
    print("=============")

    for size in signal_sizes:
        signal, _, _ = generate_test_signal(size)

        start = time.time()
        u_orig, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
        orig_time = time.time() - start

        start = time.time()
        u_opt, _, _ = VMD_opt(signal, alpha, tau, K, DC, init, tol)
        opt_time = time.time() - start

        speedup = orig_time / opt_time

        error = np.mean([np.mean(np.abs(u_orig[k] - u_opt[k])) for k in range(K)])
        result = "PASSED" if error < 1e-5 else f"FAILED (error: {error:.2e})"

        results.append({
            'Signal Size': size,
            'Original Time (s)': orig_time,
            'Optimized Time (s)': opt_time,
            'Speedup': speedup,
            'Result': result
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    df_results.to_csv('vmd_benchmark_results.csv', index=False)

    return df_results

def plot_benchmark_results(df_results):
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Signal Size'], df_results['Original Time (s)'], 'o-', label='Original VMD')
    plt.plot(df_results['Signal Size'], df_results['Optimized Time (s)'], 's-', label='Optimized VMD')
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

def visualize_decomposition(signal_size=1000):
    signal, t, true_components = generate_test_signal(signal_size)
    alpha, tau, K, DC, init, tol = 2000, 0, 3, 0, 1, 1e-7

    u_orig, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    u_opt, _, _ = VMD_opt(signal, alpha, tau, K, DC, init, tol)

    fig, axs = plt.subplots(6, 1, figsize=(14, 14), sharex=True)

    axs[0].plot(t, signal)
    axs[0].set_title('Original Signal')
    axs[0].grid()

    for i, comp in enumerate(true_components):
        axs[1].plot(t, comp, label=f'Component {i+1}')
    axs[1].set_title('True Signal Components')
    axs[1].legend()
    axs[1].grid()

    for i in range(K):
        axs[2].plot(t, u_orig[i], label=f'Mode {i+1}')
    axs[2].set_title('Original VMD Modes')
    axs[2].legend()
    axs[2].grid()

    for i in range(K):
        axs[3].plot(t, u_opt[i], label=f'Mode {i+1}')
    axs[3].set_title('Optimized VMD Modes')
    axs[3].legend()
    axs[3].grid()

    for i in range(K):
        axs[4].plot(t, u_orig[i] - u_opt[i], label=f'Diff Mode {i+1}')
    axs[4].set_title('Difference between Original and Optimized Modes')
    axs[4].legend()
    axs[4].grid()

    axs[5].plot(t, np.sum(u_orig, axis=0), label='Reconstructed Original')
    axs[5].plot(t, np.sum(u_opt, axis=0), '--', label='Reconstructed Optimized')
    axs[5].set_title('Signal Reconstruction Comparison')
    axs[5].legend()
    axs[5].grid()

    plt.tight_layout()
    plt.savefig('vmd_decomposition.png')
    plt.show()

if __name__ == "__main__":
    df_results = run_benchmark()
    plot_benchmark_results(df_results)
    visualize_decomposition()
