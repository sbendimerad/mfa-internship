import os 
import sys
import time
import numpy as np
import pandas as pd
from pandas import Series
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import ewtpy
from evaluation import compute_soi_matrix
from PyEMD import EMD, EEMD, CEEMDAN
from sktime.libs.vmdpy import VMD
from sktime.transformations.series.vmd import VmdTransformer
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))


def gridsearch_vmd(signal, fs, param_grid):
    results = []
    spectra = {}
    for params in tqdm(param_grid, desc="Benchmarking VMD"):
        Nm = params['Nmodes']
        alpha = params['alpha']
        tau = params['tau']
        DC = params.get('DC', 0)
        init = params.get('init', 1)
        tol = params.get('tol', 1e-6)

        modes, _, _ = VMD(signal, alpha, tau, Nm, DC, init, tol)
        reconstructed = np.sum(modes, axis=0)
        mae = mean_absolute_error(signal, reconstructed)
        _, mean_soi = compute_soi_matrix(modes, fs)

        results.append({
            "alpha": alpha, "tau": tau, "K": Nm, "DC": DC, "init": init, "tol": tol,
            "MAE": mae, "mean_soi": mean_soi,
        })
        spectra[(alpha, tau, Nm)] = modes

    return pd.DataFrame(results), spectra

def run_all_decompositions(signal_in, Fs, Nmodes, MaxEmdIMF, MaxVmdModes,
                           methods=None,
                           eemd_trials=50,
                           vmd_alpha=50,
                           vmd_tau=0,
                           vmd_DC=0,
                           vmd_init=0,
                           vmd_tol=1e-7,
                           return_modes=True):
    
    """
    Run multiple decomposition methods on a signal and return modes only.
    Feature extraction is done separately.

    Parameters:
        signal_in (np.array): Input 1D signal.
        Fs (int): Sampling frequency in Hz.
        Nmodes (int): Number of modes to extract.
        methods (list): List of method names to apply.
        return_modes (bool): If True, returns dictionary of modes.

    Returns:
        dict: Dictionary of extracted modes if return_modes=True.
        None: otherwise.
    """
    if methods is None:
        methods = ['EMD', 'EEMD', 'CEEMDAN', 'EWT', 'VMD', 'VMDtransformer']

    modes_out = {}

    for method in methods:
        try:
            start_time = time.time()

            # --- Decomposition ---
            if method == 'EMD':
                emd = EMD()
                IMFs = emd.emd(signal_in, max_imf=MaxEmdIMF)

            elif method == 'EEMD':
                eemd = EEMD()
                eemd.trials = eemd_trials
                IMFs = eemd(signal_in, max_imf=MaxEmdIMF)

            elif method == 'CEEMDAN':
                ceemdan = CEEMDAN()
                IMFs = ceemdan(signal_in, max_imf=MaxEmdIMF)

            elif method == 'EWT':
                ewt, _, _ = ewtpy.EWT1D(signal_in, N=MaxEmdIMF)
                IMFs = ewt.T

            elif method == 'VMD':
                IMFs, _, _ = VMD(signal_in,
                                 vmd_alpha,
                                 vmd_tau,
                                 Nmodes,
                                 vmd_DC,
                                 vmd_init,
                                 vmd_tol)

            elif method == 'VMDtransformer':
                transformer = VmdTransformer(K=None, kMax=MaxVmdModes, energy_loss_coefficient=0.01)
                IMFs = transformer.fit_transform(Series(signal_in)).to_numpy().T
                if IMFs.ndim == 1:
                    IMFs = IMFs.reshape(1, -1)

            else:
                print(f"Unknown method: {method}")
                continue

            exec_time = time.time() - start_time
            print(f"{method} decomposition done in {exec_time:.2f} seconds.")

            modes_out[method] = IMFs

        except Exception as e:
            print(f"Error processing {method}: {e}")
            continue

    if return_modes:
        return modes_out
    else:
        return None
