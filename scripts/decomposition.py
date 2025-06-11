import time
import sys
import os 
import numpy as np
from pandas import Series
from PyEMD import EMD, EEMD, CEEMDAN
import ewtpy
from sktime.libs.vmdpy import VMD
from sktime.transformations.series.vmd import VmdTransformer
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from features import extract_mode_features


def run_all_decompositions(signal_in, Fs, Nmodes,MaxEmdIMF,MaxVmdModes,
                           methods=None,
                           eemd_trials=50,
                           vmd_alpha=500,
                           vmd_tau=1,
                           vmd_DC=1,
                           vmd_init=1,
                           vmd_tol=1e-5,
                           return_modes=False):
    """
    Run multiple decomposition methods on a signal and extract features.

    Parameters:
        signal_in (np.array): Input 1D signal.
        Fs (int): Sampling frequency in Hz.
        Nmodes (int): Number of modes to extract.
        methods (list): List of method names to apply.
        return_modes (bool): If True, also returns the raw modes.

    Returns:
        dict: Feature dictionary.
        dict (optional): Dictionary of extracted modes.
    """
    if methods is None:
        methods = ['EMD', 'EEMD', 'CEEMDAN', 'EWT', 'VMD', 'VMDtransformer']

    results = {}
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
            modes_out[method] = IMFs

            # --- Feature Extraction ---
            all_feats = []
            all_labels = []

            for idx, mode in enumerate(IMFs):
                feats, labels = extract_mode_features(mode, Fs)
                all_feats.extend(feats)
                all_labels.extend([f"{label}{idx}" for label in labels])

            all_feats.append(exec_time)
            all_labels.append("decTime")
            all_feats.append(len(IMFs))
            all_labels.append("n_modes")

            results[method] = {"labels": all_labels, "values": all_feats}

        except Exception as e:
            print(f"Error processing {method}: {e}")
            continue

    return (results, modes_out) if return_modes else results
