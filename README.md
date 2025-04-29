# MFA Internship Project

This repository contains the work conducted during my Master's internship.  
The objective was to apply **Multifractal Analysis (MFA)** on signals, both directly and after applying **Variational Mode Decomposition (VMD)** and **Multivariate Variational Mode Decomposition (MVMD)**.  
It includes exploration notebooks, original and optimized versions of VMD and MVMD code, and testing scripts.

---

## 📂 Project Structure

Project organization:

- `src/`
  - `vmd/`
    - `vmd_original.py`
    - `vmd_optimized.py`
  - `mvmd/`
    - `mvmd_original.py`
    - `mvmd_optimized.py`
- `notebooks/`
  - `00_mne_tutorials/`
  - `01_vmd_mfa/`
    - `01_vmd_mfa_simulated.ipynb`
    - `02_vmd_mfa_real.ipynb`
  - `02_mvmd_mfa/`
    - `01_mvmd_mfa_simulated.ipynb`
    - `02_mvmd_mfa_real.ipynb`
- `data/`
  - `.mat` files
- `figures/`
  - (Result plots)
- `tests/`
  - `test_vmd.py`
  - `test_mvmd.py`

---

## ⚙️ Installation

It is recommended to use a Conda environment to keep dependencies isolated.

### Create and activate a Conda environment

```bash
conda create -n mfa-internship python=3.10
conda activate mfa-internship
```
### Install required Python packages

```bash
pip install -r requirements.txt
```