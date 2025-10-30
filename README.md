# Multifractal Analysis Internship Project 🧠📊

This repository contains the code and notebooks from my internship project on **multifractal analysis of time series**.  
The main focus is on applying **Variational Mode Decomposition (VMD)** and **Multivariate VMD (MVMD)** for analyzing synthetic multifractal signals.

---

## 🚀 Project Overview
- **Simulate** multifractal synthetic processes (e.g., MRW, FBM).  
- **Decompose** signals with VMD and MVMD.  
- **Extract features** (spectral envelopes, multifractal indicators).  
- **Visualize and evaluate** results with dedicated plotting scripts.  

The core logic is demonstrated in two main notebooks:  
- 📓 [VMD pipeline](notebooks/01_vmd_on_realdata_pipeline)  
- 📓 [MVMD pipeline](notebooks/02_mvmd_on_realdata_pipeline)  

Other notebooks provide simulations, animations, and wavelet-based experiments.

---

## 📂 Repository Structure
```
notebooks/          # Main pipelines & experiments
  00_simulation_pipeline.ipynb
  01_vmd_pipeline..ipynb
  02_mvmd_pipeline.ipynb
  Extra_animations.ipynb
  Extra_pywt_enveloppes.ipynb

scripts/            # Functions used inside notebooks
  decomposition.py
  evaluation.py
  extract_envelopes.py
  features.py
  mfa_utils.py
  plotting.py
  simulation.py
  mvmd/
    mvmd_original.py
    mvmd_optimized.py

results/            # Example outputs (MRW, FBM, etc.)
requirements.txt
README.md
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sbendimerad/mfa-internship.git
cd mfa-internship
pip install -r requirements.txt
```

Recommended: use a virtual environment (`conda` or `venv`).

---

## ▶️ Usage

### Open the main notebooks
Run Jupyter to explore the pipelines:
```bash
jupyter notebook
```

- `01_vmd_pipeline..ipynb` → Full pipeline using **VMD**.  
- `02_mvmd_pipeline.ipynb` → Full pipeline using **MVMD**.  

Both notebooks call functions defined in the `scripts/` folder.  

### Example: using scripts directly
```python
from scripts.decomposition import vmd
from scripts.mvmd.mvmd_optimized import mvmd
from scripts.features import extract_features
```

---

## 📊 Results
Example results (synthetic MRW, FBM, modulated processes) are available under the `results/` folder.  
Animations and plots are generated directly from the notebooks.  

---

## 👩‍💻 Author
Developed by **Sabrine Bendimerad** as part of a research internship.
