
# 🧠 MEG Signal Decomposition Project: VMD and MVMD Analysis

## 🎯 Project Overview

This repository provides a complete pipeline for analyzing **Magnetoencephalography (MEG) resting-state data** using several signal decomposition methods:

* **Empirical Mode Decomposition (EMD)**
* **Univariate Variational Mode Decomposition (VMD)**
* **Multivariate Variational Mode Decomposition (MVMD)**

The primary objective of this project is to investigate whether extracting narrowband oscillatory **modes** can improve **Multifractal Analysis (MFA)**, particularly for identifying scale-invariant properties in neural dynamics.

The **MFA** procedures rely on the **[pymultifracs library](https://github.com/neurospin/pymultifracs/tree/master/pymultifracs)**, developed within the **Mind Team (CEA NeuroSpin)** by **Merlin Dumeur**.

---

## 🧪 Decomposition Rationale

This project evaluates three decomposition strategies:

* **EMD:** Due to its lack of rigorous mathematical foundations, EMD was used only on simulated data and was not retained for real MEG analysis.

* **VMD:** VMD demonstrated strong performance and was applied to both simulated and real-world data.

* **MVMD:** As the multivariate extension of VMD, MVMD aims to identify shared oscillatory modes across multiple MEG sensors, potentially capturing **spatial interactions** and reducing the need for channel-wise decomposition.

---

## 👥 How to Use This Repository

This project supports two main use cases:

1. **Reproduce the Internship Results**
   Follow the installation steps and run the notebooks in order to replicate the full analysis pipeline.

2. **Apply EMD, VMD, or MVMD to Your Own Dataset**
   Use the simplified starting notebooks to quickly test these decomposition methods on custom time-series data.

---

# 🚀 Getting Started

## 1. 🐍 Setting up the Conda Environment

We recommend creating the environment via Conda:

```bash
# Create the environment from the provided requirements.txt
conda env create -f requirements.txt
conda activate meg-vmd
```

---

## 2. 📁 Data Acquisition (Important)

**⚠️ The required data files are NOT included in this repository.**

To run the main analysis notebooks, please contact the project maintainers to obtain the necessary files. Once acquired, place them in the following directory structure:

| Required File                               | Description                                       | Destination Folder         |
| ------------------------------------------- | ------------------------------------------------- | -------------------------- |
| `sub-01_ses-01_task-rest_proc-filt_raw.fif` | Raw MEG recording used for VMD and MVMD pipelines | `data/`                    |
| `mvmd_modes_sub-01.npz`                     | Pre-computed MVMD results for analysis            | `results/real/MVMD/modes/` |

---

# 💻 Option 1: Reproducing the Report Results

After activating the environment and placing the data in the appropriate folders, start Jupyter Lab and run the notebooks in order:

| Notebook                             | Description                | Dependencies               |
| ------------------------------------ | -------------------------- | -------------------------- |
| `00_simulation_pipeline.ipynb`       | Synthetic data benchmark   | None                       |
| `01_vmd_pipeline.ipynb`              | Univariate VMD on real MEG | `data/sub-01...raw.fif`    |
| `02_mvmd_on_realdata_pipeline.ipynb` | Analysis of MVMD results   | `results/real/MVMD/modes/` |

---

### 🛠️ Running Your Own MVMD Decomposition if you dont want to use my result

MVMD decomposition is computationally demanding and requires a GPU.

If you want to generate your own MVMD modes:

1. Navigate to the GPU-dedicated notebook:

   ```bash
   cd notebooks/notebooks_gpu/
   ```

2. Run the notebook using a machine with **GPU support** (Google Colab Pro, a CUDA-equipped workstation, or a server).

3. The notebook will output a file such as:

   ```
   mvmd_modes_sub-01.npz
   ```

   Place this file in:

   ```
   results/real/MVMD/modes/
   ```

   and rerun the analysis notebook (`02_mvmd_on_realdata_pipeline.ipynb`).

---

# 💻 Option 2: Applying Methods to Your Own Dataset

To quickly test EMD, VMD, or MVMD on your own dataset, use the starter notebooks:

* **Path:** `notebooks/extras/starting_notebooks/`
* **Content:**
  These notebooks contain simplified simulation examples and minimal workflows that can be easily adapted to custom time-series.

---

# 📬 Contact

For additional information:

* **Pipeline, notebooks, and analysis workflow:**
  **Sabrine Bendimerad**

* **Access to the MEG data used in this project:**
  **Philippe Ciuciu**

* **Multifractal Analysis (MFA) and the `pymultifracs` library:**
  **Merlin Dumeur**


