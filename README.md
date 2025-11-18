
# 🧠 MEG Signal Decomposition Project: VMD and MVMD Analysis

## 🎯 Project Overview


This repository contains the full pipeline for analyzing **Magnetoencephalography (MEG) resting-state data** using various decomposition methods: **Empirical Mode Decomposition (EMD)**, **Univariate Variational Mode Decomposition (VMD)**, and **Multivariate Variational Mode Decomposition (MVMD)**.

The overarching goal is to use these signal decomposition methods to see if extracting narrowband oscillatory **modes improves the Multifractal Analysis (MFA)** for detecting scale invariance in neural dynamics.

The **MFA** was specifically conducted using the **[pymultifracs library](https://github.com/neurospin/pymultifracs/tree/master/pymultifracs)**, which was developed inside the **MInd Team (CEA Neurospin)** by **Merlin Dumeur**.

### 🧪 Decomposition Rationale

Our work started by benchmarking three methods: EMD, VMD, and MVMD:

* **EMD:** Due to its lack of rigorous mathematical foundations, EMD was tested only on simulation data and subsequently **abandoned** for real-world application.
* **VMD:** VMD demonstrated better performance and was retained for analysis on both synthetic and real MEG data.
* **MVMD:** This is the **multivariate version of VMD**. It was adopted to see if, by analyzing the multi-channel neuro data, we could extract shared modes that **capture spatial interactions** across sensors, thereby reducing the need for channel-by-channel analysis.

### 👥 How to Use This Work

This repository supports two main use cases:

1.  **Reproduce My Internship Results:** Follow the setup and notebook steps to replicate the complete analysis pipeline.
2.  **Apply Methods to Your Own Dataset:** Use the provided starting notebooks to quickly test EMD, VMD, and MVMD on your own data.

---

## 🚀 Getting Started

### 1. 🐍 Setting up the Conda Environment

We recommend setting up the environment using Conda:

```bash
# Create the environment from my provided requirements.txt 
conda env create -f requirements.txt 
conda activate meg-vmd

````

### 2\. 📁 Data Acquisition (Crucial Step\!)

**⚠️ The necessary data files are NOT included in this repository.**

To run the analysis notebooks, you must **contact the project author** to obtain the required data files. Once obtained, please place them in the following directory structure:

| Required File | Description | Location to Place |
| :--- | :--- | :--- |
| `sub-01_ses-01_task-rest_proc-filt_raw.fif` | **Original MEG Data.** Required for notebooks `01_vmd_pipeline.ipynb` and `02_mvmd_on_realdata_pipeline.ipynb`. | `data/` |
| `mvmd_modes_sub-01.npz` | **Pre-computed MVMD Results.** Required to run `02_mvmd_on_realdata_pipeline.ipynb` (analysis notebook). | `results/real/MVMD/modes/` |

-----

## 💻 Option 1: Reproducing The Results of My Report

Once the Conda environment is active and the data files are in place, launch Jupyter Lab:

You can then run the following notebooks sequentially:

| Notebook | Focus | Dependencies |
| :--- | :--- | :--- |
| `00_simulation_pipeline.ipynb` | Synthetic Benchmark | None (Self-contained) |
| `01_vmd_pipeline.ipynb` | Univariate VMD Analysis | `data/sub-01...raw.fif` |
| `02_mvmd_on_realdata_pipeline.ipynb` | MVMD Results Analysis | `data/sub-01...raw.fif` and `results/real/MVMD/modes/mvmd_modes_sub-01.npz` |

-----

## 🛠️ Running Your Own MVMD Decomposition (Optional)

The MVMD decomposition is a computationally intensive process requiring specialized hardware. The main analysis notebook (`02_mvmd_on_realdata_pipeline.ipynb`) relies on **pre-computed results**.

If you wish to run the decomposition yourself to produce your own results:

1.  **Locate the Code:** The MVMD decomposition code is provided in the dedicated GPU directory:

    ```bash
    cd notebooks/notebooks_gpu/
    ```

2.  **Execution Requirement:** The notebook in this directory **requires GPU resources** (e.g., Google Colab Pro, dedicated server, or a local machine with a CUDA setup) to run successfully.

3.  **Output:** Running this notebook will generate a new MVMD results file (`mvmd_modes_sub-01.npz`), which you can then use to replace the pre-computed file in the `results/real/MVMD/modes/` folder before running the analysis notebook.

-----

## 💻 Option 2 : Applying Methods to Your Dataset 

If your goal is to quickly apply EMD, VMD, or MVMD to your own data, we have prepared simplified starting notebooks:

  * **Location:** Navigate to the `notebooks/extras/starting_notebooks` folder.
  * **Content:** Inside, you will find notebooks with simulation data to showcase the basic application and some analysis steps for EMD, VMD, and MVMD. You can easily adapt these notebooks to load and process your own time-series data.

