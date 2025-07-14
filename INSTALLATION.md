# Installation Guide

This document provides step-by-step instructions for setting up the environment for the **SAR Eddy Detection Demo** project. There are two methods to create your environment: using Conda (recommended) or using Python’s built-in `venv` and `pip`. Choose the option that best suits your workflow.

---

## Option 1: Using Conda (Recommended)

Using Conda is recommended because it simplifies the installation of scientific packages and handles binary dependencies smoothly.

### 1. Install Conda

If you don’t have Conda installed, we suggest using [Miniforge](https://conda-forge.org/miniforge/). Follow the instructions below for your operating system:

- **Linux (x86_64):**
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh
  ```
- **Linux (aarch64):**
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash Miniforge3-Linux-aarch64.sh
  ```
- **macOS (Intel):**
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
  bash Miniforge3-MacOSX-x86_64.sh
  ```
- **macOS (Apple Silicon):**
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
  bash Miniforge3-MacOSX-arm64.sh
  ```
- **Windows (x86_64):**
  Download the [Miniforge installer for Windows](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe) and run the installer by double-clicking it. Follow the on-screen instructions.

### 2. Create and Activate a Conda Environment

Open your terminal (or Anaconda Prompt on Windows) and run the following commands to create a new environment named `sar_eddy_env` with Python 3.10:

```bash
conda create -n sar_eddy_env python=3.10
conda activate sar_eddy_env
```

### 3. Install Required Packages

Install the necessary dependencies using Conda from the `conda-forge` channel:

```bash
conda install pandas rasterio libgdal geopandas shapely tqdm pyyaml scikit-learn xgboost -c conda-forge
conda install pytorch torchvision -c conda-forge
conda install timm -c conda-forge
```

**Platform-specific Notes:**
- **Windows Users:** If you experience issues with `rasterio` via Conda, try installing it with pip:
  ```bash
  pip install rasterio
  ```
- **macOS (Apple Silicon):** For better performance on M1/M2/M3 Macs, install the accelerated `libblas` library:
  ```bash
  conda install "libblas-*=*accelerate" -c conda-forge
  ```
- **GPU Support for PyTorch:** The above commands install the CPU version of PyTorch. For GPU support, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to select the appropriate command based on your CUDA version.

---

## Option 2: Using venv and pip

If you prefer not to use Conda, you can set up your environment with Python’s built-in `venv` and `pip`.

### 1. Create a Virtual Environment

Open your terminal and navigate to the repository root. Run:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Upgrade pip and Install Dependencies

It’s recommended to upgrade pip first, then install the required packages using the provided `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Optional:* To install the project in editable mode (so changes in `src` are immediately reflected), run:

```bash
pip install -e .
```

---

## Verifying the Installation

After setting up your environment using either method, verify the installation by running the demo pipeline:

```bash
python src/main.py --config config/inference.yaml
```

This command should execute the demo and output logs to the console, confirming that all dependencies are correctly installed.

---

## Troubleshooting

- **Python Version:** Ensure you are using Python 3.10 (or a compatible version).
- **Internet Connection:** Verify that you have an active internet connection during installation.
- **Platform-specific Issues:** Review any error messages for hints on missing dependencies or conflicts. Consult the relevant documentation links if necessary.
