

## 2. Prerequisite
### 2.1. Software
- **OS:** Linux (Ubuntu 20.04+) or Windows 10/11 with WSL2 (Recommended).
- **Python:** Version `3.10.xx` (Strictly recommended `3.10.11` to avoid dependency conflicts).
- **Package Managers:**
  - [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Required for geospatial libraries).
  - [Docker Desktop](https://www.docker.com/) & Docker Compose (If running via containers).
- **Git:** Latest version.

### 2.2. Hardware (Optional but recommend)
- **GPU:** NVIDIA GPU with minimum 4GB VRAM (6GB+ recommended).
  - **Driver:** Requires NVIDIA Driver version **>= 531.14** (on Windows) to support CUDA 12.1.
- **RAM:** Minimum 16GB (Required for loading large satellite/aerial imagery datasets).

### 2.3. System Libraries (If not using Docker)
- **CUDA Toolkit:** Version `12.1` (if using GPU).

## 3. Repository Structure
```bash
CV-AGRICULTURAL-LAND-CLASSIFICATION/
├── data/
│   ├── raw/                 # Raw Dataset
│   └── processed/           # Processed Dataset
│
├── src/
│   ├── __init__.py          
│   ├── data_module.py       # Data Preprocessing, Data Augmenting, Feature Engineering and Data Loader
│   ├── models.py            # Model Implementation
│   ├── train.py             # Script for training and evaluating works
│   ├── evaluate.py          # Scripts for final assessment on test dataset via metrics (F1, AU-ROC).
│   └── utils.py             # Common utilities
│
├── notebooks/               # Jupyter Notebooks for EDA and quick experiments
│
├── configs/                 # Hyperparameter Configuration
│   └── hyperparams.yaml
│
├── results/
│   ├── models/              # Model checkpoints (.pt)
│   └── logs/                # Training logging and history
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── environment.yml
├── README.md
└── requirements.txt
```

## 4. Workflow Description
### 1. Create Conda environment
Run the below script after navigating to the current project.
```bash
conda env create -f environment.yml
```
For example:
```powershell
PS C:\Users\VICTUS\developer\cv-agricultural-land-classification> conda env create -f environment.yml
```
