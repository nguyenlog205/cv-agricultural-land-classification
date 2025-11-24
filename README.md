# Capstone Project: Geospatial Agricultural Land Classification
## 1. Project Description
> This project develops a complete, end-to-end Deep Learning pipeline for classifying agricultural land cover from satellite imagery. The goal is to benchmark the performance of traditional Convolutional Neural Networks (CNNs) enhanced by Transfer Learning against cutting-edge Vision Transformers (ViTs) on a specialized geospatial dataset.

### 1.1. General Approach
The core solution is built around the PyTorch framework, emphasizing modularity, data efficiency, and rigorous performance evaluation.

### 1.2. Technical Stack
- **Core Framework**: PyTorch, PyTorch DataLoaders
- **Computer Vision**: torchvision.models (ResNet-50, ViT), OpenCV (cv2)
- **Data Augmentation**: Albumentations
- **Configuration**: YAML (for streamlined configuration management)
- **Tools**: Git LFS (for managing large model checkpoints)

| Feature | Description | Implemented Files|
|-|-|-|
Data Acquisition | Automated downloading and extraction of the raw satellite image dataset from the specified URL: [Satellite Image Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar). | `src/utils.py` |
| Advanced Data Preprocessing | Custom PyTorch Dataset (GeoDataset) for loading raw image data and label indexing. | `src/data_module.py` |
| Robust Data Augmentation | Implementation of a comprehensive, production-ready augmentation pipeline using the Albumentations library, including Geometric (Flip/Rotate), Color (Brightness/Contrast), Noise (ISO Noise), and Dropout techniques to enhance model generalization. | `src/data_module.py` |
| Model Architecture | Utilized Transfer Learning on a pre-trained **ResNet-50** model (ImageNet weights [cite: uploaded:src/model.py]) by freezing feature layers and modifying the final fully-connected head for agricultural classification. Also included a framework for benchmarking Vision Transformers. | `src/model.py` |
| Training Pipeline | A structured training loop (PyTorch Lightning style) manages epochs, loss calculation (CrossEntropyLoss), optimization (Adam), and checkpointing of the best model based on validation accuracy. | `src/train.py` |
| Performance | Achieved a peak validation **accuracy of 96.5%** with the fine-tuned ResNet-50 model. | |


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

### 4.1. If using Docker
> Ensure you have both `environment.yml` and `Dockerfile` before running this script.

#### 4.1.1. Build image
```powershell
docker build -t capstone-geospatial-ai-image .
```
where:
| **Parameter** | **Description**|
|----------------|------------|
|`docker build`  |Official command of Docker to read `Dockerfile` and build Image|
|`-t capstone-geospatial-ai-image`| Set **tag** and name for current Image|
|`.`| Indicate **Context Build** where contains `Dockerfile` (is the current directory)|

## REFERENCES
[1]. Sateline Image Dataset, https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar