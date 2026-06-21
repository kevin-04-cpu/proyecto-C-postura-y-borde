<div align="center">
  <img src="images_report/logo.png" alt="Logo" width="80" height="80">

  <h3 align="center">Computer vision for posture detection</h3>

  <p align="center">
    Hybrid CPU/GPU architecture using OpenMP and CUDA for classification from scratch
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#project-context">Project Context</a></li>
        <li><a href="#pipeline-architecture">Pipeline Architecture</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-and-setup">Installation and Setup</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage-and-execution">Usage and Execution</a>
      <ul>
        <li><a href="#stage-1-openmp-preprocessing">Stage 1: OpenMP Preprocessing</a></li>
        <li><a href="#stage-2-cuda-training">Stage 2: CUDA Training</a></li>
        <li><a href="#stage-3-streamlit-app">Stage 3: Streamlit App</a></li>
      </ul>
    </li>
    <li><a href="#performance-report-link">Performance Report & Benchmarks</a></li>
  </ol>
</details>

<br>
 
## About The Project

### Project Context
This project implements an end-to-end image classification pipeline built from scratch to detect sitting posture (slouched vs. upright). Instead of utilizing pre-trained deep learning frameworks, the system handles the entire lifecycle: local dataset creation, multi-core CPU preprocessing, custom GPU neural network training, and real-time deployment.  

The optimization goal focuses on benchmarking parallel computing patterns, measuring speedup, and analyzing hardware utilization limits across serial and parallel executions.

### Pipeline Architecture
The system is divided into three distinct execution phases:

![Pipeline del proyecto](images_report/pipeline.png)

### Built With

<table>
  <tr>
    <td align="center" width="150">
      <img src="https://img.shields.io/badge/C-A8B9CC?style=for-the-badge&logo=c&logoColor=white" alt="C"><br>
      <img src="https://img.shields.io/badge/OpenMP-5CA5E6?style=for-the-badge&logo=openmp&logoColor=white" alt="OpenMP">
    </td>
    <td>
      Handles low-level multi-core data-parallel execution loops over the image directory.
    </td>
  </tr>
  <tr>
    <td align="center" width="150">
      <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
    </td>
    <td>
      Manages hardware-accelerated thread scheduling, global/shared memory allocation, and matrix execution kernels on the GPU.
    </td>
  </tr>
  <tr>
    <td align="center" width="150">
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"><br>
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    </td>
    <td>
      Runs the quick prototyping layout for user interaction and real-time inference prediction.
    </td>
  </tr>
</table>
<br>

## Getting Started

### Prerequisites
Before setting up and executing this project, ensure your system meets the following hardware and software requirements:

* **Operating System**: Linux (Ubuntu 20.04 LTS or later recommended) or Windows with WSL2 configured.
* **Compiler and Flags**: `gcc` with OpenMP support (e.g., `-fopenmp`).
* **CUDA Toolkit**: NVIDIA CUDA Toolkit (v11.0 or higher) matching your GPU driver capability.
* **Python Environment**: Python 3.8 or higher.

### Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Daxdzzzy/proyecto-C-postura-y-borde
   cd computer-vision-posture-detection
   ```

2. **Download the Dataset**
Since the raw image files are excluded from this repository via `.gitignore` due to size and composition constraints, you must download the structured dataset manually.
* Download the dataset from the following link: [DOWNLOAD_DATASET_PLACEHOLDER]
* Extract the contents into the root directory of the project. Ensure the folder structure matches:

    ```text
    ├── dataset/
    │   ├── class_1_slouched/
    │   └── class_0_upright/

    ```

3. **Install Python Dependencies**
Configure the virtual environment and install the packages required for the Stage 3 graphical interface:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install streamlit numpy pillow

    ```

## Usage and Execution

This project is executed in three sequential stages. You must follow this order to properly process the data, train the model, and run the interface.

* **Stage 1: Preprocessing (OpenMP)** — Processes independent image files in a parallelized loop to convert inputs into grayscale, apply silhouette edge detection filters, resize structures to $64 \times 64$ pixels, and output a flattened matrix to disk.

* **Stage 2: Model Training (CUDA)** — Loads the flattened dataset to train a Multi-Layer Perceptron (MLP) using explicit GPU kernels for matrix multiplication, bias addition, activations (ReLU/Sigmoid), Binary Cross-Entropy loss computation, and backpropagation.

* **Stage 3: Application (Streamlit)** — A lightweight deployment application that loads the raw binaries or NumPy weight matrices to run inference on new webcam captures.

