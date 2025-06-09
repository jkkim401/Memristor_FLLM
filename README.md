# Federated LLM Fine-tuning Project (BitNet + GainCell + FL)

This project implements a federated learning system to fine-tune a Large Language Model (LLM) based on Microsoft's BitNet architecture, incorporating Gain-Cell Attention, on the MIMIC-IV-Ext-CDM medical dataset.

## Project Structure

```
/FLLM Med
├── src/                  # Source code
│   ├── __init__.py
│   ├── quantization.py     # BitNet W1.58A8 quantization logic
│   ├── gain_cell_attention.py # Gain-Cell Attention layer
│   ├── model.py            # Main BitNet+GainCell model definition
│   ├── data_utils.py       # Data loading, preprocessing, partitioning
│   ├── fl_client.py        # Flower client implementation (LoRA fine-tuning)
│   └── fl_server.py        # Flower server implementation (FedAvg strategy, simulation)
├── scripts/              # Execution scripts
│   └── main.py           # Main script to run the FL simulation
├── data/                 # Placeholder for MIMIC-IV-Ext-CDM CSV files (requires user download)
├── configs/              # Placeholder for configuration files (if any)
├── results/              # Placeholder for saving simulation results/logs
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup (Windows Environment)

1.  **Prerequisites:**
    *   **Python:** Ensure you have Python installed (version 3.9 or higher recommended). You can download it from [python.org](https://www.python.org/). Add Python to your system's PATH during installation.
    *   **Git:** Install Git from [git-scm.com](https://git-scm.com/) to clone the repository.
    *   **C++ Build Tools:** Some Python packages (especially those involving custom C++/CUDA extensions like `bitsandbytes`) require C++ build tools. Install **Microsoft C++ Build Tools**, which are part of Visual Studio. You can get them via the Visual Studio Installer:
        *   Download the [Visual Studio Installer](https://visualstudio.microsoft.com/downloads/).
        *   Run the installer and select the "Workloads" tab.
        *   Choose the "Desktop development with C++" workload.
        *   Ensure "MSVC... build tools" and "Windows SDK" are selected.
        *   Click "Install".
    *   **(Optional but Recommended) Conda:** Using Conda can help manage environments and dependencies. Install Miniconda or Anaconda from [docs.conda.io](https://docs.conda.io/projects/miniconda/en/latest/).
    *   **(Optional) CUDA Toolkit:** If you have an NVIDIA GPU and want to use GPU acceleration, install the NVIDIA CUDA Toolkit that matches the version required by PyTorch and `bitsandbytes`. Check the PyTorch installation instructions ([pytorch.org](https://pytorch.org/)) for compatible CUDA versions.

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd federated_llm
    ```

3.  **Create Virtual Environment (Recommended):**
    *   **Using venv:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **Using Conda:**
        ```bash
        conda create -n federated_llm python=3.10 # Or your preferred Python version
        conda activate federated_llm
        ```

4.  **Install Dependencies:**
    *   Upgrade pip:
        ```bash
        python -m pip install --upgrade pip
        ```
    *   Install PyTorch: Visit [pytorch.org](https://pytorch.org/) and select the appropriate command for your system (Windows, Pip, desired CUDA version or CPU). Example for CUDA 12.1:
        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
        Example for CPU only:
        ```bash
        pip3 install torch torchvision torchaudio
        ```
    *   Install other requirements:
        ```bash
        pip install -r requirements.txt
        ```
        *Note: `bitsandbytes` installation on Windows might require specific steps or pre-compiled binaries if the standard pip install fails. Check the [bitsandbytes GitHub repository](https://github.com/TimDettmers/bitsandbytes) for Windows-specific instructions if you encounter issues.* 

5.  **Download MIMIC-IV-Ext-CDM Data:**
    *   You **must** obtain access to the MIMIC-IV dataset via PhysioNet ([https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)). This involves completing required training (e.g., CITI Program) and signing a data use agreement.
    *   Once you have access, download the MIMIC-IV-Ext-CDM CSV files ([https://physionet.org/content/mimic-iv-ext-cdm/1.0/](https://physionet.org/content/mimic-iv-ext-cdm/1.0/)).
    *   Place the required CSV files (e.g., `history_of_present_illness.csv`, `physical_examination.csv`, `discharge_diagnosis.csv`) into the `/home/ubuntu/federated_llm/data` directory (or the directory specified by `--data_dir`).

## Running the Simulation

1.  **Activate your virtual environment** (if not already active).
2.  **Navigate to the `scripts` directory:**
    ```bash
    cd scripts
    ```
3.  **Run the main script:**
    ```bash
    python main.py --num_clients 16 --num_rounds 10 --batch_size 8
    ```
    *   Adjust arguments as needed:
        *   `--num_clients`: Number of federated clients.
        *   `--num_rounds`: Number of federated learning rounds.
        *   `--batch_size`: Batch size used by clients during training.
        *   `--data_dir`: Path to the directory containing MIMIC CSV files.

## Code Overview

*   **`quantization.py`:** Implements 1.58-bit weight and 8-bit activation quantization.
*   **`gain_cell_attention.py`:** Implements the Gain-Cell Attention mechanism adapted for BitNet.
*   **`model.py`:** Defines the complete LLM architecture combining BitNet, Gain-Cell Attention, and quantization.
*   **`data_utils.py`:** Handles loading, preprocessing, and partitioning of the MIMIC-IV-Ext-CDM dataset.
*   **`fl_client.py`:** Defines the Flower client using LoRA for efficient fine-tuning.
*   **`fl_server.py`:** Defines the Flower server, FedAvg aggregation strategy, and simulation setup.
*   **`main.py`:** Entry point script to start the federated learning simulation.

