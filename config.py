import os
from pathlib import Path

# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Define the data directory
data_dir = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))

# Define the model directory
model_dir = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))

# Define the log directory
log_dir = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))

# Define the checkpoint directory
checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))

# Define the output directory
output_dir = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))

# Define the cache directory
cache_dir = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / "cache"))

# Define the results directory
results_dir = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results"))
