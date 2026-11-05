"""Central configuration for CropDoc: Betel Leaf Disease Detection System.

All hyperparameters, file paths, model configurations, and thresholds are defined here
to ensure no magic numbers are scattered across the codebase.
"""

import os
from pathlib import Path

# Suppress TF oneDNN warnings and verbose logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SAMPLE_PREDICTIONS_DIR = RESULTS_DIR / "sample_predictions"
DOCS_DIR = BASE_DIR / "docs"
SRC_DIR = BASE_DIR / "src"

# Model Hyperparameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1

# Dataset Classes (will be verified dynamically from directory names during data loading)
DEFAULT_CLASS_NAMES = ["Healthy_Leaf", "Leaf_Rot", "Leaf_Spot"]

# Model File Path
MODEL_PATH = MODELS_DIR / "betel_disease_model.keras"

# Treatment Rules JSON File Path
TREATMENT_JSON_PATH = SRC_DIR / "treatment_rules.json"

# Severity Thresholds (percent of leaf surface area affected)
SEVERITY_THRESHOLDS = {
    "HEALTHY": 2.0,      # < 2% infected area is considered Healthy
    "MILD": 15.0,        # 2% - 15% infected area is Mild
    "MODERATE": 40.0,    # 15% - 40% infected area is Moderate
    # > 40% is Severe
}
