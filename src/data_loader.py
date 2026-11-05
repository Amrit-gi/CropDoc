"""Data Loading, Split, and Augmentation Module for CropDoc.

Handles reading betel leaf image directories, train/val/test splits,
data augmentation pipeline, and class mapping discovery.
"""

import os
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from config import RAW_DATA_DIR, SAMPLE_IMAGES_DIR, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT

# Suppress TF memory allocation issues & C++ logs at top level
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def find_actual_data_root(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Finds the directory that directly contains class subfolders (e.g. Healthy_Leaf, Leaf_Rot, Leaf_Spot)."""
    raw_path = Path(raw_dir)
    
    # 1. Direct check inside raw_dir
    if os.path.exists(raw_path):
        # Look for Original_Dataset or Augmented_Dataset subfolder
        for sub in ["Original_Dataset", "Augmented_Dataset"]:
            candidate = raw_path / "Betel Leaf Dataset" / sub
            if os.path.exists(candidate):
                subclasses = [c for c in os.listdir(candidate) if os.path.isdir(candidate / c) and not c.startswith('.')]
                if len(subclasses) >= 2:
                    return candidate

            candidate2 = raw_path / sub
            if os.path.exists(candidate2):
                subclasses = [c for c in os.listdir(candidate2) if os.path.isdir(candidate2 / c) and not c.startswith('.')]
                if len(subclasses) >= 2:
                    return candidate2

        classes = [c for c in os.listdir(raw_path) if os.path.isdir(raw_path / c) and not c.startswith('.')]
        if len(classes) >= 2:
            return raw_path

        # Recursively search 2 levels deep for folders with >= 2 subfolders
        for root, dirs, files in os.walk(raw_path):
            subclasses = [d for d in dirs if not d.startswith('.')]
            if len(subclasses) >= 2 and any(k in "".join(subclasses).lower() for k in ["rot", "spot", "healthy", "leaf"]):
                return Path(root)

    return raw_path


def get_class_names(raw_dir: Path = RAW_DATA_DIR) -> List[str]:
    """Discovers subfolder class names in raw dataset directory.

    Args:
        raw_dir: Path to raw dataset directory.

    Returns:
        Sorted list of class directory names.
    """
    data_root = find_actual_data_root(raw_dir)
    if os.path.exists(data_root):
        classes = [c for c in os.listdir(data_root) if os.path.isdir(data_root / c) and not c.startswith('.')]
        if classes:
            return sorted(classes)
    return ["Healthy_Leaf", "Leaf_Rot", "Leaf_Spot"]


def create_dataset_splits(
    raw_dir: Path = RAW_DATA_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = 42
):
    """Loads images from dataset directory and splits into stratified Train, Validation, and Test tf.data.Datasets.

    Lazy loads TensorFlow to prevent top-level memory allocation issues during app startup.
    """
    import tensorflow as tf

    data_root = find_actual_data_root(raw_dir)
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Raw dataset directory not found at {raw_dir}. Please place dataset in data/raw/")

    class_names = get_class_names(raw_dir)

    # 1. Load full dataset using image_dataset_from_directory
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    total_batches = tf.data.experimental.cardinality(full_ds).numpy()

    if total_batches <= 2:
        val_batches = max(1, int(total_batches * val_split))
        test_batches = max(1, int(total_batches * test_split))
        train_batches = max(1, total_batches - val_batches - test_batches)
    else:
        val_batches = int(total_batches * val_split)
        test_batches = int(total_batches * test_split)
        train_batches = total_batches - val_batches - test_batches

    train_ds = full_ds.take(train_batches)
    val_and_test_ds = full_ds.skip(train_batches)
    val_ds = val_and_test_ds.take(val_batches)
    test_ds = val_and_test_ds.skip(val_batches)

    # Performance optimization: prefetch & autotune
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def get_data_augmentation() -> Any:
    """Returns Keras Data Augmentation sequential layer block for training robust models."""
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def extract_sample_images(raw_dir: Path = RAW_DATA_DIR, output_dir: Path = SAMPLE_IMAGES_DIR, num_samples_per_class: int = 3):
    """Copies sample images per class to sample_images directory for easy demonstration and testing."""
    data_root = find_actual_data_root(raw_dir)
    if not os.path.exists(data_root):
        return

    os.makedirs(output_dir, exist_ok=True)
    class_names = get_class_names(raw_dir)

    for cname in class_names:
        c_src_dir = data_root / cname
        c_dst_dir = Path(output_dir) / cname
        os.makedirs(c_dst_dir, exist_ok=True)

        if os.path.exists(c_src_dir):
            files = [f for f in os.listdir(c_src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in files[:num_samples_per_class]:
                src_file = c_src_dir / f
                dst_file = c_dst_dir / f
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
