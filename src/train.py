"""Standalone Model Training Script for CropDoc.

Runnable via: python -m src.train
Loads dataset, builds MobileNetV2 architecture, compiles, applies training callbacks,
and saves the trained model artifact to models/betel_disease_model.keras.
"""

import os
import tensorflow as tf
from pathlib import Path
from config import MODEL_PATH, MODELS_DIR, EPOCHS, LEARNING_RATE
from src.data_loader import load_datasets
from src.model import build_model


def train_model():
    """Executes end-to-end model training pipeline."""
    print("=" * 60)
    print("CropDoc: Betel Leaf Disease Classifier Training")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/4] Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print(f"Class names detected ({num_classes}): {class_names}")

    # 2. Build & Compile Model
    print("\n[2/4] Building MobileNetV2 Transfer Learning Model...")
    model = build_model(num_classes=num_classes, learning_rate=LEARNING_RATE)
    model.summary()

    # 3. Configure Training Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Train Model
    print(f"\n[3/4] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model if checkpoint didn't trigger
    if not os.path.exists(MODEL_PATH):
        model.save(str(MODEL_PATH))

    print(f"\n[4/4] Model training complete! Saved model to: {MODEL_PATH}")
    return model, history, class_names


if __name__ == "__main__":
    train_model()
