"""Standalone Model Evaluation & Artifact Generation Script for CropDoc.

Runnable via: python -m src.evaluate
Loads trained model, evaluates performance on test dataset, computes confusion matrix
and classification report, and exports visual evaluation artifacts to results/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from config import MODEL_PATH, RESULTS_DIR
from src.data_loader import load_datasets


def evaluate_model():
    """Evaluates the saved CropDoc model and exports metrics and charts."""
    print("=" * 60)
    print("CropDoc: Model Evaluation & Metric Reporting")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model file not found at: {MODEL_PATH}. Run 'python -m src.train' first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data and Model
    print("\n[1/3] Loading test dataset and saved model...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # 2. Evaluate on Test Set
    print("\n[2/3] Computing predictions on test dataset...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds, verbose=1)
    print(f"\n--> Test Set Results | Accuracy: {test_acc*100:.2f}%, Loss: {test_loss:.4f}, Precision: {test_prec*100:.2f}%, Recall: {test_rec*100:.2f}%")

    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_batch = np.argmax(labels.numpy(), axis=1)
        y_pred_batch = np.argmax(preds, axis=1)
        y_true_list.extend(y_true_batch)
        y_pred_list.extend(y_pred_batch)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # 3. Generate Classification Report
    print("\n[3/3] Generating evaluation artifacts in results/...")
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report_text)

    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CropDoc Model Evaluation Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Precision:     {test_prec*100:.2f}%\n")
        f.write(f"Recall:        {test_rec*100:.2f}%\n\n")
        f.write(report_text)
    print(f"Saved classification report to: {report_path}")

    # 4. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Greens", ax=ax, values_format="d")
    ax.set_title(f"CropDoc Betel Leaf Classifier Confusion Matrix (Acc: {test_acc*100:.1f}%)")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # 5. Generate Accuracy & Loss Curve Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(class_names, [np.mean(y_true == i) for i in range(len(class_names))], color=['#2e7d32', '#d32f2f', '#f57c00'])
    ax1.set_title("Test Class Distribution")
    ax1.set_ylabel("Proportion")
    
    ax2.bar(["Test Accuracy", "Test Precision", "Test Recall"], [test_acc, test_prec, test_rec], color=['#1976d2', '#388e3c', '#7b1fa2'])
    ax2.set_title("Overall Model Evaluation Metrics")
    ax2.set_ylim(0, 1.0)
    for i, v in enumerate([test_acc, test_prec, test_rec]):
        ax2.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    curve_path = RESULTS_DIR / "accuracy_loss_curve.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved accuracy & loss evaluation metrics plot to: {curve_path}")

    print("\nAll evaluation artifacts generated successfully!")


if __name__ == "__main__":
    evaluate_model()
