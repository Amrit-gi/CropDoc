"""CropDoc End-to-End Diagnostic Pipeline Module.

Unified entry point chaining image validation, model prediction,
severity estimation, Grad-CAM explainability, and treatment lookup.
"""

import os
# Suppress TF oneDNN memory allocation issues & verbose C++ logs at top level
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from typing import Dict, Any, Union, Optional, Tuple
from pathlib import Path

from config import MODEL_PATH, DEFAULT_CLASS_NAMES, IMAGE_SIZE
from src.validate_image import validate_image
from src.severity import estimate_severity
from src.explainability import generate_gradcam, overlay_gradcam
from src.treatment import get_treatment_recommendation
from src.data_loader import get_class_names

# Global cached model instance to prevent re-loading per request
_CACHED_MODEL: Any = None
_CLASS_NAMES: Optional[list] = None


def get_or_load_model() -> Tuple[Any, list]:
    """Retrieves cached model or loads from disk with lazy TensorFlow loading."""
    global _CACHED_MODEL, _CLASS_NAMES
    if _CACHED_MODEL is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train model first using 'python -m src.train'")
        _CACHED_MODEL = tf.keras.models.load_model(str(MODEL_PATH))
        _CLASS_NAMES = get_class_names()
    return _CACHED_MODEL, _CLASS_NAMES or DEFAULT_CLASS_NAMES


def diagnose_leaf(
    image_input: Union[str, bytes, np.ndarray],
    model: Optional[Any] = None,
    class_names: Optional[list] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """Runs the complete CropDoc diagnostic pipeline on an input betel leaf image.

    Chains:
    1. Validation (species keyword filter, format check, blank image, cordate morphology, venation)
    2. Preprocessing & Normalization
    3. MobileNetV2 Disease Classification with Out-of-Distribution (OOD) Guard
    4. Color-Thresholding Severity Scoring
    5. Grad-CAM Activation Heatmap Overlay
    6. Rule-based Treatment Recommendation Lookup

    Args:
        image_input: Image file path, raw bytes, or BGR numpy array.
        model: Optional pre-loaded model instance.
        class_names: Optional class names list.
        filename: Optional uploaded filename for species blacklist validation.

    Returns:
        Structured diagnosis dictionary containing validation status, predicted disease,
        confidence, severity percentage/level, Grad-CAM overlays, and treatment guidelines.
    """
    # 1. Botanical and Image Validation
    is_valid, err_msg, image_bgr = validate_image(image_input, filename=filename)
    if not is_valid or image_bgr is None:
        return {
            "is_valid": False,
            "error_message": err_msg,
            "disease_predicted": None,
            "confidence": 0.0,
            "severity_percent": 0.0,
            "severity_level": None,
            "treatment": None
        }

    # Load model if not supplied
    if model is None or class_names is None:
        loaded_model, loaded_classes = get_or_load_model()
        model = model or loaded_model
        class_names = class_names or loaded_classes

    # 2. Preprocess image for classification model
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, IMAGE_SIZE)
    img_array = np.expand_dims(resized_img.astype(np.float32), axis=0)

    # 3. Model Classification
    preds = model.predict(img_array, verbose=0)[0]
    top_class_idx = int(np.argmax(preds))
    confidence = float(preds[top_class_idx])
    disease_predicted = class_names[top_class_idx]

    all_probabilities = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    # Out-of-Distribution (OOD) Softmax Uncertainty Filter:
    # If the model prediction is split and maximum confidence is low, reject as non-target specimen
    if confidence < 0.50:
        return {
            "is_valid": False,
            "error_message": f"AI Verification Rejected: Model classification confidence is low ({confidence*100:.1f}%) "
                            f"with high distributional entropy. The visual features do not match genuine Piper betle pathology.",
            "disease_predicted": None,
            "confidence": confidence,
            "severity_percent": 0.0,
            "severity_level": None,
            "treatment": None
        }

    # 4. Severity Estimation
    severity_res = estimate_severity(image_bgr)
    severity_percent = severity_res["severity_percent"]
    severity_level = severity_res["severity_level"]

    # 5. Grad-CAM Heatmap Generation
    try:
        heatmap = generate_gradcam(model, img_array, class_index=top_class_idx)
        gradcam_overlay = overlay_gradcam(image_bgr, heatmap)
    except Exception as e:
        print(f"Warning: Grad-CAM generation encountered issue: {e}")
        heatmap = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)
        gradcam_overlay = image_bgr.copy()

    # 6. Treatment Recommendation Lookup
    treatment = get_treatment_recommendation(disease_predicted, severity_level)

    return {
        "is_valid": True,
        "disease_predicted": disease_predicted,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "severity_percent": severity_percent,
        "severity_level": severity_level,
        "original_bgr": image_bgr,
        "gradcam_overlay": gradcam_overlay,
        "gradcam_heatmap": heatmap,
        "treatment": treatment
    }
