"""Grad-CAM (Gradient-weighted Class Activation Mapping) Explainability Module.

Generates visual explanations for model predictions by highlighting regions of interest
(lesions, spots, discoloration) on the input betel leaf image.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from typing import Tuple, Optional, Any


def generate_gradcam(
    model: Any,
    img_array: np.ndarray,
    class_index: Optional[int] = None,
    layer_name: str = "Conv_1"
) -> np.ndarray:
    """Generates Grad-CAM activation heatmap for a given input image array.

    Handles both single-level models and nested Functional backbone architectures (e.g. MobileNetV2).

    Args:
        model: Loaded Keras classifier model.
        img_array: Preprocessed image tensor array of shape (1, 224, 224, 3).
        class_index: Target class index. If None, auto-selects top predicted class index.
        layer_name: Target conv layer name in MobileNetV2 backbone (default 'Conv_1').

    Returns:
        2D float numpy array [0, 1] representing activation heatmap intensity.
    """
    import tensorflow as tf

    # Find base backbone sub-model if nested
    base_model = None
    top_layers = []
    
    for l in model.layers:
        if hasattr(l, "layers") and len(l.layers) > 10:
            base_model = l
        elif base_model is not None:
            top_layers.append(l)

    if base_model is not None:
        # Build intermediate model mapping base input to [conv_target_output, base_output]
        try:
            target_conv = base_model.get_layer(layer_name)
        except Exception:
            target_conv = [l for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]

        grad_base_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[target_conv.output, base_model.output]
        )

        # Preprocessing scaling for MobileNetV2 [-1, 1]
        preprocessed_input = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img_array, tf.float32))

        with tf.GradientTape() as tape:
            conv_outputs, base_out = grad_base_model(preprocessed_input)
            tape.watch(conv_outputs)
            
            # Forward pass through top classification head
            x = base_out
            for layer in top_layers:
                x = layer(x)
            predictions = x

            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    else:
        # Direct single-level model handling
        target_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

    # Apply ReLU and normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    return heatmap.numpy()


def overlay_gradcam(
    original_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """Overlays a 2D normalized Grad-CAM heatmap onto the original BGR image using JET colormap.

    Args:
        original_bgr: Original input image in BGR format (H, W, 3) uint8.
        heatmap: 2D float array [0, 1] representing activation intensity.
        alpha: Blending weight for heatmap overlay (0.0 to 1.0).

    Returns:
        Blended BGR image (H, W, 3) uint8.
    """
    h, w = original_bgr.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend heatmap with original BGR image
    blended = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

    return blended
