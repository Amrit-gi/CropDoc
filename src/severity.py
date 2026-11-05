"""Color-Thresholding Leaf Disease Severity Estimator.

This module provides standalone, explainable severity estimation for betel leaf images
using HSV color space segmentation.

METHODOLOGY:
1. Converts the RGB input image to HSV (Hue, Saturation, Value) color space.
2. Identifies total leaf region by thresholding out background (green leaf + lesion tissues).
3. Segments healthy green leaf tissue and diseased lesion tissue (Leaf Rot / Leaf Spot).
4. Calculates infection ratio:
   Infection Ratio (%) = (Lesion Pixel Count / Total Leaf Surface Pixel Count) * 100
5. Maps percentage to severity categories:
   - Healthy (< 2.0%)
   - Mild (2.0% - 15.0%)
   - Moderate (15.0% - 40.0%)
   - Severe (> 40.0%)

LIMITATIONS & TRADE-OFFS:
- Assumes the input image contains a leaf on a contrasting background.
- Lighting conditions, strong shadows, or glares may shift HSV color channels.
- While lightweight and fast (suitable for mobile deployment without heavy segmentation networks),
  it relies on color contrasts rather than semantic spatial context.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
from config import SEVERITY_THRESHOLDS


def estimate_severity(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Estimates the disease severity percentage and category for a betel leaf image.

    Args:
        image_bgr: Input image in OpenCV BGR format (H, W, 3).

    Returns:
        Dict containing:
            - 'severity_percent': float (0.0 to 100.0)
            - 'severity_level': str ('Healthy', 'Mild', 'Moderate', 'Severe')
            - 'leaf_mask': binary mask of total leaf region
            - 'lesion_mask': binary mask of detected disease lesions
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid or empty image provided to estimate_severity.")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 1. Segment Healthy Green Tissue
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2. Segment Lesion Regions (Brown, Yellow, Dark Necrotic Rot/Spot)
    lower_brown = np.array([0, 30, 20])
    upper_brown = np.array([25, 255, 220])
    lesion_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    lesion_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    combined_lesion = cv2.bitwise_or(lesion_brown, lesion_dark)

    # 3. Total Leaf Surface = Healthy Green + Lesions
    leaf_mask = cv2.bitwise_or(green_mask, combined_lesion)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.bitwise_and(combined_lesion, combined_lesion, mask=leaf_mask)

    leaf_pixel_count = int(np.count_nonzero(leaf_mask))
    lesion_pixel_count = int(np.count_nonzero(lesion_mask))

    if leaf_pixel_count == 0:
        # Fallback if leaf mask didn't isolate pixels: use total image area
        leaf_pixel_count = image_bgr.shape[0] * image_bgr.shape[1]
        leaf_mask = np.ones((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8) * 255

    # Calculate severity ratio
    severity_percent = min(100.0, (lesion_pixel_count / leaf_pixel_count) * 100.0)

    # Categorize severity level
    if severity_percent < SEVERITY_THRESHOLDS["HEALTHY"]:
        severity_level = "Healthy"
    elif severity_percent < SEVERITY_THRESHOLDS["MILD"]:
        severity_level = "Mild"
    elif severity_percent < SEVERITY_THRESHOLDS["MODERATE"]:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    return {
        "severity_percent": round(severity_percent, 2),
        "severity_level": severity_level,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
    }
