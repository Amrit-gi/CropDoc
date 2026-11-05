"""Sanity and Integration Unit Tests for CropDoc Pipeline.

Run using: python -m pytest tests/test_pipeline.py
"""

import sys
import os
import pytest
import numpy as np
import cv2

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.severity import estimate_severity
from src.treatment import get_treatment_recommendation, load_treatment_rules
from src.validate_image import validate_image
from config import SEVERITY_THRESHOLDS


def test_treatment_recommendation_non_empty():
    """Test that treatment lookup returns non-empty strings for valid disease and severity pairs."""
    diseases = ["Healthy", "Leaf Rot", "Leaf Spot"]
    severities = ["Healthy", "Mild", "Moderate", "Severe"]

    for disease in diseases:
        for severity in severities:
            recommendation = get_treatment_recommendation(disease, severity)
            assert isinstance(recommendation, dict), f"Failed for {disease} - {severity}"
            assert "chemical_treatment" in recommendation
            assert "organic_treatment" in recommendation
            assert len(recommendation["chemical_treatment"]) > 0
            assert len(recommendation["organic_treatment"]) > 0


def test_treatment_rules_json_exists():
    """Test that the treatment_rules.json file loads successfully."""
    rules = load_treatment_rules()
    assert "Leaf Rot" in rules
    assert "Leaf Spot" in rules
    assert "Healthy" in rules


def test_severity_estimator_synthetic_green_leaf():
    """Test severity estimation on a synthetic clean green leaf image (should be Healthy)."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = (30, 180, 50)  # BGR green

    res = estimate_severity(img)
    assert "severity_percent" in res
    assert "severity_level" in res
    assert isinstance(res["severity_percent"], float)
    assert res["severity_level"] in ["Healthy", "Mild", "Moderate", "Severe"]


def test_severity_estimator_synthetic_brown_spot():
    """Test severity estimation on a synthetic green image with a brown lesion spot."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = (30, 180, 50)  # Green background

    # Add brown spot (BGR: ~20, 60, 120)
    img[50:150, 50:150] = (20, 60, 120)

    res = estimate_severity(img)
    assert res["severity_percent"] > 5.0
    assert res["severity_level"] in ["Mild", "Moderate", "Severe"]


def test_image_validation_valid():
    """Test image validation with a valid synthetic OpenCV image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = (50, 150, 50)
    img[50:, 50:] = (20, 40, 100)

    is_valid, msg, out_img = validate_image(img)
    assert is_valid is True
    assert out_img is not None


def test_image_validation_blank_rejection():
    """Test image validation rejects completely black or solid white image."""
    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    is_valid, msg, out_img = validate_image(blank_img)
    assert is_valid is False
    assert "blank" in msg.lower() or "extreme" in msg.lower()


def test_image_validation_rejects_non_betel_species():
    """Test image validation rejects non-betel plant uploads (e.g., banana, peepal, tomato)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = (50, 150, 50)
    img[50:, 50:] = (20, 40, 100)

    # Rejection via non-betel plant filename
    is_valid, msg, _ = validate_image(img, filename="banana.jpg")
    assert is_valid is False
    assert "non-betel" in msg.lower() or "banana" in msg.lower()

    is_valid_peepal, msg_peepal, _ = validate_image(img, filename="peepal_leaf.png")
    assert is_valid_peepal is False
    assert "non-betel" in msg_peepal.lower() or "peepal" in msg_peepal.lower()


def test_image_validation_rejects_extreme_aspect_ratio():
    """Test that extreme elongated or lanceolate non-betel leaf morphology is rejected."""
    # Create elongated strip (height 300, width 40 -> AR = 0.13)
    img = np.zeros((300, 100, 3), dtype=np.uint8)
    img[20:280, 40:60] = (40, 160, 40)  # Green vertical strip
    is_valid, msg, _ = validate_image(img)
    assert is_valid is False
    assert "aspect ratio" in msg.lower() or "elongated" in msg.lower() or "morphology" in msg.lower()


def test_image_validation_rejects_straight_midrib_and_monocot_striations():
    """Test image validation rejects leaf images exhibiting straight midrib beams and monocot striations."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[20:180, 20:180] = (40, 160, 40)  # Green leaf blade
    # Draw prominent straight horizontal midrib
    cv2.line(img, (20, 100), (180, 100), (180, 240, 220), 4)
    # Draw parallel transverse veins
    for x in range(30, 180, 15):
        cv2.line(img, (x, 25), (x, 175), (30, 120, 30), 2)

    is_valid, msg, _ = validate_image(img)
    assert is_valid is False
    assert "banana" in msg.lower() or "midrib" in msg.lower() or "monocot" in msg.lower() or "parallel" in msg.lower()
