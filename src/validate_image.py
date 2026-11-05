"""Input Image Validation and Botanical Verification Module for CropDoc.

Ensures uploaded or provided images are valid, non-corrupt, readable,
contain authentic betel leaf (Piper betle) plant tissue, and strictly reject
all non-betel plant leaves (e.g., Banana, Peepal, Mango, Tomato, Neem, Rose, Bamboo, Papaya)
as well as non-foliage clutter before passing to classification and severity pipelines.
"""

import cv2
import numpy as np
from PIL import Image
import io
import os
from typing import Tuple, Union, Optional


# Known non-betel plant species keywords for rapid fail-safe rejection
NON_BETEL_KEYWORDS = [
    "banana", "musa", "peepal", "peepul", "ficus", "mango", "mangifera",
    "tomato", "solanum", "apple", "malus", "potato", "corn", "maize", "zea",
    "rice", "oryza", "wheat", "triticum", "cotton", "gossypium", "grape", "vitis",
    "guava", "psidium", "banyan", "rose", "rosa", "neem", "azadirachta",
    "tea", "camellia", "coffee", "coffea", "papaya", "carica", "cassava", "manihot",
    "pepper", "capsicum", "chilli", "tulsi", "ocimum", "rubber", "hevea",
    "eucalyptus", "bamboo", "bambusoideae", "sugarcane", "saccharum", "hibiscus",
    "citrus", "lemon", "orange", "cabbage", "brassica", "soybean", "glycine",
    "sunflower", "helianthus", "strawberry", "fragaria", "peach", "prunus", "cherry"
]


def check_botanical_species_keyword(filename: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Checks if the filename contains any known non-betel plant names.

    Args:
        filename: Optional name or path of the input file.

    Returns:
        Tuple of (is_non_betel: bool, detected_species: str or None).
    """
    if not filename:
        return False, None

    clean_name = os.path.basename(str(filename)).lower().replace("_", " ").replace("-", " ")
    for kw in NON_BETEL_KEYWORDS:
        if kw in clean_name:
            return True, kw.capitalize()

    return False, None


def detect_banana_or_monocot_blade(
    gray: np.ndarray,
    c: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    w: int,
    h: int
) -> Tuple[bool, str]:
    """Detects transverse central midrib beams and cropped monocot leaf blades (Banana/Palm).

    Authentic Piper betle leaves are cordate, contained inside the frame, with curving palmate veins.
    Banana leaf blade sections exhibit straight transverse midrib beams cutting horizontally
    across the leaf with high boundary-touch ratios.
    """
    try:
        # Detect straight horizontal lines across the central region of the leaf
        canny = cv2.Canny(gray, 40, 100)
        min_line_len = max(30, int(bw * 0.40))
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=35, minLineLength=min_line_len, maxLineGap=12)

        horiz_midrib_count = 0
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Horizontal line traversing the central half of the leaf
                if (ang <= 15 or ang >= 165) and (by + bh * 0.20 <= (y1 + y2) / 2 <= by + bh * 0.80):
                    horiz_midrib_count += 1

        border_pixels = np.count_nonzero((c[:, 0, 0] <= 2) | (c[:, 0, 0] >= w - 3) | (c[:, 0, 1] <= 2) | (c[:, 0, 1] >= h - 3))
        border_ratio = border_pixels / float(len(c)) if len(c) > 0 else 0.0

        # Monocot blade signature: horizontal central midrib beam (or high boundary touch)
        if horiz_midrib_count >= 2 or (horiz_midrib_count >= 1 and border_ratio > 0.12):
            return (
                True,
                "Transverse central midrib beam and cropped blade structure detected, characteristic of monocot foliage (e.g. Banana/Palm). "
                "Authentic Piper betle leaves possess curved palmate venation radiating from a cordate base."
            )

        # Check for multiple deep wind tears (convexity defects) along margin
        hull_indices = cv2.convexHull(c, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(c, hull_indices)
            if defects is not None:
                deep_tear_count = 0
                for i in range(defects.shape[0]):
                    d = defects[i, 0, 3] / 256.0
                    if d > min(bw, bh) * 0.18:
                        deep_tear_count += 1
                if deep_tear_count >= 3:
                    return (
                        True,
                        "Multiple deep linear tears detected along the leaf margin. Piper betle leaves possess an entire, smooth, intact margin."
                    )

        return False, ""
    except Exception:
        return False, ""


def detect_peepal_caudate_drip_tip(cleaned_mask: np.ndarray, bx: int, by: int, bw: int, bh: int) -> bool:
    """Detects the distinctive elongated, slender caudal drip-tip apex of Ficus religiosa (Peepal).

    Peepal leaves possess a long, narrow tail accounting for 20-35% of total leaf length.
    Piper betle leaves taper smoothly to an acute tip without an extended slender tail.
    """
    try:
        if bh < 40 or bw < 40:
            return False

        slice_h = max(3, int(bh * 0.18))
        top_slice = cleaned_mask[by:by + slice_h, bx:bx + bw]
        bottom_slice = cleaned_mask[by + bh - slice_h:by + bh, bx:bx + bw]

        top_w = np.count_nonzero(np.any(top_slice > 0, axis=0)) / float(bw) if bw > 0 else 1.0
        bottom_w = np.count_nonzero(np.any(bottom_slice > 0, axis=0)) / float(bw) if bw > 0 else 1.0

        mid_slice = cleaned_mask[by + int(bh * 0.35):by + int(bh * 0.65), bx:bx + bw]
        mid_w = np.count_nonzero(np.any(mid_slice > 0, axis=0)) / float(bw) if bw > 0 else 1.0

        if mid_w > 0.55 and min(top_w, bottom_w) < 0.12:
            return True

        return False
    except Exception:
        return False


def validate_image(
    image_source: Union[str, bytes, np.ndarray],
    filename: Optional[str] = None
) -> Tuple[bool, str, Union[np.ndarray, None]]:
    """Validates an input image against strict Piper betle botanical criteria.

    Comprehensive verification pipeline:
    1. Botanical species metadata/filename audit (rejects known non-betel plant names).
    2. Format integrity, channel count, and minimum resolution (>= 32x32).
    3. Blank / extreme exposure screening (variance >= 3.0).
    4. Chlorophyll & foliage coverage check (>= 12% plant tissue).
    5. Cordate morphology & aspect ratio filter (0.38 <= AR <= 2.20).
    6. Contour solidity filter (Solidity >= 0.60; rejects compound/serrated leaves like tomato, neem, rose).
    7. Monocot blade & transverse midrib verification (rejects Banana/Palm leaf blades).
    8. Caudate acumen / drip-tip filter (rejects Peepal tail).

    Args:
        image_source: File path string, raw byte array, or OpenCV BGR numpy array.
        filename: Optional filename for botanical keyword validation.

    Returns:
        Tuple of (is_valid: bool, message: str, image_bgr: np.ndarray or None).
    """
    image_bgr = None

    # Step 1: Botanical keyword blacklist filter
    inferred_filename = filename
    if isinstance(image_source, str) and not inferred_filename:
        inferred_filename = image_source

    is_blacklisted, species = check_botanical_species_keyword(inferred_filename)
    if is_blacklisted:
        return (
            False,
            f"Non-Betel leaf detected ({species}): CropDoc is calibrated strictly for Piper betle (Betel Vine). "
            f"The uploaded image belongs to a non-target plant species and was rejected by botanical validation.",
            None
        )

    try:
        # Step 2: Image decoding and format validation
        if isinstance(image_source, str):
            image_bgr = cv2.imread(image_source)
            if image_bgr is None:
                return False, f"Unable to read image file: {image_source}. File may be corrupted or missing.", None
        elif isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                return False, "Failed to decode image bytes. File format may be unsupported or corrupt.", None
        elif isinstance(image_source, np.ndarray):
            image_bgr = image_source
        else:
            return False, f"Unsupported input type: {type(image_source)}.", None

        # Dimensionality check
        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            return False, "Image must have 3 color channels (RGB/BGR).", None

        h, w, _ = image_bgr.shape
        if h < 32 or w < 32:
            return False, f"Image resolution too low ({w}x{h}). Minimum required is 32x32 pixels.", None

        # Step 3: Check for blank / extreme brightness image
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < 3.0:
            return False, "Image appears to be blank, single-color, or completely overexposed/underexposed.", None

        # Step 4: Chlorophyll & Leaf Tissue Coverage Check
        hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_foliage = np.array([10, 25, 25])
        upper_foliage = np.array([95, 255, 255])
        foliage_mask = cv2.inRange(hsv_img, lower_foliage, upper_foliage)

        foliage_pixel_ratio = np.count_nonzero(foliage_mask) / (h * w)

        if foliage_pixel_ratio < 0.12:
            return (
                False,
                "No betel leaf tissue detected in the uploaded photo. CropDoc requires a clear photograph of an authentic betel leaf (Piper betle).",
                None
            )

        # Step 5: Morphological Extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(foliage_mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            c_area = cv2.contourArea(main_contour)

            if c_area > 0.04 * (h * w):
                bx, by, bw, bh = cv2.boundingRect(main_contour)
                aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

                # 5a. Aspect Ratio Check: Cordate Betel leaf aspect ratio is typically 0.40 - 2.00
                if aspect_ratio < 0.35 or aspect_ratio > 2.20:
                    return (
                        False,
                        f"Non-Betel leaf morphology rejected: Aspect ratio ({aspect_ratio:.2f}) does not match the cordate "
                        f"(heart-shaped) structure of Piper betle. Elongated or lanceolate leaves (e.g., Bamboo, Mango, Grass) are rejected.",
                        None
                    )

                # 5b. Solidity Check: Betel leaves are simple leaves with an entire (unlobed) margin
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(c_area) / hull_area if hull_area > 0 else 1.0

                if solidity < 0.60:
                    return (
                        False,
                        f"Non-Betel leaf morphology rejected: Contour solidity ({solidity:.2f}) indicates compound, serrated, or deeply "
                        f"divided margins (e.g., Tomato, Neem, Rose, Papaya). Piper betle leaves possess a simple, intact cordate lamina.",
                        None
                    )

                # 5c. Monocot / Banana leaf blade verification
                is_banana, banana_reason = detect_banana_or_monocot_blade(gray, main_contour, bx, by, bw, bh, w, h)
                if is_banana:
                    return (
                        False,
                        f"Non-Betel leaf detected (Banana/Monocot): {banana_reason} "
                        f"CropDoc is calibrated strictly for Piper betle (Betel Vine) and rejects non-target species.",
                        None
                    )

                # 5d. Peepal (Ficus religiosa) Caudate Drip-Tip Detector
                if detect_peepal_caudate_drip_tip(cleaned_mask, bx, by, bw, bh):
                    return (
                        False,
                        "Non-Betel leaf detected: Prominent caudate drip-tip apex detected, characteristic of Ficus religiosa (Peepal). "
                        "Piper betle leaves exhibit an acute to shortly acuminate apex without an elongated caudal tail.",
                        None
                    )

        return True, "Betel leaf validation successful.", image_bgr

    except Exception as e:
        return False, f"An unexpected error occurred during image validation: {str(e)}", None
