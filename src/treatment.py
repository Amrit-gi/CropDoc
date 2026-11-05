"""Treatment recommendation engine for CropDoc.

Loads disease + severity lookup rules from JSON configuration and returns
actionable agronomic advice including chemical, organic, cultural, and preventive measures.
"""

import json
import os
from typing import Dict, Any, Optional
from config import TREATMENT_JSON_PATH


def load_treatment_rules(json_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the treatment rules JSON file.

    Args:
        json_path: Path to the treatment rules JSON file. Defaults to TREATMENT_JSON_PATH.

    Returns:
        Dictionary containing the disease -> severity -> recommendation mapping.
    """
    path = json_path or str(TREATMENT_JSON_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Treatment rules JSON file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    return rules


def get_treatment_recommendation(disease_name: str, severity_level: str, json_path: Optional[str] = None) -> Dict[str, str]:
    """Retrieves treatment recommendations for a given disease name and severity level.

    Args:
        disease_name: Name of the predicted disease ('Healthy', 'Leaf Rot', 'Leaf Spot').
        severity_level: Categorized severity level ('Healthy', 'Mild', 'Moderate', 'Severe').
        json_path: Optional path to treatment rules JSON.

    Returns:
        Dictionary containing status summary, chemical, organic, cultural, and preventive advice.
    """
    rules = load_treatment_rules(json_path)

    # Normalize disease key
    normalized_disease = "Healthy"
    disease_lower = disease_name.lower()
    if "rot" in disease_lower:
        normalized_disease = "Leaf Rot"
    elif "spot" in disease_lower:
        normalized_disease = "Leaf Spot"
    elif "healthy" in disease_lower:
        normalized_disease = "Healthy"

    # Normalize severity key
    normalized_severity = "Healthy"
    severity_upper = severity_level.upper()
    if "MILD" in severity_upper:
        normalized_severity = "Mild"
    elif "MODERATE" in severity_upper:
        normalized_severity = "Moderate"
    elif "SEVERE" in severity_upper:
        normalized_severity = "Severe"

    disease_data = rules.get(normalized_disease, rules.get("Healthy", {}))
    recommendation = disease_data.get(normalized_severity, disease_data.get("Healthy", {}))

    return recommendation
