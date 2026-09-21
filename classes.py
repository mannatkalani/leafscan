"""
classes.py
----------
Static reference data for the plant-disease CNN: the 38 class labels the
model was trained on (New Plant Diseases Dataset / PlantVillage-Augmented,
as loaded by `flow_from_directory`, which sorts folder names alphabetically),
plus small helpers to turn a raw label like "Tomato___Late_blight" into
something human-readable and to attach a short, practical care tip.

Keep this list in the exact order below -- it must match the class index
order Keras assigned during training (alphabetical folder order).
"""

from __future__ import annotations
from dataclasses import dataclass

CLASS_NAMES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_CROP_DISPLAY = {
    "Corn_(maize)": "Corn (Maize)",
    "Cherry_(including_sour)": "Cherry",
    "Pepper,_bell": "Bell Pepper",
}


@dataclass(frozen=True)
class ParsedClass:
    crop: str
    condition: str
    is_healthy: bool


def parse_class_name(raw: str) -> ParsedClass:
    """Split a raw label like 'Tomato___Late_blight' into crop + condition."""
    crop_raw, _, condition_raw = raw.partition("___")
    crop = _CROP_DISPLAY.get(crop_raw, crop_raw.replace("_", " "))
    is_healthy = condition_raw.strip().lower() == "healthy"
    condition = "Healthy" if is_healthy else condition_raw.replace("_", " ").replace("  ", " ").strip()
    return ParsedClass(crop=crop, condition=condition, is_healthy=is_healthy)


# Keyword -> (short cause, practical care tips). Matched against the raw
# condition string so every one of the 38 classes resolves to something
# useful without hand-writing 38 separate entries.
_TIP_LIBRARY: list[tuple[str, str, list[str]]] = [
    ("scab", "Fungal infection favored by wet spring weather.",
     ["Remove and destroy fallen leaves each autumn", "Apply a protectant fungicide from bud break", "Prune for airflow through the canopy"]),
    ("black_rot", "Fungal disease that spreads through wounds and dead wood.",
     ["Prune out mummified fruit and cankers", "Sanitize tools between cuts", "Apply fungicide during the growing season"]),
    ("rust", "Fungal pathogen that often needs a second host plant nearby.",
     ["Remove nearby alternate hosts (e.g. cedar/juniper for apple rust)", "Apply fungicide at first sign of orange pustules", "Improve air circulation"]),
    ("mildew", "Powdery fungal growth thriving in humid, still air.",
     ["Increase spacing and airflow between plants", "Avoid overhead watering", "Apply sulfur or potassium-bicarbonate spray"]),
    ("blight", "Fast-spreading fungal or bacterial infection of leaves/stems.",
     ["Remove and destroy infected foliage immediately", "Avoid overhead irrigation", "Rotate crops and apply fungicide preventively"]),
    ("leaf_spot", "Fungal spotting that weakens the leaf over time.",
     ["Remove affected leaves promptly", "Water at the base, not the foliage", "Apply a copper-based fungicide"]),
    ("spot", "Localized fungal or bacterial lesion on the leaf surface.",
     ["Remove affected leaves promptly", "Water at the base, not the foliage", "Apply a copper-based fungicide"]),
    ("bacterial", "Bacterial infection, often spread by water splash and tools.",
     ["Avoid working with wet plants", "Disinfect tools between plants", "Apply a copper-based bactericide"]),
    ("mosaic", "Viral infection, usually spread by insects or contaminated tools.",
     ["Remove and destroy infected plants", "Control aphid/insect vectors", "Disinfect tools; no chemical cure exists"]),
    ("yellow_leaf_curl", "Viral infection transmitted by whiteflies.",
     ["Control whitefly populations", "Remove infected plants promptly", "Use reflective mulch to deter whiteflies"]),
    ("greening", "Bacterial disease spread by the Asian citrus psyllid; currently incurable.",
     ["Control psyllid insect populations", "Remove and destroy infected trees", "Source certified disease-free planting stock"]),
    ("esca", "Fungal trunk disease affecting older vines.",
     ["Prune out and destroy infected wood", "Avoid pruning in wet weather", "Protect pruning wounds with a sealant"]),
    ("scorch", "Fungal or environmental stress causing leaf-edge browning.",
     ["Ensure consistent, adequate watering", "Remove severely affected leaves", "Mulch to stabilize soil moisture"]),
    ("mite", "Tiny sap-sucking pests that thrive in hot, dry conditions.",
     ["Rinse foliage with a strong water spray", "Introduce predatory mites if possible", "Apply insecticidal soap or miticide"]),
    ("mold", "Fungal growth favored by high humidity and poor ventilation.",
     ["Improve greenhouse/garden ventilation", "Avoid overhead watering", "Apply a suitable fungicide"]),
]

_HEALTHY_TIPS = [
    "Maintain a consistent watering schedule",
    "Monitor regularly for early signs of pests or disease",
    "Keep up balanced fertilization for the growth stage",
]


def get_disease_info(raw_class: str) -> dict:
    """Return {cause, tips} for a raw class label using keyword matching."""
    parsed = parse_class_name(raw_class)
    if parsed.is_healthy:
        return {"cause": "No disease detected in this specimen.", "tips": _HEALTHY_TIPS}

    condition_lower = raw_class.lower()
    for keyword, cause, tips in _TIP_LIBRARY:
        if keyword in condition_lower:
            return {"cause": cause, "tips": tips}

    return {
        "cause": "Pathology detected; consult a local agricultural extension for confirmation.",
        "tips": ["Isolate the affected plant if possible", "Remove visibly affected leaves", "Monitor spread over the next few days"],
    }
