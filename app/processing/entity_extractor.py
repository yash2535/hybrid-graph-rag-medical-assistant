from typing import Dict, List

import torch
from gliner import GLiNER

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model: GLiNER | None = None


def _get_device() -> str:
    if settings.NER_DEVICE:
        return settings.NER_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model() -> GLiNER:
    """
    Lazy-load and cache the GLiNER model.
    """
    global _model

    if _model is None:
        device = _get_device()
        logger.info(
            "Loading NER model",
            extra={
                "model_name": settings.NER_MODEL_NAME,
                "device": device,
            },
        )
        _model = GLiNER.from_pretrained(settings.NER_MODEL_NAME).to(device)

    return _model


def _empty_result() -> Dict[str, List[str]]:
    return {
        "drugs": [],
        "conditions": [],
        "biomarkers": [],
        "symptoms": [],
    }


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities from text using GLiNER.
    Safe for repeated calls and production workloads.
    """
    if not text or not isinstance(text, str):
        logger.debug("NER skipped: invalid or empty text")
        return _empty_result()

    model = _get_model()

    labels = settings.NER_LABELS

    try:
        entities = model.predict_entities(
            text,
            labels,
            threshold=settings.NER_CONFIDENCE_THRESHOLD,
        )
    except Exception:
        logger.exception("NER inference failed")
        return _empty_result()

    results = _empty_result()

    for ent in entities:
        value = ent.get("text", "").lower().strip()
        label = ent.get("label")

        if not value:
            continue

        if label == "drug" and value not in results["drugs"]:
            results["drugs"].append(value)
        elif label == "medical condition" and value not in results["conditions"]:
            results["conditions"].append(value)
        elif label == "biomarker" and value not in results["biomarkers"]:
            results["biomarkers"].append(value)
        elif label == "symptom" and value not in results["symptoms"]:
            results["symptoms"].append(value)

    logger.debug(
        "NER extraction complete",
        extra={
            "drugs": len(results["drugs"]),
            "conditions": len(results["conditions"]),
            "biomarkers": len(results["biomarkers"]),
            "symptoms": len(results["symptoms"]),
        },
    )

    return results
