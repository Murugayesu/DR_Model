"""Inference utilities: loading trained checkpoints and running calibrated predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Dict, Tuple, Union

import torch
from PIL import Image

from dr.models import DRClassifier, LABELS
from dr.preprocessing import FundusPreprocessor


def load_artifact(
    path: Union[str, Path],
    device: str = "cpu",
    allow_diagnostic: bool = False,
) -> Tuple[DRClassifier, FundusPreprocessor, Dict[str, Any]]:
    """Loads a trained diabetic retinopathy model checkpoint and its preprocessor."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if bundle.get("diagnostic_only", False) and not allow_diagnostic:
        raise ValueError("This is a diagnostic checkpoint, not a trained DR model")
    if bundle.get("labels") != LABELS:
        raise ValueError("Artifact class order differs from the five-class DR contract")

    temperature = float(bundle.get("temperature", 1.0))
    if not (0.05 <= temperature <= 20.0):
        raise ValueError("Invalid artifact temperature")

    architecture = bundle["architecture"]
    dropout = bundle.get("config", {}).get("dropout", 0.3)

    try:
        with torch.device("meta"):
            model = DRClassifier(architecture, dropout)
        model.load_state_dict(bundle["model_state_dict"], strict=True, assign=True)
    except Exception:
        # Fallback for devices/versions that don't support meta assignment
        model = DRClassifier(architecture, dropout)
        model.load_state_dict(bundle["model_state_dict"], strict=True)

    model.to(device).eval()
    preprocessing_config = bundle.get("config", {}).get("preprocessing", {})
    preprocessing = FundusPreprocessor(preprocessing_config)
    return model, preprocessing, bundle


@torch.inference_mode()
def predict_image(
    model: DRClassifier,
    preprocessing: FundusPreprocessor,
    bundle: Dict[str, Any],
    image_input: Union[str, Path, BinaryIO],
) -> Dict[str, Any]:
    """Generates five-class severity predictions and calibrated probabilities for an image."""
    with Image.open(image_input, formats=("PNG", "JPEG")) as image:
        image.load()
        device = next(model.parameters()).device
        tensor = preprocessing(image).unsqueeze(0).to(device)

    temperature = float(bundle.get("temperature", 1.0))
    logits = model(tensor).float()
    probabilities = (logits / temperature).softmax(1)[0].cpu()
    predicted = int(probabilities.argmax())

    return {
        "predicted_class": predicted,
        "predicted_label": LABELS[predicted],
        "class_probabilities": {str(index): float(probabilities[index]) for index in range(5)},
        "calibrated": bool(bundle.get("calibrated", False)),
        "preprocessing_version": preprocessing.config.get("version", "rgb-letterbox-v1"),
        "diagnostic_only": bool(bundle.get("diagnostic_only", False)),
    }
