"""Diabetic Retinopathy model architectures and loss functions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torchvision import models

LABELS: List[str] = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

WEIGHTS: Dict[str, Tuple] = {
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class DRClassifier(nn.Module):
    """Deep learning classifier for 5-class diabetic retinopathy severity grading."""

    def __init__(self, architecture: str = "resnet50", dropout: float = 0.3, base: nn.Module | None = None):
        super().__init__()
        if architecture not in WEIGHTS:
            raise ValueError(f"Unsupported architecture: {architecture}. Supported: {list(WEIGHTS.keys())}")

        base = base if base is not None else WEIGHTS[architecture][0](weights=None)
        self.architecture = architecture

        if architecture == "efficientnet_v2_s":
            self.backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten(1))
            feature_dim = base.classifier[1].in_features
        else:
            self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten(1))
            feature_dim = base.fc.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 5),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))


def build_model(
    architecture: str = "resnet50",
    dropout: float = 0.3,
    weights_path: str | Path | None = None,
    pretrained: bool = True,
) -> DRClassifier:
    """Instantiate a DRClassifier with optional pretrained backbone weights."""
    if architecture not in WEIGHTS:
        raise ValueError(f"Unsupported architecture: {architecture}")

    factory, weights_enum = WEIGHTS[architecture]
    if pretrained:
        if weights_path and Path(weights_path).is_file():
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            base = factory(weights=None)
            base.load_state_dict(state, strict=True)
        else:
            base = factory(weights=weights_enum)
    else:
        base = factory(weights=None)

    return DRClassifier(architecture=architecture, dropout=dropout, base=base)


def ranked_probability_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ordinal ranked probability score loss for ordinal diabetic retinopathy stages.

    Penalizes mistakes proportionally to the grade distance (e.g., predicting grade 0
    when true is grade 4 incurs higher penalty than predicting grade 3).
    """
    cumulative = logits.float().softmax(dim=1).cumsum(dim=1)[:, :4]
    expected = (targets[:, None] <= torch.arange(4, device=targets.device)[None, :]).float()
    return (cumulative - expected).square().mean()
