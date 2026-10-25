"""Preprocessing transforms for fundus retinal images shared by training and inference."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF

DEFAULT_PREPROCESSING: Dict[str, Any] = {
    "version": "rgb-letterbox-v1",
    "image_size": 384,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "resize": "bilinear_letterbox",
    "padding": [0, 0, 0],
    "crop": "none",
    "exif_orientation": "apply",
}


class FundusPreprocessor:
    """Preprocesses raw retinal fundus images with aspect-ratio preserving letterboxing and normalization."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = {**DEFAULT_PREPROCESSING, **(config or {})}
        self.size = int(self.config["image_size"])
        if self.size < 64 or self.size > 1024:
            raise ValueError("Image size must be between 64 and 1024")
        if self.config["resize"] != "bilinear_letterbox" or self.config["crop"] != "none":
            raise ValueError("Unsupported preprocessing geometry")
        if self.config["version"] != "rgb-letterbox-v1" or self.config["exif_orientation"] != "apply":
            raise ValueError("Unsupported preprocessing version/orientation policy")
        if len(self.config["mean"]) != 3 or len(self.config["std"]) != 3 or min(self.config["std"]) <= 0:
            raise ValueError("Invalid RGB normalization parameters")

        self.fingerprint = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(
                10,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=tuple(self.config["padding"]),
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def prepare(self, image: Image.Image) -> Image.Image:
        """Corrects orientation, converts to RGB, and letterboxes to a centered square."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Image has invalid dimensions")

        ratio = self.size / max(image.width, image.height)
        shape = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        resized = image.resize(shape, Image.Resampling.BILINEAR)

        result = Image.new("RGB", (self.size, self.size), tuple(self.config["padding"]))
        result.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return result

    def tensor(self, prepared_image: Image.Image, training: bool = False) -> torch.Tensor:
        """Converts a prepared square PIL image into a normalized PyTorch tensor."""
        if prepared_image.size != (self.size, self.size) or prepared_image.mode != "RGB":
            raise ValueError("Expected a prepared RGB square")
        if training:
            prepared_image = self.augmentation(prepared_image)
        return TF.normalize(TF.to_tensor(prepared_image), self.config["mean"], self.config["std"])

    def __call__(self, image: Image.Image, training: bool = False) -> torch.Tensor:
        """Direct pipeline call: image -> letterbox -> normalized tensor."""
        return self.tensor(self.prepare(image), training=training)
