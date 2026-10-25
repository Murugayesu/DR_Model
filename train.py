"""Standalone training script for Diabetic Retinopathy classification models.

This script trains a ResNet-50 (or EfficientNet-V2-S) backbone for 5-class
diabetic retinopathy severity grading (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).
It uses:
  - Letterbox RGB preprocessing with aspect-ratio preservation
  - Data augmentation (flips, random rotation, color jitter)
  - Ordinal ranked probability loss (penalizes distance between predicted and true severity)
  - Two-phase training: classifier warmup followed by full backbone fine-tuning
  - Saving a self-contained checkpoint bundle compatible with `dr.inference.load_artifact`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dr.models import LABELS, DRClassifier, build_model, ranked_probability_loss
from dr.preprocessing import DEFAULT_PREPROCESSING, FundusPreprocessor


class SimpleFundusDataset(Dataset):
    """Loads fundus images and severity labels from a CSV manifest."""

    def __init__(self, root_dir: Path, records: List[Tuple[str, int]], preprocessor: FundusPreprocessor, training: bool = True):
        self.root_dir = Path(root_dir)
        self.records = records
        self.preprocessor = preprocessor
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_path, grade = self.records[index]
        image_path = self.root_dir / rel_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.preprocessor(image, training=self.training)
        return tensor, torch.tensor(grade, dtype=torch.long)


def load_csv_manifest(csv_path: Path) -> List[Tuple[str, int]]:
    """Reads a CSV with columns ['image_path', 'grade'] or ['id_code', 'diagnosis']."""
    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("image_path") or f"train_images/{row.get('id_code')}.png"
            grade = int(row.get("grade", row.get("diagnosis", 0)))
            records.append((path, grade))
    return records


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains model for one epoch."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates model loss and top-1 accuracy on validation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * len(targets)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def save_artifact(
    model: DRClassifier,
    save_path: Path,
    architecture: str,
    preprocessor_config: dict,
    temperature: float = 1.0,
):
    """Exports model weights in the standardized artifact bundle format."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "architecture": architecture,
        "labels": LABELS,
        "temperature": temperature,
        "calibrated": temperature != 1.0,
        "config": {
            "dropout": 0.3,
            "preprocessing": preprocessor_config,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(bundle, save_path)
    print(f"Artifact successfully saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Diabetic Retinopathy classifier.")
    parser.add_argument("--architecture", default="resnet50", choices=["resnet50", "efficientnet_v2_s"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root directory containing images")
    parser.add_argument("--train-csv", type=Path, help="CSV manifest with image_path and grade columns")
    parser.add_argument("--val-csv", type=Path, help="Optional validation CSV manifest")
    parser.add_argument("--epochs", type=int, default=15, help="Number of full training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Epochs to train only the classification head")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for backbone")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "dr_model.pth", help="Output .pth path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build model architecture
    print(f"Initializing {args.architecture}...")
    model = build_model(architecture=args.architecture, pretrained=True).to(device)
    preprocessor = FundusPreprocessor(DEFAULT_PREPROCESSING)

    if not args.train_csv or not args.train_csv.is_file():
        print("\nNote: No --train-csv provided. To train with your dataset, run:")
        print("  python train.py --train-csv data/train.csv --data-dir data/ --epochs 10\n")
        print("Exiting without training. Pretrained model and architecture verified.")
        return

    train_records = load_csv_manifest(args.train_csv)
    train_dataset = SimpleFundusDataset(args.data_dir, train_records, preprocessor, training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_loader = None
    if args.val_csv and args.val_csv.is_file():
        val_records = load_csv_manifest(args.val_csv)
        val_dataset = SimpleFundusDataset(args.data_dir, val_records, preprocessor, training=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 1. Warmup Phase (head only)
    if args.warmup_epochs > 0:
        print(f"\n--- Phase 1: Warmup ({args.warmup_epochs} epochs) ---")
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.classifier.parameters(), lr=args.head_lr, weight_decay=1e-2)

        for epoch in range(1, args.warmup_epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, ranked_probability_loss, device)
            print(f"Warmup Epoch [{epoch}/{args.warmup_epochs}] - Loss: {loss:.4f}")

    # 2. Fine-tuning Phase (full model)
    print(f"\n--- Phase 2: Full Fine-Tuning ({args.epochs} epochs) ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, ranked_probability_loss, device)
        scheduler.step()

        val_str = ""
        if val_loader:
            val_loss, val_acc = evaluate_epoch(model, val_loader, ranked_probability_loss, device)
            val_str = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"

        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {loss:.4f}{val_str}")

    # Save trained checkpoint
    save_artifact(
        model=model,
        save_path=args.output,
        architecture=args.architecture,
        preprocessor_config=preprocessor.config,
    )


if __name__ == "__main__":
    main()
