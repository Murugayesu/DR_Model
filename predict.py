"""Command-line inference script for Diabetic Retinopathy prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on python path
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dr.inference import load_artifact, predict_image


def main():
    parser = argparse.ArgumentParser(description="Predict 5-stage Diabetic Retinopathy severity for an image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to input fundus image (PNG or JPEG)")
    parser.add_argument(
        "--checkpoint",
        default=ROOT / "artifacts" / "dr_grader_v1.pth",
        type=Path,
        help="Path to trained model artifact checkpoint",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Error: Image not found at {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model checkpoint from {args.checkpoint}...")
    model, preprocessing, bundle = load_artifact(args.checkpoint, device=args.device)

    print(f"Running inference on {args.image}...")
    result = predict_image(model, preprocessing, bundle, args.image)

    print("\n--- Prediction Result ---")
    print(f"Predicted Grade: {result['predicted_class']} ({result['predicted_label']})")
    print("Class Probabilities:")
    for grade, prob in result["class_probabilities"].items():
        print(f"  Stage {grade}: {prob * 100:.2f}%")
    print(f"Calibrated: {result['calibrated']}")
    print(f"Preprocessing: {result['preprocessing_version']}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
