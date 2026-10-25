# Diabetic Retinopathy Classification & Inference Service

A clean, self-contained deep learning model and FastAPI inference service for grading Diabetic Retinopathy (DR) severity from retinal fundus photographs.

---

## 🩺 Clinical Severity Stages

The model grades retinal images into the standard 5-stage International Clinical Diabetic Retinopathy scale:

| Grade | Clinical Stage | Description |
|:---:|:---|:---|
| **0** | **No DR** | No clinical abnormalities detected |
| **1** | **Mild DR** | Microaneurysms only |
| **2** | **Moderate DR** | More than microaneurysms, but less than severe |
| **3** | **Severe DR** | 20+ intraretinal hemorrhages in each of 4 quadrants, definite venous beading, or prominent IRMA |
| **4** | **Proliferative DR** | Neovascularization, vitreous/preretinal hemorrhage |

---

## 📁 Repository Structure

```text
dr_model/
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI service with /predict, /health, /ready
│   └── schemas.py               # Pydantic request & response schemas
├── artifacts/
│   ├── dr_grader_v1.pth         # Trained ResNet-50 PyTorch weights (~90 MB)
│   └── dr_grader_v1.calibration.json # Temperature calibration metadata
├── src/
│   └── dr/
│       ├── __init__.py          # Clean module exports
│       ├── models.py            # DRClassifier architecture, heads, loss functions
│       ├── preprocessing.py     # Aspect-preserving letterbox resizing & normalization
│       └── inference.py         # load_artifact and predict_image utilities
├── train.py                     # Standalone training script with ordinal loss & warmup
├── predict.py                   # Command-line image prediction tool
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Python package configuration
├── .gitignore                   # Standard clean gitignore
└── README.md                    # Quickstart documentation
```

---

## 🚀 Quickstart

### 1. Installation

Create and activate a Python virtual environment (Python 3.10+):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
# Or install in editable mode:
pip install -e .
```

---

### 2. Run the FastAPI Web Service

Start the inference server with Uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server automatically loads the pre-trained weights from `artifacts/dr_grader_v1.pth`.

Interactive API docs will be available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Available Endpoints

- `GET /health` - Service health status: `{"status": "ok"}`
- `GET /ready` - Verifies that the model weights are loaded and ready.
- `POST /predict` - Uploads a fundus image (PNG or JPEG) and returns severity predictions.

#### Example API Request (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_fundus.jpg"
```

#### Example JSON Response

```json
{
  "request_id": "4b27f678941f4ca2b06e30018f3a3d5f",
  "model_version": "resnet50-08c9c8c34b24",
  "status": "prediction_available",
  "predicted_class": 0,
  "predicted_label": "No DR",
  "class_probabilities": {
    "0": 0.892,
    "1": 0.071,
    "2": 0.028,
    "3": 0.006,
    "4": 0.003
  },
  "calibrated": true,
  "image_quality": "not_assessed",
  "requires_manual_review": true,
  "preprocessing_version": "rgb-letterbox-v1",
  "latency_ms": 42.15
}
```

---

### 3. CLI Prediction

You can also run predictions directly on an image from the terminal without starting the web server:

```bash
python predict.py --image path/to/eye.jpg
```

---

### 4. Training

To train or fine-tune a model on your own dataset, provide a CSV manifest with image paths and integer grades (0-4):

```bash
python train.py \
  --train-csv data/train.csv \
  --data-dir data/images/ \
  --epochs 15 \
  --batch-size 16 \
  --output artifacts/dr_model.pth
```

#### Training Features
- **Ordinal Loss:** Uses ranked probability score loss to penalize predictions farther from the ground-truth stage.
- **Two-phase Training:** Freezes the pretrained backbone for warmup epochs, then fine-tunes end-to-end with cosine annealing.
- **Standardized Artifact:** Outputs a `.pth` bundle containing architecture config, preprocessing settings, and state dict.

---

## 🛡️ License & Disclaimer

This model is intended for research, prototype development, and decision-support exploration. It should not be used as the sole basis for clinical diagnosis without clinical oversight and validation.
