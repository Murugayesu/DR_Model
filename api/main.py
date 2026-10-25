"""FastAPI service for Diabetic Retinopathy classification."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src is on python path for importing `dr`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from api.schemas import PredictionResponse
from dr.inference import load_artifact, predict_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024       # 10 MiB
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "dr_grader_v1.pth"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dr.api")


class RequestTooLarge(HTTPException):
    def __init__(self):
        super().__init__(413, "Request exceeds upload limit")


class UploadLimitMiddleware:
    """Protects the API against excessive memory consumption on uploads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/predict":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)

        if length < 0 or length > MAX_REQUEST_BYTES:
            return await JSONResponse({"detail": "Request exceeds upload limit"}, 413)(scope, receive, send)

        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > MAX_REQUEST_BYTES:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except RequestTooLarge:
            await JSONResponse({"detail": "Request exceeds upload limit"}, 413)(scope, receive, send)


def validate_image(payload: bytes):
    """Validates raw image bytes for dimensions, format, and corruption."""
    if not payload:
        raise HTTPException(400, "Image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds 10 MiB limit")
    try:
        with Image.open(io.BytesIO(payload), formats=("PNG", "JPEG")) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise HTTPException(415, "Only PNG and JPEG images are supported")
            if min(image.size) < 128:
                raise HTTPException(422, "Image dimensions must be at least 128 by 128 pixels")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds the 20 megapixel limit")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "Image could not be decoded or is corrupted") from error


class ModelService:
    """Thread-safe model service managing model lifecycle and inference."""

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        import torch

        torch.set_num_threads(2)
        logger.info(f"Loading checkpoint from {checkpoint} on {device}...")
        self.model, self.preprocessing, self.bundle = load_artifact(checkpoint, device)

        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

        self.model_version = f"{self.bundle['architecture']}-{digest.hexdigest()[:12]}"
        self.lock = threading.Lock()
        logger.info(f"Model {self.model_version} loaded successfully.")

    def predict(self, payload: bytes) -> dict:
        with self.lock:
            validate_image(payload)
            try:
                return predict_image(self.model, self.preprocessing, self.bundle, io.BytesIO(payload))
            except (UnidentifiedImageError, OSError, SyntaxError) as error:
                raise HTTPException(400, "Image could not be decoded") from error


def create_app(service: ModelService | None = None, checkpoint: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = service
        if service is None:
            path = Path(checkpoint or os.environ.get("DR_MODEL_PATH", DEFAULT_MODEL_PATH))
            device = os.environ.get("DR_DEVICE", "cpu")
            try:
                application.state.service = ModelService(path, device)
            except Exception:
                logger.exception("Model unavailable; readiness and prediction endpoints will return 503")
        yield
        application.state.service = None

    application = FastAPI(
        title="Diabetic Retinopathy Grading Service",
        description="FastAPI service for 5-stage clinical diabetic retinopathy grading from retinal fundus images.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(UploadLimitMiddleware)

    @application.get("/health")
    def health():
        """Health check endpoint to verify service uptime."""
        return {"status": "ok"}

    @application.get("/ready")
    def ready(request: Request):
        """Readiness check endpoint verifying that the model weights are loaded."""
        loaded = getattr(request.app.state, "service", None)
        if loaded is None:
            raise HTTPException(503, "Model is not loaded or unavailable")
        return {"status": "ready", "model_version": loaded.model_version}

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)):
        """Upload a fundus image (PNG/JPEG) to obtain calibrated severity grade and class probabilities."""
        loaded = getattr(request.app.state, "service", None)
        if loaded is None:
            await file.close()
            raise HTTPException(503, "Model is unavailable")

        if file.content_type not in {"image/png", "image/jpeg", "application/octet-stream"}:
            await file.close()
            raise HTTPException(415, "Upload a PNG or JPEG image")

        try:
            payload = await file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await file.close()

        validate_image(payload)
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        result = await run_in_threadpool(loaded.predict, payload)
        latency = (time.perf_counter() - started) * 1000

        logger.info(
            json.dumps({
                "request_id": request_id,
                "model_version": loaded.model_version,
                "predicted_label": result["predicted_label"],
                "latency_ms": round(latency, 2),
            })
        )

        return PredictionResponse(
            request_id=request_id,
            model_version=loaded.model_version,
            predicted_class=result["predicted_class"],
            predicted_label=result["predicted_label"],
            class_probabilities=result["class_probabilities"],
            calibrated=result["calibrated"],
            preprocessing_version=result["preprocessing_version"],
            latency_ms=latency,
        )

    return application


app = create_app()
