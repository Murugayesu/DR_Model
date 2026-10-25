"""Validated API request and response schemas for diabetic retinopathy inference."""

from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field, field_validator


class PredictionResponse(BaseModel):
    """Structured response payload returned by the /predict endpoint."""

    request_id: str
    model_version: str
    status: Literal["prediction_available"] = "prediction_available"
    predicted_class: int = Field(ge=0, le=4, description="Diabetic retinopathy grade (0 to 4)")
    predicted_label: str = Field(description="Clinical stage description")
    class_probabilities: Dict[str, float] = Field(description="Calibrated probability for each grade (0 to 4)")
    calibrated: bool = Field(description="Whether temperature scaling calibration was applied")
    image_quality: Literal["not_assessed"] = "not_assessed"
    requires_manual_review: bool = True
    preprocessing_version: str
    latency_ms: float = Field(ge=0, description="Inference latency in milliseconds")

    @field_validator("class_probabilities")
    @classmethod
    def validate_probabilities(cls, value: Dict[str, float]) -> Dict[str, float]:
        if set(value.keys()) != {str(index) for index in range(5)}:
            raise ValueError("Expected probability keys 0 through 4")
        if any(not (0.0 <= prob <= 1.0) for prob in value.values()):
            raise ValueError("Probabilities must be between 0 and 1")
        if abs(sum(value.values()) - 1.0) > 1e-4:
            raise ValueError("Probabilities must sum to 1.0")
        return value
