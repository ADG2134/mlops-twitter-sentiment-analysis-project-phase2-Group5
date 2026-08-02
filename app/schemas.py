"""Pydantic request/response models for the sentiment API."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Raw tweet text to classify.",
        examples=["I absolutely love this new phone, best purchase ever!"],
    )


class PredictResponse(BaseModel):
    text: str
    label: str = Field(..., description="'positive' or 'negative'")
    score: float = Field(..., description="Model confidence for the predicted label, 0-1")
    positive_probability: float = Field(..., description="Raw P(positive) from the model")
    model_version: str


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
