"""
Pydantic schemas for FastAPI endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str = Field(..., example="plastic")
    vietnamese_name: str = Field(..., example="Đồ nhựa / Chai nhựa")
    confidence: float = Field(..., example=0.9969)


class RecyclingDetail(BaseModel):
    vietnamese_name: str
    category: str
    bin_color: str
    instructions: List[str]
    decomposition_time: str
    impact: str


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., example="plastic")
    vietnamese_name: str = Field(..., example="Đồ nhựa / Chai nhựa")
    confidence: float = Field(..., example=0.9969)
    is_confident: bool = Field(..., example=True)
    uncertainty_message: str
    top_3: List[PredictionItem]
    recycling_info: RecyclingDetail
    inference_time_ms: float = Field(..., example=45.2)
    trace_id: Optional[str] = Field(None, example="a9c936d90d852bdd381e759a603b8438", description="Distributed OpenTelemetry trace ID")


class GradCAMResponse(PredictionResponse):
    gradcam_overlay_base64: str = Field(..., description="Base64-encoded JPEG image of the Grad-CAM heatmap overlay")


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    supported_classes_count: int = Field(..., example=12)
    uptime_seconds: Optional[float] = Field(None, example=123.45)
    memory_rss_mb: Optional[float] = Field(None, example=450.2)


class ClassesResponse(BaseModel):
    classes: List[str]
    details: Dict[str, RecyclingDetail]
