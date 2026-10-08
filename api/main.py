"""
EcoSort - Smart Trash Classifier REST API
FastAPI backend for automated waste classification and Explainable AI (Grad-CAM).
"""

import os
import sys
import io
import base64
from contextlib import asynccontextmanager
from typing import Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset import CLASSES
from src.model import load_trained_model, predict_image, generate_gradcam_heatmap
from src.utils import RECYCLING_INFO, create_gradcam_overlay, evaluate_uncertainty
from api.schemas import (
    PredictionResponse,
    PredictionItem,
    RecyclingDetail,
    GradCAMResponse,
    HealthResponse,
    ClassesResponse,
)

# Global model reference
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup and free on shutdown"""
    print("[INFO] Loading ResNet50 model for FastAPI...")
    app_state["model"] = load_trained_model()
    print("[INFO] Model loaded successfully into memory.")
    yield
    app_state.clear()


app = FastAPI(
    title="EcoSort - Smart Trash Classifier REST API",
    description="RESTful API for automated waste classification into 12 categories with Explainable AI (Grad-CAM) & recycling guidance.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pil_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """Helper to validate and parse uploaded image file"""
    if not upload_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {upload_file.content_type}. Only images are supported."
        )
    try:
        image_bytes = upload_file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image: {str(e)}"
        )


@app.get("/", tags=["Info"])
def root_info():
    """API overview and interactive docs links"""
    return {
        "service": "EcoSort Smart Trash Classifier REST API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "predict": "POST /predict",
            "predict_gradcam": "POST /predict/gradcam",
            "predict_gradcam_image": "POST /predict/gradcam/image"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Service health and model status"""
    is_loaded = "model" in app_state and app_state["model"] is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        supported_classes_count=len(CLASSES)
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Metadata"])
def get_supported_classes():
    """Retrieve all 12 supported waste classes and their recycling specifications"""
    details = {
        cls_name: RecyclingDetail(**info_data)
        for cls_name, info_data in RECYCLING_INFO.items()
    }
    return ClassesResponse(classes=CLASSES, details=details)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_waste(
    file: UploadFile = File(..., description="Image file (JPG, PNG) to classify"),
    threshold: float = Query(0.60, ge=0.1, le=1.0, description="Minimum confidence threshold for certainty")
):
    """
    Classify a waste image into one of 12 classes with confidence scores and disposal guidelines.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready.")

    image = _get_pil_image_from_upload(file)
    top_classes, top_confidences, latency = predict_image(model, image, top_k=3)

    pred_class = top_classes[0]
    conf = top_confidences[0]
    is_conf, uncert_msg = evaluate_uncertainty(conf, threshold=threshold)

    rec_info = RECYCLING_INFO.get(pred_class, {})
    top_3_items = [
        PredictionItem(
            class_name=cls_name,
            vietnamese_name=RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name),
            confidence=score
        )
        for cls_name, score in zip(top_classes, top_confidences)
    ]

    return PredictionResponse(
        predicted_class=pred_class,
        vietnamese_name=rec_info.get('vietnamese_name', pred_class),
        confidence=conf,
        is_confident=is_conf,
        uncertainty_message=uncert_msg,
        top_3=top_3_items,
        recycling_info=RecyclingDetail(**rec_info),
        inference_time_ms=round(latency, 2)
    )


@app.post("/predict/gradcam", response_model=GradCAMResponse, tags=["Explainable AI"])
async def predict_with_gradcam_json(
    file: UploadFile = File(..., description="Image file to classify with visual explanation"),
    threshold: float = Query(0.60, ge=0.1, le=1.0, description="Minimum confidence threshold")
):
    """
    Classify a waste image and return both predictions and Base64-encoded Grad-CAM heatmap overlay.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready.")

    image = _get_pil_image_from_upload(file)
    top_classes, top_confidences, latency = predict_image(model, image, top_k=3)

    pred_class = top_classes[0]
    conf = top_confidences[0]
    is_conf, uncert_msg = evaluate_uncertainty(conf, threshold=threshold)

    # Compute Grad-CAM overlay
    heatmap = generate_gradcam_heatmap(model, image)
    overlay = create_gradcam_overlay(image, heatmap)

    # Encode overlay to base64 JPEG
    buffer = io.BytesIO()
    overlay.save(buffer, format="JPEG", quality=90)
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    rec_info = RECYCLING_INFO.get(pred_class, {})
    top_3_items = [
        PredictionItem(
            class_name=cls_name,
            vietnamese_name=RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name),
            confidence=score
        )
        for cls_name, score in zip(top_classes, top_confidences)
    ]

    return GradCAMResponse(
        predicted_class=pred_class,
        vietnamese_name=rec_info.get('vietnamese_name', pred_class),
        confidence=conf,
        is_confident=is_conf,
        uncertainty_message=uncert_msg,
        top_3=top_3_items,
        recycling_info=RecyclingDetail(**rec_info),
        inference_time_ms=round(latency, 2),
        gradcam_overlay_base64=base64_str
    )


@app.post("/predict/gradcam/image", tags=["Explainable AI"])
async def predict_with_gradcam_image_stream(
    file: UploadFile = File(..., description="Image file to generate Grad-CAM overlay image")
):
    """
    Directly returns the resulting Grad-CAM heatmap overlay image (JPEG format) for direct visualization.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready.")

    image = _get_pil_image_from_upload(file)
    heatmap = generate_gradcam_heatmap(model, image)
    overlay = create_gradcam_overlay(image, heatmap)

    buffer = io.BytesIO()
    overlay.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
