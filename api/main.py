"""
EcoSort - Smart Trash Classifier REST API
FastAPI backend for automated waste classification and Explainable AI (Grad-CAM),
equipped with Structured Logging (JSON + Rotation) and Prometheus MLOps Monitoring.
"""

import os
import sys
import io
import time
import uuid
import base64
from contextlib import asynccontextmanager
from typing import Optional, Tuple
from PIL import Image
import psutil

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, status
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
from api.logger import logger, log_prediction_audit
from api.metrics import (
    HTTP_ACTIVE_REQUESTS,
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    record_inference_metrics,
    get_metrics_response,
)
from api.tracing import setup_tracing, instrument_fastapi_app, trace_span, get_current_trace_and_span_ids

# Initialize OpenTelemetry Distributed Tracing targeting Grafana Tempo
setup_tracing(service_name="ecosort-api", service_version="1.1.0")

# Global application state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup and monitor lifecycle"""
    logger.info("Initializing EcoSort API - Loading ResNet50 model weights...")
    app_state["start_time"] = time.time()
    try:
        app_state["model"] = load_trained_model()
        logger.info("Model loaded successfully into memory. Ready for inference.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}", exc_info=True)
        app_state["model"] = None

    yield

    logger.info("Shutting down EcoSort API - Releasing model resources...")
    app_state.clear()


app = FastAPI(
    title="EcoSort - Smart Trash Classifier REST API",
    description="RESTful API for automated waste classification into 12 categories with Explainable AI (Grad-CAM), Structured Logging, Prometheus Monitoring, and OpenTelemetry Tracing.",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Instrument FastAPI routes with OpenTelemetry
instrument_fastapi_app(app)

# Enable CORS for cross-origin web/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests_middleware(request: Request, call_next):
    """
    Middleware for distributed request tracing (X-Request-ID),
    structured JSON access logging, and RED metrics instrumentation.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path
    method = request.method

    HTTP_ACTIVE_REQUESTS.inc()
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        # Prometheus metrics
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(response.status_code)).inc()

        # Structured JSON log (exclude noisy /metrics and /health from INFO if desired, or log all)
        log_level = logger.debug if endpoint in ["/metrics", "/health"] else logger.info
        log_level(
            f"{method} {endpoint} completed {response.status_code} in {duration_ms:.2f}ms",
            extra={
                "request_id": request_id,
                "method": method,
                "path": endpoint,
                "status_code": response.status_code,
                "latency_ms": round(duration_ms, 2),
                "client_ip": client_ip,
            }
        )

        response.headers["X-Request-ID"] = request_id
        trace_id, _ = get_current_trace_and_span_ids()
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        return response

    except Exception as exc:
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code="500").inc()
        logger.error(
            f"Unhandled exception processing {method} {endpoint}: {str(exc)}",
            exc_info=True,
            extra={
                "request_id": request_id,
                "method": method,
                "path": endpoint,
                "status_code": 500,
                "latency_ms": round(duration_ms, 2),
                "client_ip": client_ip,
            }
        )
        raise exc
    finally:
        HTTP_ACTIVE_REQUESTS.dec()


def _get_pil_image_from_upload(upload_file: UploadFile) -> Tuple[Image.Image, int]:
    """Helper to validate, parse uploaded image file and return byte size"""
    if not upload_file.content_type or not upload_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {upload_file.content_type}. Only images are supported."
        )
    try:
        image_bytes = upload_file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image, len(image_bytes)
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
        "version": "1.2.0",
        "status": "online",
        "documentation": "/docs",
        "monitoring": "/metrics",
        "tracing": "OpenTelemetry + Grafana Tempo",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "classes": "/classes",
            "predict": "POST /predict",
            "predict_gradcam": "POST /predict/gradcam",
            "predict_gradcam_image": "POST /predict/gradcam/image"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Service health, model status, uptime, and memory usage"""
    is_loaded = "model" in app_state and app_state["model"] is not None
    start_time = app_state.get("start_time", time.time())
    uptime = time.time() - start_time

    try:
        proc = psutil.Process()
        mem_rss_mb = proc.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_rss_mb = None

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        supported_classes_count=len(CLASSES),
        uptime_seconds=round(uptime, 2),
        memory_rss_mb=round(mem_rss_mb, 2) if mem_rss_mb else None
    )


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """
    Prometheus scraping endpoint.
    Exposes API request throughput, latencies, model predictions by class, confidence scores, and memory.
    """
    return get_metrics_response()


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
    request: Request,
    file: UploadFile = File(..., description="Image file (JPG, PNG) to classify"),
    threshold: float = Query(0.60, ge=0.1, le=1.0, description="Minimum confidence threshold for certainty")
):
    """
    Classify a waste image into one of 12 classes with confidence scores and disposal guidelines.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready.")

    with trace_span("image.decode_and_validate", {"file.name": file.filename}) as span:
        image, file_size = _get_pil_image_from_upload(file)
        span.set_attribute("image.width", image.width)
        span.set_attribute("image.height", image.height)
        span.set_attribute("image.file_size_bytes", file_size)

    with trace_span("model.inference", {"model.architecture": "resnet50"}) as span:
        top_classes, top_confidences, latency = predict_image(model, image, top_k=3)
        span.set_attribute("prediction.class", top_classes[0])
        span.set_attribute("prediction.confidence", float(top_confidences[0]))

    pred_class = top_classes[0]
    conf = top_confidences[0]

    with trace_span("uncertainty.evaluation", {"threshold": threshold}):
        is_conf, uncert_msg = evaluate_uncertainty(conf, threshold=threshold)

    # 1. Update Prometheus ML metrics
    record_inference_metrics(
        predicted_class=pred_class,
        confidence=conf,
        is_confident=is_conf,
        latency_ms=latency
    )

    rec_info = RECYCLING_INFO.get(pred_class, {})
    top_3_items = [
        PredictionItem(
            class_name=cls_name,
            vietnamese_name=RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name),
            confidence=score
        )
        for cls_name, score in zip(top_classes, top_confidences)
    ]

    req_id = getattr(request.state, "request_id", "direct")
    client_ip = request.client.host if request.client else "unknown"
    trace_id, _ = get_current_trace_and_span_ids()

    # 2. Append to structured prediction audit log
    with trace_span("audit_log.write", {"request_id": req_id}):
        log_prediction_audit(
            request_id=req_id,
            client_ip=client_ip,
            filename=file.filename or "unknown",
            file_size_bytes=file_size,
            image_width=image.width,
            image_height=image.height,
            predicted_class=pred_class,
            confidence=conf,
            is_confident=is_conf,
            uncertainty_message=uncert_msg,
            inference_time_ms=latency,
            top_3=[{"class": item.class_name, "confidence": round(item.confidence, 4)} for item in top_3_items],
            trace_id=trace_id
        )

    return PredictionResponse(
        predicted_class=pred_class,
        vietnamese_name=rec_info.get('vietnamese_name', pred_class),
        confidence=conf,
        is_confident=is_conf,
        uncertainty_message=uncert_msg,
        top_3=top_3_items,
        recycling_info=RecyclingDetail(**rec_info),
        inference_time_ms=round(latency, 2),
        trace_id=trace_id
    )


@app.post("/predict/gradcam", response_model=GradCAMResponse, tags=["Explainable AI"])
async def predict_with_gradcam_json(
    request: Request,
    file: UploadFile = File(..., description="Image file to classify with visual explanation"),
    threshold: float = Query(0.60, ge=0.1, le=1.0, description="Minimum confidence threshold")
):
    """
    Classify a waste image and return both predictions and Base64-encoded Grad-CAM heatmap overlay.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready.")

    with trace_span("image.decode_and_validate", {"file.name": file.filename}) as span:
        image, file_size = _get_pil_image_from_upload(file)
        span.set_attribute("image.width", image.width)
        span.set_attribute("image.height", image.height)
        span.set_attribute("image.file_size_bytes", file_size)

    with trace_span("model.inference", {"model.architecture": "resnet50"}) as span:
        top_classes, top_confidences, latency = predict_image(model, image, top_k=3)
        span.set_attribute("prediction.class", top_classes[0])
        span.set_attribute("prediction.confidence", float(top_confidences[0]))

    pred_class = top_classes[0]
    conf = top_confidences[0]

    with trace_span("uncertainty.evaluation", {"threshold": threshold}):
        is_conf, uncert_msg = evaluate_uncertainty(conf, threshold=threshold)

    # Compute Grad-CAM overlay
    with trace_span("gradcam.generate", {"target_class": pred_class}):
        heatmap = generate_gradcam_heatmap(model, image)
        overlay = create_gradcam_overlay(image, heatmap)
        buffer = io.BytesIO()
        overlay.save(buffer, format="JPEG", quality=90)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Update Prometheus metrics
    record_inference_metrics(
        predicted_class=pred_class,
        confidence=conf,
        is_confident=is_conf,
        latency_ms=latency
    )

    rec_info = RECYCLING_INFO.get(pred_class, {})
    top_3_items = [
        PredictionItem(
            class_name=cls_name,
            vietnamese_name=RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name),
            confidence=score
        )
        for cls_name, score in zip(top_classes, top_confidences)
    ]

    req_id = getattr(request.state, "request_id", "direct")
    client_ip = request.client.host if request.client else "unknown"
    trace_id, _ = get_current_trace_and_span_ids()

    # Append to structured prediction audit log
    with trace_span("audit_log.write", {"request_id": req_id}):
        log_prediction_audit(
            request_id=req_id,
            client_ip=client_ip,
            filename=file.filename or "unknown",
            file_size_bytes=file_size,
            image_width=image.width,
            image_height=image.height,
            predicted_class=pred_class,
            confidence=conf,
            is_confident=is_conf,
            uncertainty_message=uncert_msg,
            inference_time_ms=latency,
            top_3=[{"class": item.class_name, "confidence": round(item.confidence, 4)} for item in top_3_items],
            trace_id=trace_id
        )

    return GradCAMResponse(
        predicted_class=pred_class,
        vietnamese_name=rec_info.get('vietnamese_name', pred_class),
        confidence=conf,
        is_confident=is_conf,
        uncertainty_message=uncert_msg,
        top_3=top_3_items,
        recycling_info=RecyclingDetail(**rec_info),
        inference_time_ms=round(latency, 2),
        gradcam_overlay_base64=base64_str,
        trace_id=trace_id
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

    with trace_span("image.decode_and_validate", {"file.name": file.filename}):
        image, _ = _get_pil_image_from_upload(file)

    with trace_span("model.inference", {"model.architecture": "resnet50"}):
        top_classes, top_confidences, latency = predict_image(model, image, top_k=1)
    
    # Record inference latency
    record_inference_metrics(
        predicted_class=top_classes[0],
        confidence=top_confidences[0],
        is_confident=True,
        latency_ms=latency
    )

    with trace_span("gradcam.generate", {"target_class": top_classes[0]}):
        heatmap = generate_gradcam_heatmap(model, image)
        overlay = create_gradcam_overlay(image, heatmap)
        buffer = io.BytesIO()
        overlay.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
