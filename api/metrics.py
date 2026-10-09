"""
Prometheus Metrics Exporter for EcoSort API.
Defines system, API HTTP, and Machine Learning model operational metrics.
"""

import time
import psutil
from fastapi import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)

# -------------------------------------------------------------------------
# 1. HTTP Traffic & Latency Metrics (RED pattern: Rate, Errors, Duration)
# -------------------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "ecosort_http_requests_total",
    "Total count of HTTP requests handled by the API",
    ["method", "endpoint", "status_code"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "ecosort_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

HTTP_ACTIVE_REQUESTS = Gauge(
    "ecosort_http_active_requests",
    "Number of HTTP requests currently being processed"
)

# -------------------------------------------------------------------------
# 2. Machine Learning Operational Metrics (MLOps)
# -------------------------------------------------------------------------
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "ecosort_model_inference_duration_seconds",
    "Model inference latency in seconds (excluding network/upload I/O)",
    ["model_name"],
    buckets=(0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0)
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "ecosort_model_predictions_total",
    "Total number of predictions categorized by waste class",
    ["predicted_class", "is_confident"]
)

MODEL_CONFIDENCE_SCORE = Histogram(
    "ecosort_model_confidence_score",
    "Distribution of prediction confidence scores (useful for detecting data drift)",
    buckets=(0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
)

MODEL_UNCERTAINTY_TOTAL = Counter(
    "ecosort_model_uncertainty_total",
    "Total number of low-confidence or uncertain predictions",
    ["predicted_class"]
)

# -------------------------------------------------------------------------
# 3. System Resource Utilization
# -------------------------------------------------------------------------
SYSTEM_MEMORY_USAGE_BYTES = Gauge(
    "ecosort_system_memory_usage_bytes",
    "Memory (Resident Set Size) consumed by the API process in bytes"
)

SYSTEM_CPU_USAGE_PERCENT = Gauge(
    "ecosort_system_cpu_usage_percent",
    "Current CPU utilization percentage of the API process"
)

# Keep track of process
_current_process = psutil.Process()


def update_system_metrics():
    """Update process-level RAM and CPU usage gauges"""
    try:
        mem_info = _current_process.memory_info()
        SYSTEM_MEMORY_USAGE_BYTES.set(mem_info.rss)
        SYSTEM_CPU_USAGE_PERCENT.set(_current_process.cpu_percent(interval=None))
    except Exception:
        pass


def record_inference_metrics(predicted_class: str, confidence: float, is_confident: bool, latency_ms: float):
    """Convenience helper to update all model-related Prometheus metrics in one call"""
    # Pure inference time in seconds
    MODEL_INFERENCE_DURATION_SECONDS.labels(model_name="resnet50_waste").observe(latency_ms / 1000.0)

    # Prediction count per class
    MODEL_PREDICTIONS_TOTAL.labels(
        predicted_class=predicted_class,
        is_confident=str(is_confident).lower()
    ).inc()

    # Confidence distribution histogram
    MODEL_CONFIDENCE_SCORE.observe(confidence)

    # Low-confidence count
    if not is_confident:
        MODEL_UNCERTAINTY_TOTAL.labels(predicted_class=predicted_class).inc()


def get_metrics_response() -> Response:
    """Generate Prometheus exposition format payload"""
    update_system_metrics()
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )
