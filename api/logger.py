"""
Structured Logging Module for EcoSort API.
Provides JSON formatting, console colorized output, file rotation, and prediction audit logging.
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any

# Root directory of logs
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

APP_LOG_FILE = os.path.join(LOG_DIR, "api.log")
PREDICTION_LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")


class JsonFormatter(logging.Formatter):
    """
    Format log records as structured JSON for easy ingestion by
    ELK, Loki, Datadog, CloudWatch, etc.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom fields attached to record (e.g., request_id, latency_ms)
        for key in ["request_id", "method", "path", "status_code", "latency_ms", "client_ip"]:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        # OpenTelemetry Trace Correlation (Traces-to-Logs correlation)
        try:
            from api.tracing import get_current_trace_and_span_ids
            trace_id, span_id = get_current_trace_and_span_ids()
            if trace_id:
                log_obj["trace_id"] = trace_id
                log_obj["span_id"] = span_id
        except Exception:
            pass

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(name: str = "ecosort-api") -> logging.Logger:
    """
    Configure API logger with both Console output (readable) and
    Rotating File output (JSON structured).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # 1. Console Handler (Human-readable for dev & terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 2. File Handler (Structured JSON with rotation: 10MB per file, 5 backups)
    file_handler = RotatingFileHandler(
        APP_LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger("ecosort-api")


def log_prediction_audit(
    request_id: str,
    client_ip: str,
    filename: str,
    file_size_bytes: int,
    image_width: int,
    image_height: int,
    predicted_class: str,
    confidence: float,
    is_confident: bool,
    uncertainty_message: str,
    inference_time_ms: float,
    top_3: list,
    trace_id: Optional[str] = None
):
    """
    Log an individual model prediction to an append-only JSONL file.
    Used for audit trails, MLOps monitoring, and offline data drift analysis.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "trace_id": trace_id,
        "client_ip": client_ip,
        "input": {
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "dimensions": [image_width, image_height],
        },
        "output": {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "is_confident": is_confident,
            "uncertainty_message": uncertainty_message,
            "inference_time_ms": round(inference_time_ms, 2),
            "top_3": top_3,
        }
    }

    try:
        with open(PREDICTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write prediction audit log: {str(e)}")
