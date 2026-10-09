"""
Quick verification script for EcoSort Logging & Monitoring.
Sends a test image to the API and displays formatted logs and Prometheus metrics.
"""

import os
import sys
from fastapi.testclient import TestClient

# Ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from api.main import app

def run_test():
    print("=" * 60)
    print("  EcoSort AI - Logging & Monitoring Verification Test")
    print("=" * 60)

    sample_image = "data/battery/battery1.jpg"
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        return

    with TestClient(app) as client:
        # 1. Health check
        health_resp = client.get("/health")
        print(f"\n[1] Health Check: Status {health_resp.status_code}")
        print("    Response:", health_resp.json())

        # 2. Predict image
        print(f"\n[2] Testing Inference with '{sample_image}'...")
        with open(sample_image, "rb") as f:
            pred_resp = client.post("/predict", files={"file": ("battery1.jpg", f, "image/jpeg")})
        print(f"    Inference Status: {pred_resp.status_code}")
        print(f"    Request Header (X-Request-ID): {pred_resp.headers.get('x-request-id')}")
        print(f"    Trace Header (X-Trace-ID):     {pred_resp.headers.get('x-trace-id')}")
        pred_data = pred_resp.json()
        print(f"    Prediction: {pred_data.get('predicted_class')} (Confidence: {pred_data.get('confidence'):.2%})")
        print(f"    Payload Trace ID: {pred_data.get('trace_id')}")
        print(f"    Latency: {pred_data.get('inference_time_ms')} ms")

        # 3. Prometheus metrics
        print("\n[3] Prometheus Metrics Sample (/metrics):")
        metrics_resp = client.get("/metrics")
        for line in metrics_resp.text.splitlines():
            if line.startswith("ecosort_") and not line.startswith("ecosort_http_request_duration_seconds_bucket"):
                print(f"    {line}")

    print("\n" + "=" * 60)
    print(" Verification complete!")
    print(" Log files generated at:")
    print("   - logs/api.log           (Structured JSON requests)")
    print("   - logs/predictions.jsonl (ML audit & data drift log)")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
