# 📊 Hướng Dẫn Logging, Monitoring & Distributed Tracing - EcoSort AI

Hệ thống Observability (Quan sát toàn diện) hoàn chỉnh chuẩn **Production & MLOps** dành cho mô hình phân loại rác thải EcoSort, bao gồm đầy đủ 3 trụ cột: **Logs**, **Metrics**, và **Traces**.

---

## 🏛️ 1. Kiến Trúc Tổng Quan (Architecture)

```
[Client / Mobile / Web]
         │  HTTP Request (Header: X-Request-ID, X-Trace-ID)
         ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      FastAPI Inference Backend                         │
 │                                                                        │
 │  ┌───────────────────────┐   ┌───────────────────┐   ┌───────────────┐ │
 │  │ OpenTelemetry Tracing │   │ Prometheus Client │   │  JSON Logger  │ │
 │  │   (Spans & Context)   │   │   (RED & MLOps)   │   │  (Rotation)   │ │
 │  └───────────┬───────────┘   └─────────┬─────────┘   └───────┬───────┘ │
 └──────────────┼─────────────────────────┼─────────────────────┼─────────┘
                │ OTLP gRPC (4317)        │ Scrape /metrics     │
                ▼                         ▼                     ▼
     ┌─────────────────────┐   ┌─────────────────────┐ ┌──────────────────┐
     │    Grafana Tempo    │   │  Prometheus Server  │ │ logs/api.log     │
     │     (Port 3200)     │   │     (Port 9090)     │ │ predictions.jsonl│
     └──────────┬──────────┘   └──────────┬──────────┘ └──────────────────┘
                │                         │
                └───────────┬─────────────┘
                            │ Query Traces & Metrics
                            ▼
                 ┌─────────────────────┐
                 │  Grafana Dashboard  │
                 │     (Port 3000)     │
                 │ (Metrics & Tracing) │
                 └─────────────────────┘
```

---

## 📁 2. Cấu Trúc Thư Mục Đã Triển Khai

- [api/tracing.py](file:///D:/hoc_AI/smart-trash-classifier/api/tracing.py): Cấu hình OpenTelemetry TracerProvider, OTLP gRPC exporter tới Grafana Tempo, context manager `trace_span` bóc tách từng công đoạn inference.
- [api/logger.py](file:///D:/hoc_AI/smart-trash-classifier/api/logger.py): Structured JSON logging, tự động xoay vòng file (10MB x 5 backups), tương quan tự động với `trace_id` và `span_id`.
- [api/metrics.py](file:///D:/hoc_AI/smart-trash-classifier/api/metrics.py): Định nghĩa bộ chỉ số Prometheus (RED metrics, ML inference metrics, System gauges).
- [api/main.py](file:///D:/hoc_AI/smart-trash-classifier/api/main.py): Tích hợp OpenTelemetry FastAPI middleware, bóc tách span con cho `image.decode_and_validate`, `model.inference`, `uncertainty.evaluation`, `gradcam.generate`, `audit_log.write`.
- [monitoring/tempo/tempo.yml](file:///D:/hoc_AI/smart-trash-classifier/monitoring/tempo/tempo.yml): Cấu hình backend lưu trữ trace Grafana Tempo tiếp nhận OTLP gRPC và HTTP.
- [monitoring/prometheus/prometheus.yml](file:///D:/hoc_AI/smart-trash-classifier/monitoring/prometheus/prometheus.yml): Cấu hình Prometheus scraper cào metrics từ `/metrics` mỗi 5s.
- [monitoring/grafana/provisioning/datasources/datasource.yml](file:///D:/hoc_AI/smart-trash-classifier/monitoring/grafana/provisioning/datasources/datasource.yml): Provisioning tự động 2 data source: Prometheus và Tempo.
- [monitoring/grafana/dashboards/ecosort_dashboard.json](file:///D:/hoc_AI/smart-trash-classifier/monitoring/grafana/dashboards/ecosort_dashboard.json): Dashboard Grafana thiết kế sẵn cho EcoSort AI.
- [docker-compose.yml](file:///D:/hoc_AI/smart-trash-classifier/docker-compose.yml): Cụm dịch vụ tích hợp gồm API, Web, Prometheus, Tempo và Grafana.

---

## 🔍 3. Distributed Tracing (OpenTelemetry + Grafana Tempo)

### 3.1. Phân Tách Spans Trong Pipeline AI

Mỗi request gửi tới endpoint `/predict` hoặc `/predict/gradcam` đều được OpenTelemetry tạo 1 Root Span và các Child Spans chi tiết:

```text
[Trace: POST /predict] ────────────────────────────────────────────────────────── (300ms)
  ├── span: image.decode_and_validate ─────────── (15ms)
  │     [attributes: file.name, dimensions, file_size_bytes]
  ├── span: model.inference ───────────────────── (270ms)
  │     [attributes: model.architecture, prediction.class, prediction.confidence]
  ├── span: uncertainty.evaluation ────────────── (1ms)
  │     [attributes: threshold=0.6]
  ├── span: gradcam.generate ──────────────────── (60ms) (chỉ có khi gọi /gradcam)
  │     [attributes: target_class]
  └── span: audit_log.write ───────────────────── (2ms)
        [attributes: request_id]
```

### 3.2. Liên Kết Chéo (Traces-to-Logs Correlation)

Khi có request, hệ thống tự động:
1. Gán header `X-Trace-ID` và `X-Request-ID` trả về cho Client.
2. Tự động tiêm `trace_id` và `span_id` vào từng dòng log trong [logs/api.log](file:///D:/hoc_AI/smart-trash-classifier/logs/api.log) và [logs/predictions.jsonl](file:///D:/hoc_AI/smart-trash-classifier/logs/predictions.jsonl).
3. Khi debug lỗi hoặc kiểm toán độ tin cậy thấp, chỉ cần copy `trace_id` dán vào mục **Explore -> Tempo** trên Grafana để xem toàn bộ sơ đồ thời gian thực thi (Flamegraph / Waterfall).

---

## 📑 4. Định Dạng Log (Structured JSON)

### 4.1. Log Hoạt Động API (`logs/api.log`)
```json
{
  "timestamp": "2026-10-09T02:55:20.065335Z",
  "level": "INFO",
  "logger": "ecosort-api",
  "message": "POST /predict completed 200 in 300.31ms",
  "request_id": "19518dfc-3fbf-4318-b218-9ea97ffb2164",
  "trace_id": "95b6c21c0cfe9e271e8fdc45a55710bb",
  "span_id": "a70bc974f9a6c5c1",
  "method": "POST",
  "path": "/predict",
  "status_code": 200,
  "latency_ms": 300.31,
  "client_ip": "127.0.0.1"
}
```

### 4.2. Log Audit Inference (`logs/predictions.jsonl`)
```json
{
  "timestamp": "2026-10-09T02:55:20.065335Z",
  "request_id": "19518dfc-3fbf-4318-b218-9ea97ffb2164",
  "trace_id": "95b6c21c0cfe9e271e8fdc45a55710bb",
  "client_ip": "127.0.0.1",
  "input": {
    "filename": "battery1.jpg",
    "file_size_bytes": 7405,
    "dimensions": [280, 180]
  },
  "output": {
    "predicted_class": "battery",
    "confidence": 1.0,
    "is_confident": true,
    "uncertainty_message": "Độ tin cậy cao, kết quả nhận diện đáng tin cậy.",
    "inference_time_ms": 276.93,
    "top_3": [
      {"class": "battery", "confidence": 1.0},
      {"class": "metal", "confidence": 0.0},
      {"class": "cardboard", "confidence": 0.0}
    ]
  }
}
```

---

## 📈 5. Các Chỉ Số Giám Sát (Prometheus Metrics)

| Tên Metric | Kiểu | Mô tả | Ứng Dụng Thực Tế |
|---|---|---|---|
| `ecosort_http_requests_total` | Counter | Tổng request theo method, path, status_code | Đo RPS, tỷ lệ lỗi 4xx/5xx |
| `ecosort_http_request_duration_seconds` | Histogram | Thời gian xử lý toàn bộ request | Tính P50, P95, P99 request latency |
| `ecosort_http_active_requests` | Gauge | Số lượng request đang xử lý đồng thời | Theo dõi tải tức thời |
| `ecosort_model_inference_duration_seconds` | Histogram | Thời gian model tính toán (trừ upload I/O) | Giám sát hiệu năng phần cứng/model |
| `ecosort_model_predictions_total` | Counter | Số lượt dự đoán chia theo từng loại rác | Phát hiện phân bố rác và data drift |
| `ecosort_model_confidence_score` | Histogram | Phân bố điểm tự tin (0.0 → 1.0) | Cảnh báo khi confidence bị sụt giảm bất thường |
| `ecosort_model_uncertainty_total` | Counter | Số lần dự đoán dưới ngưỡng tin cậy | Giám sát chất lượng dữ liệu đầu vào |
| `ecosort_system_memory_usage_bytes` | Gauge | RAM process đang sử dụng (RSS bytes) | Phát hiện Memory Leak |
| `ecosort_system_cpu_usage_percent` | Gauge | % CPU của process | Phát hiện nghẽn CPU |

---

## 🚀 6. Hướng Dẫn Khởi Chạy

### Cách 1: Khởi Chạy Toàn Bộ Stack Với Docker Compose (Khuyên Dùng)
*(Bao gồm: FastAPI, Web App, Prometheus, Grafana Tempo và Grafana)*

1. **Khởi động toàn bộ container**:
   ```powershell
   docker compose up -d --build
   ```
2. **Truy cập các dịch vụ**:
   - **FastAPI Backend Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Streamlit Web App**: [http://localhost:8501](http://localhost:8501)
   - **Prometheus UI**: [http://localhost:9090](http://localhost:9090)
   - **Grafana Dashboard & Tracing**: [http://localhost:3000](http://localhost:3000)
     - Tài khoản: `admin` / Mật khẩu: `admin`
     - **Xem Metrics**: Menu Dashboards -> `EcoSort - Waste Classification MLOps`.
     - **Xem Tracing**: Menu Explore -> Chọn Data source `Tempo` -> Search theo Service `ecosort-api` hoặc dán trực tiếp `trace_id`.

### Cách 2: Chạy Kiểm Thử Nhanh (Verification Test)
Chạy script kiểm tra tự động toàn bộ luồng inference, log JSON, trace header và prometheus metrics:
```powershell
python test_monitoring.py
```
