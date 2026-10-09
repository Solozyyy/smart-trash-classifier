"""
Smart Trash Classifier - Streamlit Web Application
AI-powered waste classification with Explainable AI (Grad-CAM), Camera capture,
and Full Observability (Arize Phoenix Cloud Tracing, Local Logging & MLOps Monitoring Dashboard).
"""

import os
import sys
import json
import time
import uuid
from PIL import Image
import streamlit as st
import pandas as pd
import psutil
from dotenv import load_dotenv

load_dotenv()

# Ensure repository root and app directory are on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_DIR = os.path.abspath(os.path.dirname(__file__))
for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Sync secrets.toml to os.environ for Streamlit Cloud deployment
if hasattr(st, "secrets"):
    for k in ["PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_PROJECT_NAME"]:
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])

from download_model import ensure_model_exists
from src.dataset import CLASSES
from src.model import load_trained_model, predict_image, generate_gradcam_heatmap
from src.utils import RECYCLING_INFO, create_gradcam_overlay, evaluate_uncertainty
from opentelemetry import trace
from api.tracing import setup_tracing, trace_span, get_current_trace_and_span_ids, flush_tracing
from api.logger import logger, log_prediction_audit, APP_LOG_FILE, PREDICTION_LOG_FILE
from api.metrics import record_inference_metrics

# Initialize Distributed Tracing (Arize Phoenix Cloud or Local Tempo)
setup_tracing(service_name="ecosort-streamlit-app")

# Streamlit Page Config
st.set_page_config(
    page_title="EcoSort AI - Waste Classifier & Observability",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme-Adaptive)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.6rem;
        color: #2E7D32;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .prediction-box-certain {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e7e34 0%, #28a745 100%);
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .prediction-box-uncertain {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #d39e00 0%, #e0a800 100%);
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .trace-badge {
        background: rgba(0, 128, 128, 0.08);
        border: 1px solid rgba(0, 128, 128, 0.3);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.8rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Cached loader for trained ResNet50 model with cloud auto-download fallback"""
    with st.spinner("⏳ Đang chuẩn bị mô hình AI (lần đầu có thể mất 1-2 phút)..."):
        model_path = ensure_model_exists()
        return load_trained_model(model_path)


def load_predictions_history(limit: int = 50) -> pd.DataFrame:
    """Read local prediction audit log into a pandas DataFrame"""
    if not os.path.exists(PREDICTION_LOG_FILE):
        return pd.DataFrame()
    records = []
    try:
        with open(PREDICTION_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append({
                            "Thời gian (UTC)": record.get("timestamp", "")[:19].replace("T", " "),
                            "Ảnh": record.get("input", {}).get("filename", "unknown"),
                            "Loại rác": record.get("output", {}).get("predicted_class", "unknown"),
                            "Độ tin cậy": f"{record.get('output', {}).get('confidence', 0.0)*100:.1f}%",
                            "Tự tin": "✅ Có" if record.get("output", {}).get("is_confident") else "⚠️ Thấp",
                            "Độ trễ (ms)": record.get("output", {}).get("inference_time_ms", 0.0),
                            "Trace ID": record.get("trace_id", "N/A"),
                        })
                    except Exception:
                        pass
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.iloc[::-1].head(limit)
        return df
    except Exception:
        return pd.DataFrame()


def load_recent_api_logs(limit: int = 40) -> list:
    """Read recent lines from api.log"""
    if not os.path.exists(APP_LOG_FILE):
        return []
    try:
        with open(APP_LOG_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            return lines[-limit:]
    except Exception:
        return []


def main():
    st.markdown('<p class="main-header">♻️ EcoSort AI - Smart Trash Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Phân loại rác thải tự động • Giải thích AI (Grad-CAM) • Giám sát Arize Phoenix Cloud</p>',
        unsafe_allow_html=True
    )

    phoenix_key = os.getenv("PHOENIX_API_KEY")
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com")
    phoenix_project = os.getenv("PHOENIX_PROJECT_NAME", "ecosort-trash-classifier")
    phoenix_ui_url = phoenix_endpoint if "app.phoenix.arize.com" in phoenix_endpoint else "https://app.phoenix.arize.com"

    # Sidebar settings and info
    with st.sidebar:
        st.header("⚙️ Cấu hình Nhận diện")
        confidence_threshold = st.slider(
            "Ngưỡng tin cậy tối thiểu (Uncertainty Threshold)",
            min_value=0.40,
            max_value=0.90,
            value=0.60,
            step=0.05,
            help="Nếu độ tin cậy < ngưỡng này, hệ thống sẽ cảnh báo không chắc chắn."
        )

        enable_gradcam = st.checkbox(
            "🔥 Bật Explainable AI (Grad-CAM Heatmap)",
            value=True,
            help="Hiển thị vùng ảnh mà mô hình tích chập tập trung quan sát."
        )

        gradcam_alpha = st.slider(
            "Độ mờ Heatmap (Alpha Overlay)",
            min_value=0.2,
            max_value=0.8,
            value=0.45,
            step=0.05,
            disabled=not enable_gradcam
        )

        st.markdown("---")
        st.header("🔭 Cloud Observability")
        if phoenix_key:
            st.success("🟢 **Arize Phoenix Cloud:** Đã kết nối")
            st.caption(f"Project: `{phoenix_project}`")
            st.link_button("🌐 Mở Phoenix Cloud Dashboard", phoenix_ui_url, use_container_width=True)
        else:
            st.info("🟡 **Local Tracing Mode**")
            st.caption("Chưa có PHOENIX_API_KEY, xuất trace cục bộ.")

        st.markdown("---")
        st.header("📊 Thông tin Mô hình")
        st.write("""
        - **Kiến trúc:** ResNet50 (Transfer Learning)
        - **Độ chính xác:** ~93.63% (12 nhóm rác)
        - **Tracing:** OpenTelemetry + Arize Phoenix
        """)

        st.markdown("---")
        st.caption("EcoSort AI System | Production MLOps")

    # Main Tabs: 1. Classifier, 2. Monitoring & Logs
    tab_classifier, tab_monitoring = st.tabs([
        "♻️ Phân Loại Rác & XAI",
        "📊 Giám Sát & Nhật Ký (MLOps Dashboard)"
    ])

    # Load model
    try:
        with st.spinner("Đang nạp mô hình Deep Learning..."):
            model = get_model()
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình: {str(e)}")
        st.stop()

    # =========================================================================
    # TAB 1: PHÂN LOẠI RÁC & XAI
    # =========================================================================
    with tab_classifier:
        st.markdown("### 📥 Chọn phương thức cung cấp ảnh")
        input_tab1, input_tab2 = st.tabs(["📁 Tải ảnh từ thiết bị (Upload File)", "📷 Chụp trực tiếp bằng Camera"])

        image_source = None
        with input_tab1:
            uploaded_file = st.file_uploader(
                "Kéo & thả ảnh rác thải vào đây (JPG, JPEG, PNG)",
                type=['jpg', 'jpeg', 'png'],
                key="file_uploader"
            )
            if uploaded_file is not None:
                image_source = uploaded_file

        with input_tab2:
            camera_file = st.camera_input("Chụp ảnh vật thể rác qua Camera / Webcam", key="camera_input")
            if camera_file is not None:
                image_source = camera_file

        if image_source is not None:
            try:
                image = Image.open(image_source)
            except Exception as err:
                st.error(f"Không thể mở file ảnh: {err}")
                return

            request_id = str(uuid.uuid4())
            filename = getattr(image_source, "name", "camera_capture.jpg")
            filesize = getattr(image_source, "size", len(image_source.getvalue()) if hasattr(image_source, "getvalue") else 0)

            # Start OpenTelemetry Root Trace targeting Phoenix Cloud
            with trace_span("streamlit.classify_trash", {
                "openinference.span.kind": "chain",
                "input.value": f"Image: {filename} ({image.width}x{image.height}px, {filesize} bytes)",
                "request.id": request_id,
                "file.name": filename,
                "client.type": "streamlit_web"
            }) as root_span:

                with trace_span("image.decode_and_validate") as span:
                    span.set_attribute("image.width", image.width)
                    span.set_attribute("image.height", image.height)
                    span.set_attribute("image.size_bytes", filesize)

                with trace_span("model.inference", {"model.architecture": "resnet50"}) as span:
                    top_classes, top_confidences, inf_time = predict_image(model, image, top_k=3)
                    predicted_class = top_classes[0]
                    confidence = top_confidences[0]
                    span.set_attribute("prediction.class", predicted_class)
                    span.set_attribute("prediction.confidence", float(confidence))

                with trace_span("uncertainty.evaluation", {"threshold": confidence_threshold}) as span:
                    is_certain, uncertainty_msg = evaluate_uncertainty(confidence, threshold=confidence_threshold)
                    span.set_attribute("prediction.is_confident", is_certain)

                # Update Prometheus Metrics
                record_inference_metrics(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    is_confident=is_certain,
                    latency_ms=inf_time
                )

                # Grad-CAM heatmap
                heatmap = None
                overlay_img = None
                if enable_gradcam:
                    with trace_span("gradcam.generate", {"target_class": predicted_class}):
                        heatmap = generate_gradcam_heatmap(model, image, class_idx=CLASSES.index(predicted_class))
                        overlay_img = create_gradcam_overlay(image, heatmap, alpha=gradcam_alpha)

                trace_id, span_id = get_current_trace_and_span_ids()

                # Audit Log
                with trace_span("audit_log.write"):
                    top_3_items = [{"class": c, "confidence": round(float(cf), 4)} for c, cf in zip(top_classes, top_confidences)]
                    log_prediction_audit(
                        request_id=request_id,
                        client_ip="streamlit-client",
                        filename=filename,
                        file_size_bytes=filesize,
                        image_width=image.width,
                        image_height=image.height,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        is_confident=is_certain,
                        uncertainty_message=uncertainty_msg,
                        inference_time_ms=inf_time,
                        top_3=top_3_items,
                        trace_id=trace_id
                    )
                    logger.info(
                        f"Streamlit prediction completed: {predicted_class} ({confidence*100:.1f}%) in {inf_time:.1f}ms",
                        extra={
                            "request_id": request_id,
                            "method": "STREAMLIT",
                            "path": "/classify",
                            "status_code": 200,
                            "latency_ms": round(inf_time, 2),
                            "client_ip": "streamlit-client",
                        }
                    )

                # Set OpenInference formatted output & OK status for Phoenix
                viet_name_out = RECYCLING_INFO.get(predicted_class, {}).get('vietnamese_name', predicted_class)
                root_span.set_attribute(
                    "output.value",
                    f"Predicted: {predicted_class} ({confidence*100:.1f}%) | Tiếng Việt: {viet_name_out} | Tin cậy: {is_certain}"
                )
                root_span.set_status(trace.Status(trace.StatusCode.OK))

            # Force flush spans immediately to Arize Phoenix Cloud
            flush_tracing()

            # Display Results
            col_img, col_pred = st.columns([1, 1], gap="medium")

            with col_img:
                st.markdown("#### 📸 Ảnh đầu vào")
                st.image(image, use_container_width=True, caption=f"Kích thước gốc: {image.size[0]}x{image.size[1]}px")

            with col_pred:
                st.markdown("#### 🎯 Kết quả phân loại")

                box_class = "prediction-box-certain" if is_certain else "prediction-box-uncertain"
                viet_name = RECYCLING_INFO.get(predicted_class, {}).get('vietnamese_name', predicted_class)

                st.markdown(f"""
                <div class="{box_class}">
                    <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Loại rác dự đoán</div>
                    <h2 style="margin: 0.3rem 0; font-size: 2.2rem; color: white;">{viet_name}</h2>
                    <div style="font-size: 1.1rem; opacity: 0.95; color: white;">Tên quốc tế: <b>{predicted_class.upper()}</b></div>
                    <h3 style="margin: 0.5rem 0 0 0; color: white;">Độ tin cậy: {confidence*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                if not is_certain:
                    st.warning(f"⚠️ **Cảnh báo độ tin cậy thấp:** {uncertainty_msg}")
                else:
                    st.success("✅ Mô hình tự tin với kết quả nhận diện này.")

                st.markdown("##### 📊 Top 3 lớp có xác suất cao nhất:")
                for i, (cls_name, conf) in enumerate(zip(top_classes, top_confidences), 1):
                    vn_label = RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name)
                    st.write(f"**{i}. {vn_label}** (`{cls_name}`): **{conf*100:.2f}%**")
                    st.progress(float(conf))

                st.caption(f"⚡ Thời gian suy luận: **{inf_time:.1f} ms**")

                # Trace Link Badge
                if trace_id:
                    st.markdown(f"""
                    <div class="trace-badge">
                        <div>
                            <span style="font-size: 0.85rem; font-weight: bold; color: #008080;">🔭 Arize Phoenix Trace ID:</span><br>
                            <code style="font-size: 0.95rem; font-weight: bold;">{trace_id}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.link_button("🔍 Mở Trace này trên Phoenix Cloud", phoenix_ui_url, use_container_width=True)

            # Grad-CAM Section
            if enable_gradcam and overlay_img is not None:
                st.markdown("---")
                st.markdown("### 🔥 Giải thích AI bằng Grad-CAM (Visual Interpretability)")

                cam_options = {}
                for i, (cls_name, conf) in enumerate(zip(top_classes, top_confidences), 1):
                    vn_label = RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name)
                    label_text = f"Top {i}: {vn_label} ({cls_name}) - {conf*100:.1f}%"
                    cam_options[label_text] = (cls_name, CLASSES.index(cls_name))

                selected_cam_label = st.radio(
                    "🔍 Chọn góc nhìn phân tích theo lớp:",
                    options=list(cam_options.keys()),
                    index=0,
                    horizontal=True
                )
                target_cls_name, target_cls_idx = cam_options[selected_cam_label]

                if target_cls_idx != CLASSES.index(predicted_class):
                    with st.spinner(f"Đang tính toán Grad-CAM Heatmap cho lớp {target_cls_name}..."):
                        heatmap = generate_gradcam_heatmap(model, image, class_idx=target_cls_idx)
                        overlay_img = create_gradcam_overlay(image, heatmap, alpha=gradcam_alpha)

                col_cam1, col_cam2 = st.columns(2, gap="medium")
                with col_cam1:
                    st.image(image, use_container_width=True, caption="Ảnh đầu vào gốc")
                with col_cam2:
                    st.image(
                        overlay_img,
                        use_container_width=True,
                        caption=f"Bản đồ nhiệt Grad-CAM theo góc nhìn lớp: {target_cls_name.upper()}"
                    )

            # Recycling Information Guide
            st.markdown("---")
            st.markdown("### ♻️ Hướng dẫn Xử lý & Tái chế Rác thải")
            info = RECYCLING_INFO.get(predicted_class, {})
            rec_col1, rec_col2 = st.columns(2, gap="medium")

            with rec_col1:
                with st.container(border=True):
                    st.markdown("#### 📌 Phân loại")
                    st.markdown(f"**{info.get('category', 'N/A')}**")
                    st.markdown("---")
                    st.markdown("#### 🗑️ Thùng rác quy định")
                    st.markdown(f"**{info.get('bin_color', 'N/A')}**")
                    st.markdown("---")
                    st.markdown("#### ⏱️ Thời gian phân hủy tự nhiên")
                    st.markdown(f"**{info.get('decomposition_time', 'N/A')}**")

            with rec_col2:
                with st.container(border=True):
                    st.markdown("#### 📋 Hướng dẫn xử lý đúng cách")
                    instructions = info.get('instructions', [])
                    if instructions:
                        for inst in instructions:
                            st.markdown(f"- {inst}")
                    else:
                        st.write("Không có hướng dẫn chi tiết.")

            with st.container(border=True):
                st.markdown("#### 🌍 Tác động Môi trường & Giá trị Tái chế")
                st.markdown(f"{info.get('impact', 'N/A')}")

        else:
            st.info("👆 Vui lòng kéo thả file ảnh ở tab bên trên hoặc bấm vào tab Camera để chụp ảnh rác!")
            st.markdown("### 📦 12 Nhóm rác được hỗ trợ:")
            cols = st.columns(4)
            for idx, (cls_key, cls_val) in enumerate(RECYCLING_INFO.items()):
                col_target = cols[idx % 4]
                with col_target:
                    st.markdown(f"- **{cls_val['vietnamese_name']}** (`{cls_key}`)")

    # =========================================================================
    # TAB 2: GIÁM SÁT & NHẬT KÝ (MLOPS DASHBOARD)
    # =========================================================================
    with tab_monitoring:
        st.markdown("### 📊 MLOps Dashboard & Nhật Ký Kiểm Toán")
        st.markdown(f"""
        Toàn bộ tương tác phân loại rác được đồng bộ thời gian thực lên **[Arize Phoenix Cloud]({phoenix_ui_url})**.
        Bạn có thể xem các chỉ số tổng hợp tại đây hoặc mở trực tiếp trang quản trị Cloud.
        """)

        col_top_a, col_top_b = st.columns([3, 1])
        with col_top_b:
            if phoenix_key:
                st.link_button("🚀 Mở Arize Phoenix Cloud", phoenix_ui_url, use_container_width=True)

        # 1. Load history DataFrame
        df_history = load_predictions_history(limit=100)

        if not df_history.empty:
            # Metrics Row
            total_preds = len(df_history)
            confident_count = (df_history["Tự tin"] == "✅ Có").sum()
            confident_pct = (confident_count / total_preds) * 100 if total_preds > 0 else 0
            avg_latency = df_history["Độ trễ (ms)"].mean() if total_preds > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎯 Tổng lượt phân loại", total_preds)
            m2.metric("✅ Tỷ lệ tự tin", f"{confident_pct:.1f}%")
            m3.metric("⚡ Độ trễ TB (Latency)", f"{avg_latency:.1f} ms")
            m4.metric("⚠️ Cảnh báo không chắc chắn", f"{total_preds - confident_count} lượt")

            st.markdown("---")

            # Charts Row
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📈 Phân bố các loại rác đã nhận diện")
                class_counts = df_history["Loại rác"].value_counts()
                st.bar_chart(class_counts)

            with c2:
                st.markdown("##### ⏱️ Thời gian phản hồi suy luận (Latency ms)")
                latency_series = df_history[["Độ trễ (ms)"]].reset_index(drop=True)
                st.line_chart(latency_series)

            # Audit Table
            st.markdown("---")
            st.markdown("##### 📋 Lịch sử kiểm toán gần nhất (Audit Trail Table)")
            st.dataframe(
                df_history,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Chưa có lượt dự đoán nào được ghi nhận. Hãy qua Tab 1 để tải ảnh phân loại rác!")

        # 2. System Hardware Metrics
        st.markdown("---")
        st.markdown("##### 💻 Sức khỏe Tài nguyên Máy chủ (System Health)")
        try:
            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            cpu_pct = proc.cpu_percent(interval=None)
            total_sys_ram = psutil.virtual_memory().percent
        except Exception:
            mem_mb, cpu_pct, total_sys_ram = 0, 0, 0

        h1, h2, h3 = st.columns(3)
        h1.metric("Bộ nhớ RAM Process", f"{mem_mb:.1f} MB")
        h2.metric("Mức chiếm CPU Process", f"{cpu_pct:.1f}%")
        h3.metric("RAM Hệ thống", f"{total_sys_ram:.1f}%")

        # 3. Live Log Viewer
        st.markdown("---")
        with st.expander("📜 Xem nhật ký ứng dụng trực tiếp (Live Logs - logs/api.log)", expanded=False):
            log_lines = load_recent_api_logs(limit=30)
            if log_lines:
                for line in reversed(log_lines):
                    st.code(line, language="json")
            else:
                st.caption("Chưa có log trong logs/api.log")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        EcoSort Project • ResNet50 Transfer Learning + Grad-CAM XAI • OpenTelemetry & Arize Phoenix
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
