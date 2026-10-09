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
from datetime import datetime, timezone, timedelta
from PIL import Image
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Ensure repository root and app directory are on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_DIR = os.path.abspath(os.path.dirname(__file__))
for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Helper to retrieve config from Streamlit Secrets or Environment
def get_config_val(key: str, default: str = "") -> str:
    val = ""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key])
    except Exception:
        pass
    if not val:
        val = os.getenv(key, default)
    if val:
        val = str(val).strip("'\" \t\r\n")
    return val or default

_PHOENIX_API_KEY = get_config_val("PHOENIX_API_KEY")
_PHOENIX_COLLECTOR = get_config_val("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/s/hnkhoa04/v1/traces")
_PHOENIX_PROJECT = get_config_val("PHOENIX_PROJECT_NAME", "ecosort-trash-classifier")

# Sync to os.environ as fallback
if _PHOENIX_API_KEY:
    os.environ["PHOENIX_API_KEY"] = _PHOENIX_API_KEY
if _PHOENIX_COLLECTOR:
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = _PHOENIX_COLLECTOR
if _PHOENIX_PROJECT:
    os.environ["PHOENIX_PROJECT_NAME"] = _PHOENIX_PROJECT

from download_model import ensure_model_exists
from src.dataset import CLASSES
from src.model import load_trained_model, predict_image, generate_gradcam_heatmap
import importlib
import src.utils
importlib.reload(src.utils)
from src.utils import RECYCLING_INFO, create_gradcam_overlay, evaluate_uncertainty
from opentelemetry import trace

try:
    import api.tracing
    importlib.reload(api.tracing)
    from api.tracing import setup_tracing, trace_span, get_current_trace_and_span_ids, flush_tracing, get_tracing_status
except Exception:
    from api.tracing import setup_tracing, trace_span, get_current_trace_and_span_ids
    def flush_tracing(timeout_millis: int = 3000):
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            try:
                provider.force_flush(timeout_millis)
            except Exception:
                pass
    def get_tracing_status():
        return {"active": False, "provider": "none"}

from api.logger import logger, log_prediction_audit, APP_LOG_FILE, PREDICTION_LOG_FILE
from api.metrics import record_inference_metrics

# Initialize Distributed Tracing (Arize Phoenix Cloud or Local Tempo)
setup_tracing(
    service_name="ecosort-streamlit-app",
    api_key=_PHOENIX_API_KEY,
    endpoint=_PHOENIX_COLLECTOR,
    project_name=_PHOENIX_PROJECT,
)

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


def format_timestamp_vn(iso_ts: str) -> str:
    """Convert ISO timestamp to Vietnam local time (UTC+7) formatted string"""
    if not iso_ts:
        return ""
    try:
        clean_ts = iso_ts.rstrip("Z").split(".")[0]
        dt = datetime.fromisoformat(clean_ts)
        # Shift UTC to Vietnam timezone UTC+7
        dt_vn = dt + timedelta(hours=7)
        return dt_vn.strftime("%H:%M • %d/%m/%Y")
    except Exception:
        return iso_ts[:16].replace("T", " ")


def load_predictions_history(limit: int = 50) -> pd.DataFrame:
    """Read local prediction audit log into a user-friendly DataFrame"""
    if not os.path.exists(PREDICTION_LOG_FILE):
        return pd.DataFrame()
    records = []
    try:
        with open(PREDICTION_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        raw_cls = record.get("output", {}).get("predicted_class", "unknown")
                        cls_info = RECYCLING_INFO.get(raw_cls, {})
                        vn_name = cls_info.get("vietnamese_name", raw_cls.capitalize())
                        bin_color = cls_info.get("bin_color", "Thùng tái chế")
                        category = cls_info.get("category", "Rác sinh hoạt")
                        conf = record.get("output", {}).get("confidence", 0.0)

                        records.append({
                            "Thời gian": format_timestamp_vn(record.get("timestamp", "")),
                            "Loại rác": vn_name,
                            "Thùng rác quy định": bin_color,
                            "Nhóm phân loại": category,
                            "Độ tin cậy": f"{conf * 100:.1f}%",
                            "_raw_class": raw_cls,
                            "_category": category,
                        })
                    except Exception:
                        pass
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.iloc[::-1].head(limit)
        return df
    except Exception:
        return pd.DataFrame()


def main():
    st.markdown('<p class="main-header">♻️ EcoSort AI - Smart Trash Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Phân loại rác thải tự động • Hướng dẫn xử lý & tái chế • Giải thích AI (Grad-CAM)</p>',
        unsafe_allow_html=True
    )

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
        st.header("📊 Thông tin Mô hình")
        st.write("""
        - **Kiến trúc:** ResNet50 (Transfer Learning)
        - **Độ chính xác:** ~93.63% (12 nhóm rác)
        - **XAI:** Grad-CAM Visual Heatmap
        """)

        st.markdown("---")
        st.caption("EcoSort AI System • Smart Waste Classifier")

    # Main Tabs: 1. Classifier, 2. Green Journal
    tab_classifier, tab_journal = st.tabs([
        "♻️ Phân Loại Rác & XAI",
        "🌱 Nhật Ký Sống Xanh"
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

        st.info("""
        📸 **Mẹo chụp ảnh để AI nhận diện chuẩn xác nhất:**
        - 🔍 **Chụp cận cảnh:** Đưa vật thể lại gần sao cho rác chiếm phần lớn (**60 – 70%**) khung hình.
        - 🧹 **Hạn chế nền rối:** Đặt vật thể trên mặt bàn hoặc sàn trơn, tránh để lẫn nhiều đồ đạc khác vào khung hình.
        - ✋ **Tránh che khuất:** Nếu cầm trên tay, hạn chế để các ngón tay che lấp các đặc điểm nhận dạng của vật thể.
        - 💡 **Đủ ánh sáng:** Giữ camera ổn định, chụp ở nơi đủ sáng, tránh ảnh bị mờ hoặc ngược sáng.
        """, icon="💡")

        input_tab1, input_tab2 = st.tabs(["📁 Tải ảnh từ thiết bị (Upload File)", "📷 Chụp trực tiếp bằng Camera"])

        image_source = None
        with input_tab1:
            st.caption("👉 *Nên chọn ảnh chụp cận cảnh, góc nhìn rõ nét của vật thể.*")
            uploaded_file = st.file_uploader(
                "Kéo & thả ảnh rác thải vào đây (JPG, JPEG, PNG)",
                type=['jpg', 'jpeg', 'png'],
                key="file_uploader"
            )
            if uploaded_file is not None:
                image_source = uploaded_file

        with input_tab2:
            st.caption("👉 *Đưa vật thể lại gần, căn giữa khung hình camera và bấm nút chụp.*")
            camera_file = st.camera_input(
                "Chụp ảnh vật thể rác qua Camera / Webcam",
                key="camera_input",
                help="Đưa vật thể lại gần camera và căn giữa khung hình"
            )
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
    # TAB 2: NHẬT KÝ SỐNG XANH
    # =========================================================================
    with tab_journal:
        st.markdown("### 🌱 Nhật Ký Sống Xanh (Lịch Sử Của Bạn)")
        st.caption("Theo dõi thói quen phân loại rác thải và hành trình chung tay bảo vệ môi trường của bạn.")

        df_history = load_predictions_history(limit=100)

        if not df_history.empty:
            total_items = len(df_history)

            # Find top trash type
            top_series = df_history["Loại rác"].value_counts()
            top_trash = top_series.index[0] if not top_series.empty else "N/A"
            top_count = top_series.iloc[0] if not top_series.empty else 0

            # Calculate recyclable percentage
            recyclable_count = sum(
                1 for cat in df_history.get("_category", [])
                if any(kw in str(cat).lower() for kw in ["tái chế", "hữu cơ", "carton", "giấy"])
            )
            recycle_pct = (recyclable_count / total_items) * 100 if total_items > 0 else 0

            # Friendly Metrics Cards
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("🎯 Đã phân loại", f"{total_items} món rác")
            with col_m2:
                pct_str = f"chiếm {top_count/total_items*100:.0f}%" if total_items > 0 else ""
                st.metric("🏆 Xuất hiện nhiều nhất", top_trash, f"{top_count} lần ({pct_str})")
            with col_m3:
                st.metric("♻️ Tỷ lệ tái chế / hữu cơ", f"{recycle_pct:.1f}%")

            st.markdown("---")

            # Chart
            st.markdown("##### 📊 Thống kê các nhóm rác bạn hay gặp:")
            st.bar_chart(df_history["Loại rác"].value_counts())

            # Table
            st.markdown("---")
            st.markdown("##### 📋 Chi tiết các lần phân loại gần nhất:")
            display_cols = ["Thời gian", "Loại rác", "Thùng rác quy định", "Nhóm phân loại", "Độ tin cậy"]
            st.dataframe(
                df_history[display_cols],
                use_container_width=True,
                hide_index=True
            )

            # Green Eco Tip
            st.markdown("---")
            st.success("""
            💡 **Mẹo sống xanh từ EcoSort:**
            - **Rác tái chế (Nhựa, Kim loại, Hộp sữa):** Hãy tráng sạch và để ráo nước trước khi bỏ vào thùng tái chế để bảo vệ chất lượng vật liệu tái chế.
            - **Pin & Rác điện tử:** Tuyệt đối không bỏ chung vào thùng rác gia đình. Hãy gom vào hộp riêng và mang đến các điểm thu gom siêu thị/trường học.
            - **Rác hữu cơ:** Có thể tận dụng bã cà phê và vỏ hoa quả làm phân bón tự nhiên cho cây cảnh trong nhà!
            """)
        else:
            st.info("👋 Bạn chưa phân loại món rác nào. Hãy chuyển sang **Tab 1** để chụp hoặc tải ảnh rác đầu tiên nhé!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        EcoSort AI • Phân loại rác thông minh với Trí tuệ nhân tạo (ResNet50 & Grad-CAM)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
