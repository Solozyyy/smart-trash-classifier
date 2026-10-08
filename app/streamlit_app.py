"""
Smart Trash Classifier - Streamlit Web Application
AI-powered waste classification with Explainable AI (Grad-CAM), Camera capture, and Recycling database.
"""

import os
import sys
from PIL import Image
import streamlit as st

# Ensure repository root is on sys.path for src imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset import CLASSES
from src.model import load_trained_model, predict_image, generate_gradcam_heatmap
from src.utils import RECYCLING_INFO, create_gradcam_overlay, evaluate_uncertainty

# Streamlit Page Config
st.set_page_config(
    page_title="Smart Trash Classifier & XAI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme-Adaptive)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #2E7D32;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        text-align: center;
        opacity: 0.85;
        margin-bottom: 1.8rem;
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
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Cached loader for trained ResNet50 model"""
    return load_trained_model()


def main():
    st.markdown('<p class="main-header">♻️ Smart Trash Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Phân loại rác thải tự động & Giải thích quyết định AI bằng Grad-CAM</p>',
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
        - **Độ chính xác kiểm thử:** ~93.63%
        - **Số lớp phân loại:** 12 nhóm rác thải
        - **Giải thích AI:** Grad-CAM (`conv5_block3_out` logits)
        """)

        st.markdown("---")
        st.caption("EcoSort System | Powered by TensorFlow & Streamlit")

    # Load model with status indication
    try:
        with st.spinner("Đang nạp mô hình Deep Learning..."):
            model = get_model()
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình: {str(e)}")
        st.stop()

    # Input Method: File Upload or Live Camera
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

        # Perform inference
        top_classes, top_confidences, inf_time = predict_image(model, image, top_k=3)
        predicted_class = top_classes[0]
        confidence = top_confidences[0]

        is_certain, uncertainty_msg = evaluate_uncertainty(confidence, threshold=confidence_threshold)

        # Main Layout: Image & Predictions
        col_img, col_pred = st.columns([1, 1], gap="medium")

        with col_img:
            st.markdown("#### 📸 Ảnh đầu vào")
            st.image(image, use_container_width=True, caption=f"Kích thước gốc: {image.size[0]}x{image.size[1]}px")

        with col_pred:
            st.markdown("#### 🎯 Kết quả phân loại")

            # Prediction badge with certainty styling
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

            # Certainty banner
            if not is_certain:
                st.warning(f"⚠️ **Cảnh báo độ tin cậy thấp:** {uncertainty_msg}")
            else:
                st.success("✅ Mô hình tự tin với kết quả nhận diện này.")

            # Top 3 probabilities
            st.markdown("##### 📊 Top 3 lớp có xác suất cao nhất:")
            for i, (cls_name, conf) in enumerate(zip(top_classes, top_confidences), 1):
                vn_label = RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name)
                st.write(f"**{i}. {vn_label}** (`{cls_name}`): **{conf*100:.2f}%**")
                st.progress(float(conf))

            st.caption(f"⚡ Thời gian suy luận (Inference Latency): **{inf_time:.1f} ms**")

        # Grad-CAM Section
        if enable_gradcam:
            st.markdown("---")
            st.markdown("### 🔥 Giải thích AI bằng Grad-CAM (Visual Interpretability)")
            st.write(
                "Bản đồ nhiệt **Grad-CAM (Gradient-weighted Class Activation Mapping)** hiển thị khu vực đặc trưng "
                "khiến mạng nơ-ron đưa ra quyết định phân loại:"
            )

            # Allow user to inspect heatmap according to each of the Top-3 classes
            cam_options = {}
            for i, (cls_name, conf) in enumerate(zip(top_classes, top_confidences), 1):
                vn_label = RECYCLING_INFO.get(cls_name, {}).get('vietnamese_name', cls_name)
                label_text = f"Top {i}: {vn_label} ({cls_name}) - {conf*100:.1f}%"
                cam_options[label_text] = (cls_name, CLASSES.index(cls_name))

            selected_cam_label = st.radio(
                "🔍 Chọn góc nhìn phân tích theo lớp:",
                options=list(cam_options.keys()),
                index=0,
                horizontal=True,
                help="Xem vùng đặc trưng mà model tập trung nhận diện theo từng lớp dự đoán."
            )
            target_cls_name, target_cls_idx = cam_options[selected_cam_label]

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

            st.info(
                f"💡 **Cách phân tích:** Đang hiển thị vùng kích hoạt đặc trưng cho lớp **{target_cls_name.upper()}**. "
                "Vùng màu **đỏ/cam/vàng** là nơi mô hình tập trung nhiều nhất, vùng **xanh lam** ít có ảnh hưởng."
            )

        # Recycling Information Guide (Native Theme-adaptive Containers)
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

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        EcoSort Project • ResNet50 Transfer Learning + Grad-CAM XAI • Streamlit & TensorFlow
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
