"""
Recycling database, Grad-CAM visualization, and utility helpers.
"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

# Comprehensive Recycling Information for all 12 classes
RECYCLING_INFO: Dict[str, Dict[str, Any]] = {
    'battery': {
        'vietnamese_name': 'Pin & Ắc quy',
        'category': '🔋 Chất thải nguy hại (Hazardous Waste)',
        'bin_color': 'Đỏ / Thùng gom pin chuyên dụng',
        'instructions': [
            'Tuyệt đối không vứt chung với rác sinh hoạt thông thường',
            'Bọc kín 2 đầu cực bằng băng dính cách điện để tránh cháy nổ',
            'Mang đến các điểm thu gom rác điện tử hoặc thùng gom pin siêu thị'
        ],
        'decomposition_time': '100+ năm (gây ô nhiễm vĩnh viễn nguồn nước ngầm)',
        'impact': 'Chứa chì, thủy ngân, cadimi cực độc. 1 viên pin có thể làm ô nhiễm 500 lít nước và 1m3 đất trong 50 năm.'
    },
    'biological': {
        'vietnamese_name': 'Rác hữu cơ / Thực phẩm',
        'category': '🌱 Rác hữu cơ dễ phân hủy',
        'bin_color': 'Xanh lá cây (Organic Bin)',
        'instructions': [
            'Ủ phân hữu cơ vi sinh (compost) tại nhà nếu có điều kiện',
            'Loại bỏ bao bì nilon hoặc dị vật trước khi bỏ vào thùng',
            'Buộc kín túi đựng để tránh bốc mùi hôi thối và thu hút côn trùng'
        ],
        'decomposition_time': '2 tuần - 2 tháng',
        'impact': 'Nếu chôn lấp chung sẽ sinh khí Methane (CH4) gây hiệu ứng nhà kính gấp 25 lần CO2.'
    },
    'brown-glass': {
        'vietnamese_name': 'Thủy tinh màu nâu',
        'category': '♻️ Thủy tinh tái chế',
        'bin_color': 'Vàng / Ô phân loại thủy tinh màu',
        'instructions': [
            'Tráng rửa sạch cặn chất lỏng bên trong',
            'Tháo nắp chai kim loại hoặc nhựa',
            'Tránh làm vỡ để đảm bảo an toàn cho nhân viên thu gom'
        ],
        'decomposition_time': 'Hơn 1 triệu năm',
        'impact': 'Tái chế thủy tinh giúp tiết kiệm 30% năng lượng so với việc sản xuất từ cát thô.'
    },
    'cardboard': {
        'vietnamese_name': 'Bìa carton / Thùng giấy',
        'category': '♻️ Giấy & Carton tái chế',
        'bin_color': 'Xanh dương (Recyclable Paper)',
        'instructions': [
            'Gấp phẳng thùng để tiết kiệm thể tích thùng rác',
            'Bóc băng dính dán và ghim bấm nếu có',
            'Giữ khô ráo, thùng bị ướt/dính dầu mỡ không thể tái chế'
        ],
        'decomposition_time': '2 - 3 tháng',
        'impact': 'Tái chế 1 tấn bìa carton cứu sống 17 cây xanh và tiết kiệm 4.000 kWh điện.'
    },
    'clothes': {
        'vietnamese_name': 'Quần áo / Vải dệt',
        'category': '👕 Đồ cũ tái sử dụng / Tái chế',
        'bin_color': 'Thùng quyên góp từ thiện / Tái chế vải',
        'instructions': [
            'Nếu còn lành lặn: Giặt sạch và quyên góp cho các quỹ từ thiện',
            'Nếu rách nát: Cắt làm giẻ lau hoặc gửi các cơ sở tái chế sợi vải',
            'Không vứt bừa bãi ra môi trường tự nhiên'
        ],
        'decomposition_time': 'Vải sợi tự nhiên: 5 tháng | Sợi tổng hợp (Polyester): 40 - 200 năm',
        'impact': 'Ngành dệt may tiêu thụ lượng nước khổng lồ và sợi tổng hợp giải phóng hạt vi nhựa vào đại dương.'
    },
    'green-glass': {
        'vietnamese_name': 'Thủy tinh màu xanh lá',
        'category': '♻️ Thủy tinh tái chế',
        'bin_color': 'Vàng / Ô phân loại thủy tinh màu',
        'instructions': [
            'Súc sạch cặn đồ uống còn sót lại',
            'Tháo rời nắp và vòng đệm cổ chai',
            'Phân loại riêng theo màu sắc để tăng giá trị tái chế'
        ],
        'decomposition_time': 'Hơn 1 triệu năm',
        'impact': 'Thủy tinh có thể tái chế vô hạn lần mà không làm giảm chất lượng sản phẩm.'
    },
    'metal': {
        'vietnamese_name': 'Kim loại / Vỏ lon',
        'category': '♻️ Kim loại tái chế giá trị cao',
        'bin_color': 'Vàng / Thùng kim loại',
        'instructions': [
            'Rửa sạch thức ăn, đồ uống còn đọng lại',
            'Dẫm bẹp vỏ lon nhôm để giảm thể tích lưu trữ',
            'Cẩn thận các cạnh sắc nhọn để tránh đứt tay'
        ],
        'decomposition_time': 'Lon nhôm: 80 - 200 năm | Kim loại dày: 500 năm',
        'impact': 'Tái chế nhôm tiết kiệm tới 95% năng lượng so với sản xuất nhôm mới từ quặng bauxite.'
    },
    'paper': {
        'vietnamese_name': 'Giấy báo / Tài liệu / Sách vở',
        'category': '♻️ Giấy tái chế',
        'bin_color': 'Xanh dương (Paper Bin)',
        'instructions': [
            'Giữ giấy sạch sẽ và khô ráo',
            'Tháo bỏ kẹp ghim sắt hoặc bìa bọc nilon',
            'Xé nhỏ giấy tờ bảo mật trước khi bỏ vào thùng tái chế'
        ],
        'decomposition_time': '2 - 6 tuần',
        'impact': 'Tái chế giấy giảm 73% ô nhiễm không khí so với sản xuất từ bột gỗ tươi.'
    },
    'plastic': {
        'vietnamese_name': 'Đồ nhựa / Chai nhựa',
        'category': '♻️ Nhựa tái chế',
        'bin_color': 'Vàng (Plastic Bin)',
        'instructions': [
            'Kiểm tra mã ký hiệu nhựa dưới đáy (PET 1, HDPE 2, PP 5 dễ tái chế)',
            'Tráng sạch cặn nước và bóp dẹp chai',
            'Tháo nắp và nhãn mác nếu có thể'
        ],
        'decomposition_time': '450 - 1000 năm',
        'impact': 'Khoảng 8 triệu tấn nhựa thải vào đại dương mỗi năm, phân rã thành vi nhựa đe dọa chuỗi thức ăn sinh học.'
    },
    'shoes': {
        'vietnamese_name': 'Giày dép cũ',
        'category': '👟 Đồ dùng tái sử dụng / Tái chế',
        'bin_color': 'Thùng đồ cũ / Thu gom giày',
        'instructions': [
            'Buộc 2 chiếc giày lại với nhau để không bị thất lạc',
            'Nếu còn dùng được: Vệ sinh sạch sẽ và gửi tặng người khó khăn',
            'Nhiều thương hiệu giày có chương trình thu hồi giày cũ tái chế đế cao su'
        ],
        'decomposition_time': '25 - 50 năm (phần đế cao su/nhựa)',
        'impact': 'Giày dép chứa hỗn hợp nhiều vật liệu dính keo nên rất khó phân hủy nếu bị chôn lấp.'
    },
    'trash': {
        'vietnamese_name': 'Rác vô cơ không tái chế / Rác thải chung',
        'category': '🗑️ Rác thải chung (General Waste)',
        'bin_color': 'Đen / Xám (Landfill Bin)',
        'instructions': [
            'Chỉ bỏ các vật dụng không thể tái chế hoặc tái sử dụng',
            'Đóng gói cẩn thận tránh rò rỉ rác ra ngoài',
            'Cố gắng hạn chế tối đa việc phát sinh loại rác này'
        ],
        'decomposition_time': 'Tùy vật liệu (từ vài năm đến hàng trăm năm)',
        'impact': 'Rác thải đưa về bãi chôn lấp gây quá tải diện tích và tạo khí thải nhà kính.'
    },
    'white-glass': {
        'vietnamese_name': 'Thủy tinh trong suốt (Trắng)',
        'category': '♻️ Thủy tinh tái chế (Loại cao cấp nhất)',
        'bin_color': 'Vàng / Thủy tinh trong suốt',
        'instructions': [
            'Rửa sạch dầu mỡ, cặn nước',
            'Tách riêng với thủy tinh màu để đảm bảo độ tinh khiết khi nấu lại',
            'Tránh lẫn gốm sứ hoặc thủy tinh chịu nhiệt (như nắp nồi)'
        ],
        'decomposition_time': 'Hơn 1 triệu năm',
        'impact': 'Thủy tinh trong suốt có giá trị kinh tế và tỷ lệ tái chế cao nhất trong các loại thủy tinh.'
    }
}


def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = 'jet'
) -> Image.Image:
    """
    Superimpose Grad-CAM heatmap onto the original PIL Image.
    Returns:
        PIL Image: Original image overlaid with colored heatmap.
    """
    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(
        original_image.size,
        resample=Image.Resampling.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Colorize heatmap using matplotlib colormap
    if hasattr(mpl, 'colormaps'):
        color_mapper = mpl.colormaps[colormap]
    elif hasattr(cm, 'get_cmap'):
        color_mapper = getattr(cm, 'get_cmap')(colormap)
    else:
        import matplotlib.pyplot as plt
        color_mapper = plt.get_cmap(colormap)
    colored_heatmap = color_mapper(heatmap_resized)[:, :, :3]  # Strip alpha channel
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Convert original image to RGB numpy array
    if original_image.mode != 'RGB':
        orig = original_image.convert('RGB')
    else:
        orig = original_image
    orig_np = np.array(orig)

    # Alpha blending: overlay = heatmap * alpha + original * (1 - alpha)
    blended = np.uint8(colored_heatmap * alpha + orig_np * (1.0 - alpha))
    return Image.fromarray(blended)


def evaluate_uncertainty(confidence: float, threshold: float = 0.60) -> Tuple[bool, str]:
    """
    Check if model prediction exceeds confidence threshold.
    Returns:
        is_confident (bool)
        message (str)
    """
    if confidence >= threshold:
        return True, "Độ tin cậy cao, kết quả nhận diện đáng tin cậy."
    else:
        return False, (
            f"Độ tin cậy thấp ({confidence * 100:.1f}% < {threshold * 100:.0f}%). "
            "Vật thể có thể bị che khuất, mờ, góc chụp lạ, hoặc không nằm trong 12 lớp rác được huấn luyện. "
            "Khuyến nghị: Chụp lại gần hơn hoặc kiểm tra thủ công."
        )
