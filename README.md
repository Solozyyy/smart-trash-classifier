Smart Trash Classifier - Phân loại rác thải bằng Computer Vision

Dự án phân loại ảnh rác thải thành các nhóm (nhựa, giấy, kim loại, thủy tinh, rác hữu cơ...) sử dụng transfer learning với ResNet18. Mục tiêu: xây dựng một hệ thống nhận diện rác đơn giản, có thể ứng dụng cho các trạm phân loại rác tự động hoặc app hướng dẫn phân loại rác tại nhà.

1. Bài toán

Phân loại rác thủ công tốn thời gian và dễ sai sót. Dự án này xây một model deep learning có thể nhận diện loại rác từ một bức ảnh, hỗ trợ phân loại tự động hoặc giáo dục người dùng cách phân loại đúng.

Input: Ảnh chụp một vật thể rác
Output: Nhãn loại rác (VD: nhựa, giấy, kim loại...) kèm độ tin cậy (confidence)

2. Dataset

Tên: Garbage Classification Dataset
Nguồn: Kaggle — https://www.kaggle.com/datasets/mostafaabla/garbage-classification
Mô tả: Bộ ảnh rác đã gán nhãn theo nhiều loại, phổ biến gồm:


Cardboard (bìa carton)
Glass (thủy tinh)
Metal (kim loại)
Paper (giấy)
Plastic (nhựa)
Trash (rác không tái chế được)


Số lượng: Vài nghìn ảnh, chia đều tương đối giữa các lớp (kiểm tra kỹ số lượng ảnh mỗi lớp khi tải về, vì có thể mất cân bằng - class imbalance)

Cách tải:

bashpip install kaggle
# Upload file kaggle.json (API key từ Kaggle account settings) vào thư mục ~/.kaggle/
kaggle datasets download -d mostafaabla/garbage-classification
unzip garbage-classification.zip -d garbage_data

3. Các bước thực hiện

Bước 1: Chuẩn bị môi trường


Dùng Google Colab (free GPU: Runtime > Change runtime type > GPU)
Cài đặt: torch, torchvision, matplotlib, kaggle


Bước 2: Tải và khám phá dữ liệu (EDA)


Tải dataset theo hướng dẫn ở mục 2
Đếm số ảnh mỗi lớp, xem có mất cân bằng không
Xem thử vài ảnh mẫu mỗi lớp để hiểu dữ liệu


Bước 3: Tiền xử lý & Augmentation


Resize ảnh về 224x224 (kích thước chuẩn cho ResNet)
Augmentation: lật ngang, xoay nhẹ, để tăng đa dạng dữ liệu train
Chuẩn hóa (normalize) theo mean/std của ImageNet vì dùng model pretrained


Bước 4: Xây dựng model (Transfer Learning)


Dùng ResNet18 pretrained trên ImageNet
Đóng băng (freeze) các layer gốc, chỉ train lại layer phân loại cuối
Lý do: dataset nhỏ, transfer learning giúp học nhanh và tránh overfitting


Bước 5: Huấn luyện (Training)


Chia dữ liệu train/validation (80/20)
Train khoảng 10-15 epoch, theo dõi accuracy và loss trên tập validation
Lưu lại model có validation accuracy tốt nhất


Bước 6: Đánh giá kết quả


Vẽ biểu đồ accuracy/loss theo epoch
Xem confusion matrix để biết model hay nhầm lẫn giữa lớp nào với lớp nào (VD: nhựa và giấy dễ nhầm)
Tính thêm precision/recall/F1 cho từng lớp, đặc biệt quan trọng nếu dataset mất cân bằng


Bước 7: Demo trực quan


Dùng Gradio hoặc Streamlit để tạo giao diện đơn giản: upload ảnh → model dự đoán loại rác
Giúp người xem (kể cả người không biết code) thấy được kết quả thực tế


Bước 8: Viết báo cáo/README


Ghi rõ: vấn đề, dataset, kiến trúc model, kết quả, hạn chế, hướng cải thiện
Đây là phần quan trọng nhất khi apply — thể hiện khả năng tư duy và trình bày, không chỉ code chạy được


#4. Kết quả (điền sau khi train xong)


Validation accuracy: ...
Model hay nhầm lẫn giữa các lớp: ...
Nhận xét: ...


#5. Hướng cải thiện tiếp theo (tùy chọn, làm nếu có thời gian)


Fine-tune sâu hơn (unfreeze thêm layer của ResNet thay vì chỉ train layer cuối)
Thử model khác (EfficientNet, MobileNet) để so sánh
Xử lý class imbalance bằng weighted loss hoặc oversampling
Deploy thành web app thật, hoặc app mobile đơn giản
Mở rộng thêm loại rác (rác điện tử, pin...) nếu tìm được dataset phù hợp


#6. Công nghệ sử dụng


Python, PyTorch, torchvision
ResNet18 (transfer learning)
Google Colab (training môi trường free GPU)
Gradio (demo)

