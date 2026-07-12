# Smart Trash Classifier — Smart Trash Classifier

> Phân loại rác thải tự động bằng Deep Learning & Computer Vision

Dự án phân loại ảnh rác thải thành **12 loại khác nhau** sử dụng **ResNet50** với Transfer Learning. Hệ thống có thể nhận diện loại rác từ ảnh chụp, hỗ trợ trạm phân loại tự động hoặc ứng dụng di động.

---

## 12 Loại Rác Được Phân Loại

| # | Loại rác | English | # | Loại rác | English |
|---|----------|---------|---|----------|---------|
| 1 | Pin | Battery | 7 | Kim loại | Metal |
| 2 | Hữu cơ | Biological | 8 | Giấy | Paper |
| 3 | Thủy tinh nâu | Brown Glass | 9 | Nhựa | Plastic |
| 4 | Bìa carton | Cardboard | 10 | Giày dép | Shoes |
| 5 | Quần áo | Clothes | 11 | Rác khác | Trash |
| 6 | Thủy tinh xanh | Green Glass | 12 | Thủy tinh trắng | White Glass |

---

## Quy Trình Thực Hiện

### **Bước 1: EDA - Exploratory Data Analysis**
- Mount Google Drive và giải nén dataset
- Khám phá phân bố dữ liệu (12 classes)
- Phát hiện class imbalance (clothes: 5000+, battery: 600+)
- Visualize samples từ mỗi class

### **Bước 2: Data Balancing**
- **Target**: 2000 ảnh/class
- **Undersampling**: Giảm `clothes` từ 5000+ → 2000
- **Augmentation**: Tăng các class thiếu lên 2000
  - Rotation: ±20°
  - Width/Height shift: ±10%
  - Shear: ±10%
  - Zoom: ±10%
  - Horizontal flip: Random

### **Bước 3: Data Splitting**
- **Train**: 70% (~16,800 ảnh)
- **Validation**: 15% (~3,600 ảnh)
- **Test**: 15% (~3,600 ảnh)
- Sử dụng `train_test_split` từ sklearn

### **Bước 4: Data Preprocessing**
- Resize về **224x224** (input size cho ResNet)
- Normalize bằng `resnet_preprocess_input`
- **Train augmentation**:
  - Rotation: ±15°
  - Width/Height shift: ±10%
  - Shear: ±10%
  - Zoom: ±10%
  - Horizontal flip
  - Fill mode: nearest
- **Val/Test**: Chỉ normalize (không augment)

### **Bước 5: Model Architecture**
- **Base Model**: ResNet50 (pretrained ImageNet)
  - *Note*: Dùng ResNet50 
  - Freeze toàn bộ base layers (không train lại)
- **Custom Head**:
  ```
  GlobalAveragePooling2D → Flatten
  → Dense(512, ReLU) → Dropout(0.5)
  → Dense(256, ReLU) → Dropout(0.5)
  → Dense(12, Softmax)
  ```
- **Total params**: ~25M (23M frozen, 2M trainable)

### **Bước 6: Training**
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch size**: 32
- **Epochs**: 20 (với Early Stopping)
- **Callbacks**:
  - ModelCheckpoint: Lưu best model theo `val_accuracy`
  - EarlyStopping: Patience=5, restore best weights

### **Bước 7: Evaluation**
- **Test Accuracy**: _%_ (sẽ cập nhật sau khi train)
- **Confusion Matrix**: Phân tích lỗi giữa các classes
- **Classification Report**: Precision, Recall, F1-Score
- **ROC Curve & AUC**: Đánh giá từng class
- **Training History**: Loss & Accuracy curves

### **Bước 8: Inference Demo**
- Dự đoán trên ảnh ngẫu nhiên từ test set
- Hiển thị ảnh + True label + Predicted label
- Confidence score cho mỗi prediction

---

## Dataset

### **Thông Tin Chung**
- **Nguồn**: Custom dataset (từ Kaggle)
- **Tổng số ảnh ban đầu**: ~15,000 ảnh
- **Sau khi balance**: ~24,000 ảnh (2000/class × 12)
- **Số classes**: 12 loại rác
- **Format**: JPG images
- **Phân chia**:
  - Train: 70% (~16,800 ảnh)
  - Validation: 15% (~3,600 ảnh)
  - Test: 15% (~3,600 ảnh)

### **Phân Bố Ban Đầu (Imbalanced)**
| Class | Số lượng | Xử lý |
|-------|----------|-------|
| Clothes | 5000+ | Undersampling → 2000 |
| Shoes | ~2000 | Giữ nguyên |
| Biological | ~1000 | Augmentation → 2000 |
| Battery | ~600 | Augmentation → 2000 |
| Others | 700-1000 | Augmentation → 2000 |

---

## Model Architecture Details

### **ResNet50 Base (Frozen)**
```python
Input: (224, 224, 3)
↓
ResNet50 (pretrained ImageNet, frozen)
↓
GlobalAveragePooling2D
↓
Flatten
```

### **Custom Classification Head (Trainable)**
```python
Dense(512, activation='relu')
↓
Dropout(0.5)
↓
Dense(256, activation='relu')
↓
Dropout(0.5)
↓
Dense(12, activation='softmax')  # 12 classes
```

### **Training Configuration**
```python
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = (224, 224)
```

---

## Cài Đặt & Sử Dụng

### **Yêu Cầu Hệ Thống**
- Python 3.8+
- TensorFlow 2.x
- Google Colab (free T4 GPU) **khuyến nghị**
- RAM: 12GB+ (cho training)
- Storage: 5GB+ (cho dataset)

### **1. Clone Repository**
```bash
git clone https://github.com/your-username/smart-trash-classifier.git
cd smart-trash-classifier
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Chuẩn Bị Dataset**

**Trên Google Colab**:
1. Upload `data.zip` lên Google Drive: `/MyDrive/data.zip`
2. Đảm bảo cấu trúc:
   ```
   data.zip
   ├── battery/
   ├── biological/
   ├── brown-glass/
   └── ... (12 folders)
   ```

**Trên Local**:
1. Giải nén `data.zip` vào thư mục `data/`
2. Cấu trúc tương tự

### **4. Chạy Training**

**Option A: Chạy từng notebook riêng (Khuyến nghị học tập)**
```
01_eda.ipynb              → Khám phá dữ liệu
02_data_balancing.ipynb   → Cân bằng dữ liệu
03_train_resnet50.ipynb   → Training model (sẽ tạo)
04_evaluation.ipynb       → Đánh giá kết quả (sẽ tạo)
```

**Option B: Chạy notebook đầy đủ (Nhanh)**
```
garbage_classifier_full.ipynb  → Full pipeline (18 cells)
```

**Trên Google Colab**:
1. Mở Colab: https://colab.research.google.com/
2. File → Open notebook → GitHub
3. Paste: `https://github.com/your-username/smart-trash-classifier`
4. Chọn notebook và Run All

---

## Cấu Trúc Project

```
smart-trash-classifier/
├── notebooks/
│   ├── garbage_classifier_full.ipynb     # Full pipeline (18 cells)
│   ├── 01_eda.ipynb                      # EDA only
│   ├── 02_data_balancing.ipynb           # Data preprocessing
│   ├── 03_train_resnet50.ipynb           # Training (TODO)
│   └── 04_evaluation.ipynb               # Evaluation (TODO)
├── src/
│   ├── model.py                          # Model definition
│   ├── dataset.py                        # Dataset utilities
│   └── utils.py                          # Helper functions
├── evaluation_plots/                     # Generated plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── classification_report.txt
├── weights/
│   └── best_model.h5                     # Trained model (download separately)
├── config.py                             # Configuration loader
├── .env.example                          # Environment template
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md
```

---

##  Download Model & Results

### **Trained Model Weights**
File `best_model.h5` quá lớn để push lên GitHub. Tải tại:

- **Google Drive**: [Download best_model.h5](#) _(Chưa có - sẽ update sau khi train)_
- **GitHub Release**: [v1.0.0](#) _(Chưa có)_

**Cách sử dụng**:
```bash
# 1. Tải file best_model.h5
# 2. Đặt vào thư mục weights/
mv ~/Downloads/best_model.h5 ./weights/

# 3. Load model trong Python
from tensorflow.keras.models import load_model
model = load_model('weights/best_model.h5')
```

### **Evaluation Plots**
Các biểu đồ được tạo tự động sau khi training và lưu vào `evaluation_plots/`:
- `training_history.png` - Loss & Accuracy curves
- `confusion_matrix.png` - Confusion matrix
- `roc_curves.png` - ROC curves cho 12 classes
- `classification_report.txt` - Precision, Recall, F1

---

## Kết Quả Training

_(Sẽ được cập nhật sau khi hoàn thành training trên Colab)_

### **Model Performance**
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | _% | _% | _% |
| Loss | _ | _ | _ |
| F1-Score (Macro) | _ | _ | _ |

### **Per-Class Performance**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Battery | _ | _ | _ | _ |
| Biological | _ | _ | _ | _ |
| ... | _ | _ | _ | _ |

### **Confusion Matrix**
![Confusion Matrix](evaluation_plots/confusion_matrix.png)

### **Training History**
![Training Curves](evaluation_plots/training_history.png)

### **ROC Curves**
![ROC Curves](evaluation_plots/roc_curves.png)

---

## Requirements

```txt
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.0.0
```

Cài đặt:
```bash
pip install -r requirements.txt
```

---

## Configuration

### **File `.env.example`**
```env
# Environment
ENVIRONMENT=colab  # hoặc "local"
