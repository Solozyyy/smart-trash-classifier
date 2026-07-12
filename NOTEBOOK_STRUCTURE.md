# Chi Tiết Cấu Trúc Notebook `garbage_classifier_full.ipynb`

## Tổng Quan
- **Tổng số cells**: 18 cells (10 code + 8 markdown)
- **Runtime**: Google Colab với GPU T4
- **Thời gian chạy**: ~2-3 hours
- **Output**: Model weights + Evaluation plots

---

## Chi Tiết Từng Cell

### **CELL 0** (Code): Mount Drive & Load Data
```python
from google.colab import drive
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns, os, zipfile

# Mount Google Drive
drive.mount('/content/drive')

# Unzip data
zip_path = '/content/drive/MyDrive/data.zip'
extract_path = '/content/data_extracted/'
# Extract all files...
```
**Output**: 12 categories phát hiện
```
['brown-glass', 'battery', 'metal', 'green-glass', 
 'shoes', 'trash', 'plastic', 'white-glass', 
 'clothes', 'cardboard', 'paper', 'biological']
```

---

### **CELL 1** (Code): EDA - Visualize Data Distribution
```python
import matplotlib.image as mpimg, random

# Đếm số lượng ảnh mỗi class
categories = os.listdir(extract_path)
data_stats = {}
for cat in categories:
    if os.path.isdir(path):
        data_stats[cat] = len(os.listdir(path))

# Vẽ barplot
plt.figure(figsize=(12, 5))
sns.barplot(x=list(data_stats.keys()), y=list(data_stats.values()))

# Hiển thị 5 samples ngẫu nhiên
plt.figure(figsize=(20, 10))
# ... display random images
```
**Output**:
- Barplot phân bố dữ liệu
- 5 ảnh samples ngẫu nhiên

---

### **CELL 2** (Code): Check Imbalance Ratio
```python
import pandas as pd

# Tạo DataFrame statistics
stats_df = pd.DataFrame(list(data_stats.items()), 
                        columns=['Category', 'Count'])
stats_df['Percentage'] = (stats_df['Count'] / stats_df['Count'].sum() * 100).round(2)
stats_df = stats_df.sort_values(by='Count', ascending=False)

# Tính imbalance ratio
imbalance_ratio = stats_df['Count'].max() / stats_df['Count'].min()
print(f"Lớp nhiều nhất gấp {imbalance_ratio:.2f} lần lớp ít nhất.")
```
**Output**: 
- Bảng thống kê chi tiết
- Imbalance ratio: ~8x (clothes 5000 vs battery 600)

---

### **CELL 3** (Code): Data Balancing
```python
import shutil, random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config
SOURCE_DIR = '/content/data_extracted/'
BALANCED_DIR = '/content/balanced_data/'
TARGET_COUNT = 2000

# Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Process each category:
# - clothes: Undersample 5000 → 2000
# - Others < 1500: Augment → 2000
# - Others >= 1500: Keep original
```
**Output**: Balanced dataset tại `/content/balanced_data/` (~24,000 ảnh)

---

### **CELL 4** (Code): Verify Balanced Data
```python
# Kiểm tra lại số lượng sau khi balance
final_stats = {cat: len(os.listdir(os.path.join(BALANCED_DIR, cat))) 
               for cat in categories}

# Vẽ barplot mới
plt.figure(figsize=(12, 5))
sns.barplot(x=list(final_stats.keys()), y=list(final_stats.values()))
plt.title('Số lượng ảnh sau khi cân bằng (Balanced Dataset)')
```
**Output**: Barplot với ~2000 ảnh/class

---

### **CELL 5** (Markdown): Section Header
```markdown
## 2. Chia dữ liệu thành tập huấn luyện, xác thực và kiểm tra (70/15/15)
```

---

### **CELL 6** (Code): Train/Val/Test Split
```python
from sklearn.model_selection import train_test_split

# Define directories
TRAIN_DIR = '/content/train_data/'
VAL_DIR = '/content/val_data/'
TEST_DIR = '/content/test_data/'

# Split ratio: 70/15/15
# Sử dụng train_test_split 2 lần:
# 1. Split balanced → train (70%) + temp (30%)
# 2. Split temp → val (50% of 30% = 15%) + test (50% of 30% = 15%)

# Copy files vào các thư mục tương ứng
```
**Output**: 
- `/content/train_data/` (~16,800 ảnh)
- `/content/val_data/` (~3,600 ảnh)
- `/content/test_data/` (~3,600 ảnh)

---

### **CELL 7** (Markdown): Section Header
```markdown
## 3. Tiền xử lý dữ liệu và Augmentation
```

---

### **CELL 8** (Code): Data Generators Setup
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input

# Config
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# Train generator (with augmentation)
train_datagen = ImageDataGenerator(
    preprocessing_function=resnet_preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Val/Test generators (only preprocessing)
val_test_datagen = ImageDataGenerator(
    preprocessing_function=resnet_preprocess_input
)

# Create generators
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, ...)
validation_generator = val_test_datagen.flow_from_directory(VAL_DIR, ...)
test_generator = val_test_datagen.flow_from_directory(TEST_DIR, ...)
```
**Output**: 
- Found ~16,800 images in train
- Found ~3,600 images in validation
- Found ~3,600 images in test

---

### **CELL 9** (Markdown): Model Architecture Explanation
```markdown
## 4. Xây dựng mô hình (Transfer Learning với ResNet)

Sử dụng ResNet50 - mạnh mẽ và phù hợp cho Transfer Learning.
```

---

### **CELL 10** (Code): Build Model
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)  # 12 classes

model = Model(inputs=base_model.input, outputs=predictions)
```
**Output**: Model summary (~25M params total, 2M trainable)

---

### **CELL 11** (Markdown): Section Header
```markdown
## 5. Huấn luyện mô hình
```

---

### **CELL 12** (Code): Train Model
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)
```
**Output**: 
- Training history (loss, accuracy per epoch)
- Best model saved to `best_model.h5`

---

### **CELL 13** (Markdown): Section Header
```markdown
## 6. Đánh giá kết quả trên tập kiểm tra
```

---

### **CELL 14** (Code): Comprehensive Evaluation
```python
import matplotlib.pyplot as plt, numpy as np, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Create plots directory
plots_dir = 'evaluation_plots'
os.makedirs(plots_dir, exist_ok=True)

# 1. Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 2. Get predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# 3. Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.savefig(f'{plots_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')

# 4. Classification Report
report = classification_report(y_true, y_pred_classes, target_names=class_names)
with open(f'{plots_dir}/classification_report.txt', 'w') as f:
    f.write(report)

# 5. ROC Curves for each class
plt.figure(figsize=(15, 12))
for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
plt.savefig(f'{plots_dir}/roc_curves.png', dpi=300, bbox_inches='tight')

# 6. Training History
plt.figure(figsize=(14, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.savefig(f'{plots_dir}/training_history.png', dpi=300, bbox_inches='tight')
```
**Output**: 
- `confusion_matrix.png`
- `classification_report.txt`
- `roc_curves.png`
- `training_history.png`

---

### **CELL 15** (Markdown): Section Header
```markdown
## 7. Dự đoán trên ảnh ngẫu nhiên từ tập kiểm tra
```

---

### **CELL 16** (Code): Demo Predictions
```python
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def display_image_prediction(image_array, true_label, predicted_label, class_names):
    plt.imshow(image_array)
    color = 'green' if true_label == predicted_label else 'red'
    plt.title(f'True: {class_names[true_label]} | Pred: {class_names[predicted_label]}', 
              color=color, fontsize=14, weight='bold')
    plt.axis('off')

# Lấy 10 ảnh ngẫu nhiên từ test set
random_indices = random.sample(range(len(test_generator.filenames)), 10)

plt.figure(figsize=(20, 10))
for i, idx in enumerate(random_indices):
    # Load image
    img_path = os.path.join(TEST_DIR, test_generator.filenames[idx])
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    
    # Predict
    img_preprocessed = resnet_preprocess_input(img_array.reshape(1, *img_array.shape))
    prediction = model.predict(img_preprocessed)
    predicted_class = np.argmax(prediction)
    true_class = test_generator.classes[idx]
    
    # Display
    plt.subplot(2, 5, i + 1)
    display_image_prediction(img_array/255., true_class, predicted_class, class_names)

plt.tight_layout()
plt.savefig(f'{plots_dir}/sample_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
```
**Output**: 
- Grid 2×5 với 10 predictions
- Green title = correct, Red title = wrong

---

### **CELL 17** (Code): Download Results
```python
import shutil
from google.colab import files

# Zip evaluation plots
shutil.make_archive('/content/evaluation_plots', 'zip', '/content/evaluation_plots')

# Download zip file
files.download('/content/evaluation_plots.zip')
```
**Output**: Auto-download `evaluation_plots.zip` về máy

---

##  Outputs Summary

### **Files Created**:
1. `best_model.h5` - Trained model weights (~98MB)
2. `evaluation_plots/confusion_matrix.png`
3. `evaluation_plots/classification_report.txt`
4. `evaluation_plots/roc_curves.png`
5. `evaluation_plots/training_history.png`
6. `evaluation_plots/sample_predictions.png`
7. `evaluation_plots.zip` - All plots zipped

### **Directories Created**:
- `/content/data_extracted/` - Original unzipped data
- `/content/balanced_data/` - Balanced dataset (2000/class)
- `/content/train_data/` - Training set (70%)
- `/content/val_data/` - Validation set (15%)
- `/content/test_data/` - Test set (15%)
- `/content/evaluation_plots/` - All evaluation plots

---

##  Execution Flow

```
START
  ↓
Cell 0-1: Load & Explore Data
  ↓
Cell 2: Detect Imbalance
  ↓
Cell 3-4: Balance Data (2000/class)
  ↓
Cell 6: Split Train/Val/Test (70/15/15)
  ↓
Cell 8: Setup Data Generators
  ↓
Cell 10: Build ResNet50 Model
  ↓
Cell 12: Train Model (20 epochs, early stopping)
  ↓
Cell 14: Evaluate & Generate Plots
  ↓
Cell 16: Demo Predictions
  ↓
Cell 17: Download Results
  ↓
END
```

---

##  Estimated Runtime

| Cell(s) | Task | Time |
|---------|------|------|
| 0 | Mount & Unzip | ~1 min |
| 1-2 | EDA | ~30 sec |
| 3-4 | Data Balancing | ~10 min |
| 6 | Train/Val/Test Split | ~2 min |
| 8 | Setup Generators | ~10 sec |
| 10 | Build Model | ~5 sec |
| 12 | Training (20 epochs) | **2-3 hours** |
| 14 | Evaluation | ~5 min |
| 16 | Demo Predictions | ~1 min |
| 17 | Download | ~10 sec |
| **TOTAL** | | **~2.5-3.5 hours** |

---

##  Storage Requirements

- Original data.zip: ~1.5GB
- Balanced data: ~5GB
- Train/Val/Test split: ~5GB (duplicates)
- Model weights: ~98MB
- Evaluation plots: ~5MB
- **Total Colab storage needed**: ~12GB

---

##  Learning Path

**Beginners**: Run cells **sequentially** và đọc output mỗi cell để hiểu workflow

**Intermediate**: Modify hyperparameters trong Cell 8, 10, 12 để experiment

**Advanced**: Fork notebook và thử các model khác (EfficientNet, MobileNet)

---

** Note**: Notebook này là **complete end-to-end pipeline**. Có thể chạy "Run All" và chờ ~3 hours để có model trained!
