"""
Dataset configuration, classes, and preprocessing utilities for Smart Trash Classifier.
"""

from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# 12 Classes recognized by the model
CLASSES: List[str] = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)


def validate_and_convert_image(image: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB format (handles RGBA PNGs, Grayscale, CMYK, etc.)
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def preprocess_pil_image(image: Image.Image, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess a PIL Image for ResNet50 inference:
    1. Ensure RGB
    2. Resize to (224, 224)
    3. Convert to float32 [0, 255]
    4. Expand batch dimension to (1, 224, 224, 3)
    5. Apply ResNet ImageNet mean subtraction & RGB->BGR color conversion
    """
    image = validate_and_convert_image(image)
    resized_img = image.resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return resnet_preprocess_input(img_batch)
