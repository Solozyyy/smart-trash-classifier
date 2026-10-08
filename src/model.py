"""
Model architecture, loading, inference, and Grad-CAM implementation.
"""

import os
import time
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from src.dataset import CLASSES, preprocess_pil_image

DEFAULT_WEIGHT_PATHS = [
    r'D:\hoc_AI\smart-trash-classifier\weights\best_model_converted.h5',
    os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model_converted.h5'),
    os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.h5'),
    'weights/best_model_converted.h5',
    'weights/best_model.h5',
]


def build_model(num_classes: int = 12, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> keras.Model:
    """
    Build the ResNet50 model architecture matching training setup.
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models

    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    predictions = layers.Dense(num_classes, activation='softmax', name='dense_1')(x)

    return models.Model(inputs=base_model.input, outputs=predictions, name='waste_classifier_resnet50')


def load_trained_model(model_path: Optional[str] = None) -> keras.Model:
    """
    Load the trained model with multiple path fallback and compatibility modes.
    """
    candidate_paths = [model_path] if model_path else DEFAULT_WEIGHT_PATHS

    for path in candidate_paths:
        if not path:
            continue
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            try:
                # Load with safe_mode=False to handle Keras format compatibility
                model = keras.models.load_model(norm_path, compile=False, safe_mode=False)
                return model
            except Exception:
                # Fallback: rebuild architecture and load weights
                try:
                    model = build_model()
                    model.load_weights(norm_path, by_name=True, skip_mismatch=True)
                    return model
                except Exception:
                    continue

    raise FileNotFoundError("Could not find or load a valid model weights file in the specified paths.")


def predict_image(
    model: keras.Model,
    image: Image.Image,
    top_k: int = 3
) -> Tuple[List[str], List[float], float]:
    """
    Perform inference on a single PIL image.
    Returns:
        top_k_classes: List of predicted class names
        top_k_confidences: List of corresponding probabilities
        inference_time_ms: Latency in milliseconds
    """
    img_tensor = preprocess_pil_image(image)

    start_time = time.perf_counter()
    raw_predictions = model(img_tensor, training=False)[0].numpy()
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    top_indices = np.argsort(raw_predictions)[-top_k:][::-1]
    top_k_classes = [CLASSES[i] for i in top_indices]
    top_k_confidences = [float(raw_predictions[i]) for i in top_indices]

    return top_k_classes, top_k_confidences, inference_time_ms


def get_gradcam_model(
    model: keras.Model,
    last_conv_layer_name: str = 'conv5_block3_out'
) -> keras.Model:
    """
    Build a gradient model returning [conv_activation, pre_softmax_logits].
    Uses linear logits (before softmax) as recommended in original Grad-CAM paper
    to prevent gradient saturation and cross-class interference.
    """
    if hasattr(model, '_cached_grad_model'):
        return model._cached_grad_model

    last_conv_layer = model.get_layer(last_conv_layer_name)
    dense_layer = model.get_layer('dense_1')
    dropout_layer = model.get_layer('dropout')

    dense_linear = keras.layers.Dense(
        units=dense_layer.units,
        activation=None,
        name='dense_linear_cam'
    )
    logits_output = dense_linear(dropout_layer.output)
    dense_linear.set_weights(dense_layer.get_weights())

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, logits_output]
    )
    model._cached_grad_model = grad_model
    return grad_model


def generate_gradcam_heatmap(
    model: keras.Model,
    image: Image.Image,
    class_idx: Optional[int] = None,
    last_conv_layer_name: str = 'conv5_block3_out'
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the input PIL image using un-saturated logits.
    Returns a 2D numpy array with values normalized in [0, 1].
    """
    img_tensor = preprocess_pil_image(image)
    grad_model = get_gradcam_model(model, last_conv_layer_name=last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, logits = grad_model(img_tensor, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(logits[0]))
        loss = logits[:, class_idx]

    # Gradient of target class score with respect to conv feature map
    grads = tape.gradient(loss, conv_outputs)

    # Channel-wise mean of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of forward activation maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: keep only features having a positive impact on the class
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()
