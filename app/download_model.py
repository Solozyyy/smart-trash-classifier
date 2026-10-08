"""
Helper script to download model weights from Google Drive if not present locally.
Used for Streamlit Cloud deployment where weights files are not committed to Git.
"""

import os
import sys

# File ID from Google Drive: https://drive.google.com/file/d/1WepUV4f8a_Q8UYFiRiOySyBMvETpO82z/view?usp=sharing
GDRIVE_FILE_ID = "1WepUV4f8a_Q8UYFiRiOySyBMvETpO82z"

# Define target paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model_converted.h5")


def ensure_model_exists(target_path: str = MODEL_PATH, file_id: str = GDRIVE_FILE_ID) -> str:
    """
    Check if the model file exists and is non-empty.
    If missing, download it automatically from Google Drive via gdown.
    
    Returns:
        str: Absolute path to the model file.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Check if target model already exists and is valid (> 10MB)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10 * 1024 * 1024:
        return target_path

    # Check fallback: original best_model.h5
    fallback_path = os.path.join(WEIGHTS_DIR, "best_model.h5")
    if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 10 * 1024 * 1024:
        return fallback_path

    print(f"[Model Loader] Model weights not found at '{target_path}'.")
    print(f"[Model Loader] Downloading model from Google Drive (ID: {file_id})...")

    try:
        import gdown
    except ImportError:
        raise ImportError("Package 'gdown' is required to download weights. Please install it via 'pip install gdown'.")

    # Download file using gdown
    gdown.download(id=file_id, output=target_path, quiet=False)

    if not os.path.exists(target_path) or os.path.getsize(target_path) < 10 * 1024 * 1024:
        raise RuntimeError(
            f"Failed to download model or file is incomplete ({target_path}). "
            "Please check the Google Drive sharing permissions and internet connection."
        )

    print(f"[Model Loader] Model successfully downloaded to: {target_path}")
    return target_path


if __name__ == "__main__":
    ensure_model_exists()
