"""
Configuration loader for Smart Trash Classifier
Automatically detects Colab vs Local environment and loads appropriate paths
"""

import os
from pathlib import Path


class Config:
    """
    Configuration class that auto-detects environment and loads settings
    
    Usage:
        from config import config
        
        print(config.DATA_DIR)
        print(config.TARGET_COUNT)
    """
    
    def __init__(self):
        # Detect environment (Colab or Local)
        self.IS_COLAB = self._detect_colab()
        
        # Load environment variables from .env file
        self._load_env_file()
        
        # Setup paths based on environment
        if self.IS_COLAB:
            self._setup_colab_paths()
        else:
            self._setup_local_paths()
        
        # Load training hyperparameters
        self._load_training_config()
        
        print(f"✅ Config loaded: {'Colab' if self.IS_COLAB else 'Local'} environment")
    
    def _detect_colab(self):
        """Auto-detect if running on Google Colab"""
        try:
            from google.colab import drive
            return True
        except ImportError:
            return False
    
    def _load_env_file(self):
        """Load variables from .env file (if exists)"""
        env_path = Path('.env')
        
        if not env_path.exists():
            print("⚠️  File .env không tồn tại. Sử dụng giá trị mặc định.")
            print("💡 Tạo file .env từ .env.example: cp .env.example .env")
            return
        
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    def _setup_colab_paths(self):
        """Setup paths for Google Colab environment"""
        self.DRIVE_PATH = os.getenv('COLAB_DRIVE_PATH', '/content/drive/MyDrive/data.zip')
        self.DATA_DIR = os.getenv('COLAB_DATA_DIR', '/content/data_extracted/')
        self.BALANCED_DIR = os.getenv('COLAB_BALANCED_DIR', '/content/balanced_data/')
        self.WEIGHTS_DIR = os.getenv('COLAB_WEIGHTS_DIR', '/content/weights/')
        
        # Create directories if not exist
        for directory in [self.DATA_DIR, self.BALANCED_DIR, self.WEIGHTS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_local_paths(self):
        """Setup paths for Local environment"""
        project_root = os.getenv('LOCAL_PROJECT_ROOT', os.getcwd())
        
        # Validate project root
        if project_root == 'your_project_path_here':
            print("⚠️  Cảnh báo: LOCAL_PROJECT_ROOT chưa được cấu hình!")
            print("📝 Vui lòng chỉnh sửa file .env và đổi LOCAL_PROJECT_ROOT")
            project_root = os.getcwd()
        
        self.DATA_DIR = os.path.join(project_root, os.getenv('LOCAL_DATA_DIR', 'data'))
        self.BALANCED_DIR = os.path.join(project_root, os.getenv('LOCAL_BALANCED_DIR', 'data_balanced'))
        self.WEIGHTS_DIR = os.path.join(project_root, os.getenv('LOCAL_WEIGHTS_DIR', 'weights'))
        
        # Create directories if not exist
        for directory in [self.BALANCED_DIR, self.WEIGHTS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Check if data directory exists
        if not os.path.exists(self.DATA_DIR):
            print(f"⚠️  Cảnh báo: Không tìm thấy thư mục data tại {self.DATA_DIR}")
            print("📝 Vui lòng giải nén data vào thư mục này")
    
    def _load_training_config(self):
        """Load training hyperparameters"""
        self.TARGET_COUNT = int(os.getenv('TARGET_IMAGES_PER_CLASS', 2000))
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
        self.EPOCHS = int(os.getenv('EPOCHS', 15))
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
        self.IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
        
        # Model config
        self.MODEL_NAME = os.getenv('MODEL_NAME', 'resnet18')
        self.PRETRAINED = os.getenv('PRETRAINED', 'true').lower() == 'true'
        self.RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
    
    def print_config(self):
        """Print all configuration values"""
        print("\n" + "="*50)
        print("📋 CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Environment:     {'Google Colab' if self.IS_COLAB else 'Local'}")
        print(f"Data Dir:        {self.DATA_DIR}")
        print(f"Balanced Dir:    {self.BALANCED_DIR}")
        print(f"Weights Dir:     {self.WEIGHTS_DIR}")
        print(f"\nTraining Config:")
        print(f"  Model:         {self.MODEL_NAME}")
        print(f"  Pretrained:    {self.PRETRAINED}")
        print(f"  Target Images: {self.TARGET_COUNT}")
        print(f"  Batch Size:    {self.BATCH_SIZE}")
        print(f"  Epochs:        {self.EPOCHS}")
        print(f"  Learning Rate: {self.LEARNING_RATE}")
        print(f"  Image Size:    {self.IMAGE_SIZE}")
        print(f"  Random Seed:   {self.RANDOM_SEED}")
        print("="*50 + "\n")


# Global config instance
config = Config()


# Quick test
if __name__ == "__main__":
    config.print_config()
