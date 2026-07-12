"""
Configuration loader for Smart Trash Classifier
Loads settings from config.yaml and auto-detects environment
"""

import os
import yaml
from pathlib import Path


class Config:
    """
    Configuration class that loads from config.yaml
    Auto-detects Colab vs Local environment
    
    Usage:
        from config import config
        
        print(config.DATA_DIR)
        print(config.BATCH_SIZE)
        config.print_config()
    """
    
    def __init__(self, config_file='config.yaml'):
        """Initialize configuration"""
        # Detect environment
        self.IS_COLAB = self._detect_colab()
        
        # Load config from YAML
        self.cfg = self._load_yaml(config_file)
        
        # Setup paths based on environment
        self._setup_paths()
        
        # Load training config
        self._load_training_config()
        
        # Load model config
        self._load_model_config()
        
        print(f"✅ Config loaded: {'Colab' if self.IS_COLAB else 'Local'} environment")
    
    def _detect_colab(self):
        """Auto-detect if running on Google Colab"""
        return 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ
    
    def _load_yaml(self, config_file):
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                "Please ensure config.yaml exists in the project root."
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_paths(self):
        """Setup paths based on environment (Colab or Local)"""
        env_key = 'colab' if self.IS_COLAB else 'local'
        paths = self.cfg['paths'][env_key]
        
        # Main paths
        if self.IS_COLAB:
            self.DRIVE_ZIP_PATH = paths['drive_zip']
            self.DATA_EXTRACTED_DIR = paths['data_extracted']
            self.BALANCED_DATA_DIR = paths['balanced_data']
            self.TRAIN_DATA_DIR = paths['train_data']
            self.VAL_DATA_DIR = paths['val_data']
            self.TEST_DATA_DIR = paths['test_data']
            self.MODEL_SAVE_PATH = paths['model_save']
            self.PLOTS_DIR = paths['plots']
        else:
            self.DATA_ROOT = paths['data_root']
            self.DATA_EXTRACTED_DIR = paths['data_extracted']
            self.BALANCED_DATA_DIR = paths['balanced_data']
            self.TRAIN_DATA_DIR = paths['train_data']
            self.VAL_DATA_DIR = paths['val_data']
            self.TEST_DATA_DIR = paths['test_data']
            self.MODEL_SAVE_PATH = paths['model_save']
            self.PLOTS_DIR = paths['plots']
        
        # Backward compatibility aliases
        self.DATA_DIR = self.DATA_EXTRACTED_DIR
        self.BALANCED_DIR = self.BALANCED_DATA_DIR
        self.WEIGHTS_DIR = os.path.dirname(self.MODEL_SAVE_PATH) or './weights'
    
    def _load_training_config(self):
        """Load training hyperparameters"""
        train_cfg = self.cfg['training']
        dataset_cfg = self.cfg['dataset']
        
        # Training params
        self.BATCH_SIZE = train_cfg['batch_size']
        self.EPOCHS = train_cfg['epochs']
        self.LEARNING_RATE = train_cfg['learning_rate']
        
        # Dataset params
        self.TARGET_COUNT = dataset_cfg['target_samples_per_class']
        self.NUM_CLASSES = dataset_cfg['num_classes']
        self.CLASSES = dataset_cfg['classes']
        
        # Data split
        self.TRAIN_RATIO = dataset_cfg['split']['train']
        self.VAL_RATIO = dataset_cfg['split']['val']
        self.TEST_RATIO = dataset_cfg['split']['test']
        
        # Random seed
        self.RANDOM_SEED = self.cfg['random']['seed']
    
    def _load_model_config(self):
        """Load model configuration"""
        model_cfg = self.cfg['model']
        
        self.MODEL_NAME = model_cfg['architecture']
        self.INPUT_SHAPE = tuple(model_cfg['input_shape'])
        self.IMAGE_SIZE = self.INPUT_SHAPE[0]  # Backward compatibility
        self.PRETRAINED_WEIGHTS = model_cfg['pretrained_weights']
        self.INCLUDE_TOP = model_cfg['include_top']
    
    def get_augmentation_config(self, mode='train'):
        """
        Get data augmentation configuration
        
        Args:
            mode: 'train' or 'balancing'
        
        Returns:
            dict: Augmentation parameters
        """
        return self.cfg['augmentation'][mode]
    
    def get_callbacks_config(self):
        """Get training callbacks configuration"""
        return self.cfg['training']['callbacks']
    
    def print_config(self):
        """Print all configuration values"""
        print("\n" + "="*60)
        print(" CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Environment:        {'Google Colab' if self.IS_COLAB else 'Local'}")
        
        print(f"\n Paths:")
        if self.IS_COLAB:
            print(f"  Drive ZIP:        {self.DRIVE_ZIP_PATH}")
        print(f"  Data Extracted:   {self.DATA_EXTRACTED_DIR}")
        print(f"  Balanced Data:    {self.BALANCED_DATA_DIR}")
        print(f"  Train Data:       {self.TRAIN_DATA_DIR}")
        print(f"  Val Data:         {self.VAL_DATA_DIR}")
        print(f"  Test Data:        {self.TEST_DATA_DIR}")
        print(f"  Model Save:       {self.MODEL_SAVE_PATH}")
        print(f"  Plots:            {self.PLOTS_DIR}")
        
        print(f"\n Dataset:")
        print(f"  Classes:          {self.NUM_CLASSES}")
        print(f"  Target/Class:     {self.TARGET_COUNT}")
        print(f"  Split:            {self.TRAIN_RATIO:.0%} / {self.VAL_RATIO:.0%} / {self.TEST_RATIO:.0%}")
        
        print(f"\n Model:")
        print(f"  Architecture:     {self.MODEL_NAME}")
        print(f"  Input Shape:      {self.INPUT_SHAPE}")
        print(f"  Pretrained:       {self.PRETRAINED_WEIGHTS}")
        
        print(f"\n Training:")
        print(f"  Batch Size:       {self.BATCH_SIZE}")
        print(f"  Epochs:           {self.EPOCHS}")
        print(f"  Learning Rate:    {self.LEARNING_RATE}")
        print(f"  Random Seed:      {self.RANDOM_SEED}")
        
        print("="*60 + "\n")


# Global config instance
config = Config()


# Quick test
if __name__ == "__main__":
    config.print_config()
    
    # Test augmentation config
    print("\n🔄 Augmentation Config (train):")
    print(config.get_augmentation_config('train'))
    
    print("\n📞 Callbacks Config:")
    print(config.get_callbacks_config())
