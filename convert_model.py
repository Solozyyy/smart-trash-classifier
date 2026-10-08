"""
Convert old TensorFlow model to new format compatible with TF 2.12+
Run this script ONCE to convert the model
"""

import tensorflow as tf
from tensorflow import keras
import os

print("TensorFlow version:", tf.__version__)

# Path to old model
old_model_path = "weights/best_model.h5"
new_model_path = "weights/best_model_converted.h5"

print(f"\n🔄 Converting model from: {old_model_path}")
print(f"📥 Will save to: {new_model_path}")

try:
    # Try loading with old Keras API
    print("\n1️⃣ Attempting to load with tf.keras.models.load_model...")
    
    # Custom object scope to handle old layers
    with tf.keras.utils.custom_object_scope({}):
        model = tf.keras.models.load_model(
            old_model_path,
            compile=False
        )
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model summary:")
    print(f"   - Total layers: {len(model.layers)}")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    
    # Save in new format
    print(f"\n2️⃣ Saving model in new format...")
    model.save(new_model_path, save_format='h5')
    print(f"✅ Model saved to: {new_model_path}")
    
    # Verify new model
    print(f"\n3️⃣ Verifying converted model...")
    test_model = tf.keras.models.load_model(new_model_path, compile=False)
    print("✅ Converted model loads successfully!")
    
    print("\n" + "="*50)
    print("✅ CONVERSION SUCCESSFUL!")
    print("="*50)
    print(f"\n📝 Next steps:")
    print(f"1. Update streamlit_app.py to use: '{new_model_path}'")
    print(f"2. Or rename: {new_model_path} → {old_model_path}")
    print(f"3. Delete old model if conversion is good")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("\n🔄 Trying alternative method...")
    
    # Alternative: Load weights only and rebuild
    try:
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras import layers, models
        
        print("\n4️⃣ Rebuilding model architecture...")
        
        # Rebuild exact architecture from your training
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base
        base_model.trainable = False
        
        # Add custom head - EXACT match from training
        x = base_model.output  # Shape: (None, 7, 7, 2048)
        x = layers.Flatten()(x)  # Shape: (None, 100352) = 7*7*2048
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(12, activation='softmax')(x)  # Only 2 Dense layers!
        
        model = models.Model(inputs=base_model.input, outputs=predictions)
        
        print(f"✅ Architecture rebuilt!")
        print(f"   - Total layers: {len(model.layers)}")
        
        # Try to load weights
        print("\n5️⃣ Loading weights from old model...")
        model.load_weights(old_model_path, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded (with skip_mismatch=True)")
        
        # Save new model
        print(f"\n6️⃣ Saving rebuilt model...")
        model.save(new_model_path, save_format='h5')
        print(f"✅ Rebuilt model saved to: {new_model_path}")
        
        print("\n" + "="*50)
        print("✅ REBUILD SUCCESSFUL!")
        print("="*50)
        print(f"\n⚠️  Note: Some weights may have been skipped")
        print(f"   Test the model before using in production")
        
    except Exception as e2:
        print(f"\n❌ Alternative method also failed: {str(e2)}")
        print("\n💡 Suggestion: Re-train the model with current TensorFlow version")
