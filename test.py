import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_source_dir = os.path.join(_current_dir, 'Source')
if _source_dir not in sys.path:
    sys.path.append(_source_dir)


try:
    from tensorflow.keras.models import load_model
    print("TensorFlow/Keras found for model loading test.")
except ImportError:
    print("TensorFlow/Keras NOT found for model loading test. Please install it.")
    sys.exit(1) # Exit if TensorFlow isn't there

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_BASE_DIR, 'models', 'em_recog.h5')

print(f"Attempting to load model from: {model_path}")

try:
   
    model = load_model(model_path, compile=False)
    

    print("Model loaded successfully!")
    model.summary() 
    print("Model appears to be a valid Keras .h5 file.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("This indicates the .h5 file is either corrupted, incomplete, or incompatible with your TensorFlow version (even with compile=False).")
