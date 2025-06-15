import cv2
import numpy as np
import os
import warnings 

#Checking for errors in the environment
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    _tensorflow_available = True
except ImportError:
    print("Warning: TensorFlow/Keras not found. EmotionRecognizer will operate in dummy mode.")
    _tensorflow_available = False

class EmotionAnalyzer:
    

    def __init__(self, model_path, emotions_list, input_shape=(48, 48), channels=1):
       
        self.model_path = model_path
        self.emotions = emotions_list
        self.input_shape = input_shape
        self.channels = channels
        self.emotion_classifier = None

        if not _tensorflow_available:
            print("EmotionAnalyzer: TensorFlow/Keras is not installed. Cannot load real model.")
            return

        if not os.path.exists(self.model_path):
            print(f"Warning: Emotion model file not found at: {self.model_path}")
            print("EmotionAnalyzer will operate in dummy mode.")
            return

        try:
            
            self.emotion_classifier = load_model(self.model_path, compile=False)

            print(f"EmotionAnalyzer: Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading emotion model from {self.model_path}: {e}")
            print("EmotionAnalyzer will operate in dummy mode.")
            self.emotion_classifier = None

    def preprocess_face(self, face_roi_gray):
       
        # Resize to the input size expected by the emotion model
        face_roi_resized = cv2.resize(face_roi_gray, self.input_shape, interpolation=cv2.INTER_AREA)

        # Normalize pixel values to be between 0 and 1
        face_roi_normalized = face_roi_resized.astype("float") / 255.0

        
        # For a single grayscale image (48x48, 1 channel), this becomes (1, 48, 48, 1)
        face_roi_processed = np.expand_dims(np.expand_dims(face_roi_normalized, -1), 0)

        return face_roi_processed

    def predict_emotion(self, face_roi_gray):
        
        if self.emotion_classifier is None:
            # Dummy prediction if the model couldn't be loaded or TensorFlow isn't available
            print("Using dummy emotion prediction (model not loaded).")
            random_emotion_index = np.random.randint(0, len(self.emotions))
            return self.emotions[random_emotion_index], 0.85 # Dummy probability

        try:
            processed_face = self.preprocess_face(face_roi_gray)
            
            # Perform prediction
            preds = self.emotion_classifier.predict(processed_face)[0]
            
            # Get the emotion label and its probability
            emotion_probability = np.max(preds)
            emotion_label = self.emotions[np.argmax(preds)]

            return emotion_label, emotion_probability

        except Exception as e:
            print(f"Error during emotion prediction: {e}")
            return "Error", 0.0 # Return an error state

#Main function to test the Emotion Analyzer
if __name__ == "__main__":
    

    _current_dir = os.path.dirname(__file__) # This is 'Source/'
    # Go up one level to the project root, then into 'models/'
    _project_root = os.path.abspath(os.path.join(_current_dir, '..'))
    _model_path = os.path.join(_project_root, 'models', 'em_recog.h5')
    
    # CRITICAL: This list MUST match the order of your model's output classes.
    # If your model's output neurons correspond to emotions in a different order,
    # you MUST change this list accordingly.
    _emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    print(f"Attempting to initialize EmotionRecognizer with model path: {_model_path}")
    recognizer = EmotionRecognizer(
        model_path=_model_path,
        emotions_list=_emotions,
        input_shape=(48, 48), # Common for FER2013, adjust if your model expects different
        channels=1
    )

    # Simple test: create a dummy grayscale image (e.g., 100x100)
    dummy_face_roi = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    print("\n--- Testing prediction with a dummy face ROI ---")
    predicted_emotion, confidence = recognizer.predict_emotion(dummy_face_roi)
    print(f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}")

    if recognizer.emotion_classifier is None:
        print("\nNote: Since no real model was loaded, predictions are random (dummy).")
        print("To enable real predictions, ensure TensorFlow is installed and")
        print("a valid 'em_recog.h5' file is available at the specified path.")