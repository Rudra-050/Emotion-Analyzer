

import os
import cv2 

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


HAARCASCADE_FACE_PATH = os.path.join(_BASE_DIR, 'models', 'haarcascade_frontalface_default.xml')
EMOTION_MODEL_PATH = os.path.join(_BASE_DIR, 'models', 'em_recog.h5')


print(f"Debug: HAARCASCADE_FACE_PATH resolved to: {HAARCASCADE_FACE_PATH}") 
print(f"Debug: EMOTION_MODEL_PATH resolved to: {EMOTION_MODEL_PATH}") 

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

MODEL_INPUT_SHAPE = (48, 48)


MODEL_CHANNELS = 1 


FACE_SCALE_FACTOR = 1.1       
FACE_MIN_NEIGHBORS = 5       
FACE_MIN_SIZE = (80, 80)      


WEBCAM_INDEX = 0             
DISPLAY_WINDOW_NAME = "Emotion Analyzer"
FONT = cv2.FONT_HERSHEY_SIMPLEX 
FONT_SCALE = 0.9
FONT_THICKNESS = 2


COLOR_BLUE = (255, 0, 0)      
COLOR_GREEN = (0, 255, 0)     
COLOR_RED = (0, 0, 255)       
COLOR_YELLOW = (0, 255, 255)  
COLOR_ORANGE = (0, 165, 255)
COLOR_PURPLE = (128, 0, 128)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)


EMOTION_COLORS = {
    "Happy": COLOR_GREEN,
    "Sad": COLOR_BLUE,
    "Angry": COLOR_RED,
    "Surprise": COLOR_YELLOW,
    "Fear": COLOR_PURPLE,
    "Disgust": COLOR_ORANGE,
    "Neutral": COLOR_CYAN,
    "Error": COLOR_WHITE # For error states
}


def print_config():
    """Prints all configuration parameters."""
    print("\n--- Application Configuration ---")
    print(f"Base Directory: {_BASE_DIR}")
    print(f"Face Cascade Path: {HAARCASCADE_FACE_PATH}")
    print(f"Emotion Model Path: {EMOTION_MODEL_PATH}")
    print(f"Emotions List: {EMOTIONS_LIST}")
    print(f"Model Input Shape: {MODEL_INPUT_SHAPE}")
    print(f"Model Channels: {MODEL_CHANNELS}")
    print(f"Face Scale Factor: {FACE_SCALE_FACTOR}")
    print(f"Face Min Neighbors: {FACE_MIN_NEIGHBORS}")
    print(f"Face Min Size: {FACE_MIN_SIZE}")
    print(f"Webcam Index: {WEBCAM_INDEX}")
    print(f"Display Window Name: {DISPLAY_WINDOW_NAME}")
    print("-----------------------------------\n")


if __name__ == "__main__":
 
    
    print_config()