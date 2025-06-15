

import cv2
import os
import sys


_current_dir = os.path.dirname(os.path.abspath(__file__))
_source_dir = os.path.join(_current_dir, 'Source')
if _source_dir not in sys.path:
    sys.path.append(_source_dir)


import config
from face_detect import FaceDetector
from emotion import EmotionAnalyzer

def run_emotion_detection():
    
    print("Starting Emotion Analyzer...")
    config.print_config() 

   
    try:
        face_detector = FaceDetector(cascade_path=config.HAARCASCADE_FACE_PATH)
    except (FileNotFoundError, IOError) as e:
        print(f"Error initializing FaceDetector: {e}")
        print("Please ensure the face cascade XML file is correctly placed and accessible.")
        return # Exit if face detector cannot be initialized
    except Exception as e:
        print(f"An unexpected error occurred during FaceDetector initialization: {e}")
        return

    
    try:
        emotion_analyzer = EmotionAnalyzer(
            model_path=config.EMOTION_MODEL_PATH,
            emotions_list=config.EMOTIONS_LIST,
            input_shape=config.MODEL_INPUT_SHAPE,
            channels=config.MODEL_CHANNELS
        )
    except Exception as e:
        print(f"Error initializing Emotion Analyzer: {e}")
        print("Please ensure TensorFlow is installed and the emotion model is valid.")
       
        pass 

    # --- Open Webcam ---
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {config.WEBCAM_INDEX}.")
        print("Please check if the webcam is connected and not in use by another application.")
        return

    print(f"\nWebcam opened successfully. Displaying '{config.DISPLAY_WINDOW_NAME}' window. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting application.")
            break

       
        frame = cv2.flip(frame, 1)

        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       
        faces = face_detector.detect_faces(
            gray_frame, # Added grayscale image to detector for efficiency
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE
        )

        for (x, y, w, h) in faces:
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_BLUE, 2)

            
            face_roi_gray = gray_frame[y:y + h, x:x + w]

            
            emotion_label, emotion_probability = emotion_analyzer.predict_emotion(face_roi_gray)
            
    
            text = f"{emotion_label}: {emotion_probability:.2f}"
            
            # color for the emotion label
            text_color = config.EMOTION_COLORS.get(emotion_label, config.COLOR_WHITE) # Default to white if not found

            # text above the face rectangle
            cv2.putText(frame, text, (x, y - 10),
                        config.FONT, config.FONT_SCALE, text_color, config.FONT_THICKNESS, cv2.LINE_AA)

        # Display the processed frame
        cv2.imshow(config.DISPLAY_WINDOW_NAME, frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Exiting application.")
            break

    # --- Release Resources ---
    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped. Resources released.")

if __name__ == "__main__":
    run_emotion_detection()