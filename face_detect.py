

import cv2
import os

class FaceDetector:

    

    def __init__(self, cascade_path):
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Face cascade file not found at: {cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise IOError(f"Failed to load face cascade from: {cascade_path}. Check file integrity.")
        
        print(f"FaceDetector initialized with cascade: {cascade_path}")

    def detect_faces(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)):
        
        # Keeping the image grayscale for detection
        if len(image.shape) == 3: # Check if it's a color image (height, width, channels)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image # Already grayscale

        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces


if __name__ == "__main__":
   
    _current_dir = os.path.dirname(__file__) # This is 'Source/'
    # Go up one level to the project root, then into 'models/'
    _project_root = os.path.abspath(os.path.join(_current_dir, '..'))
    _cascade_file_path = os.path.join(_project_root, 'models', 'haarcascade_frontalface_default.xml')

    print(f"Attempting to load face cascade from: {_cascade_file_path}")

    try:
        face_detector = FaceDetector(cascade_path=_cascade_file_path)

        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam for testing FaceDetector.")
            exit()

        print("Webcam opened. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame during test. Exiting.")
                break

            detected_faces = face_detector.detect_faces(frame)

            for (x, y, w, h) in detected_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue rectangle

            cv2.imshow('Face Detection Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Face detection test finished.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'haarcascade_frontalface_default.xml' is in the specified path.")
    except IOError as e:
        print(f"Error initializing FaceDetector: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")