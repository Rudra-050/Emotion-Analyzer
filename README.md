Emotion Analyzer


🌟 Project Overview

Emotion Analyzer is a real-time facial emotion recognition system that processes live webcam feeds to detect faces and classify their expressions into core human emotions. Built with Python, it leverages powerful computer vision and deep learning techniques to provide instant visual feedback on emotional states, making it a compelling demonstration of AI's ability to interpret non-verbal communication.

✨ Features

Real-time Face Detection: Utilizes OpenCV's Haar Cascades to accurately identify and localize faces in a live video stream.
Emotion Classification: Employs a pre-trained deep learning model (TensorFlow/Keras) to classify seven universal emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
Live Visualization: Overlays bounding boxes around detected faces and displays the predicted emotion label with its confidence score directly on the video feed.
Configurable Parameters: Easily adjust detection thresholds, model paths, and display settings via a centralized config.py file.

🚀 Technologies Used

Python 3.x

OpenCV (opencv-python): For video capture, image processing, face detection, and drawing.

TensorFlow/Keras: For loading and running the deep learning emotion recognition model.

NumPy: For numerical operations on image data.

📊Model Used

haarcascade_frontalface_default.xml: This is a standard OpenCV cascade. You can often find it within your OpenCV installation directory (e.g., opencv/data/haarcascades/) or download it from the official OpenCV GitHub repository. Place it in the models/ directory.

em_recog.h5: This is the  emotion recognition model placed  in the models/ directory. Crucially, this model was trained on the same 7 emotions in the exact order specified in config.py's EMOTIONS_LIST.
