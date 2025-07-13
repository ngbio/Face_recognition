# Face Emotion Detector

This is a real-time AI-based face emotion detector built with Python, OpenCV, and TensorFlow.  
It uses a pre-trained convolutional neural network (CNN) model trained on the FER2013 dataset to classify emotions like **Happy**, **Sad**, **Angry**, **Neutral**, and more from a live webcam feed.

## Features
- Real-time face detection using Haar Cascades
- Emotion classification using Mini-XCEPTION CNN
- Live webcam video processing
- Fast and lightweight performance

## Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Haar Cascades

## Run Locally
```bash
git clone https://github.com/yourusername/face-emotion-detector.git
cd face-emotion-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
