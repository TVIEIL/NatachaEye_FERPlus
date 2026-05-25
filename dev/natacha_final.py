import cv2
import numpy as np
import subprocess
import threading
from collections import deque
from ultralytics import YOLO

# --- CONFIGURATION ---
width, height = 1280, 720
MODEL_PATH = "emotion-ferplus.onnx"
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
model = YOLO("yolov8n.pt")
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# --- DÉMARRAGE GSTREAMER INTÉGRÉ ---
# On force la conversion en BGR ici, dans le pipeline, pour que OpenCV reçoive du BGR pur
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"nvvidconv ! videoconvert ! video/x-raw, width={width}, height={height}, format=BGR ! "
    f"filesink location=/dev/stdout"
)
process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("🚀 Natacha V2 prête. Capture et détection intégrées.")

try:
    while True:
        # Lecture des pixels
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3: continue
        #frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3).copy()

        # Détection YOLO
        results = model(frame, stream=True, verbose=False)
        
        # Affichage
        cv2.putText(frame, "Natacha V2 - Thierry", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Natacha Eye", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    process.kill()
    cv2.destroyAllWindows()
