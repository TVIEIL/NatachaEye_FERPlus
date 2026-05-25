import cv2
import numpy as np
import subprocess
import time
import psutil
from ultralytics import YOLO

# --- CONFIGURATION ---
width, height = 1280, 720
model = YOLO("yolov8n.pt") 

# Pipeline : le GPU envoie du BGR propre vers le stdout
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"nvvidconv ! videoconvert ! video/x-raw, width={width}, height={height}, format=BGR ! "
    f"filesink location=/dev/stdout"
)

# On lance le pipeline via subprocess
process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("🚀 Natacha est éveillée (Mode Pipe Robuste).")

try:
    while True:
        # Lecture du flux binaire brut
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3: continue
        
        # Transformation en image numpy
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3).copy()
        
        # --- CORRECTION COULEURS ---
        # Si c'est bleu/violet, on inverse les canaux
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- DÉTECTION (CPU pour la stabilité) ---
        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- DASHBOARD ---
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        cv2.putText(frame, f"Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Natacha Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    process.kill()
    cv2.destroyAllWindows()
