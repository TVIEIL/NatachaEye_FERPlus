import cv2
import numpy as np
import subprocess
import threading
import time
import psutil
from ultralytics import YOLO

# --- INITIALISATION GLOBALE (Pour éviter le NameError) ---
# On charge le modèle une seule fois ici
model = YOLO("yolov8n.pt") 

# --- CONFIGURATION PIPELINE ---
width, height = 1280, 720
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"nvvidconv ! videoconvert ! video/x-raw, width={width}, height={height}, format=BGR ! "
    f"filesink location=/dev/stdout"
)
process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("🚀 Natacha est éveillée et consciente.")

# --- VARIABLES DIVERSES ---
prev_time = 0

try:
    while True:
        # 1. Capture
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3: continue
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3).copy()
        
        # 2. Inversion BGR -> RGB (pour corriger les couleurs)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Calculs (FPS & Temp)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        
        # 4. Inférence YOLO (Mode CPU pour garantir la stabilité)
        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 5. Dashboard
        cv2.putText(frame, f"FPS: {fps:.1f} | Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Natacha Eye", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    process.kill()
    cv2.destroyAllWindows()
