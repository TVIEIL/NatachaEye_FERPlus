import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# --- CONFIGURATION ---
width, height = 1280, 720
model = YOLO("yolov8n.pt")

# Pipeline NV12 (YUV brut) - Plus de conversion hasardeuse ici
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12 ! "
    f"nvvidconv ! video/x-raw, width={width}, height={height}, format=NV12 ! "
    f"filesink location=/dev/stdout"
)

process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("🚀 Natacha V2 connectée (Mode NV12).")

try:
    while True:
        # NV12 = 1.5 octets par pixel (Width * Height * 1.5)
        raw_size = int(width * height * 1.5)
        raw_frame = process.stdout.read(raw_size)
        
        if len(raw_frame) != raw_size: continue
        
        # 1. On reformate le buffer YUV
        yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height * 3 // 2, width)
        
        # 2. On convertit proprement en BGR avec OpenCV
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)
        
        # Détection YOLO
        results = model(frame, stream=True, verbose=False)
        
        # Affichage
        cv2.putText(frame, "Natacha V2 - Thierry (OK)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Natacha Eye", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    process.kill()
    cv2.destroyAllWindows()
