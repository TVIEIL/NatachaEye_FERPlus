import cv2
import numpy as np
import subprocess

# On définit la résolution
width = 1280
height = 720

# Pipeline GStreamer qui envoie les pixels bruts (format BGR) vers stdout
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! "
    f"videoconvert ! video/x-raw, format=BGR ! "
    f"filesink location=/dev/stdout"
)

# Lancement du processus GStreamer
process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("Succès : Flux capturé par GStreamer, lecture en cours...")

try:
    while True:
        # On lit le nombre exact d'octets pour une frame (1280*720*3)
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3:
            continue
            
        # Conversion des octets bruts en image OpenCV
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
        
        cv2.imshow("Natacha Eye (Pipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    process.kill()
    cv2.destroyAllWindows()
