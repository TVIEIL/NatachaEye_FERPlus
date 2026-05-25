import cv2
import numpy as np
import subprocess

width = 1280
height = 720

# Pipeline : Sortie brute NV12 (YUV420 semi-planar)
# Pas de conversion, pas de padding, vitesse maximale.
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12 ! "
    f"filesink location=/dev/stdout"
)

process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("Flux NV12 capturé... Conversion YUV -> BGR activée.")

try:
    while True:
        # NV12 = width * height * 1.5 octets
        raw_size = int(width * height * 1.5)
        raw_frame = process.stdout.read(raw_size)
        
        if len(raw_frame) != raw_size:
            continue
            
        # Conversion du buffer brut en image OpenCV
        # On utilise le format natif NV12
        yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height * 3 // 2, width)
        
        # conversion magique de NV12 vers BGR
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)
        
        cv2.imshow("Natacha Eye (NV12 Native)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    process.kill()
    cv2.destroyAllWindows()
