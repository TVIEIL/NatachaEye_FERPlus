import cv2
import numpy as np
import subprocess

width = 1280
height = 720

# Pipeline FORCÉ en BGR pur (3 octets par pixel, fini le 'x')
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! "
    f"videoconvert ! video/x-raw, format=BGR ! "
    f"filesink location=/dev/stdout"
)

process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

print("Flux capturé... Correction des couleurs activée.")

try:
    while True:
        # On lit exactement 3 octets par pixel (BGR)
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3:
            continue
            
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
        
        # SI LES COULEURS SONT ENCORE INVERSÉES :
        # OpenCV attend du BGR. Si ton image a toujours les couleurs bizarres,
        # décommente la ligne ci-dessous pour inverser proprement :
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Natacha Eye (FIXED)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    process.kill()
    cv2.destroyAllWindows()
