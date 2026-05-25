import cv2
import numpy as np
import subprocess
import threading
import time
import psutil
from ultralytics import YOLO

# --- CONFIGURATION MODÈLE ---
# On initialise le modèle ICI, avant toute utilisation
#model = YOLO("yolov8n.pt").to("cuda:0")
model_yolo = YOLO("yolov8n.pt").to("cpu")

# --- CONFIGURATION ---
width, height = 1280, 720
# Chargement des modèles
#model_yolo = YOLO("yolov8n.pt")
#model_yolo = YOLO("yolov8n.pt", device='cpu')
#model_yolo = YOLO("yolov8n.pt").to("cuda:0")

net_emotion = cv2.dnn.readNetFromONNX("emotion-ferplus.onnx")
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Variables globales pour le thread
emotion_actuelle = "Analyse..."
verrou = threading.Lock()

def detecter_emotion_thread(roi):
    global emotion_actuelle
    # Préparation de l'image pour le modèle
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
    net_emotion.setInput(blob)
    output = net_emotion.forward()
    # Récupération de l'indice max
    index = np.argmax(output[0])
    with verrou:
        emotion_actuelle = EMOTION_LABELS[index]

# --- PIPELINE ---
pipeline = (
    f"gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! "
    f"nvvidconv ! videoconvert ! video/x-raw, width={width}, height={height}, format=BGR ! "
    f"filesink location=/dev/stdout"
)
process = subprocess.Popen(pipeline.split(), stdout=subprocess.PIPE, bufsize=10**8)

# Variables pour FPS et Temp
prev_time = 0

print("🚀 Natacha est éveillée et consciente.")

try:
    while True:
        # 1. Capture
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3: continue
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3).copy()
        
        # 2. Calculs (FPS & Temp)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        
        # 3. YOLO (Détection Personne)
        #results = model_yolo(frame, stream=True, verbose=False)
        # Remplace ton appel model(...) par celui-ci :
        #results = model.predict(frame, device=0, stream=True, verbose=False)
        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                if model_yolo.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Extraction visage (on prend le haut du corps)
                    roi_visage = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
                    if roi_visage.size > 0:
                        # Lancer l'analyse émotion en thread séparé
                        if not any(t.name == 'emotion_thread' for t in threading.enumerate()):
                            threading.Thread(target=detecter_emotion_thread, args=(roi_visage,), name='emotion_thread', daemon=True).start()

        # 4. Affichage Dashboard
        cv2.putText(frame, f"FPS: {fps:.1f} | Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion_actuelle}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Natacha Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    process.kill()
    cv2.destroyAllWindows()
