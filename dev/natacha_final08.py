import cv2
import numpy as np
import threading
import time
import psutil
from ultralytics import YOLO

# --- INITIALISATION ---
model = YOLO("yolov8n.pt") 
net_emotion = cv2.dnn.readNetFromONNX("emotion-ferplus.onnx")
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Variables globales pour l'émotion
emotion_actuelle = "Initialisation..."
verrou = threading.Lock() # Pour éviter les conflits de données

def analyser_emotion_thread(roi):
    global emotion_actuelle
    # Préparation de l'image pour le modèle FER+
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
    
    net_emotion.setInput(blob)
    output = net_emotion.forward()
    
    # Récupération de l'indice de l'émotion la plus probable
    index = np.argmax(output[0])
    
    with verrou:
        emotion_actuelle = EMOTION_LABELS[index]

# --- FLUX VIDÉO ---
cap = cv2.VideoCapture(1) # Ton /dev/video1 (le pont GStreamer)

print("🚀 Natacha est éveillée et prête à analyser tes émotions.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Inférence YOLO (Détection de personne)
        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Extraction du visage (on prend le haut du corps)
                    # On ajuste les coordonnées pour cibler la zone tête
                    roi_visage = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
                    
                    if roi_visage.size > 0:
                        # Lancer l'analyse en thread séparé si aucun thread n'est déjà en cours
                        if not any(t.name == 'emotion_thread' for t in threading.enumerate()):
                            threading.Thread(target=analyser_emotion_thread, args=(roi_visage,), name='emotion_thread', daemon=True).start()

        # Dashboard
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        
        cv2.putText(frame, f"Emotion: {emotion_actuelle}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Temp: {temp:.1f}C", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Natacha Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
