#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import threading
import time
import psutil
import os
from collections import deque
from ultralytics import YOLO

# --- CONFIGURATION ONNX ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion-ferplus.onnx")

if not os.path.exists(MODEL_PATH):
    print(f"CRITIQUE: Le fichier modèle n'existe pas : {MODEL_PATH}")
    exit()

# Chargement modèles
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
model = YOLO("yolov8n.pt") # Charge YOLO pour la détection
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Variables globales
emotion_detectee = "Neutre"
all_emotions = {k: 0.0 for k in EMOTION_LABELS}
verrou = threading.Lock()
history_size = 5
emotions_history = deque(maxlen=history_size)
last_emotion_check = 0
prev_time = 0
CHECK_INTERVAL = 1.0 

def process_emotion_onnx(roi):
    global emotion_detectee, all_emotions
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    probabilities = np.exp(output[0]) / np.sum(np.exp(output[0]))
    results_dict = dict(zip(EMOTION_LABELS, probabilities))
    
    emotions_history.append(results_dict)
    avg_emotions = {k: 0 for k in EMOTION_LABELS}
    for em_dict in emotions_history:
        for k, v in em_dict.items():
            avg_emotions[k] += v / len(emotions_history)
            
    top_em = max(avg_emotions, key=avg_emotions.get)
    
    with verrou:
        emotion_detectee = top_em
        all_emotions = avg_emotions

# --- INITIALISATION CAMÉRA ---
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=True sync=False"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

print("🚀 Natacha V2 est en ligne avec le moteur ONNX !")

while True:
    ret, frame = cap.read()
    print(f"DEBUG: ret={ret}") # Ajoute cette ligne
    if not ret: 
        print("DEBUG: Pas de frame capturée, le pipeline GStreamer bloque.")
        continue
        

    
    # Détection YOLO
    results = model(frame, stream=True, verbose=False)
    
    # Affichage DASHBOARD (Calculs)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
    
    # Dessin interface
    cv2.rectangle(frame, (10, 10), (350, 350), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f} | Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Thierry | {emotion_detectee}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Affichage émotions
    start_y = 120
    for i, (emotion, score) in enumerate(all_emotions.items()):
        current_y = start_y + (i * 25)
        text = f"{emotion}: {score:.2f}"
        color = (0, 255, 0) if emotion == emotion_detectee else (200, 200, 200)
        cv2.putText(frame, text, (20, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Logique Détection
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                roi_visage = frame[max(0, y1):y1+int((y2-y1)*0.5), x1:x2]
                
                if roi_visage.size > 0 and (time.time() - last_emotion_check) > CHECK_INTERVAL:
                    threading.Thread(target=process_emotion_onnx, args=(roi_visage,), daemon=True).start()
                    last_emotion_check = time.time()
                break
        
    cv2.imshow("Natacha Eye", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
