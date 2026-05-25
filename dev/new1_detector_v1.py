#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import threading
import time
import psutil
from collections import deque

# --- CONFIGURATION ONNX ---
# Assure-toi que ce fichier est dans le même dossier
MODEL_PATH = "emotion-ferplus.onnx"
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Variables globales
emotion_detectee = "Neutre"
all_emotions = {k: 0.0 for k in EMOTION_LABELS}
verrou = threading.Lock()
history_size = 5
emotions_history = deque(maxlen=history_size)

def process_emotion_onnx(roi):
    global emotion_detectee, all_emotions
    
    # Prétraitement
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
    
    # Inférence
    net.setInput(blob)
    output = net.forward()
    
    # Softmax
    probabilities = np.exp(output[0]) / np.sum(np.exp(output[0]))
    results_dict = dict(zip(EMOTION_LABELS, probabilities))
    
    # Lissage
    emotions_history.append(results_dict)
    avg_emotions = {k: 0 for k in EMOTION_LABELS}
    for em_dict in emotions_history:
        for k, v in em_dict.items():
            avg_emotions[k] += v / len(emotions_history)
            
    top_em = max(avg_emotions, key=avg_emotions.get)
    
    with verrou:
        emotion_detectee = top_em
        all_emotions = avg_emotions

# --- BOUCLE PRINCIPALE (Résumé) ---
# ... (pipeline caméra identique) ...

while True:
    ret, frame = cap.read()
    if not ret: continue
    
    # ... (logique de détection YOLO identique) ...
    
    # AFFICHAGE PROPRE (Zéro chevauchement)
    cv2.rectangle(frame, (10, 10), (350, 350), (0, 0, 0), -1)
    
    # Infos système
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
    
    cv2.putText(frame, f"FPS: {fps:.1f} | Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Thierry | {emotion_detectee}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Liste des émotions avec enumerate (La correction pour le chevauchement)
    start_y = 120
    for i, (emotion, score) in enumerate(all_emotions.items()):
        current_y = start_y + (i * 25) # Espacement vertical fixe
        text = f"{emotion}: {score:.2f}"
        color = (0, 255, 0) if emotion == emotion_detectee else (200, 200, 200)
        cv2.putText(frame, text, (20, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    cv2.imshow("Natacha Eye", frame)
    # ...
