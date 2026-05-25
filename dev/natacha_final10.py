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

emotion_actuelle = "Analyse..."
verrou = threading.Lock()

def get_color_name(b, g, r):
    # Si le total est trop bas, c'est sombre
    if (b + g + r) < 100: return "Noir"
    # Si les trois sont proches et élevés, c'est blanc/gris
    if abs(b-g) < 20 and abs(g-r) < 20 and r > 100: return "Blanc/Gris"
    
    # On cherche la couleur dominante
    max_val = max(b, g, r)
    
    if max_val == b and b > (g + 10) and b > (r + 10): return "Bleu"
    if max_val == r and r > (g + 10) and r > (b + 10): return "Rouge"
    if max_val == g and g > (b + 10) and g > (r + 10): return "Vert"
    
    return "Autre"

def analyser_emotion_thread(roi):
    global emotion_actuelle
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
    net_emotion.setInput(blob)
    output = net_emotion.forward()
    index = np.argmax(output[0])
    with verrou:
        emotion_actuelle = EMOTION_LABELS[index]

# --- FLUX VIDÉO ---
cap = cv2.VideoCapture(1) 

print("🚀 Natacha est éveillée : Analyse émotion + style vestimentaire active.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # --- ROI PLUS PRÉCIS ---
                    # On ignore le haut (visage) et le bas (pantalon/hanches)
                    # Le torse se situe en gros entre 30% et 70% de la hauteur totale de la boîte
                    # --- ROI PLUS BAS ---
                    # On commence à 50% de la hauteur de la boîte (pour être sous le cou)
                    # On finit à 90% (pour rester sur le buste)
                    h_tot = y2 - y1
                    w_tot = x2 - x1

                    # Calculs ajustés
                    # On réduit la hauteur pour n'avoir qu'une fine bande sur le torse
                    roi_y1 = y1 + int(h_tot * 0.70) 
                    roi_y2 = y1 + int(h_tot * 0.90) # Hauteur de seulement 20% du rectangle total
                    roi_x1 = x1 + int(w_tot * 0.35) # On resserre aussi sur les côtés pour éviter les bords
                    roi_x2 = x2 - int(w_tot * 0.35)

                    # Extraction
                    torso_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

                    # --- NOUVEAU : On dessine le ROI en VERT pour visualiser la zone de scan ---
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
                    
                    couleur_text = "..."
                    
                    # 1. Analyse Couleur Veste (Torse au centre)
                    h_roi = int((y2 - y1) * 0.3)
                    w_roi = int((x2 - x1) * 0.4)
                    roi_x1 = x1 + int((x2 - x1) * 0.3)
                    roi_y1 = y1 + int((y2 - y1) * 0.2)
                    
                    torso_roi = frame[roi_y1:roi_y1+h_roi, roi_x1:roi_x1+w_roi]
                    
                    couleur_text = "..."
                    if torso_roi.size > 0:
                        b, g, r_mean = np.mean(torso_roi, axis=(0, 1))
                        couleur_text = get_color_name(b, g, r_mean)

                    # 2. Analyse Émotion (Visage en haut)
                    roi_visage = frame[y1:y1+int((y2-y1)*0.4), x1:x2]
                    if roi_visage.size > 0:
                        if not any(t.name == 'emotion_thread' for t in threading.enumerate()):
                            threading.Thread(target=analyser_emotion_thread, args=(roi_visage,), name='emotion_thread', daemon=True).start()

                    # 3. Affichage
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Veste: {couleur_text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Dashboard global
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        cv2.putText(frame, f"Emotion: {emotion_actuelle} | Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Natacha Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
