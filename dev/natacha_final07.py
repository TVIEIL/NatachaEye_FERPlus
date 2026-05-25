import cv2
import time
import psutil
from ultralytics import YOLO

# Chargement du modèle
model = YOLO("yolov8n.pt") 

# On ouvre simplement la caméra virtuelle index 1
cap = cv2.VideoCapture(1)

print("🚀 Natacha est connectée à /dev/video1 (Flux fluide).")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Calculs
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        
        # Inférence YOLO (sur CPU pour ne pas saturer le GPU qui fait déjà le traitement vidéo)
        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Affichage
        cv2.putText(frame, f"Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Natacha Eye", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
