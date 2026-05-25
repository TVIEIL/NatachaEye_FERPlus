import cv2
import numpy as np
import threading
import time
import psutil
from collections import deque
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import os
from dotenv import load_dotenv


# --- CONFIGURATION AUTOMATIQUE ---
FICHIER_ENV = ".env"
if not os.path.exists(FICHIER_ENV):
    with open(FICHIER_ENV, "w", encoding="utf-8") as f:
        f.write("MQTT_BROKER=localhost\nMQTT_PORT=1883\nMQTT_TOPIC=natacha/oeil_detection\nYOLO_CONFIDENCE=0.45\nANALYSE_EMOTION=True\nAFFICHAGE_GRAPHIQUE=False\n")

load_dotenv()
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "natacha/oeil_detection")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.45))
ANALYSE_EMOTION = os.getenv("ANALYSE_EMOTION", "True").lower() == "true"
AFFICHAGE_GRAPHIQUE = os.getenv("AFFICHAGE_GRAPHIQUE", "True").lower() == "true"


print("📡 Connexion au réseau MQTT...")
client_mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
try:
    client_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
    client_mqtt.loop_start()
except Exception as e:
    print(f"⚠️ Mode déconnecté réseau ({e})")


# Une mémoire pour garder les 20 dernières couleurs vues
couleurs_buffer = deque(maxlen=20)

# --- INITIALISATION ---
model = YOLO("yolov8n.pt") 
net_emotion = cv2.dnn.readNetFromONNX("emotion-ferplus.onnx")
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

emotion_actuelle = "Analyse..."
verrou = threading.Lock()


def get_color_name_autonome(b, g, r):
    # Ajouter la mesure actuelle dans le buffer
    couleurs_buffer.append((b, g, r))
    
    # Calculer la médiane
    b_med = np.median([c[0] for c in couleurs_buffer])
    g_med = np.median([c[1] for c in couleurs_buffer])
    r_med = np.median([c[2] for c in couleurs_buffer])
    
    somme = b_med + g_med + r_med + 1e-6
    b_ratio = b_med / somme
    g_ratio = g_med / somme
    r_ratio = r_med / somme
    
    # --- LA CORRECTION DE LA PIÈCE ---  (lumiere jaune riche en couleur rouge)
    # Puisque R est à 0.38 et B à 0.31, il y a un écart de 0.07.
    # On donne un bonus au bleu pour annuler cet écart.
    bonus_bleu = 0.08 
    b_ratio_corrige = b_ratio + bonus_bleu
    
    # Debug pour vérifier la correction (on verra les chiffres ajustés)
    # print(f"DEBUG RATIOS -> B_corrige:{b_ratio_corrige:.2f} | R:{r_ratio:.2f}")

    # Classification robuste basée sur les ratios corrigés
    if (b_ratio + g_ratio + r_ratio) < 0.1: return "Noir"
    
    # Maintenant, on compare avec le ratio corrigé
    if b_ratio_corrige > r_ratio and b_ratio_corrige > g_ratio: 
        return "Bleu"
    
    if r_ratio > b_ratio_corrige + 0.05 and r_ratio > g_ratio: 
        return "Rouge"
        
    if g_ratio > b_ratio_corrige + 0.05 and g_ratio > r_ratio: 
        return "Vert"
    
    return "Autre"
    
    

def get_color_name(b, g, r):
    # --- DEBUG : Affiche les valeurs en temps réel dans ton terminal ---
    #print(f"DEBUG -> B: {b:.0f} | G: {g:.0f} | R: {r:.0f}")

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
#cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(1)


# Initialisation des variables pour le MQTT (juste avant le while)
dernier_envoi_mqtt = 0
intervalle_mqtt_standard = 10 
statut_precedent = None
couleur_veste_precedente = None

print("🚀 Natacha est éveillée : Analyse émotion + style vestimentaire active.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        results = model.predict(frame, device="cpu", stream=True, verbose=False)
        
        # 1. On initialise le flag à False à chaque nouvelle image
        personne_detectee = False
        
        for r in results:
            for box in r.boxes:
                if model.names[int(box.cls[0])] == "person":
                    personne_detectee = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 1. CALCUL DES ZONES
                    #veste_x1 = x1 + int((x2 - x1) * 0.3)
                    #veste_y1 = y1 + int((y2 - y1) * 0.55)
                    #veste_x2 = x2 - int((x2 - x1) * 0.3)
                    #veste_y2 = y1 + int((y2 - y1) * 0.85)
                    #visage_y2 = y1 + int((y2 - y1) * 0.4) 
                    
                    h_personne = y2 - y1
                    w_personne = x2 - x1
                    
                    # On définit la zone du torse (la veste)
                    # On augmente les ratios pour descendre
                    # On passe à 0.75 pour démarrer le rectangle plus bas sur le torse
                    # On passe à 0.98 pour l'étendre jusqu'au bas du torse
                    veste_y1 = y1 + int(h_personne * 0.75) 
                    veste_y2 = y1 + int(h_personne * 0.98) 
                    veste_x1 = x1 + int(w_personne * 0.3)
                    veste_x2 = x2 - int(w_personne * 0.3)
                    
                    visage_y2 = y1 + int((y2 - y1) * 0.4) 
                    
                    # 2. ANALYSE COULEUR (AVANT D'AFFICHER)
                    torso_roi = frame[veste_y1:veste_y2, veste_x1:veste_x2]
                    couleur_text = "..." # On initialise par défaut
                    
                    if torso_roi.size > 0:
                        b, g, r = np.mean(torso_roi, axis=(0, 1))
                        couleur_text = get_color_name_autonome(b, g, r)

                    # 3. DESSIN (UNIQUEMENT APRÈS LE CALCUL)
                    # Rectangle Veste (Vert)
                    cv2.rectangle(frame, (veste_x1, veste_y1), (veste_x2, veste_y2), (0, 255, 0), 2)
                    # Texte Veste
                    cv2.putText(frame, f"Veste: {couleur_text}", (veste_x1, veste_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 4. EMOTION (AVEC LA ZONE VISAGE)
                    if ANALYSE_EMOTION:
                        roi_visage = frame[y1:visage_y2, x1:x2]
                        if roi_visage.size > 0:
                            if not any(t.name == 'emotion_thread' for t in threading.enumerate()):
                                threading.Thread(target=analyser_emotion_thread, args=(roi_visage,), name='emotion_thread', daemon=True).start()
                    else:
                        emotion_actuelle = "Desactive"

                    # Rectangle principal (Bleu)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    
        #  Gestion de l'absence
        if personne_detectee:
            # Exemple : envoyer un message MQTT ou réinitialiser l'affichage
            # client_mqtt.publish(MQTT_TOPIC, "Personne absente")
            cv2.putText(frame, "Present", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
           cv2.putText(frame, "Absent", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           emotion_actuelle = "En attente..." # Reset de l'émotion si personne
           
        # ... (ton code de détection se termine ici) ...
        # Fin de : for r in results:

        # --- 2. LOGIQUE D'ENVOI MQTT ---
        statut_actuel = "present" if personne_detectee else "absent"
        couleur_veste = couleur_text if personne_detectee else "inconnue"
        # On récupère la confiance si détectée, sinon 0
        conf_trouvee = float(box.conf[0]) if personne_detectee else 0.0 
        
        # On force l'envoi si le statut ou la couleur change
        force_envoi_mqtt = (statut_actuel != statut_precedent) or (couleur_veste != couleur_veste_precedente)
        temps_actuel = time.time()

        if force_envoi_mqtt or (temps_actuel - dernier_envoi_mqtt > intervalle_mqtt_standard):
            paquet = {
                "cible": "interlocuteur", 
                "statut": statut_actuel,
                "confiance": round(conf_trouvee, 2) if statut_actuel == "present" else 0.0,
                "couleur_vetement": couleur_veste, 
                "emotion": emotion_actuelle, 
                "timestamp": int(temps_actuel)
            }
            client_mqtt.publish(MQTT_TOPIC, json.dumps(paquet))
            
            dernier_envoi_mqtt = temps_actuel
            statut_precedent = statut_actuel
            couleur_veste_precedente = couleur_veste
            print(f"📡 [MQTT] Statut: {statut_actuel} | Veste: {couleur_veste} | Humeur: {emotion_actuelle}")


        # Dashboard global
        temp = psutil.sensors_temperatures()['cpu-thermal'][0].current if 'cpu-thermal' in psutil.sensors_temperatures() else 0
        cv2.putText(frame, f"Emotion: {emotion_actuelle} | Temp: {temp:.1f}C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # --- GESTION AFFICHAGE CONDITIONNEL ---
        if AFFICHAGE_GRAPHIQUE:
            cv2.imshow("Natacha Eye", frame)
            # On ne peut quitter avec 'q' que si la fenêtre est affichée
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Si pas d'affichage, on fait une petite pause pour ne pas saturer le CPU
            time.sleep(0.01)

finally:
    cap.release()
    cv2.destroyAllWindows()
