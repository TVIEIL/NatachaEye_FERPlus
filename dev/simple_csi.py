import cv2

# On ouvre la caméra virtuelle (qui est maintenant le pont vers ta caméra CSI)
# Essaye 1 ou 0 selon comment le système a assigné /dev/video1
cap = cv2.VideoCapture(1) 

if not cap.isOpened():
    # Si ça ne marche pas, essaye l'index 0
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra virtuelle.")
else:
    print("Succès : Flux détecté ! Appuyez sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture.")
            break
        cv2.imshow("Natacha Eye (Loopback)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
