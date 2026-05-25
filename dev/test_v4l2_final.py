import cv2

# On utilise uniquement V4L2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Réglage basique pour forcer la résolution (V4L2 sur Jetson a besoin de ça)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Échec de l'ouverture avec V4L2.")
else:
    print("Succès ! Ouverture V4L2 OK.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture.")
            break
        cv2.imshow("Natacha Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
