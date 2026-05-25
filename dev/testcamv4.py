import cv2



# Pipeline optimisé avec 'videoconvert' pour forcer le format BGR
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=True sync=False"
)

print("Tentative d'ouverture de la caméra...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERREUR : Impossible d'ouvrir la caméra.")
else:
    print("SUCCÈS : Caméra ouverte ! Appuyez sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture du flux.")
            break
        cv2.imshow("Test Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
