import cv2
# Cherche si GStreamer est supporté dans la build d'OpenCV
print("GStreamer supporté : " in cv2.getBuildInformation()) 
# Ou plus simplement :
print(cv2.getBuildInformation().split("GStreamer:")[1].split("\n")[0])
