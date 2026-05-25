#!/bin/bash
cd /home/jetson/oeil_natacha_2

# 1. Nettoyage : On tue tout ce qui traîne proprement
sudo pkill -9 gst-launch-1.0
sudo pkill -9 python3
sleep 1

# 2. Chargement du pont virtuel
sudo modprobe v4l2loopback devices=1 video_nr=1 card_label="NatachaCam"
sudo chmod 666 /dev/video1

# 3. Lancement du pipeline en BACKGROUND (&)
echo "Lancement du pipeline vers /dev/video1..."
# Le "&" à la fin envoie la commande en arrière-plan
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1' ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! 'video/x-raw, format=YUY2' ! v4l2sink device=/dev/video1 sync=false > /dev/null 2>&1 &

# On récupère le PID (l'identifiant) du processus GStreamer tout juste lancé
GST_PID=$!

# Petit délai pour laisser le temps à /dev/video1 de se créer
sleep 2

# 4. Lancement de Natacha (au premier plan)
echo "Lancement de Natacha..."
conda run -n oeil_natacha_v2 python3 natacha_final13.py

# 5. Nettoyage automatique à la fermeture (quand tu feras Ctrl+C)
# On tue le pipeline GStreamer en utilisant le PID qu'on a stocké
echo "Arrêt de Natacha et du pipeline..."
kill $GST_PID
exit
