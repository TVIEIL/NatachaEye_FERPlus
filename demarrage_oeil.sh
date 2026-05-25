#!/bin/bash
# ==============================================================================
# @MeshTag       : Natacha_Oeil
# @Project       : NatachaEye_FERPlus
# @Version       : 1.0.1
# @Author        : Thierry VIEIL
# @Licence       : Apache License 2.0
# @Description   : Pipeline GStreamer et lancement du module de vision
# ==============================================================================

# 1. Se placer dynamiquement là où est le script (indépendant du chemin complet)
cd "$(dirname "$0")"

# Fonction de nettoyage appelée lors d'un Ctrl+C
cleanup() {
    echo -e "\nArrêt propre demandé..."
    kill $GST_PID 2>/dev/null
    sudo pkill -9 python3
    exit
}

# Associe le Ctrl+C (SIGINT) et l'arrêt (SIGTERM) à la fonction cleanup
trap cleanup SIGINT SIGTERM

# 2. Nettoyage initial
sudo pkill -9 gst-launch-1.0
sudo pkill -9 python3
sleep 1

# 3. Chargement du pont virtuel
sudo modprobe v4l2loopback devices=1 video_nr=1 card_label="NatachaCam"
sudo chmod 666 /dev/video1

# 4. Lancement du pipeline en background
echo "Lancement du pipeline vers /dev/video1..."
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1' ! nvvidconv ! 'video/x-raw, format=BGRx' ! videoconvert ! 'video/x-raw, format=YUY2' ! v4l2sink device=/dev/video1 sync=false > /dev/null 2>&1 &

GST_PID=$!
sleep 2

# 5. Lancement de Natacha
echo "Lancement de Natacha..."
conda run -n oeil_natacha_v2 python3 natacha_final13.py

# Le script attend ici grâce au trap, il ne se fermera qu'avec Ctrl+C
