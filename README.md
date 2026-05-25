# NatachaEye_FERPlus

**NatachaEye_FERPlus** est une extension visuelle dédiée au projet **Natacha**, l'assistant multimodal. Ce module est conçu pour fournir à l'intelligence centrale une perception en temps réel de son interlocuteur (présence, émotions, style vestimentaire) via un pipeline optimisé pour les plateformes NVIDIA Jetson.

---

## 🛠️ Hardware
Ce système est optimisé pour les spécifications suivantes :

* **Carte :** NVIDIA Jetson NX SUPER (8 Go de RAM).
* **Caméra :** Module caméra IMX219 (interface MIPI CSI).
* **Pipeline :** Utilisation de l'accélération matérielle NVIDIA (GStreamer / `nvarguscamerasrc`).

---

## 🧠 Intelligence Artificielle : FER vs FER+

Cette version marque une évolution majeure par rapport au projet initial (FER).

### Historique
* **FER (Facial Expression Recognition) :** Basé sur le dataset historique Kaggle. Bien qu'efficace, il souffrait de labels bruités et imprécis, rendant la classification parfois erratique sur des expressions ambiguës.
* **FER+ :** Une version raffinée et améliorée du dataset FER.

### Pourquoi FER+ est supérieur ?
* **Labels de qualité :** Les images ont été re-labellisées par des humains avec une méthodologie rigoureuse, prenant en compte l'ambiguïté des expressions.
* **Robustesse :** Le modèle FER+ offre une bien meilleure généralisation et une précision accrue sur les expressions complexes, rendant les interactions de Natacha beaucoup plus fluides.
* **Optimisation :** Nous utilisons un modèle au format **ONNX**, permettant une inférence rapide et efficace sur le moteur DNN d'OpenCV.

---

## 🚀 Installation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/TVIEIL/NatachaEye_FERPlus.git](https://github.com/TVIEIL/NatachaEye_FERPlus.git)
   cd NatachaEye_FERPlus
   
2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt

3. **Configuration :**
   Copie le fichier .env.example en .env et adapte les paramètres MQTT :
   ```bash
   cp .env.example .env
   # Édite .env avec tes paramètres (MQTT_BROKER, etc.)
   ```

4. **Lancement :**
  ```bash
  ./demarrage_oeil.sh
  ```

## 🛠️ Debugging

Si Natacha ne "voit" rien ou ne communique pas :

1.**Vérifier le pipeline caméra**

Si aucune image n'est capturée, teste ton pipeline GStreamer seul :

```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1' ! nvvidconv ! videoconvert ! autovideosink
```

2.**Vérifier la connexion MQTT**

Assure-toi que ton broker est actif. Utilise ce listener pour vérifier le trafic en direct :

```bash
mosquitto_sub -h localhost -t "natacha/oeil_detection" -v
```

3.**Consulter les Logs**

Si le script se ferme brutalement, vérifie les erreurs :

```bash
conda run -n oeil_natacha_v2 python3 natacha_final13.py > natacha_logs.txt 2>&1
cat natacha_logs.txt
```
4.**Mode Headless**

Si tu as modifié le fichier .env pour AFFICHAGE_GRAPHIQUE=False, assure-toi que le script ne tente pas d'ouvrir de fenêtres OpenCV, ce qui provoquerait une erreur de segmentation.


Projet développé par Thierry VIEIL.
 
