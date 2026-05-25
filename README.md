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
