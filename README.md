# Dashboard : Augmentation de Données et Super Résolution

Ce projet est une application interactive construite avec **Streamlit**, permettant :

- D'explorer et visualiser des techniques d'augmentation de données appliquées aux images.
- D'appliquer la **super résolution** d'images en utilisant l'algorithme **BSRGAN**.

## Fonctionnalités

### 1. **Augmentation de Données**

Permet de générer plusieurs versions augmentées d'une image en utilisant différentes transformations :

- **Flip Horizontal** : Retourne l'image horizontalement.
- **Rotation Aléatoire** : Effectue une rotation dans une plage aléatoire de ±45 degrés.
- **Zoom Out** : Réduit la taille de l'image tout en ajoutant des bordures.
- **Perspective** : Applique une déformation perspective simulant une inclinaison.

L'utilisateur peut :

- Téléverser une image.
- Sélectionner le nombre d'exemples augmentés à générer.
- Visualiser les résultats directement sur le dashboard.

---

### 2. **Super Résolution**

Utilise le modèle **BSRGAN** pour améliorer la résolution des images téléchargées :

- Facteur de mise à l'échelle fixe à **4x**.
- Téléchargement de l'image super-résolue au format PNG.

---

## Installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/akdavid/Developpez-une-preuve-de-concept.git
   cd Developpez-une-preuve-de-concept
   ```

2. Configurez l'environnement virtuel avec les dépendances requises. Par exemple, avec conda :

   ```bash
   conda create -n dashboard_sr python=3.9 -y
   conda activate dashboard_sr
   pip install -r requirements.txt
   ```

3. Assurez-vous que le modèle **BSRGAN** est disponible dans le chemin suivant :
   ```
   BSRGAN/model_zoo/BSRGAN.pth
   ```
   > Vous pouvez télécharger le modèle depuis [le dépôt officiel BSRGAN](https://github.com/cszn/BSRGAN).

---

## Lancer l'application

1. Activez l'environnement virtuel :

   ```bash
   conda activate dashboard_sr
   ```

2. Lancez l'application Streamlit :

   ```bash
   streamlit run dashboard.py
   ```

3. Accédez au dashboard via l'URL générée, par exemple :
   ```
   http://localhost:8501
   ```

---

## Utilisation

### Augmentation de Données

1. Téléversez une image au format JPG/PNG/JPEG.
2. Définissez le nombre d'exemples générés via le **slider**.
3. Visualisez les résultats directement dans le dashboard.

### Super Résolution

1. Téléversez une image au format JPG/PNG/JPEG.
2. Le modèle **BSRGAN** applique automatiquement un facteur d'échelle de **4x**.
3. Téléchargez l'image améliorée via le bouton de téléchargement.

---

## Dépendances principales

- **Streamlit** : Interface interactive et visualisation.
- **PyTorch** : Framework pour le modèle BSRGAN.
- **BSRGAN** : Modèle de super résolution basé sur RRDBNet.
- **TorchVision** : Transformations d'images pour l'augmentation de données.
- **Pillow** : Manipulation d'images.
