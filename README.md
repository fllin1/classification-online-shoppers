# 🔮 Prédiction des intentions d'achat en ligne

Ce projet est basé sur la base de données [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

- [🔮 Prédiction des intentions d'achat en ligne](#-prédiction-des-intentions-dachat-en-ligne)
  - [📖 Description](#-description)
  - [📋 Étapes du projet :](#-étapes-du-projet-)
    - [Exploration des données :](#exploration-des-données-)
    - [Prétraitement des données :](#prétraitement-des-données-)
    - [Modélisation :](#modélisation-)
    - [Évaluation des modèles :](#évaluation-des-modèles-)
    - [Optimisation et validation :](#optimisation-et-validation-)
    - [Analyse des résultats :](#analyse-des-résultats-)
  - [⚙️ Installation](#️-installation)
    - [Prérequis :](#prérequis-)
    - [Installation des dépendances :](#installation-des-dépendances-)
  - [🎰 Utilisation](#-utilisation)
  - [🌇 Structure du projet](#-structure-du-projet)


## 📖 Description
Ce projet a pour objectif d'analyser le dataset "Online Shoppers Purchasing Intention" disponible sur l'UCI Machine Learning Repository. L'objectif principal est de développer un modèle de classification capable de prédire si une session en ligne aboutira à un achat (Revenue = True) ou non (Revenue = False).

## 📋 Étapes du projet :

### Exploration des données :

Analyse descriptive des variables du dataset.
Visualisations pour comprendre les distributions et les relations entre les variables.

### Prétraitement des données :

1. Encodage des variables catégorielles.
2. Normalisation ou standardisation des variables numériques si nécessaire.
3. Gestion du déséquilibre des classes.
4. Division des données en ensembles d'entraînement et de test.

### Modélisation :

- Implémentation de différents algorithmes de classification, tels que :
 - Régression Logistique
 - Arbres de décision
 - Forêts aléatoires
 - XGBoost
 - Réseaux de neurones avec PyTorch
*Utilisation de techniques pour gérer le déséquilibre des classes, comme le sur-échantillonnage avec SMOTE ou l'ajustement des poids de classe.*

### Évaluation des modèles :

- Utilisation de métriques adaptées pour les données déséquilibrées :
 - Précision
 - Rappel
 - F1-Score
 - Matrice de Confusion
 - ROC-AUC

*Finir les deux derniers points*

### Optimisation et validation :

- Ajustement des hyperparamètres avec des techniques comme la Grid Search.
- Validation croisée pour assurer la robustesse du modèle.

### Analyse des résultats :

- Identification des variables les plus influentes dans la prédiction.
- Discussion sur les performances des différents modèles.
- Réflexion sur les limitations rencontrées et les améliorations possibles.

## ⚙️ Installation

### Prérequis :
Ce projet a été réalisé sur `Python 3.12`

### Installation des dépendances :

```bash
pip install -r requirements.txt
```

## 🎰 Utilisation

1. Téléchargement du dataset :

Téléchargez le dataset [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) et placez-le dans le répertoire data/ du projet.

2. Exécution du script principal :

```bash
python main.py
```

3. Résultats :

Les résultats, y compris les visualisations et les métriques de performance, seront générés dans le répertoire results/.

## 🌇 Structure du projet
- `data/` : Contient le dataset utilisé pour le projet.
- `notebooks/` : Contient les notebooks Jupyter pour l'exploration et les essais.
- `scripts/` : Contient les scripts Python pour le prétraitement, la modélisation et l'évaluation.
- `results/` : Contient les sorties du modèle, les graphiques et les rapports.
- `requirements.txt` : Liste des dépendances du projet.
- `README.md` : Bonne lecture.