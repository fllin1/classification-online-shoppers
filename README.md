# 🔮 Prédiction des intentions d'achat en ligne

Ce projet est basé sur la base de données [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

- [🔮 Prédiction des intentions d'achat en ligne](#-prédiction-des-intentions-dachat-en-ligne)
  - [🪐 Description](#-description)
  - [⚙️ Installation](#️-installation)
    - [Prérequis](#prérequis)
    - [Installation des dépendances](#installation-des-dépendances)
  - [🗃️ Étapes du projet: Classification](#️-étapes-du-projet-classification)
    - [1. Exploration des données](#1-exploration-des-données)
    - [2. Prétraitement des données](#2-prétraitement-des-données)
    - [3. Modélisation](#3-modélisation)
    - [4. Évaluation des modèles](#4-évaluation-des-modèles)
    - [5. Optimisation et validation](#5-optimisation-et-validation)
    - [6. Analyse des résultats](#6-analyse-des-résultats)
  - [🚀 Utilisation](#-utilisation)
  - [🕋 Structure du projet](#-structure-du-projet)
  - [🗝️ Keywords](#️-keywords)


## 🪐 Description

Ce projet à pour but de résoudre un problème classique de **Machine Learning**, en faisant intervenir des techniques d'apprentissage, de classification, de régression ainsi que de clustering. 

Nous procéderons tout d'abord à une **exploration des données** (EDA) puis implémenterons des méthodes de **classification** pour la prédiction d'une variable binaire.

## ⚙️ Installation

### Prérequis
Ce projet a été réalisé sur `Python 3.12`

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 🗃️ Étapes du projet: Classification

Pour suivre les étapes de la classification, il suffit de lancer le notebook `notebook/classification.ipynb`.

### 1. Exploration des données

- Analyse descriptive des variables du dataset.
- Visualisations pour comprendre les distributions et les relations entre les variables.

### 2. Prétraitement des données

- Encodage des variables catégorielles.
- Normalisation ou standardisation des variables numériques si nécessaire.
- Gestion du déséquilibre des classes.
- Division des données en ensembles d'entraînement et de test.

### 3. Modélisation

- Implémentation de différents algorithmes de classification, tels que :
   - Régression Logistique
   - Arbres de décision
   - Forêts aléatoires
   - XGBoost
   - Réseaux de neurones avec PyTorch

*Utilisation de techniques pour gérer le déséquilibre des classes, comme le sur-échantillonnage avec SMOTE ou l'ajustement des poids de classe.*

### 4. Évaluation des modèles

Utilisation de métriques adaptées pour les données déséquilibrées :
- Précision
- Rappel
- F1-Score
- Matrice de Confusion
- ROC-AUC

### 5. Optimisation et validation

- Ajustement des hyperparamètres avec des techniques comme la Grid Search.
- Validation croisée pour assurer la robustesse du modèle.

### 6. Analyse des résultats

- Identification des variables les plus influentes dans la prédiction.
- Discussion sur les performances des différents modèles.
  - Réflexion sur les limitations rencontrées et les améliorations possibles.

## 🚀 Utilisation

1. Téléchargement du dataset :

Téléchargez le dataset [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) et placez-le dans le répertoire data/ du projet.

2. Exécution du script principal :

```bash
python main.py
```

3. Résultats :

Les résultats, y compris les visualisations et les métriques de performance, seront générés dans le répertoire results/.

## 🕋 Structure du projet
- `data/` : Contient le dataset utilisé pour le projet.
- `notebooks/` : Contient les notebooks Jupyter pour l'exploration et les essais.
- `scripts/` : Contient les scripts Python pour le prétraitement, la modélisation et l'évaluation.
- `results/` : Contient les sorties du modèle, les graphiques et les rapports.
- `requirements.txt` : Liste des dépendances du projet.
- `README.md` : Bonne lecture.

## 🗝️ Keywords

Imbalanced dataset, Feature selection, Online shopper's purchase intention, Real time prediction, Classification Methods