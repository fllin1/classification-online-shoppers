# 🔮 Prediction of Online Shopping Intentions

This project is based on the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

- [🔮 Prediction of Online Shopping Intentions](#-prediction-of-online-shopping-intentions)
- [🪐 Description](#-description)
- [⚙️ Installation](#️-installation)
  - [Prerequisites](#prerequisites)
  - [Dependency installation](#dependency-installation)
- [🗃️ Project Steps: Classification](#️-project-steps-classification)
  - [1. Data Exploration](#1-data-exploration)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Modeling](#3-modeling)
  - [4. Model Evaluation](#4-model-evaluation)
  - [5. Optimization and Validation](#5-optimization-and-validation)
  - [6. Results Analysis](#6-results-analysis)
- [🚀 Usage](#-usage)
- [🕋 Project Structure](#-project-structure)
- [🗝️ Keywords](#️-keywords)


# 🪐 Description

This project aims to solve a classic **Machine Learning** problem by involving techniques such as learning, classification, regression, and clustering.

We will first perform a **data exploration** (EDA) and then implement **classification** methods to predict a binary variable.

# ⚙️ Installation

## Prerequisites
This project was built with `Python 3.12`.

## Dependency installation

```bash
pip install -r requirements.txt
```

# 🗃️ Project Steps: Classification

To follow the classification steps, simply run the notebook `notebook/classification.ipynb`.

## 1. Data Exploration

- Descriptive analysis of the dataset variables.
- Visualizations to understand distributions and relationships between variables.

## 2. Data Preprocessing

- Encoding of categorical variables.
- Normalization or standardization of numerical variables if necessary.
- Handling class imbalance.
- Splitting the data into training and test sets.

## 3. Modeling

- Implementing various classification algorithms, such as:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - XGBoost
   - Neural Networks with PyTorch

Techniques to handle class imbalance, such as oversampling with SMOTE or adjusting class weights, will be used.

## 4. Model Evaluation

Using metrics tailored for imbalanced data:
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

## 5. Optimization and Validation

- Hyperparameter tuning using techniques such as Grid Search.
- Cross-validation to ensure the model's robustness.

## 6. Results Analysis

- Identifying the most influential features in the prediction.
- Discussing the performance of different models.
  - Reflecting on limitations encountered and possible improvements.

# 🚀 Usage

1. Download the dataset:

Download the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) and place it in the project’s `data/` directory.

2. Run the main script to execute the model and generate results:

```bash

python main.py

```

3. Results:

The results, including visualizations and performance metrics, will be generated in the `results/` directory.

# 🕋 Project Structure
- `data/` : Contains the dataset used for the project.
- `notebooks/` : Contains Jupyter notebooks for exploration and experimentation.
- `scripts/` : Contains Python scripts for preprocessing, modeling, and evaluation.
- `results/` : Contains model outputs, graphs, and reports.
- `requirements.txt` : List of project dependencies.
- `README.md` : Documentation and project overview.

# 🗝️ Keywords

Imbalanced dataset, Feature selection, Online shopper's purchase intention, Real time prediction, Classification Methods