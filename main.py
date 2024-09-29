# Data manipulation
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             make_scorer, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Scipy
from scipy.stats import ks_2samp

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# UCI ML Repo
from ucimlrepo import fetch_ucirepo

# Custom scripts
from scripts.data_preprocessing import data_preprocessing, fetch_data
from scripts.data_visualisation import data_visualisation, results_visualisation
from scripts.models import (logistic_regression, random_forest, 
                            decision_tree, xgboost, train_model)
from scripts.eval import classification_report
import os


def main() -> None:
    """
    This function runs the entire machine learning pipeline
    
    Args:
        None
    
    Returns:
        None
    """
    # Create the "results" directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Fetch the dataset
    X, y, variables = fetch_data()
    
    # Data visualisation
    data_visualisation(X, y)

    # Data preprocessing
    X_train, X_test, y_train, y_test = data_preprocessing(X, y)
    
    # Train NN model
    model = train_model(X_train, y_train)
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    # Create a TensorDataset
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # Create a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_preds = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            all_preds.extend(predicted.numpy()) 
    
    lr = logistic_regression(X_train, y_train)
    dt = decision_tree(X_train, y_train)
    rf = random_forest(X_train, y_train)
    xgb_model = xgboost(X_train, y_train)
    
    # Model evaluation
    predictions = {
        'Logistic Regression': lr.predict(X_test),  # Predictions from Logistic Regression
        'Decision Tree': dt.predict(X_test),        # Predictions from Decision Tree
        'Random Forest': rf.predict(X_test),        # Predictions from Random Forest
        'XGBoost': xgb_model.predict(X_test),       # Predictions from XGBoost
        'NN (PyTorch)': all_preds  # Predictions from the PyTorch Neural Network
    }
    
    # Convert the test set to a PyTorch tensor
    X_test_tensor = torch.FloatTensor(X_test.values.astype(float))

    # Perform inference with the neural network and calculate probabilities using the sigmoid function
    with torch.no_grad():
        logits = model(X_test_tensor)  # Get the raw logits from the model
        probs_pytorch = F.sigmoid(logits).numpy()[:, 1]  # Apply sigmoid and convert to probabilities for class 1

    # Dictionary to store the predicted probabilities for each model
    probabilities = {
        'Logistic Regression': lr.predict_proba(X_test)[:, 1],  # Probabilities for Logistic Regression
        'Decision Tree': dt.predict_proba(X_test)[:, 1],        # Probabilities for Decision Tree
        'Random Forest': rf.predict_proba(X_test)[:, 1],        # Probabilities for Random Forest
        'XGBoost': xgb_model.predict_proba(X_test)[:, 1],       # Probabilities for XGBoost
        'Neural Network (PyTorch)': probs_pytorch               # Probabilities from the PyTorch Neural Network
}
    
    # Results visualisation
    results_visualisation(predictions, y_test, probabilities)


if __name__ == '__main__':
    main()