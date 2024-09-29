import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def logistic_regression(X, y) -> LogisticRegression:
    """
    This function fits a Logistic Regression model to the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
        
    Returns:
        LogisticRegression: The Logistic Regression model
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model


def random_forest(X, y) -> RandomForestClassifier:
    """
    This function fits a Random Forest model to the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
    
    Returns:
        RandomForestClassifier: The Random Forest model
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


def decision_tree(X, y) -> DecisionTreeClassifier:
    """
    This function fits a Decision Tree model to the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
    
    Returns:
        DecisionTreeClassifier: The Decision Tree model
    """
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


def xgboost(X, y) -> xgb.XGBClassifier:
    """
    This function fits an XGBoost model to the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
        
    Returns:
        xgb.XGBClassifier: The XGBoost model
    """
    model = xgb.XGBClassifier()
    model.fit(X, y)
    return model


class NeuralNet(nn.Module):
    """
    This class defines a simple neural network model using PyTorch
    
    Args:
        nn.Module: The base class for all neural network models in PyTorch
    """
    # Define the layers and activation functions in the constructor
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64) # Input layer with 64 neurons
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2) # Output layer with 2 classes
        self.relu = nn.ReLU()
    
    # Define the forward pass
    def forward(self, x):
        out = self.relu(self.layer1(x)) # ReLU activation
        out = self.relu(self.layer2(out)) # ReLU activation
        out = self.output(out)
        return out
    

def train_model(X_train, y_train, num_epochs = 20) -> None:
    """
    This function trains a simple neural network model using PyTorch
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
        num_epochs (int): The number of epochs for training the model
        
    Returns:
        predictions (list): A list of predicted classes for the test set
    """
    # Convert the variables to a tensor
    X_train_tensor = torch.tensor(X_train.values.astype(float), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    # Create a TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = NeuralNet(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate the loss
            
            optimizer.zero_grad() # Zero the gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights
            
    return model