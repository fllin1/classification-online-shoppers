import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
import numpy as np


def fetch_data() -> pd.DataFrame:
    """
    Fetches the Online Shoppers Purchasing Intention dataset from the UCIML repository.

    Returns:
        pd.DataFrame: A tuple containing the feature matrix (X), target vector (y), 
                      and the variables/metadata associated with the dataset.
    """
    # Import the dataset using fetch_ucirepo function from the UCIML repository
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 

    variables = online_shoppers_purchasing_intention_dataset.variables 
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets
    
    return X, y, variables


def data_preprocessing(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This function preprocesses the dataset by encoding categorical variables and splitting the dataset into training and testing sets
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the preprocessed feature matrix (X) and target vector (y)
    """
    data = pd.concat([X, y], axis=1)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Convert the categorical variables to the appropriate data type
    data['Weekend'] = data['Weekend'].astype(int)
    data['Revenue'] = data['Revenue'].astype(int)
    
    # One-hot encode the categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Standardize the numerical features
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # Split the dataset into training and testing sets
    X = data.drop('Revenue', axis=1)
    y = data['Revenue']

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, 
        y_resampled, 
        test_size=0.3, # 30% of the data is used for testing
        random_state=42,
        stratify=y_resampled # Stratification 
        )
    
    return X_train, X_test, y_train, y_test
