import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             make_scorer, roc_auc_score, roc_curve)


def numerical_features(X: pd.DataFrame) -> None:
    """
    This function returns the numerical features in the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        
    Returns:
        plot: A plot of the numerical features in the dataset
    """
    # Select the numeric features from the dataset
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Determine the number of rows and columns for subplots
    n = len(numeric_features)
    cols = 4  # Number of columns for the grid layout
    rows = n // cols + (n % cols > 0)  # Number of rows, accounting for remaining features

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))  # Define the figure size
    axes = axes.flatten()  # Flatten the axes array to easily iterate through

    # Plot each numeric feature's distribution
    for i, feature in enumerate(numeric_features):
        sns.histplot(X[feature], kde=True, ax=axes[i])  # Plot histogram and KDE for each feature
        axes[i].set_title(f'Distribution of {feature}')  # Set title for each subplot

    # Remove unused subplots if there are any extra axes
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    # Adjust the layout for better readability
    plt.tight_layout()
    plt.savefig('results/numerical_features.png', dpi=300, bbox_inches='tight')
     

def categorical_features(X: pd.DataFrame) -> None:
    """
    This function returns the categorical features in the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        
    Returns:
        plot: A plot of the categorical features in the dataset
    """
    # Select the categorical features from the dataset
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Determine the number of rows and columns for subplots
    n = len(categorical_features)
    cols = 4  # Number of columns for the grid layout
    rows = n // cols + (n % cols > 0)  # Number of rows, accounting for remaining features

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))  # Define the figure size
    axes = axes.flatten()  # Flatten the axes array to easily iterate through

    # Plot each categorical feature's distribution
    for i, feature in enumerate(categorical_features):
        sns.countplot(y=feature, data=X, ax=axes[i])  # Plot countplot for each feature
        axes[i].set_title(f'Count of {feature}')  # Set title for each subplot

    # Remove unused subplots if there are any extra axes
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('results/numerical_features.png', dpi=300, bbox_inches='tight')
    

def correlation_matrix (X: pd.DataFrame) -> None:
    """
    This function returns the correlation matrix of the dataset
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        
    Returns:
        plot: A plot of the correlation matrix of the dataset
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create a figure with a specified size
    plt.figure(figsize=(12, 10))

    # Compute the correlation matrix for the numeric features
    correlation = X[numeric_features].corr()

    # Create a heatmap to visualize the correlation matrix
    sns.heatmap(correlation, annot=True, cmap='coolwarm')  # 'coolwarm' colormap for better visualization

    plt.title('Correlation Matrix')
    plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    

def filter_percentiles(df, feature, lower_percentile=0.10, upper_percentile=0.90) -> pd.DataFrame:
    """
    This function filters the dataframe based on the lower and upper percentiles of a feature
    
    Args:
        df (pd.DataFrame): The dataframe to filter
        feature (str): The feature to filter
        lower_percentile (float): The lower percentile to filter the feature
        upper_percentile (float): The upper percentile to filter the feature
        
    Returns:
        pd.DataFrame: The filtered dataframe based on the lower and upper percentiles
    """
    lower_bound = df[feature].quantile(lower_percentile)
    upper_bound = df[feature].quantile(upper_percentile)
    # Return the filtered dataframe based on the lower and upper percentiles
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]


def features_against_target (X: pd.DataFrame, y: pd.Series) -> None:
    """
    This function returns the relationship between features and the target variable
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
        
    Returns:
        plot: A plot of the relationship between features and the target variable
    """
    data = pd.concat([X, y], axis=1)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    plt.figure(figsize=(16, 12))

    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(4, 4, i) 
        # Filter the data based on the 10th and 90th percentiles of the feature
        data_filtered = filter_percentiles(data, feature)
        sns.violinplot(x='Revenue', y=feature, data=data_filtered, split=True) 
        plt.title(f'{feature} vs Revenue (80% values)')  

    plt.tight_layout()
    plt.savefig('results/numerical_features.png', dpi=300, bbox_inches='tight')
     

def data_visualisation(X: pd.DataFrame, y: pd.Series) -> None:
    """
    This function visualizes the dataset using various plots
    
    Args:
        X (pd.DataFrame): The feature matrix of the dataset
        y (pd.Series): The target vector of the dataset
        
    Returns:
        None
    """
    # Visualize the numerical features in the dataset
    numerical_features(X)
    
    # Visualize the categorical features in the dataset
    categorical_features(X)
    
    # Visualize the correlation matrix of the dataset
    correlation_matrix(X)
    
    # Visualize the relationship between features and the target variable
    features_against_target(X, y)
    
    # Explains the plots
    print("The first plot shows the distribution of numerical features in the dataset.")
    print("The second plot shows the count of categorical features in the dataset.")
    print("The third plot shows the correlation matrix of the dataset.")
    print("The fourth plot shows the relationship between features and the target variable.")
    print("All plots are saved in the results folder.")
    
    
def class_report(predictions, y_test) -> None:
    """
    This function visualizes the classification report for different models
    
    Args:
        df (pd.DataFrame): The classification report dataframe
        
    Returns:
        plot: A plot of the classification report for different models
    """
    # List to store results for each model
    results = []

    # Loop through each model and its predictions
    for model_name, y_pred in predictions.items():
        # Get the classification report as a dictionary
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Extract precision, recall, and f1-score for both classes (0 and 1)
        precision_0 = report['0']['precision']
        precision_1 = report['1']['precision']
        recall_0 = report['0']['recall']
        recall_1 = report['1']['recall']
        f1_0 = report['0']['f1-score']
        f1_1 = report['1']['f1-score']

        # Append the results for each model to the results list
        results.append({
            'Model': model_name,
            'Precision False': precision_0,
            'Precision True': precision_1,
            'Recall False': recall_0,
            'Recall True': recall_1,
            'F1-Score False': f1_0,
            'F1-Score True': f1_1
        })
        
    df = pd.DataFrame(results)
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics = ['Precision False', 'Recall False', 'F1-Score False', 'Precision True', 'Recall True', 'F1-Score True']
    titles = ['Precision False', 'Recall False', 'F1-Score False', 'Precision True', 'Recall True', 'F1-Score True']

    for i, metric in enumerate(metrics):
        row, col = divmod(i, 3)  
        sns.barplot(x='Model', y=metric, data=df, ax=axes[row, col], palette='viridis', hue='Model')
        axes[row, col].tick_params(axis='x', rotation=30)
        axes[row, col].set_title(titles[i])
        axes[row, col].set_ylabel('Score')
        axes[row, col].set_xlabel('Model')
        axes[row, col].set_ylim((0.8, 1))

    plt.tight_layout()
    plt.savefig('results/classification_report.png', dpi=300, bbox_inches='tight')
      

def confusion_matrix_plot(predictions, y_test, y_pred) -> None:
    """
    This function visualizes the confusion matrix for different models
    
    Args:
        y_test (pd.Series): The true target vector
        y_pred (pd.Series): The predicted target vector
        
    Returns:
        plot: A plot of the confusion matrix for different models
    """
    # Create a figure with a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loop through each model and its predictions to plot the confusion matrices
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i // 3, i % 3]  # Determine the correct subplot location
        cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', ax=ax)  # Plot the confusion matrix heatmap
        ax.set_title(f'Confusion Matrix - {model_name}')  # Set title
        ax.set_xlabel('Predictions')  # Set x-axis label
        ax.set_ylabel('True Labels')  # Set y-axis label

    # Remove the unused last subplot (bottom right)
    axes[1][2].axis('off')

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
     

def roc_curve_plot(y_test, probabilities) -> None:
    """
    This function visualizes the ROC curve for different models
    
    Args:
        y_test (pd.Series): The true target vector
        probabilities (dict): A dictionary containing the model name and its predicted probabilities
    
    Returns:
        plot: A plot of the ROC curve for different models
    """
    # Create a figure for plotting the ROC curves
    plt.figure(figsize=(10, 8))

    # Loop through each model and its probabilities to plot the ROC curve
    for model_name, probs in probabilities.items():
        fpr, tpr, thresholds = roc_curve(y_test, probs)  # Calculate FPR and TPR
        roc_auc = roc_auc_score(y_test, probs)  # Calculate the AUC score
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')  # Plot the ROC curve

    # Plot the random guessing line
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for Multiple Models')

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
     
    
def results_visualisation(predictions, y_test, probabilities) -> None:
    """
    This function visualizes the results of the classification models
    
    Args:
        predictions (dict): A dictionary containing the model name and its predictions
        y_test (pd.Series): The true target vector
        probabilities (dict): A dictionary containing the model name and its predicted probabilities
        
    Returns:
        None
    """

    # Visualize the classification report for different models
    class_report(predictions, y_test)
    
    for y_pred in predictions.values():
        # Visualize the confusion matrix for different models
        confusion_matrix_plot(predictions, y_test, y_pred)
    
    # Visualize the ROC curve for different models
    roc_curve_plot(y_test, probabilities)
    
    # Explains the plots
    print("The first plot shows the classification report for different models.")
    print("The second plot shows the confusion matrix for different models.")
    print("The third plot shows the ROC curve for different models.")
    print("All plots are saved in the results folder.")