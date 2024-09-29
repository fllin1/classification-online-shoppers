import pandas as pd
from sklearn.metrics import classification_report


def classification_report(predictions, y_test, y_pred) -> None:
    """
    This function returns the classification report for each model
    
    Args:
        predictions (dict): A dictionary containing the model name and its predictions
        y_test (pd.Series): The true target vector
        y_pred (pd.Series): The predicted target vector
        
    Returns:
        DataFrame: A DataFrame containing the classification report for each model
    """
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

    return pd.DataFrame(results)