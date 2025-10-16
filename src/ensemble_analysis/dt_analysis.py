import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import load_digits
from pathlib import Path
from typing import Dict, Tuple
import sys
import json

from .dt_hyperparameter_search import DecisionTreeClassifier


def load_ensemble_data() -> Dict[str, Tuple[np.array]]:
    """Helper function to load the training and testing data from the ensemble learning datasets

    Returns:
        Dict[str, Tuple[np.array]]: data set names mapped to their training and testing data
    """
    # Start with the provided German credit data set
    data_dict = {}
    with open(Path("data/credit.csv"), 'r') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        X = data[:,[i for i in range(len(data[0])-1)]]
        y = data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        data_dict["credit"] = (X_train, y_train, X_test, y_test)
    
    # Now load digits
    digits_X, digits_y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y)
    data_dict["digits"] = (X_train, y_train, X_test, y_test)
    
    return data_dict 
    
    
def run_ensemble_tests():
    tree_depths = [1, 5, sys.maxsize]
    
    data = load_ensemble_data()
    
    # Map data set to the best tree depth and corresponding accuracy
    results = {}
    for dataset_name, dataset in data.items():
        best_accuracy = float('-inf')
        best_depth = None
        X_train, y_train, X_test, y_test = dataset
        for depth in tree_depths:
            dt = DecisionTreeClassifier(max_depth=depth)
            num_splits = 5
            kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
            avg_accuracy = 0
            # Over all splits, determine the best accuracy
            for (train_indices, test_inidices) in kf.split(X_train):
                train_X_split, train_y_split = X_train[train_indices], y_train[train_indices]
                dt.fit(train_X_split, train_y_split)
                test_X_split, test_y_split = X_train[test_inidices], y_train[test_inidices]
                test_y_pred_split = dt.predict(test_X_split)
                avg_accuracy += np.sum(test_y_pred_split == test_y_split) / len(test_y_split)
            avg_accuracy /= num_splits
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_depth = depth
        
        # Now that we have discovered the best depth with KFold, train a new tree on it
        dt = DecisionTreeClassifier(max_depth=best_depth)
        dt.fit(X_train, y_train)
        y_preds = dt.predict(X_test)
        test_accuracy = np.sum(y_preds == y_test) / len(y_test)
        results[dataset_name] = (f"Best Depth: {best_depth}", f"Corresponding Test Accuracy: {test_accuracy}", f"Corresponding CV Accuracy: {best_accuracy}")
    
    return results


if __name__=="__main__":
    results = run_ensemble_tests()
    
    with open(Path("results/decision_tree_accuracies.json"), 'w') as f:
        json.dump(results, f, indent=4)