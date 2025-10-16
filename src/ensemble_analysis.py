import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import xgboost
from pathlib import Path
from typing import Dict, Tuple

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
        data_dict["credit"] = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Now load digits
    digits_X, digits_y = load_digits(return_X_y=True)
    data_dict["digits"] = train_test_split(digits_X, digits_y, test_size=0.2, random_state=42)
    
    return data_dict 
    