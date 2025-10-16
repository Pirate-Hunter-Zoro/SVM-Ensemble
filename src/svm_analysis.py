from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from pathlib import Path
import json


def load_data() -> Dict[str, Tuple[np.array, np.array, np.array, np.array]]:
    """Function to load all of the training and testing data relevant to testing the SVM algorithm

    Returns:
        Dict[str, Tuple[np.array, np.array, np.array, np.array]]: training input, training output, testing input, and testing output for all relevant data sets
    """
    data_set_paths = {
        "linear_separable": ((Path("data/linear_separable/test_inputs.csv"),Path("data/linear_separable/train_inputs.csv")),(Path("data/linear_separable/test_targets.csv"), Path("data/linear_separable/train_targets.csv"))),
        "linear_overlap": ((Path("data/linear_overlap/test_inputs.csv"),Path("data/linear_overlap/train_inputs.csv")),(Path("data/linear_overlap/test_targets.csv"), Path("data/linear_overlap/train_targets.csv"))),
        "ellipse": ((Path("data/ellipse/test_inputs.csv"),Path("data/ellipse/train_inputs.csv")),(Path("data/ellipse/test_targets.csv"), Path("data/ellipse/train_targets.csv"))),
    }
    spirals_path = Path("data/spirals.csv")
    
    data_sets = {
        "linear_separable": None,
        "linear_overlap": None,
        "ellipse": None,
        "spirals": None,
    }
    for dataset_label, paths in data_set_paths.items():
        # (test input, train input), (test target, train target) 
        with open(paths[0][0], 'r') as f:
            test_input = np.loadtxt(f, delimiter=',')
        with open(paths[0][1], 'r') as f:
            train_input = np.loadtxt(f, delimiter=',')
        with open(paths[1][0], 'r') as f:
            test_labels = np.loadtxt(f, delimiter=',')
        with open(paths[1][1], 'r') as f:
            train_labels = np.loadtxt(f, delimiter=',')
        data_sets[dataset_label] = (train_input, train_labels, test_input, test_labels)
    # We must split the spirals data into training and testing ourself
    with open(spirals_path, 'r') as f:
        all_spirals_data = np.loadtxt(f, delimiter=",", skiprows=1)
        input_features = all_spirals_data[:, [1,2]]
        labels = all_spirals_data[:, 3]
        X_train, X_test, y_train, y_test = train_test_split(input_features, labels, random_state=42, test_size=0.2)
        data_sets['spirals'] = (X_train, y_train, X_test, y_test)
     
    return data_sets

def plot_decision_boundary(clf: SVC, X: np.array, y: np.array, title_str: str, filename_str: str):
    """Plot and save the classification results for the given input and expected output values, including the original points and the model's decision boundary

    Args:
        clf (SVC): model that classifies
        X (np.array): input data
        y (np.array): expected outputs
        title_str (str): title for plot
        filename_str (str): file name for plot
    """
    min_x1 = np.min(X[:,0])
    max_x1 = np.max(X[:,0])
    h_x1 = (max_x1 - min_x1) / 200
    min_x2 = np.min(X[:,1])
    max_x2 = np.max(X[:,1])
    h_x2 = (max_x2 - min_x2) / 200
    x1_v, x2_v = np.meshgrid(np.arange(min_x1, max_x1, h_x1), np.arange(min_x2, max_x2, h_x2))
    # reshape the meshgrid into shape (N,2) where N is the number of coordinates that go into our meshgrid
    all_meshgrid = np.array([x1_v.flatten(),x2_v.flatten()]).T
    # For each (x1,x2) point combo in our mesh grid, we have a predicted class for that point
    mesh_predictions = clf.predict(all_meshgrid).reshape((len(x1_v),len(x2_v)))
    # plots decision boundary where probabilities become maximum for different classes when crossing
    plt.contourf(x1_v, x2_v, mesh_predictions, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    plt.title(title_str)
    plt.savefig(f"results/plots/{filename_str}")
    plt.close()

def run_svm_analysis(datasets: Dict[str, Tuple[np.array,np.array,np.array,np.array]]) -> Dict[str, float]:
    """Helper function to run the support vector machine analysis on the datasets and report their accuracies

    Args:
        datasets (Dict[str, Tuple[np.array,np.array,np.array,np.array]]): Each datasets train input, train output, test input, and test output values

    Returns:
        Dict[str, float]: Accuracies for each data set
    """
    accuracies = {}
    
    linear_sets = ['linear_separable', 'linear_overlap']
    for name in linear_sets:
        X_train, y_train, X_test, y_test = datasets[name]
        clf = SVC(kernel='linear', C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        plot_decision_boundary(clf, X_train, y_train, title_str=f"Decision Boundary for {name}", filename_str=f"{name}_decision_boundary.png")
        accuracies[name] = accuracy
        
    # We have a bunch of model configurations we need to test for the non linear data sets
    nonlinear_sets = ['ellipse', 'spirals']
    kernels = ['poly', 'rbf']
    C_values = [1.0, 1000.0]
    for name in nonlinear_sets:
        X_train, y_train, X_test, y_test = datasets[name]
        for C_value in C_values:
            for kernel in kernels:
                if kernel == 'poly':
                    clf = SVC(kernel=kernel, degree=2, C=C_value, random_state=42)
                else:
                    clf = SVC(kernel=kernel, C=C_value, random_state=42)
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                plot_decision_boundary(clf, X_train, y_train, title_str=f"Decision Boundary for {name}", filename_str=f"{name}_{C_value}_{kernel}_decision_boundary.png")
                accuracies[f"{name}_{C_value}_{kernel}"] = accuracy
    
    return accuracies


if __name__ == "__main__":
    data = load_data()
    accuracies = run_svm_analysis(data)
    with open(Path("results/svm_accuracies.json"), 'w') as f:
        json.dump(accuracies, f, indent=4)