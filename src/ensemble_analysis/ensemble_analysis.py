import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

class RandomForestClassifier:
    
    def __init__(self, n_estimators:int, max_depth:int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
    
    def fit(self, X: np.array, y: np.array):
        """Fit the model according to the training data and their classifications

        Args:
            X (np.array): input observations
            y (np.array): respective classes of observations
        """
        # At each split, sample the some features - how many? The square root of the total number of features
        num_features_to_sample = int(math.sqrt(X.shape[1]))
        for _ in range(self.n_estimators):
            # Sample with replacement from the features - make the sample size as large as the number of features
            observation_sample_indices = np.random.randint(0, X.shape[0], size=(X.shape[0]))
            sample_X = X[observation_sample_indices]
            sample_y = y[observation_sample_indices]
            dt = DecisionTreeClassifier(max_depth=self.max_depth, max_features=num_features_to_sample)
            dt.fit(sample_X, sample_y)
            self.estimators.append(dt)
    
    def predict(self, X: np.array) -> np.array:
        """Predict classes of the given inputs

        Args:
            X (np.array): inputs to predict

        Returns:
            np.array: respective predicted classes of inputs
        """
        predictions = np.array([dt.predict(X) for dt in self.estimators])
        return stats.mode(predictions, axis=0)[0]
    
    
class AdaboostClassifier:
    
    def __init__(self, n_estimators:int, max_depth:int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        self.alpha_weights = []
        
    def fit(self, X: np.array, y: np.array):
        """Fit the model according to the training data and their classifications

        Args:
            X (np.array): input observations
            y (np.array): respective classes of observations
        """
        weights = np.array([1/X.shape[0] for _ in range(X.shape[0])])
        self.unique_classes = np.unique(y)
        for _ in range(self.n_estimators):
            self.alpha_weights.append(np.copy(weights))
            dt = DecisionTreeClassifier(max_depth=self.max_depth)
            dt.fit(X, y, sample_weight=weights)
            self.estimators.append(dt)
            # Now see which observations were classified incorrectly
            y_hat = dt.predict(X)
            # Find sum of weights of misclassified observations
            weight_error = np.sum(weights[y_hat!=y])
            alpha = 0.5*np.log((1-weight_error)/weight_error)
            # TODO - Decrease weight of correctly classified samples ONLY
            
            # Normalize the weights
            weights /= np.sum(weights)
        self.alpha_weights = np.array(self.alpha_weights)
    
    def predict(self, X: np.array) -> np.array:
        """Predict classes of the given inputs

        Args:
            X (np.array): inputs to predict

        Returns:
            np.array: respective predicted classes of inputs
        """
        pass