import numpy as np
import sys
from typing import Self
from scipy import stats

class DecisionTreeNode:
    
    def __init__(self, feature_index:int, threshold:float, left: Self=None, right: Self=None, value: int=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Always None unless leaf node

class DecisionTreeClassifier:
    
    def __init__(self, max_depth:int=sys.maxsize, split_criterion:str='gini_index'):
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        
    def fit(self, X:np.array, y:np.array):
        """Method to build an underlying decision tree around supplied data

        Args:
            X (np.array): inputs
            y (np.array): corresponding outputs
        """
        self.attributes = X.shape[1] # number of features of each observation
        self.root = self._build_tree(X, y, 0)
        
    def _build_tree(self, X_subset: np.array, y_subset: np.array, current_depth: int) -> DecisionTreeNode:
        """Recursive helper method to build a decision tree from a given set of observations with their classes

        Args:
            X_subset (np.array): input observations
            y_subset (np.array): corresponding classes for observations
            current_depth (int): current depth of the tree

        Returns:
            DecisionTreeNode: resulting tree node
        """
        # Start with the stopping conditions
        if len(np.unique(y_subset))==1:
            # All observations are the same class
            return DecisionTreeNode(-1,-1,None,None,y_subset[0])
        elif current_depth > self.max_depth:
            # We're not going any farther - take majority
            return DecisionTreeNode(-1,-1,None,None,stats.mode(y_subset)[0])
        else:
            # Find the best split
            current_gini_score = self._calculate_gini_score(y_subset)
            next_gini_score, feature, feature_value = self._find_best_split(X_subset, y_subset)
            if feature == None or feature_value == None or next_gini_score >= current_gini_score:
                # No more helpful splits
                return DecisionTreeNode(-1,-1,None,None,stats.mode(y_subset)[0])
            else:
                # No longer base case
                X_left = X_subset[X_subset[:,feature] < feature_value]
                X_right = X_subset[X_subset[:,feature] >= feature_value]
                y_left = y_subset[X_subset[:,feature] < feature_value]
                y_right = y_subset[X_subset[:,feature] >= feature_value]
                left = self._build_tree(X_left, y_left, current_depth+1)
                right = self._build_tree(X_right, y_right, current_depth+1)
                return DecisionTreeNode(feature, feature_value, left, right)
    
    def _find_best_split(self, X_subset: np.array, y_subset: np.array) -> tuple[float, int, float]:
        """Determine the best attribute to split on going by the classifier's split criterion and the input observation subset paired with their classes

        Args:
            X_subset (np.array): input observations
            y_subset (np.array): classes of observations

        Returns:
            Tuple[float, int, float]: resulting gini score, index of best attribute to split on, and the best value to split it at
        """
        # Determine the best feature of X to split on
        if self.split_criterion == 'gini_index':
            best_gini_score = float('inf')
            best_attribute_to_split_on = None
            best_split_threshold = None
            for feature in range(X_subset.shape[1]):
                feature_values = np.unique(X_subset[:,feature])
                for v_idx in range(feature_values.shape[0]):
                    feature_value = feature_values[v_idx]
                    # Partition X_subset and y_subset into the values less than and greater than this feature value
                    y_left = y_subset[X_subset[:,feature] < feature_value]
                    y_right = y_subset[X_subset[:,feature] >= feature_value]
                    if len(y_left) == 0 or len(y_right) == 0:
                        gini_score = float('inf')
                    else:
                        gini_score = len(y_left)/len(y_subset)*self._calculate_gini_score(y_left) + \
                            len(y_right)/len(y_subset)*self._calculate_gini_score(y_right)
                    if gini_score < best_gini_score:
                        best_gini_score = gini_score
                        best_attribute_to_split_on = feature
                        best_split_threshold = feature_value
            return (best_gini_score, best_attribute_to_split_on, best_split_threshold)
        elif self.split_criterion == 'entropy':
            pass
        else:
            pass
        
    def _calculate_gini_score(self, y_subset:np.array) -> float:
        """Given a set of classes corresponding to observations, find the numeric gini index value

        Args:
            y_subset (np.array): classes for set of observations
            
        Returns:
            float: gini index value for said set of class observations
        """
        unique_classes = np.unique(y_subset)
        class_counts = np.zeros((len(unique_classes),),dtype=int)
        for i in range(len(unique_classes)):
            class_counts[i] = np.sum(y_subset==unique_classes[i])
        # For the calculations below - turn into squared probability
        class_probs_squared = np.zeros((len(unique_classes),),dtype=float)
        for i in range(len(unique_classes)):
            class_probs_squared[i] = (class_counts[i]/len(y_subset))**2 
        # Gini index formula - one minus the sum of each class probability squared
        return 1-np.sum(class_probs_squared)
    
    def predict(self, X:np.array) -> np.array:
        """Method to predict classes for the given input observations

        Args:
            X (np.array): input observations

        Returns:
            np.array: class predictions for input observations
        """
        pass