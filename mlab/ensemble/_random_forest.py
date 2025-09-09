import numpy as np
from ..base import BaseClassifier
import pandas as pd
from ..trees._decision_tree import DecisionTreeClassifier
from ..trees._tree_utils import safe_indexing
class RandomForestClassifier(BaseClassifier):
    """
    RandomForestClassifier is an ensemble learning method that constructs multiple decision trees
    during training and outputs the mode of the classes as the prediction.
    """

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
        """
        Initialize the RandomForestClassifier.

        :param n_estimators: The number of trees in the forest.
        :param max_depth: The maximum depth of the tree.
        :param max_features: The number of features to consider when looking for the best split.
        :param random_state: Seed used by the random number generator.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        # Use numpy's recommended random generator
        self.random_state = np.random.default_rng(random_state)
        self.estimators_ = []

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the data.

        :param X: Feature matrix.
        :param y: Target values.
        :return: Bootstrap sample of input features and y.
        """
        n_samples = X.shape[0]
        indices = self.random_state.integers(0, n_samples, n_samples)
        if np.max(indices) >= n_samples or np.min(indices) < 0:
            raise ValueError(f"Generated indices out of bounds: {indices}")

        X_sample = safe_indexing(X, indices, axis=0)
        y_sample = safe_indexing(y, indices, axis=0)

        return X_sample, y_sample

    def _select_features(self, X):
        """
        Select a subset of features.

        :param X: Feature matrix.
        :return: Indices of selected features.
        """
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        features = self.random_state.choice(n_features, max_features, replace=False)
        if np.max(features) >= n_features or np.min(features) < 0:
            raise ValueError(f"Selected features out of bounds: {features}")

        return features

    def fit(self, X, y):
        """
        Fit the RandomForestClassifier model.

        :param X: Feature matrix.
        :param y: Target values.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(f"y must be a pandas Series or numpy ndarray, got {type(y)}")

        X = np.array(X)
        y = np.array(y)
        unique_classes = np.unique(y)
        if X.shape[0] <= 2 or len(unique_classes) == X.shape[0]:
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=1, min_samples_leaf=1, max_features=self.max_features)
            tree.fit(X, y)
            self.estimators_ = [(tree, np.arange(X.shape[1]))]
            return self
        self.estimators_ = []
        n_samples = X.shape[0]
        # Precompute all bootstrap indices for all trees at once
        all_indices = self.random_state.choice(n_samples, size=(self.n_estimators, n_samples), replace=True)
        for i in range(self.n_estimators):
            x_sample = safe_indexing(X, all_indices[i], axis=0)
            y_sample = safe_indexing(y, all_indices[i], axis=0)
            features = self._select_features(x_sample)
            selected_x_sample = safe_indexing(x_sample, features, axis=1)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=2, min_samples_leaf=1, max_features=self.max_features)
            tree.fit(selected_x_sample, y_sample)
            self.estimators_.append((tree, features))
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        :param X: Feature matrix.
        :return: Predicted class labels.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}")
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, (tree, features) in enumerate(self.estimators_):
            if np.max(features) >= X.shape[1] or np.min(features) < 0:
                raise ValueError(f"Prediction features out of bounds: {features}")

            selected_X = safe_indexing(X, features, axis=1)
            predictions[:, i] = tree.predict(selected_X)
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)

