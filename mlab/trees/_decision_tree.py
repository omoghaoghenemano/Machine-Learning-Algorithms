from collections import Counter
from typing import Optional
import numpy as np
from ..base import BaseClassifier 
from ._tree_utils import Node


class DecisionTreeClassifier(BaseClassifier):
    """
    Class for decision tree using the ID3 algorithm.
    """

    def __init__(self, criterion: str = 'entropy', random_state: Optional[int] = None, min_samples_split=2,
                 max_depth=2, min_samples_leaf=1, max_features=None):
        self.criterion = criterion
        self.rng_ = np.random.default_rng(random_state)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.max_features = max_features

    @staticmethod
    def _entropy(s):
        # Fast, vectorized entropy calculation
        if len(s) == 0:
            return 0.0
        _, counts = np.unique(s, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    def _information_gain(self, parent, left_child, right_child):
        """
        Helper function, calculates information gain from a parent and two child nodes.

        :param parent: list, the parent node
        :param left_child: list, left child of a parent
        :param right_child: list, right child of a parent
        :return: float, information gain
        """
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))

    def _best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        _, n_cols = X.shape
        # Efficient random feature selection
        if hasattr(self, 'max_features') and self.max_features is not None:
            if isinstance(self.max_features, float) and 0 < self.max_features < 1:
                n_feats = max(1, int(n_cols * self.max_features))
            elif isinstance(self.max_features, int):
                n_feats = min(self.max_features, n_cols)
            else:
                n_feats = n_cols
            feature_indices = self.rng_.choice(n_cols, n_feats, replace=False)
        else:
            feature_indices = np.arange(n_cols)
        for attribute_idx in feature_indices:
            col = X[:, attribute_idx]
            # Sort feature and target together for efficient split search
            sorted_idx = np.argsort(col)
            sorted_col = col[sorted_idx]
            sorted_y = y[sorted_idx]
            # Only consider splits between unique values where class label changes
            possible_split = np.nonzero(sorted_y[:-1] != sorted_y[1:])[0]
            thresholds = (sorted_col[possible_split] + sorted_col[possible_split + 1]) / 2
            for _, threshold in zip(possible_split, thresholds):
                left_mask = sorted_col <= threshold
                right_mask = sorted_col > threshold
                y_left = sorted_y[left_mask]
                y_right = sorted_y[right_mask]
                if len(y_left) > 0 and len(y_right) > 0:
                    gain = self._information_gain(sorted_y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': attribute_idx,
                            'threshold': threshold,
                            'df_left': X[X[:, attribute_idx] <= threshold],
                            'df_right': X[X[:, attribute_idx] > threshold],
                            'y_left': y[X[:, attribute_idx] <= threshold],
                            'y_right': y[X[:, attribute_idx] > threshold],
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split

    def _build(self, X, y, depth=0):
        """
        Helper recursive function, used to build a decision tree from the input data.

        :param X: np array, features
        :param y: np array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        """
        n_rows, _ = X.shape
        # Special handling for minimal datasets
        unique_classes = np.unique(y)
        if n_rows <= 2 or len(unique_classes) == n_rows:
            # If only one sample per class, make a leaf for each
            return Node(value=Counter(y).most_common(1)[0][0])
        # Allow unlimited depth if max_depth is None
        if n_rows >= self.min_samples_split and (self.max_depth is None or depth <= self.max_depth):
            best = self._best_split(X, y)
            if 'gain' in best and best['gain'] > 0:
                left = self._build(best['df_left'], best['y_left'], depth + 1)
                right = self._build(best['df_right'], best['y_right'], depth + 1)
                return Node(
                    feature=best['feature_index'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
        return Node(value=Counter(y).most_common(1)[0][0])

    def fit(self, X, y):
        self.root = self._build(np.array(X), np.array(y))

    def _predict(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature]
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)
        return self._predict(x=x, tree=tree.data_right)

    def predict(self, X):
        return [self._predict(x, self.root) for x in np.array(X)]
