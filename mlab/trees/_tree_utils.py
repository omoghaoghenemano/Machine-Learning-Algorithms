import pandas as pd
import numpy as np

def safe_indexing(input_feats, indices, axis=0):

    if isinstance(input_feats, pd.DataFrame):
        if axis == 0:
            return input_feats.iloc[indices].values
        elif axis == 1:
            return input_feats.iloc[:, indices].values
    elif isinstance(input_feats, pd.Series):
        return input_feats.iloc[indices].values
    elif isinstance(input_feats, np.ndarray):
        if axis == 0:
            return input_feats[indices]
        elif axis == 1:
            return input_feats[:, indices]
    else:
        raise TypeError(f"Input data must be a pandas DataFrame, Series, or a numpy ndarray. Got {type(input_feats)} instead.")

class Node:

    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None, max_features=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value
        self.max_features = max_features