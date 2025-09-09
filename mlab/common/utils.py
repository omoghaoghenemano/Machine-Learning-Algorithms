import numpy as np
import pandas as pd

def oversample_minority(X, y, random_state=42):
    np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    Xy_balanced = []
    for label in unique:
        X_class = X[y == label]
        n_samples = X_class.shape[0]
        if n_samples < max_count:
            idxs = np.random.choice(n_samples, size=max_count, replace=True)
            X_upsampled = X_class.iloc[idxs]
        else:
            X_upsampled = X_class
        y_upsampled = np.full((max_count,), label)
        Xy_balanced.append((X_upsampled, y_upsampled))
    
    X_bal = pd.concat([x for x, _ in Xy_balanced], axis=0).reset_index(drop=True)
    y_bal = np.concatenate([y for _, y in Xy_balanced])
    y_bal = pd.DataFrame(y_bal, columns=['domain'])
    return X_bal, y_bal
