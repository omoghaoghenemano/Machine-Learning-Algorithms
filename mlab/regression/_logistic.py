# pylint: disable=missing-docstring
import numpy as np

from mlab.base import BaseClassifier


class LogisticRegression:
    def __init__(self, max_iterations=100, penalty=None, alpha=0.01):
        """
        Parameters
        ----------
        max_iterations : int
            Number of Newton iterations.
        penalty : {'l1', 'l2', None}
            Type of regularization.
        alpha : float
            Regularization strength.
        """
        self.max_iterations = max_iterations
        self.penalty = penalty
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y = y.flatten()

        for _ in range(self.max_iterations):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # === Gradient ===
            grad = (1 / n_samples) * X.T @ (y_pred - y)

            # Apply L2 regularization to gradient
            if self.penalty == 'l2':
                grad += (self.alpha / n_samples) * self.weights
            elif self.penalty == 'l1':
                grad += (self.alpha / n_samples) * np.sign(self.weights)

            # === Hessian ===
            R = np.diag((y_pred * (1 - y_pred)))
            H = (1 / n_samples) * X.T @ R @ X

            if self.penalty == 'l2':
                H += (self.alpha / n_samples) * np.eye(n_features)  # regularize Hessian
            # L1 regularization doesn't affect Hessian in closed-form Newton step

            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)

            self.weights -= H_inv @ grad

            # === Bias update (no regularization) ===
            db = float(np.mean(y_pred - y))
            self.bias -= db

            # === Cost (cross-entropy + regularization) ===
            cost = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            if self.penalty == 'l2':
                cost += (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.penalty == 'l1':
                cost += (self.alpha / n_samples) * np.sum(np.abs(self.weights))
            self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    

class SGDLogisticRegression(BaseClassifier):
    """
    SGD Logistic Regression with optional L1/L2 regularization, mini-batch support,
    automatic class imbalance handling, and feature scaling.
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iterations=1000,
        batch_size=32,
        tol=1e-4,
        penalty=None,  # 'l1', 'l2', or None
        alpha=0.0,      # Regularization strength
        random_state=None,
        verbose=0,
        scale_data=True,
        class_weight="balanced",  # None or 'balanced' or dict
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.tol = tol
        self.penalty = penalty
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        self.scale_data = scale_data
        self.class_weight = class_weight

        self.weights = None
        self.bias = None
        self.cost_history = []
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _is_imbalanced(self, y, threshold=0.4):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            # For simplicity, handle only binary imbalance auto-detection here
            return False
        ratio = counts.min() / counts.max()
        return ratio < threshold


    def _scale(self, X, fit=False):
        if fit:
            self.scaler_mean_ = X.mean(axis=0)
            self.scaler_std_ = X.std(axis=0) + 1e-8
        return (X - self.scaler_mean_) / self.scaler_std_
    def _compute_class_weights(self, y):
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight[label] for label in y])
        
        # Auto-detect imbalance and apply weights if needed
        if self.class_weight == 'balanced' or (self.class_weight is None and self._is_imbalanced(y)):
            classes, counts = np.unique(y, return_counts=True)
            weights = np.ones_like(y, dtype=float)
            total = len(y)
            for c, count in zip(classes, counts):
                weights[y == c] = total / (2 * count)
            return weights

        # Otherwise, uniform weights
        return np.ones_like(y, dtype=float)

    def _apply_regularization(self, dw):
        if self.penalty == 'l2':
            return dw + self.alpha * self.weights
        elif self.penalty == 'l1':
            return dw + self.alpha * np.sign(self.weights)
        return dw  # No penalty

    def _check_multicollinearity(self, X):
        corr = np.corrcoef(X, rowvar=False)
        high_corr = np.abs(corr[np.triu_indices_from(corr, 1)]) > 0.95
        if np.any(high_corr):
            if self.penalty is None:
                if self.verbose:
                    print("Warning: Multicollinearity detected. Switching to L2 regularization.")
                self.penalty = 'l2'
                self.alpha = max(self.alpha, 0.01)

    def _compute_regularization_cost(self):
        if self.penalty == 'l2':
            return 0.5 * self.alpha * np.sum(self.weights ** 2)
        elif self.penalty == 'l1':
            return self.alpha * np.sum(np.abs(self.weights))
        return 0.0
    
    def _stratified_batches(self, X, y, sample_weights, batch_size, rng):
        """Yield stratified mini-batches if data is imbalanced."""
        classes = np.unique(y)
        idxs = [np.where(y == c)[0] for c in classes]
        n_samples = len(y)
        n_batches = int(np.ceil(n_samples / batch_size))
        per_class = [max(1, int(batch_size * len(i) / n_samples)) for i in idxs]

        for _ in range(n_batches):
            batch_idxs = []
            for i, c_idxs in enumerate(idxs):
                chosen = rng.choice(c_idxs, per_class[i], replace=len(c_idxs) < per_class[i])
                batch_idxs.extend(chosen)
            batch_idxs = rng.permutation(batch_idxs)[:batch_size]
            yield X[batch_idxs], y[batch_idxs], sample_weights[batch_idxs]

    def _fit_epoch(self, X, y, sample_weights, rng):
        """Run one epoch of SGD updates."""
        if self._is_imbalanced(y):
            batch_iter = self._stratified_batches(X, y, sample_weights, self.batch_size, rng)
        else:
            n_samples = X.shape[0]
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            weights_shuffled = sample_weights[indices]
            batch_iter = (
                (X_shuffled[start:end], y_shuffled[start:end], weights_shuffled[start:end])
                for start in range(0, n_samples, self.batch_size)
                for end in [start + self.batch_size]
            )
        for X_batch, y_batch, w_batch in batch_iter:
            self._sgd_update(X_batch, y_batch, w_batch)

    def _sgd_update(self, X_batch, y_batch, w_batch):
        """Perform a single SGD update step."""
        linear_model = X_batch @ self.weights + self.bias
        y_pred = self._sigmoid(linear_model)
        error = y_pred - y_batch
        dw = (X_batch.T @ (w_batch * error)) / X_batch.shape[0]
        dw = self._apply_regularization(dw)
        db = float(np.mean(w_batch * error))
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        epsilon = 1e-5

        if self.scale_data:
            X = self._scale(X, fit=True)

        self._check_multicollinearity(X)

        _, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        prev_cost = float('inf')

        # Automatically compute sample weights if class_weight is set
        sample_weights = self._compute_class_weights(y)

        for epoch in range(self.max_iterations):
            self._fit_epoch(X, y, sample_weights, rng)

            linear_model_full = X @ self.weights + self.bias
            y_pred_full = np.clip(self._sigmoid(linear_model_full), 1e-15, 1 - 1e-15)
            cost = float(-np.mean(sample_weights * (y * np.log(y_pred_full + epsilon) + (1 - y) * np.log(1 - y_pred_full + epsilon))))
            cost += float(self._compute_regularization_cost())
            self.cost_history.append(cost)

            if self.verbose and (epoch % 100 == 0 or epoch == self.max_iterations - 1):
                print(f"Epoch {epoch + 1}/{self.max_iterations}, Cost: {cost:.6f}")

            if abs(prev_cost - cost) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break
            prev_cost = cost

        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scale_data:
            X = self._scale(X)
        return self._sigmoid(X @ self.weights + self.bias)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.scale_data:
            X = self._scale(X)
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
