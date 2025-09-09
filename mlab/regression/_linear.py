# pylint: disable=missing-docstring
import numpy as np

from mlab.base import BaseRegressor


# pylint: disable=too-many-instance-attributes, invalid-name
class LinearRegression(BaseRegressor):
    """Linear regression model using the normal equation."""

    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear regression model using the normal equation with ridge regularization.

        Parameters:
        X (np.ndarray): Training features.
        y (np.ndarray): Target values.

        Returns:
        self: Fitted model.
        """
        feature_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
        A = feature_with_intercept.T.dot(feature_with_intercept)
        # Add small ridge regularization to the diagonal (except intercept)
        reg = 1e-6 * np.eye(A.shape[0])
        reg[0, 0] = 0  # Don't regularize the intercept
        A_reg = A + reg
        self.weights = (
            np.linalg.inv(A_reg)
            .dot(feature_with_intercept.T)
            .dot(y)
        )
        return self

    def predict(self, X: np.ndarray):
        """
        Predict using the linear regression model.

        Parameters:
        X (np.ndarray): Test features.

        Returns:
        np.ndarray: Predicted values.
        """
        feature_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
        return feature_with_intercept @ self.weights

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.

        Returns:
        float: Mean Squared Error.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error (MAE) between true and predicted values.

        Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.

        Returns:
        float: Mean Absolute Error.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_type: str) -> float:
        """
        Evaluate the model using Mean Squared Error (MSE) or Mean Absolute Error (MAE).

        Parameters:
        X (np.ndarray): Test features.
        y (np.ndarray): True target values.
        eval_type (str): Evaluation type ('mse' or 'mae').

        Returns:
        float: Evaluation score.
        """
        y_pred = self.predict(X)
        if eval_type == "mse":
            return self.mean_squared_error(y, y_pred)
        elif eval_type == "mae":
            return self.mean_absolute_error(y, y_pred)
        else:
            raise ValueError("Invalid evaluation type. Choose 'mse' or 'mae'.")


# pylint: disable=too-many-instance-attributes, invalid-name
class SGDRegression:
    """
    Production-ready SGD Regressor with support for L1, L2, and ElasticNet regularization,
    mini-batch training, early stopping, learning rate schedules, and serialization.

    Parameters
    ----------
    config : dict, optional
        Dictionary of hyperparameters. Includes:
            - loss: str, default='squared_error'
            - penalty: str, default='l2'
            - alpha: float, default=0.0001
            - l1_ratio: float, default=0.15
            - fit_intercept: bool, default=True
            - max_iter: int, default=1000
            - tol: float, default=1e-3
            - shuffle: bool, default=True
            - verbose: int, default=0
            - random_state: int or None
            - learning_rate: str, default='invscaling'
            - eta0: float, default=0.01
            - power_t: float, default=0.25
            - early_stopping: bool, default=False
            - validation_fraction: float, default=0.1
            - n_iter_no_change: int, default=5
            - warm_start: bool, default=False
            - batch_size: int, default=32
    """

    def __init__(self, config=None):
        config = config or {}

        # Load hyperparameters
        self.loss = config.get("loss", "squared_error")
        self.penalty = config.get("penalty", "l2")
        self.alpha = config.get("alpha", 0.0001)
        self.l1_ratio = config.get("l1_ratio", 0.15)
        self.fit_intercept = config.get("fit_intercept", True)
        self.max_iter = config.get("max_iter", 1000)
        self.tol = config.get("tol", 1e-3)
        self.shuffle = config.get("shuffle", True)
        self.verbose = config.get("verbose", 0)
        self.random_state = config.get("random_state", None)
        self.learning_rate = config.get("learning_rate", "invscaling")
        self.eta0 = config.get("eta0", 0.01)
        self.power_t = config.get("power_t", 0.25)
        self.early_stopping = config.get("early_stopping", False)
        self.validation_fraction = config.get("validation_fraction", 0.1)
        self.n_iter_no_change = config.get("n_iter_no_change", 5)
        self.warm_start = config.get("warm_start", False)
        self.batch_size = config.get("batch_size", 32)

        # Initialize internal state
        self.coef_ = None
        self.intercept_ = 0.0 if self.fit_intercept else None
        self.n_iter_ = 0
        self.cost_history_ = []
        self.best_weights_ = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _initialize_weights(self, n_features):
        """Initialize weights and intercept."""
        self.coef_ = np.random.randn(n_features) * 0.01
        if self.fit_intercept:
            self.intercept_ = 0.0

    def _apply_penalty(self, grad):
        """Apply regularization penalty to gradient."""
        if self.penalty == "l2":
            return grad + self.alpha * self.coef_
        elif self.penalty == "l1":
            return grad + self.alpha * np.sign(self.coef_)
        elif self.penalty == "elasticnet":
            l1 = self.alpha * self.l1_ratio * np.sign(self.coef_)
            l2 = self.alpha * (1 - self.l1_ratio) * self.coef_
            return grad + l1 + l2
        else:
            return grad  # No penalty will be applied

    def _compute_loss(self, X, y):
        """Compute the loss value on the given data."""
        preds = self.predict(X)
        return np.mean((y - preds) ** 2)

    def _shuffle_data(self, X, y):
        """Shuffle X and y consistently."""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def _update_learning_rate(self, epoch):
        """Update learning rate according to schedule."""
        if self.learning_rate == "invscaling":
            return self.eta0 / (1 + epoch) ** self.power_t
        return self.eta0  # Default learning rate

    def _display_training_info(self, epoch, loss):
        """Display training information."""
        if self.verbose:
            print(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {loss:.5f}")

    def _validate_dimensions(self, X, y):
        """Validate dimensions of X and y."""
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

    def _check_for_early_stopping(self, no_improvement):
        """Check if early stopping criteria are met."""
        if self.early_stopping and no_improvement >= self.n_iter_no_change:
            if self.verbose:
                print("Early stopping triggered.")
            return True
        return False

    def _update_early_stopping(self, loss, best_loss, no_improvement):
        """Update early stopping counters and best weights."""
        if loss < best_loss - self.tol:  # Check for improvement
            best_loss = loss
            no_improvement = 0
            self.best_weights_ = (self.coef_.copy(), self.intercept_)
        else:
            no_improvement += 1
        return best_loss, no_improvement

    def fit(self, X, y):
        """
        Train the model using SGD.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self._validate_dimensions(X, y)
        n_samples, n_features = X.shape
        if self.coef_ is None or not self.warm_start:
            self._initialize_weights(n_features)

        val_size = (
            int(self.validation_fraction * n_samples) if self.early_stopping else 0
        )
        X_train, y_train = X[val_size:], y[val_size:]
        X_val, y_val = X[:val_size], y[:val_size]

        best_loss = float("inf")
        no_improvement = 0
        eta = self.eta0

        for epoch in range(self.max_iter):
            if self.shuffle:
                X_train, y_train = self._shuffle_data(X_train, y_train)
            self._run_epoch(X_train, y_train, eta)
            loss = self._compute_loss(
                X_val if self.early_stopping else X_train,
                y_val if self.early_stopping else y_train,
            )
            self.cost_history_.append(loss)

            self._display_training_info(epoch, loss)

            best_loss, no_improvement = self._update_early_stopping(
                loss, best_loss, no_improvement
            )

            if self._check_for_early_stopping(no_improvement):
                break

            eta = self._update_learning_rate(epoch)

        if self.best_weights_:
            self.coef_, self.intercept_ = self.best_weights_

        self.n_iter_ = epoch + 1
        return self

    def _run_epoch(self, X, y, eta):
        """Run one full epoch of mini-batch updates."""
        for start in range(0, X.shape[0], self.batch_size):
            end = start + self.batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            for i in range(X_batch.shape[0]):
                xi = X_batch[i : i + 1]
                yi = y_batch[i : i + 1]
                pred = xi @ self.coef_ + (self.intercept_ if self.fit_intercept else 0)
                error = pred - yi
                grad = error * xi
                grad = self._apply_penalty(grad.flatten())
                self.coef_ -= eta * grad
                if self.fit_intercept:
                    self.intercept_ -= eta * error

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        X = np.asarray(X)
        preds = X @ self.coef_
        if self.fit_intercept:
            preds += self.intercept_
        return preds
