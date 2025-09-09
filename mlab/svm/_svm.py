import numpy as np
from typing import Optional

class BaseSVM():
    """
    Base class for Support Vector Machines (SVM).
    Provides kernel computation and common SVM attributes.
    """
    def __init__(self, C: float = 1.0, lr: float = 1.0, tol: float = 1e-4, max_iter: int = 10000,
                 kernel: str = 'linear', gamma: float = 1.0, verbose:bool = False, random_state: Optional[int] = None, degree: int = 3):
        """
        Initialize the BaseSVM.
        Args:
            C (float): Regularization parameter.
            lr (float): Learning rate.
            tol (float): Tolerance for stopping criterion.
            max_iter (int): Maximum number of iterations.
            kernel (str): Kernel type ('linear', 'rbf', 'poly').
            gamma (float): Kernel coefficient for 'rbf' and 'poly'.
            verbose (bool): Verbosity flag.
            random_state (Optional[int]): Random seed.
            degree (int): Degree for polynomial kernel.
        """
        self.C = C
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.weight_ = None
        self.bias_ = 0
        assert kernel in ('linear', 'rbf', 'poly'), 'Kernel must be either linear, rbf, or poly'
        self.kernel = kernel
        self.gamma = gamma
        self.verbose = verbose
        self.random_state = random_state
        self.degree = degree
        if random_state is not None:
            np.random.seed(random_state)

        self.support_vectors_ = None
        self.support_ = None
        self.alpha_ = None
        self.intercept_ = None
        self.n_support_ = None

    def _get_kernel(self, X, Y=None):
        """
        Compute the kernel matrix between X and Y.
        Args:
            X (np.ndarray): First input array.
            Y (np.ndarray, optional): Second input array.
        Returns:
            np.ndarray: Kernel matrix.
        Raises:
            ValueError: If kernel computation fails due to invalid input or numerical issues.
        """
        try:
            if self.kernel == 'linear':
                if Y is None:
                    return X @ X.T
                return X @ Y.T
            elif self.kernel == 'rbf':
                if Y is None:
                    Y = X
                X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
                Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
                K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
                return np.exp(-self.gamma * K)
            elif self.kernel == 'poly':
                if Y is None:
                    Y = X
                return (self.gamma * (X @ Y.T) + 1) ** self.degree
            else:
                raise ValueError("Unsupported kernel")
        except Exception as e:
            raise ValueError(f"Kernel computation failed: {e}") from e

    def _rbf_kernel(self, X, Y=None):
        """
        Compute the RBF (Gaussian) kernel between X and Y.
        Args:
            X (np.ndarray): First input array.
            Y (np.ndarray, optional): Second input array.
        Returns:
            np.ndarray: RBF kernel matrix.
        Raises:
            ValueError: If kernel computation fails due to invalid input or numerical issues.
        """
        try:
            if Y is None:
                Y = X
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
            K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            return np.exp(-self.gamma * K)
        except Exception as e:
            raise ValueError(f"RBF kernel computation failed: {e}") from e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained SVM model.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted values.
        Raises:
            RuntimeError: If prediction fails due to missing model parameters or numerical issues.
        """
        try:
            if self.kernel == 'rbf' and self.support_vectors_ is not None:
                X = self._rbf_kernel(X, self.support_vectors_)
            elif self.kernel == 'poly' and self.support_vectors_ is not None:
                X = (self.gamma * (X @ self.support_vectors_.T) + 1) ** self.degree
            elif self.kernel == 'linear' and self.support_vectors_ is not None:
                X = X @ self.support_vectors_.T
            return X.dot(self.weight_) + self.bias_
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

class SVM(BaseSVM):
    """
    Linear Support Vector Machine (SVM) classifier using gradient descent.
    """
    def __init__(self, C=1.0, lr=0.001, tol=1e-4, max_iter=1000,
                 kernel='linear', gamma=1.0, verbose=False,
                 random_state=None, degree=3):
        """
        Initialize the SVM classifier.
        Args:
            C (float): Regularization parameter.
            lr (float): Learning rate.
            tol (float): Tolerance for stopping criterion.
            max_iter (int): Maximum number of iterations.
            kernel (str): Kernel type (only 'linear' supported).
            gamma (float): Not used for linear kernel.
            verbose (bool): Verbosity flag.
            random_state (Optional[int]): Random seed.
            degree (int): Not used for linear kernel.
        """
        super().__init__(C=C, lr=lr, tol=tol, max_iter=max_iter,
                         kernel=kernel, gamma=gamma, verbose=verbose,
                         random_state=random_state, degree=degree)
        # Validate and clip hyperparameters
        self.C = max(C, 1e-8)  # Avoid zero or negative C
        self.gamma = max(gamma, 1e-8)
        self._scaler_mean = None
        self._scaler_scale = None

    def _scale_features(self, X, training=True):
        """
        Standardize features by removing the mean and scaling to unit variance.
        Args:
            X (np.ndarray): Input data.
            training (bool): Whether to fit scaler on X.
        Returns:
            np.ndarray: Scaled data.
        Raises:
            ValueError: If scaling fails due to division by zero or numerical issues.
        """
        try:
            if training:
                self._scaler_mean = X.mean(axis=0)
                self._scaler_scale = X.std(axis=0)
                self._scaler_scale[self._scaler_scale == 0] = 1.0
            X_scaled = (X - self._scaler_mean) / self._scaler_scale
            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                raise ValueError("Feature scaling resulted in NaN or Inf values.")
            return X_scaled
        except Exception as e:
            raise ValueError(f"Feature scaling failed: {e}") from e

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Fit the SVM model using gradient descent.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation data.
            y_val (np.ndarray, optional): Validation labels.
            early_stopping_rounds (int): Early stopping patience.
        Returns:
            self
        Raises:
            ValueError: If fitting fails due to invalid input or convergence issues.
        """
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)
            if X.size == 0 or y.size == 0:
                raise ValueError("Empty input data.")

            # Scale features
            X = self._scale_features(X, training=True)

            unique_labels = np.unique(y)
            if len(unique_labels) != 2:
                raise ValueError("This SVM supports binary classification only.")
            y_bin = np.where(y == unique_labels[0], -1, 1)

            n_samples, n_features = X.shape
            self.weight_ = np.zeros(n_features)
            self.bias_ = 0

            best_val_loss = float('inf')
            no_improve_count = 0
            rng = np.random.default_rng(self.random_state)

            for iteration in range(self.max_iter):
                indices = rng.permutation(n_samples)
                total_loss = 0
                for idx in indices:
                    x_i = X[idx]
                    y_i = y_bin[idx]
                    decision = np.dot(self.weight_, x_i) + self.bias_
                    loss = max(0, 1 - y_i * decision)
                    total_loss += loss
                    if y_i * decision >= 1:
                        dw = 2 * (1 / self.C) * self.weight_
                        db = 0
                    else:
                        dw = 2 * (1 / self.C) * self.weight_ - y_i * x_i
                        db = -y_i

                    # Clip gradients for stability
                    dw = np.clip(dw, -1e5, 1e5)
                    self.weight_ -= self.lr * dw
                    self.bias_ -= self.lr * db

                    if np.isnan(self.weight_).any() or np.isinf(self.weight_).any():
                        raise ValueError(f"Numerical instability detected in weights at iteration {iteration}.")

                # Early stopping on validation loss
                if X_val is not None and y_val is not None:
                    X_val_scaled = (np.asarray(X_val, dtype=np.float64) - self._scaler_mean) / self._scaler_scale
                    y_val_bin = np.where(y_val == unique_labels[0], -1, 1)
                    val_decision = X_val_scaled @ self.weight_ + self.bias_
                    val_losses = np.maximum(0, 1 - y_val_bin * val_decision)
                    val_loss = np.mean(val_losses)
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= early_stopping_rounds:
                            if self.verbose:
                                print(f"Early stopping at iteration {iteration} due to validation loss stagnation.")
                            break
                else:
                    if total_loss < self.tol:
                        if self.verbose:
                            print(f"Converged at iteration {iteration} with training loss {total_loss:.6f}.")
                        break

            # Identify support vectors: margin ≤ 1 + some slack
            margins = np.abs(X @ self.weight_ + self.bias_)
            support_mask = margins <= 1 + 1e-3
            self.support_ = np.where(support_mask)[0]
            self.support_vectors_ = X[self.support_]
            self.alpha_ = np.ones(len(self.support_)) / len(self.support_)
            self.intercept_ = self.bias_
            self.n_support_ = len(self.support_)
            self.classes_ = unique_labels  # pylint: disable=attribute-defined-outside-init

            return self
        except Exception as e:
            raise ValueError(f"SVM fitting failed: {e}") from e

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted class labels.
        Raises:
            RuntimeError: If prediction fails due to missing model parameters or numerical issues.
        """
        try:
            X = np.asarray(X, dtype=np.float64)
            if self._scaler_mean is None or self._scaler_scale is None:
                raise RuntimeError("Model is not fitted yet. No scaler parameters available.")
            X_scaled = (X - self._scaler_mean) / self._scaler_scale
            decision = X_scaled @ self.weight_ + self.bias_
            preds = np.where(decision >= 0, self.classes_[1], self.classes_[0])
            return preds
        except Exception as e:
            raise RuntimeError(f"SVM prediction failed: {e}") from e

class SVC(BaseSVM):
    """
    Support Vector Classifier (SVC) using kernel methods and dual optimization.
    """
    def __init__(self, C=1.0, lr=0.01, tol=1e-4, max_iter=500, kernel='linear', gamma=1.0, verbose=False, random_state: Optional[int] = None, degree: int = 3):
        """
        Initialize the SVC classifier.
        Args:
            C (float): Regularization parameter.
            lr (float): Learning rate.
            tol (float): Tolerance for stopping criterion.
            max_iter (int): Maximum number of iterations.
            kernel (str): Kernel type ('linear', 'rbf', 'poly').
            gamma (float): Kernel coefficient for 'rbf' and 'poly'.
            verbose (bool): Verbosity flag.
            random_state (Optional[int]): Random seed.
            degree (int): Degree for polynomial kernel.
        """
        super().__init__(C=C, lr=lr, tol=tol, max_iter=max_iter, kernel=kernel, gamma=gamma, verbose=verbose, random_state=random_state, degree=degree)
        self.sv_y = None
        self._scaler_mean = None
        self._scaler_scale = None

    def _update_lr(self, iteration):
        """
        Update the learning rate with decay.
        Args:
            iteration (int): Current iteration.
        Returns:
            float: Updated learning rate.
        """
        decay_rate = 0.001
        return np.clip(self.lr / (1 + decay_rate * iteration), 1e-8, 10.0)

    def fit(self, X, y, _X_val=None, _y_val=None, early_stopping_rounds=10):
        """
        Fit the SVC model using dual optimization.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            _X_val, _y_val: Not used.
            early_stopping_rounds (int): Early stopping patience.
        Returns:
            self
        Raises:
            ValueError: If fitting fails due to invalid input or convergence issues.
        """
        X = np.asarray(X, dtype=np.float64)
        y = y.astype(float)
        # Standardize features
        self._scaler_mean = X.mean(axis=0)
        self._scaler_scale = X.std(axis=0)
        self._scaler_scale[self._scaler_scale == 0] = 1.0
        X = (X - self._scaler_mean) / self._scaler_scale
        # Check for NaN or inf in X or y
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Input X contains NaN or inf values.")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("Input y contains NaN or inf values.")
        n_samples, n_features = X.shape
        self.alpha_ = np.zeros(n_samples)
        self.support_ = np.array([], dtype=int)
        self.support_vectors_ = np.empty((0, n_features))
        self.sv_y = np.array([])
        self.n_support_ = 0
        self.intercept_ = 0.0
        self.weight_ = None
        K = self._get_kernel(X)
        best_obj = float('inf')
        no_improve_count = 0
        min_C = 1e-8
        C = np.clip(self.C, min_C, 1e8)
        for iteration in range(self.max_iter):
            lr = self._update_lr(iteration)
            decision = (self.alpha_ * y) @ K
            gradient = 1 - y * decision
            candidates = np.where((self.alpha_ < C) & (gradient > self.tol))[0]
            if len(candidates) == 0:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            self.alpha_[candidates] += lr * gradient[candidates]
            self.alpha_ = np.clip(self.alpha_, 0, C)
            # Check for NaN/inf in alpha_
            if np.isnan(self.alpha_).any() or np.isinf(self.alpha_).any():
                raise ValueError(f"NaN or inf encountered in alpha_ at iteration {iteration}.")
            dual_obj = 0.5 * (self.alpha_ * y) @ (K @ (self.alpha_ * y)) - np.sum(self.alpha_)
            if self.verbose and iteration % 100 == 0:
                print(f"Iter {iteration}, Dual Obj: {dual_obj:.4f}, lr: {lr:.6f}")
            if dual_obj < best_obj - self.tol:
                best_obj = dual_obj
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stopping_rounds:
                    if self.verbose:
                        print(f"Early stopping at iteration {iteration}, dual objective stalled.")
                    break
        # Identify support vectors properly: alpha > 1e-5 (non-zero)
        sv_mask = self.alpha_ > 1e-5
        if np.any(sv_mask):
            self.support_ = np.where(sv_mask)[0]
            self.support_vectors_ = X[sv_mask]
            self.sv_y = y[sv_mask]
            self.alpha_ = self.alpha_[sv_mask]
            self.n_support_ = len(self.support_)
            self.intercept_ = np.mean(
                self.sv_y - np.sum((self.alpha_ * self.sv_y)[:, None] * self._get_kernel(self.support_vectors_), axis=0)
            )
            if self.kernel == 'linear':
                self.weight_ = np.sum((self.alpha_ * self.sv_y)[:, None] * self.support_vectors_, axis=0)
            else:
                self.weight_ = None
        else:
            self.support_ = np.array([], dtype=int)
            self.support_vectors_ = np.empty((0, n_features))
            self.sv_y = np.array([])
            self.alpha_ = np.array([])
            self.n_support_ = 0
            self.intercept_ = 0.0
            self.weight_ = None
        return self

    def decision_function(self, X):
        """
        Compute the decision function for samples in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Decision function values.
        """
        X = np.asarray(X, dtype=np.float64)
        if self._scaler_mean is not None and self._scaler_scale is not None:
            X = (X - self._scaler_mean) / self._scaler_scale
        if self.n_support_ == 0 or self.alpha_.size == 0:
            return np.zeros(X.shape[0])
        K = self._get_kernel(X, self.support_vectors_)
        return (K @ (self.alpha_ * self.sv_y)) + self.intercept_

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted class labels (-1 or 1).
        """
        dec = self.decision_function(X)
        if self.n_support_ == 0 or self.alpha_.size == 0:
            return np.full(X.shape[0], -1)
        return np.sign(dec)

class SVR(BaseSVM):
    """
    Support Vector Regression (SVR) using kernel methods and dual optimization.
    """
    def __init__(self, C=1.0, epsilon=0.1, lr=0.01, tol=1e-4, max_iter=500, kernel='linear', gamma=1.0, verbose=False, random_state: Optional[int] = None, degree: int = 3):
        """
        Initialize the SVR regressor.
        Args:
            C (float): Regularization parameter.
            epsilon (float): Epsilon-tube within which no penalty is associated.
            lr (float): Learning rate.
            tol (float): Tolerance for stopping criterion.
            max_iter (int): Maximum number of iterations.
            kernel (str): Kernel type ('linear', 'rbf', 'poly').
            gamma (float): Kernel coefficient for 'rbf' and 'poly'.
            verbose (bool): Verbosity flag.
            random_state (Optional[int]): Random seed.
            degree (int): Degree for polynomial kernel.
        """
        super().__init__(C=C, lr=lr, tol=tol, max_iter=max_iter, kernel=kernel, gamma=gamma, verbose=verbose, random_state=random_state, degree=degree)
        self.epsilon = epsilon
        self.alpha_star_ = None
        self._scaler_mean = None
        self._scaler_scale = None
    def _update_lr(self, iteration):
        """
        Update the learning rate with decay.
        Args:
            iteration (int): Current iteration.
        Returns:
            float: Updated learning rate.
        """
        decay_rate = 0.001
        return np.clip(self.lr / (1 + decay_rate * iteration), 1e-8, 10.0)
    def fit(self, X, y, early_stopping_rounds=10):
        """
        Fit the SVR model using dual optimization.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training targets.
            _X_val, _y_val: Not used.
            early_stopping_rounds (int): Early stopping patience.
        Returns:
            self
        Raises:
            ValueError: If fitting fails due to invalid input or convergence issues.
        """
        X = np.asarray(X, dtype=np.float64)
        y = y.astype(float)
        # Standardize features
        self._scaler_mean = X.mean(axis=0)
        self._scaler_scale = X.std(axis=0)
        self._scaler_scale[self._scaler_scale == 0] = 1.0
        X = (X - self._scaler_mean) / self._scaler_scale
        # Check for NaN or inf in X or y
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Input X contains NaN or inf values.")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("Input y contains NaN or inf values.")
        n_samples, n_features = X.shape

        self.alpha_ = np.zeros(n_samples)
        self.alpha_star_ = np.zeros(n_samples)

        K = self._get_kernel(X)

        best_loss = float('inf')
        no_improve_count = 0
        min_C = 1e-8
        C = np.clip(self.C, min_C, 1e8)
        for iteration in range(self.max_iter):
            lr = self._update_lr(iteration)

            pred = (self.alpha_ - self.alpha_star_) @ K
            error = pred - y

            update_pos = error > self.epsilon
            update_neg = error < -self.epsilon

            self.alpha_[update_pos] += lr
            self.alpha_star_[update_neg] += lr

            self.alpha_ = np.clip(self.alpha_, 0, C)
            self.alpha_star_ = np.clip(self.alpha_star_, 0, C)

            loss = np.mean(np.maximum(0, np.abs(error) - self.epsilon)) + \
                   0.5 * (self.alpha_ @ K @ self.alpha_ + self.alpha_star_ @ K @ self.alpha_star_)

            if self.verbose and iteration % 100 == 0:
                print(f"Iter {iteration}, Loss: {loss:.4f}, lr: {lr:.6f}")

            if loss < best_loss - self.tol:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stopping_rounds:
                    if self.verbose:
                        print(f"Early stopping at iteration {iteration}, loss stalled.")
                    break

        self.alpha_ = self.alpha_ - self.alpha_star_
        sv_mask = np.abs(self.alpha_) > 1e-5
        if np.any(sv_mask):
            self.support_ = np.where(sv_mask)[0]
            self.support_vectors_ = X[sv_mask]
            self.alpha_ = self.alpha_[sv_mask]
            self.n_support_ = len(self.support_)
            self.intercept_ = np.mean([
                y[i] - np.sum(self.alpha_ * K[i, sv_mask])
                for i in self.support_
            ])
            if self.kernel == 'linear':
                self.weight_ = np.zeros(n_features)
                for i, alpha_i in enumerate(self.alpha_):
                    self.weight_ += alpha_i * self.support_vectors_[i]
                self.bias_ = self.intercept_
            else:
                self.weight_ = None
                self.bias_ = self.intercept_
        else:
            self.support_ = np.array([], dtype=int)
            self.support_vectors_ = np.empty((0, n_features))
            self.alpha_ = np.array([])
            self.n_support_ = 0
            self.intercept_ = 0.0
            self.weight_ = None
            self.bias_ = 0.0

        return self

    def predict(self, X):
        """
        Predict regression targets for samples in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted regression values.
        """
        X = np.asarray(X, dtype=np.float64)
        if self._scaler_mean is not None and self._scaler_scale is not None:
            X = (X - self._scaler_mean) / self._scaler_scale
        if self.n_support_ == 0 or self.alpha_.size == 0:
            return np.zeros(X.shape[0])
        K = self._get_kernel(X, self.support_vectors_)
        return np.dot(K, self.alpha_) + self.intercept_
