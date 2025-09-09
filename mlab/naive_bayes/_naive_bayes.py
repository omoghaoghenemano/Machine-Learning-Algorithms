import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        """
        Initialize MultinomialNB.
        Parameters:
            alpha (float): Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """
        Fit the model using training data.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            y (np.ndarray): Target vector (n_samples,)
        Raises:
            ValueError: If X or y is empty.
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("Input data X and y must not be empty.")
        # X: shape (n_samples, n_features), y: shape (n_samples,)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.n_features_ = n_features

        class_count = np.zeros(n_classes, dtype=np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_count[idx] = X_c.shape[0]
            feature_count[idx, :] = X_c.sum(axis=0)

        # Compute class log prior
        self.class_log_prior_ = np.log(class_count / class_count.sum())

        # Compute smoothed likelihoods
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

    def predict_log_proba(self, X):
        """
        Return log-probabilities for each class for input X.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
        Returns:
            np.ndarray: Log-probabilities (n_samples, n_classes)
        """
        # X: shape (n_samples, n_features)
        jll = (X @ self.feature_log_prob_.T) + self.class_log_prior_
        return jll

    def predict_proba(self, X):
        """
        Return probabilities for each class for input X.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
        Returns:
            np.ndarray: Probabilities (n_samples, n_classes)
        """
        log_probs = self.predict_log_proba(X)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        jll = self.predict_log_proba(X)
        return self.classes_[np.argmax(jll, axis=1)]
    

class GaussianNB:
    def __init__(self):
        """
        Initialize GaussianNB.
        """
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # mean
        self.sigma_ = None  # variance

    def fit(self, X, y):
        """
        Fit the model using training data.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            y (np.ndarray): Target vector (n_samples,)
        Raises:
            ValueError: If X or y is empty.
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("Input data X and y must not be empty.")
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.sigma_[idx, :] = X_c.var(axis=0)
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]

    def _log_gaussian_pdf(self, X, mean, var):
        """
        Compute log Gaussian PDF for X given mean and variance.
        """
        # X: (n_samples, n_features), mean/var: (n_features,)
        eps = 1e-9  # for numerical stability
        return -0.5 * np.log(2. * np.pi * (var + eps)) - ((X - mean) ** 2) / (2. * (var + eps))

    def predict_log_proba(self, X):
        """
        Return log-probabilities for each class for input X.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
        Returns:
            np.ndarray: Log-probabilities (n_samples, n_classes)
        """
        log_probs = []
        for idx in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[idx])
            log_likelihood = self._log_gaussian_pdf(X, self.theta_[idx], self.sigma_[idx]).sum(axis=1)
            log_probs.append(log_prior + log_likelihood)
        return np.vstack(log_probs).T  # shape (n_samples, n_classes)

    def predict_proba(self, X):
        """
        Return probabilities for each class for input X.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
        Returns:
            np.ndarray: Probabilities (n_samples, n_classes)
        """
        log_probs = self.predict_log_proba(X)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features)
        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]