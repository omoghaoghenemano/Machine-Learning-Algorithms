import numpy as np

KMEANS_INIT_METHODS = ('random', 'kmeans++')

def _vectorized_median_centroids(x, labels, n_clusters, old_centroids):
    """Compute the median for each cluster in a vectorized way."""
    new_centroids = np.copy(old_centroids)
    for i in range(n_clusters):
        mask = labels == i
        if np.any(mask):
            new_centroids[i] = np.median(x[mask], axis=0)
    return new_centroids

class _BaseKMeans:
    def __init__(self, n_clusters, max_iters=100, random_state=42, tol=1e-4,
                 use_median=False, max_no_improve=10, n_init=15):
        self.n_clusters = int(n_clusters)
        self.max_iters = max_iters
        self.random_seed = int(random_state)
        self.centroids = None
        self.wcss_history = []
        self.labels = None
        self._rng = np.random.default_rng(self.random_seed)
        self.tol = tol
        self.use_median = use_median
        self.max_no_improve = max_no_improve
        self.n_init = n_init

    def _init_centroids(self, x):
        raise NotImplementedError("Subclasses must implement _init_centroids.")

    def _compute_inertia(self, x, centroids):
        distances = np.sum((x[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.take_along_axis(distances, labels[:, None], axis=1))
        return inertia
    
    def _handle_type_error(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError("Input data x must be a numpy ndarray.")
        if x.ndim != 2:
            raise ValueError(
                f"Input data x must be 2D (n_samples, n_features), got shape {x.shape}")
        if x.shape[0] == 0:
            raise ValueError(
                "Cannot fit KMeans on an empty dataset. The input data contains 0 samples.")
        if not np.isfinite(x).all():
            raise ValueError("Input data contains NaN or Inf values.")
        n_samples = x.shape[0]
        unique_points = set(tuple(row) for row in x)
        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError(
                f"n_clusters must be in [1, n_samples]. Got n_clusters={self.n_clusters}, n_samples={n_samples}")
        if len(unique_points) < self.n_clusters:
            raise ValueError("Not enough unique points to initialize centroids")

    def fit(self, x):
        self._handle_type_error(x)
        self.centroids = self._init_centroids(x)
        no_improve = 0
        best_inertia = float('inf')
        for _ in range(self.max_iters):
            distances = np.sum((x[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            counts = np.bincount(labels, minlength=self.n_clusters)
            if self.use_median:
                new_centroids = _vectorized_median_centroids(x, labels, self.n_clusters, self.centroids)
            else:
                new_centroids = np.zeros_like(self.centroids)
                for i in range(self.n_clusters):
                    if counts[i] > 0:
                        new_centroids[i] = np.sum(x[labels == i], axis=0) / counts[i]
            empty = np.nonzero(counts == 0)[0]
            if empty.size > 0:
                rand_idx = self._rng.integers(x.shape[0], size=empty.size)
                noise = self._rng.normal(
                    scale=1e-4 * np.std(x, axis=0), size=(empty.size, x.shape[1]))
                new_centroids[empty] = x[rand_idx] + noise
            if np.allclose(self.centroids, new_centroids, rtol=self.tol, atol=self.tol):
                self.centroids = new_centroids
                break
            self.centroids = new_centroids
            inertia = np.sum((x - self.centroids[labels]) ** 2)
            self.wcss_history.append(inertia)
            if inertia < best_inertia - self.tol:
                best_inertia = inertia
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.max_no_improve:
                break
        if not self.wcss_history:
            distances = np.sum((x[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            inertia = np.sum((x - self.centroids[labels]) ** 2)
            self.wcss_history.append(inertia)
        self.labels = labels
        return self

    def predict(self, x):
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        distances = np.linalg.norm(
            x[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2) ** 2
        return np.argmin(distances, axis=1)

class KMeans(_BaseKMeans):
    def _init_centroids(self, x):
        n_samples = x.shape[0]
        best_centroids = None
        best_inertia = float('inf')
        mini_iters = 10
        
        rng = np.random.default_rng(self.random_seed)
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = x[indices].copy()
        for _ in range(mini_iters):
            distances = np.linalg.norm(
                x[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2) ** 2
            labels = np.argmin(distances, axis=1)
            centroids = _vectorized_median_centroids(x, labels, self.n_clusters, centroids)
        distances = np.linalg.norm(
            x[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2) ** 2
        labels = np.argmin(distances, axis=1)
        inertia = np.sum((x - centroids[labels]) ** 2)
        if inertia < best_inertia:
            best_centroids = centroids
            best_inertia = inertia
        return best_centroids

    def fit(self, x):
        return super().fit(x)

    def predict(self, x):
        return super().predict(x)

class KMeansPlusPlus(_BaseKMeans):
    def _init_centroids(self, x):
        n_samples, n_features = x.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=x.dtype)
        first_idx = self._rng.integers(n_samples)
        centroids[0] = x[first_idx]
        closest_dist_sq = np.sum((x - centroids[0]) ** 2, axis=1)
        n_local_trials = 10
        for c_id in range(1, self.n_clusters):
            probs = closest_dist_sq / closest_dist_sq.sum()
            candidate_ids = self._rng.choice(
                n_samples, size=n_local_trials, p=probs, replace=False)
            candidates = x[candidate_ids]
            dists = np.sum(
                (x[:, np.newaxis, :] - candidates[np.newaxis, :, :]) ** 2, axis=2)
            potentials = np.minimum(dists, closest_dist_sq[:, np.newaxis]).sum(axis=0)
            best_candidate = candidates[np.argmin(potentials)]
            centroids[c_id] = best_candidate
            new_distances = np.sum((x - best_candidate) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_distances)
        return centroids

    def predict(self, x):
        return super().predict(x)

    def fit(self, x):
        return super().fit(x)
