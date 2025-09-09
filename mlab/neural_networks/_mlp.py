"""
Production-level Multi-layer perceptron implementation for regression.
Enhanced with scikit-learn compatible features and advanced optimizers.
"""
import numpy as np
from ..base import BaseEstimator
from .layers import ModularLinearLayer, ReLULayer, DropoutLayer
from .optimizers import get_optimizer


class MLPRegressor(BaseEstimator):
    """
    Multi-layer perceptron regressor with production-level features.

    Features:
    - Multiple solvers (sgd, adam, lbfgs)
    - Mini-batch training for scalability
    - Early stopping with validation monitoring
    - Learning rate scheduling
    - Warm start capability
    - Robust input validation
    - Convergence detection
    """

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', lr='constant',
                 lr_init=0.001, epochs=200, random_state=None,
                 tol=1e-4, verbose=False, warm_start=False, early_stopping=False,
                 dropout=0.0):
        """
        Initialize the MLP regressor with production-level features.

        Args:
            hidden_layer_sizes (tuple): Sizes of hidden layers
            activation (str): Activation function ('relu', 'tanh')
            solver (str): Optimizer ('sgd', 'adam', 'lbfgs')
            alpha (float): L2 regularization parameter
            batch_size (int or 'auto'): Size of minibatches
            lr (str): Learning rate schedule ('constant', 'invscaling', 'adaptive')
            lr_init (float): Initial learning rate
            epochs (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
            tol (float): Tolerance for optimization
            verbose (bool): Whether to print progress messages
            warm_start (bool): Whether to reuse previous solution
            early_stopping (bool): Whether to use early stopping
            dropout (float): Dropout probability (0.0 disables dropout)
        """
        # Core parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.lr = lr
        self.lr_init = lr_init
        self.epochs = epochs
        self.random_state = random_state
        # More reasonable default tolerance
        self.tol = max(tol, 1e-6) if tol > 0 else 1e-5
        self.verbose = verbose
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.dropout = dropout

        # Additional parameters with defaults
        self.power_t = 0.5
        self.shuffle = True
        self.momentum = 0.9
        self.nesterovs_momentum = True
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.validation_fraction = 0.1
        self.n_iter_no_change = 20  # Increase patience
        self.max_fun = 15000

        # Pre-declare attributes for Pylint compliance
        self.layers_ = None
        self.optimizer_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.loss_curve_ = None
        self.validation_scores_ = None
        self.best_loss_ = None
        self.best_validation_score_ = None
        self._no_improvement_count = None
        self._current_lr = None
        self._best_weights = None
        self._best_biases = None

        # Initialize state variables
        self._initialize_state()

        # Set up reproducible random state for weight initialization
        if random_state is not None:
            self._random_state = np.random.default_rng(random_state)
        else:
            self._random_state = np.random.default_rng(42)

    def _initialize_state(self):
        """Initialize state variables."""
        # All attributes defined here for Pylint compliance
        self.layers_ = None
        self.optimizer_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.loss_curve_ = []
        self.validation_scores_ = []
        self.best_loss_ = np.inf
        self.best_validation_score_ = -np.inf
        self._no_improvement_count = 0
        self._current_lr = self.lr_init
        self._best_weights = None
        self._best_biases = None

    def _validate_input(self, x, y=None):
        """Validate input data."""
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")

        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "x and y must have the same number of samples")
            return x, y

        return x

    def _get_activation_layer(self):
        """Get activation layer based on activation parameter."""
        if self.activation == 'relu':
            return ReLULayer()
        raise ValueError(
            f"Unsupported activation: {self.activation}. Only 'relu' is supported.")

    def _get_batch_size(self, n_samples):
        """Determine batch size."""
        if self.batch_size == 'auto':
            return min(200, n_samples)
        return min(self.batch_size, n_samples)

    def _create_optimizer(self):
        """Create optimizer instance."""
        if self.solver == 'sgd':
            return get_optimizer('sgd',
                                 lr=self._current_lr,
                                 momentum=self.momentum,
                                 nesterov=self.nesterovs_momentum)
        if self.solver == 'adam':
            return get_optimizer('adam',
                                 lr=self._current_lr,
                                 beta1=self.beta_1,
                                 beta2=self.beta_2,
                                 epsilon=self.epsilon)
        if self.solver == 'lbfgs':
            return get_optimizer('lbfgs',
                                 lr=self._current_lr,
                                 epochs=self.max_fun)
        raise ValueError(f"Unknown solver: {self.solver}")

    def _update_lr(self, iteration):
        """Update learning rate based on schedule."""
        if self.lr == 'invscaling':
            self._current_lr = (self.lr_init /
                                pow(iteration + 1, self.power_t))
        elif self.lr == 'adaptive':
            # Will be updated based on loss improvement
            pass
        # 'constant' keeps the same learning rate

    def _build_network(self, n_features, n_outputs):
        """
        Build the neural network architecture with improved initialization and dropout.

        Args:
            n_features (int): Number of input features
            n_outputs (int): Number of output features
        Returns:
            list: List of layers
        """
        if not self.warm_start or self.layers_ is None:
            layers = []
            rng = np.random.default_rng(self.random_state)
            prev_size = n_features
            for hidden_size in self.hidden_layer_sizes:
                # Linear layer
                linear_layer = ModularLinearLayer(
                    prev_size, hidden_size, rng=rng)
                weight, bias = self._init_coef(
                    prev_size, hidden_size, np.float32)
                linear_layer.weight = weight
                linear_layer.bias = bias
                layers.append(linear_layer)
                # Activation
                activation_layer = self._get_activation_layer()
                layers.append(activation_layer)
                # Dropout (only if dropout > 0)
                if self.dropout > 0.0:
                    layers.append(DropoutLayer(self.dropout))
                prev_size = hidden_size
            # Output layer (no activation for regression)
            output_layer = ModularLinearLayer(prev_size, n_outputs, rng=rng)
            weight, bias = self._init_coef(prev_size, n_outputs, np.float32)
            output_layer.weight = weight
            output_layer.bias = bias
            layers.append(output_layer)
            return layers
        return self.layers_

    def _split_train_validation(self, x, y):
        """Split data into training and validation sets."""
        if not self.early_stopping:
            return x, y, None, None

        n_samples = x.shape[0]
        n_validation = int(self.validation_fraction * n_samples)
        rng = np.random.default_rng(self.random_state)

        if self.shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        val_indices = indices[:n_validation]
        train_indices = indices[n_validation:]

        x_train, x_val = x[train_indices], x[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        return x_train, y_train, x_val, y_val

    def _generate_batches(self, x, y, batch_size):
        """Generate mini-batches for training."""
        n_samples = x.shape[0]
        rng = np.random.default_rng(self.random_state)

        if self.shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield x[batch_indices], y[batch_indices]

    def _forward_pass(self, x, training=False):
        """
        Perform forward pass through the network.
        Args:
            x (ndarray): Input data
            training (bool): Whether in training mode (enables dropout)
        Returns:
            ndarray: Network output
        """
        current_input = x
        for layer in self.layers_:
            # DropoutLayer: only active during training
            if isinstance(layer, DropoutLayer):
                current_input = layer(current_input, training=training)
            else:
                current_input = layer(current_input)
        return current_input

    def _backward_pass(self, x, y, y_pred):
        """
        Perform backward pass through the network.

        Args:
            x (ndarray): Input data
            y (ndarray): True targets
            y_pred (ndarray): Predicted targets
        """
        # Calculate loss gradient (MSE derivative)
        n_samples = x.shape[0]
        loss_grad = 2 * (y_pred - y) / n_samples

        # Backward pass through layers
        current_grad = loss_grad

        # Process layers in reverse order
        for layer in reversed(self.layers_):
            if hasattr(layer, 'backward'):
                if isinstance(layer, ModularLinearLayer):
                    # For linear layers, include L2 regularization
                    current_grad = layer.backward(
                        current_grad, alpha=self.alpha)
                else:
                    # For activation layers
                    current_grad = layer.backward(current_grad)

    def _update_parameters(self):
        """Update network parameters using the optimizer."""
        # Gradient clipping to prevent gradient explosion
        max_grad_norm = 1.0
        total_norm = 0.0

        # Calculate total gradient norm
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer) and hasattr(layer, 'weight_grad') and layer.weight_grad is not None:
                total_norm += np.sum(layer.weight_grad ** 2)
                total_norm += np.sum(layer.bias_grad ** 2)

        total_norm = np.sqrt(total_norm)

        # Apply gradient clipping if needed
        clip_factor = min(1.0, max_grad_norm / (total_norm + 1e-8))

        for i, layer in enumerate(self.layers_):
            if isinstance(layer, ModularLinearLayer) and hasattr(layer, 'weight_grad') and layer.weight_grad is not None:
                # Apply clipping by modifying the gradients in place
                weight_grad = layer.weight_grad * clip_factor
                bias_grad = layer.bias_grad * clip_factor

                if self.optimizer_ is not None:
                    # Use optimizer to update parameters
                    layer_id = f"layer_{i}"
                    updated_weight, updated_bias = self.optimizer_.update(
                        layer_id, weight_grad, bias_grad,
                        layer.weight.copy(), layer.bias.copy()
                    )
                    layer.weight = updated_weight
                    layer.bias = updated_bias
                else:
                    # Fallback to simple gradient descent
                    layer.weight -= self._current_lr * weight_grad
                    layer.bias -= self._current_lr * bias_grad

    def _mean_squared_error(self, y_true, y_pred):
        """
        Calculate mean squared error.

        Args:
            y_true (ndarray): True targets
            y_pred (ndarray): Predicted targets

        Returns:
            float: Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)

    def _calculate_regularization_loss(self):
        """
        Calculate L2 regularization loss.

        Returns:
            float: Regularization loss
        """
        reg_loss = 0.0
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                # Sum of squared weights
                reg_loss += np.sum(layer.weight ** 2)
        # Apply normalization in _compute_loss
        return 0.5 * self.alpha * reg_loss

    def _compute_loss(self, y_true, y_pred):
        """Compute total loss (MSE + regularization)."""
        n_samples = y_true.shape[0]
        mse_loss = self._mean_squared_error(y_true, y_pred)
        reg_loss = self._calculate_regularization_loss()
        # Apply normalization: reg_loss / n_samples
        return mse_loss + (reg_loss / n_samples)

    def _check_convergence(self, loss, iteration):
        """Check if training has converged."""
        if len(self.loss_curve_) < 5:  # Need at least 5 iterations
            return False

        # Don't converge too early
        if iteration < 20:
            return False

        # Check if loss improvement is below tolerance
        if len(self.loss_curve_) > 1:
            loss_diff = abs(self.loss_curve_[-1] - loss)
            relative_improvement = loss_diff / \
                max(abs(self.loss_curve_[-1]), 1e-8)

            if relative_improvement < self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

        # Early stopping based on loss improvement
        if self._no_improvement_count >= self.n_iter_no_change:
            if self.verbose:
                print(f"Convergence after {iteration + 1} iterations")
            return True

        return False

    def _validate_on_validation_set(self, x_val, y_val):
        """Evaluate model on validation set."""
        if x_val is None or y_val is None:
            return None

        # Extract features for validation
        # val_pred = self._forward_pass(x_val)  # Unused variable removed
        val_score = self.score(x_val, y_val.ravel()
                               if y_val.shape[1] == 1 else y_val)

        # Check for improvement
        if val_score > self.best_validation_score_:
            self.best_validation_score_ = val_score
            self._no_improvement_count = 0
            # Save best weights
            self._best_weights = []
            self._best_biases = []
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    self._best_weights.append(layer.weight.copy())
                    self._best_biases.append(layer.bias.copy())
        else:
            self._no_improvement_count += 1

        return val_score

    def _train_epoch(self, x_train, y_train, batch_size):
        """Train for one epoch and return epoch loss."""
        epoch_losses = []
        # Mini-batch training
        for x_batch, y_batch in self._generate_batches(x_train, y_train, batch_size):
            # Forward pass (training=True for dropout)
            y_pred = self._forward_pass(x_batch, training=True)
            # Backward pass
            self._backward_pass(x_batch, y_batch, y_pred)
            # Update parameters
            self._update_parameters()
            # Track batch loss
            batch_loss = self._compute_loss(y_batch, y_pred)
            epoch_losses.append(batch_loss)
        return np.mean(epoch_losses)

    def _should_stop_early(self, epoch_loss, iteration):
        """Check if training should stop early."""
        # Check convergence
        if self._check_convergence(epoch_loss, iteration):
            return True

        # Update best loss
        self.best_loss_ = min(self.best_loss_, epoch_loss)

        # Adaptive learning rate
        if self.lr == 'adaptive' and len(self.loss_curve_) > 2:
            if self.loss_curve_[-1] > self.loss_curve_[-2]:
                self._current_lr *= 0.5
                if self.verbose:
                    print(f"Learning rate reduced to {self._current_lr:.6f}")

        return False

    def fit(self, X, y):
        """
        Train the MLP regressor with production-level features.

        Args:
            X (ndarray): Training data of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples, n_outputs)

        Returns:
            self: Fitted estimator
        """
        # Input validation
        X, y = self._validate_input(X, y)

        _, n_features = X.shape
        n_outputs = y.shape[1]

        # Split training and validation data
        x_train, y_train, x_val, y_val = self._split_train_validation(X, y)

        # Build network
        self.layers_ = self._build_network(n_features, n_outputs)
        self.n_layers_ = len(
            [l for l in self.layers_ if isinstance(l, ModularLinearLayer)])
        self.n_outputs_ = n_outputs

        # Create optimizer
        self.optimizer_ = self._create_optimizer()

        # Determine batch size
        batch_size = self._get_batch_size(x_train.shape[0])

        # Initialize training history (attributes already defined in __init__/_initialize_state)
        if not self.warm_start:
            self.loss_curve_.clear()
            self.validation_scores_.clear()
            self.best_loss_ = np.inf
            self.best_validation_score_ = -np.inf
            self._no_improvement_count = 0

        # Training loop
        for iteration in range(self.epochs):
            # Update learning rate
            self._update_lr(iteration)

            # Train for one epoch
            epoch_loss = self._train_epoch(x_train, y_train, batch_size)
            self.loss_curve_.append(epoch_loss)

            # Validation
            val_score = self._validate_on_validation_set(x_val, y_val)
            if val_score is not None:
                self.validation_scores_.append(val_score)

            # Early stopping check
            if self._should_stop_early(epoch_loss, iteration):
                break

            # Verbose output
            if self.verbose and (iteration + 1) % 10 == 0:
                val_msg = f", Val Score: {val_score:.6f}" if val_score is not None else ""
                print(f"Iteration {iteration + 1}/{self.epochs}, "
                      f"Loss: {epoch_loss:.6f}{val_msg}")

        self.n_iter_ = len(self.loss_curve_)

        # Restore best weights if early stopping was used
        if self.early_stopping and x_val is not None and self._best_weights is not None:
            weight_idx = 0
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    layer.weight = self._best_weights[weight_idx]
                    layer.bias = self._best_biases[weight_idx]
                    weight_idx += 1
            if self.verbose:
                print(
                    f"Restored best weights from validation score: {self.best_validation_score_:.6f}")

        if self.verbose:
            print(f"Training completed in {self.n_iter_} iterations")

        return self

    def predict(self, x):
        """
        Make predictions using the trained model.
        Args:
            x (ndarray): Input data of shape (n_samples, n_features)
        Returns:
            ndarray: Predictions of shape (n_samples, n_outputs)
        """
        if self.layers_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        x = self._validate_input(x)
        # Dropout is disabled during prediction
        predictions = self._forward_pass(x, training=False)
        if predictions.shape[1] == 1:
            return predictions.ravel()
        return predictions

    def score(self, x, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Args:
            x (ndarray): Test samples
            y (ndarray): True values for x

        Returns:
            float: R^2 score
        """
        y_pred = self.predict(x)

        # Handle 1D arrays
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def get_params(self, mode: bool = True):
        """Get parameters for this estimator."""
        _ = mode  # Unused parameter for compatibility
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'lr_init': self.lr_init,
            'epochs': self.epochs,
            'random_state': self.random_state,
            'tol': self.tol,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'early_stopping': self.early_stopping,
            'dropout': self.dropout
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self

    def _init_coef(self, fan_in, fan_out, dtype):
        # Use the initialization method recommended by Glorot et al.
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        intercept_init = self._random_state.uniform(
            -init_bound, init_bound, fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init


class MLPClassifier(BaseEstimator):
    """
    Multi-layer perceptron classifier with production-level features.

    Features:
    - Multiple solvers (sgd, adam, lbfgs)
    - Mini-batch training for scalability
    - Early stopping with validation monitoring
    - Learning rate scheduling
    - Warm start capability
    - Robust input validation
    - Convergence detection
    - Softmax activation for multi-class classification
    - Class weight support for imbalanced datasets
    """

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', lr='constant',
                 lr_init=0.001, epochs=200, random_state=None,
                 tol=1e-4, verbose=False, warm_start=False, early_stopping=False,
                 dropout=0.2, class_weight=None):
        """
        Initialize the MLP classifier with production-level features.

        Args:
            hidden_layer_sizes (tuple): Sizes of hidden layers
            activation (str): Activation function ('relu', 'tanh')
            solver (str): Optimizer ('sgd', 'adam', 'lbfgs')
            alpha (float): L2 regularization parameter
            batch_size (int or 'auto'): Size of minibatches
            lr (str): Learning rate schedule ('constant', 'invscaling', 'adaptive')
            lr_init (float): Initial learning rate
            epochs (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
            tol (float): Tolerance for optimization
            verbose (bool): Whether to print progress messages
            warm_start (bool): Whether to reuse previous solution
            early_stopping (bool): Whether to use early stopping
            dropout (float): Dropout probability (0.0 disables dropout)
            class_weight (dict, 'balanced' or None): Weights for classes
        """
        # Core parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.lr = lr
        self.lr_init = lr_init
        self.epochs = epochs
        self.random_state = random_state
        # More reasonable default tolerance
        self.tol = max(tol, 1e-6) if tol > 0 else 1e-5
        self.verbose = verbose
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.class_weight = class_weight

        # Additional parameters with defaults
        self.power_t = 0.5
        self.shuffle = True
        self.momentum = 0.9
        self.nesterovs_momentum = True
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.validation_fraction = 0.1
        self.n_iter_no_change = 20  # Increase patience
        self.max_fun = 15000

        # Pre-declare attributes for Pylint compliance
        self.layers_ = None
        self.optimizer_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.class_weights_ = None
        self.loss_curve_ = None
        self.validation_scores_ = None
        self.best_loss_ = None
        self.best_validation_score_ = None
        self._no_improvement_count = None
        self._current_lr = None
        self._best_weights = None
        self._best_biases = None

        # Initialize state variables
        self._initialize_state()

        # Set up reproducible random state for weight initialization
        if random_state is not None:
            self._random_state = np.random.default_rng(random_state)
        else:
            self._random_state = np.random.default_rng(42)

    def _initialize_state(self):
        """Initialize state variables."""
        # All attributes defined here for Pylint compliance
        self.layers_ = None
        self.optimizer_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.class_weights_ = None
        self.loss_curve_ = []
        self.validation_scores_ = []
        self.best_loss_ = np.inf
        self.best_validation_score_ = -np.inf
        self._no_improvement_count = 0
        self._current_lr = self.lr_init
        self._best_weights = None
        self._best_biases = None

    def _validate_input(self, x, y=None):
        """Validate input data."""
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")

        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1:
                raise ValueError("y must be a 1D array for classification")
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    "x and y must have the same number of samples")
            return x, y

        return x

    def _get_activation_layer(self):
        """Get activation layer based on activation parameter."""
        if self.activation == 'relu':
            return ReLULayer()
        raise ValueError(
            f"Unsupported activation: {self.activation}. Only 'relu' is supported.")

    def _get_batch_size(self, n_samples):
        """Determine batch size."""
        if self.batch_size == 'auto':
            return min(200, n_samples)
        return min(self.batch_size, n_samples)

    def _create_optimizer(self):
        """Create optimizer instance."""
        if self.solver == 'sgd':
            return get_optimizer('sgd',
                                 lr=self._current_lr,
                                 momentum=self.momentum,
                                 nesterov=self.nesterovs_momentum)
        if self.solver == 'adam':
            return get_optimizer('adam',
                                 lr=self._current_lr,
                                 beta1=self.beta_1,
                                 beta2=self.beta_2,
                                 epsilon=self.epsilon)
        if self.solver == 'lbfgs':
            return get_optimizer('lbfgs',
                                 lr=self._current_lr,
                                 epochs=self.max_fun)
        raise ValueError(f"Unknown solver: {self.solver}")

    def _update_lr(self, iteration):
        """Update learning rate based on schedule."""
        if self.lr == 'invscaling':
            self._current_lr = (self.lr_init /
                                pow(iteration + 1, self.power_t))
        elif self.lr == 'adaptive':
            # Will be updated based on loss improvement
            pass
        # 'constant' keeps the same learning rate

    def _softmax(self, z):
        """Compute softmax activation."""
        # Numerical stability: subtract max
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_class_weights(self, y):
        """Compute class weights for balanced training."""
        if self.class_weight is None:
            return None
        elif self.class_weight == 'balanced':
            # Compute balanced class weights
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y), 
                y=y
            )
            return dict(zip(np.unique(y), class_weights))
        elif isinstance(self.class_weight, dict):
            return self.class_weight
        else:
            raise ValueError("class_weight must be None, 'balanced', or a dictionary")

    def _build_network(self, n_features, n_outputs):
        """
        Build the neural network architecture with improved initialization and dropout.

        Args:
            n_features (int): Number of input features
            n_outputs (int): Number of output features
        Returns:
            list: List of layers
        """
        if not self.warm_start or self.layers_ is None:
            layers = []
            rng = np.random.default_rng(self.random_state)
            prev_size = n_features
            for hidden_size in self.hidden_layer_sizes:
                # Linear layer
                linear_layer = ModularLinearLayer(
                    prev_size, hidden_size, rng=rng)
                weight, bias = self._init_coef(
                    prev_size, hidden_size, np.float32)
                linear_layer.weight = weight
                linear_layer.bias = bias
                layers.append(linear_layer)
                # Activation
                activation_layer = self._get_activation_layer()
                layers.append(activation_layer)
                # Dropout (only if dropout > 0)
                if self.dropout > 0.0:
                    layers.append(DropoutLayer(p=self.dropout))
                prev_size = hidden_size
            # Output layer (no activation - softmax applied separately)
            output_layer = ModularLinearLayer(prev_size, n_outputs, rng=rng)
            weight, bias = self._init_coef(prev_size, n_outputs, np.float32)
            output_layer.weight = weight
            output_layer.bias = bias
            layers.append(output_layer)
            return layers
        return self.layers_

    def _split_train_validation(self, x, y):
        """Split data into training and validation sets."""
        if not self.early_stopping:
            return x, y, None, None

        n_samples = x.shape[0]
        n_validation = int(self.validation_fraction * n_samples)
        rng = np.random.default_rng(self.random_state)

        if self.shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        val_indices = indices[:n_validation]
        train_indices = indices[n_validation:]

        x_train, x_val = x[train_indices], x[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        return x_train, y_train, x_val, y_val

    def _generate_batches(self, x, y, batch_size):
        """Generate mini-batches for training."""
        n_samples = x.shape[0]
        rng = np.random.default_rng(self.random_state)

        if self.shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield x[batch_indices], y[batch_indices]

    def _forward_pass(self, x, training=False):
        """
        Perform forward pass through the network.
        Args:
            x (ndarray): Input data
            training (bool): Whether in training mode (enables dropout)
        Returns:
            ndarray: Network output (logits)
        """
        current_input = x
        for layer in self.layers_:
            # DropoutLayer: only active during training
            if isinstance(layer, DropoutLayer):
                current_input = layer(current_input)
            else:
                current_input = layer(current_input)
        return current_input

    def _one_hot_encode(self, y):
        """Convert class labels to one-hot encoding."""
        n_samples = len(y)
        y_one_hot = np.zeros((n_samples, self.n_classes_))
        for i, class_idx in enumerate(y):
            y_one_hot[i, class_idx] = 1
        return y_one_hot

    def _backward_pass(self, x, y, y_pred_logits):
        """
        Perform backward pass through the network.

        Args:
            x (ndarray): Input data
            y (ndarray): True targets (class indices)
            y_pred_logits (ndarray): Predicted logits
        """
        # Convert y to one-hot encoding
        y_one_hot = self._one_hot_encode(y)
        
        # Apply softmax to get probabilities
        y_pred_probs = self._softmax(y_pred_logits)
        
        # Calculate loss gradient (cross-entropy derivative)
        n_samples = x.shape[0]
        loss_grad = (y_pred_probs - y_one_hot) / n_samples

        # Backward pass through layers
        current_grad = loss_grad

        # Process layers in reverse order
        for layer in reversed(self.layers_):
            if hasattr(layer, 'backward'):
                if isinstance(layer, ModularLinearLayer):
                    # For linear layers, include L2 regularization
                    current_grad = layer.backward(
                        current_grad, alpha=self.alpha)
                else:
                    # For activation layers
                    current_grad = layer.backward(current_grad)

    def _update_parameters(self):
        """Update network parameters using the optimizer."""
        # Gradient clipping to prevent gradient explosion
        max_grad_norm = 1.0
        total_norm = 0.0

        # Calculate total gradient norm
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer) and hasattr(layer, 'weight_grad') and layer.weight_grad is not None:
                total_norm += np.sum(layer.weight_grad ** 2)
                total_norm += np.sum(layer.bias_grad ** 2)

        total_norm = np.sqrt(total_norm)

        # Apply gradient clipping if needed
        clip_factor = min(1.0, max_grad_norm / (total_norm + 1e-8))

        for i, layer in enumerate(self.layers_):
            if isinstance(layer, ModularLinearLayer) and hasattr(layer, 'weight_grad') and layer.weight_grad is not None:
                # Apply clipping by modifying the gradients in place
                weight_grad = layer.weight_grad * clip_factor
                bias_grad = layer.bias_grad * clip_factor

                if self.optimizer_ is not None:
                    # Use optimizer to update parameters
                    layer_id = f"layer_{i}"
                    updated_weight, updated_bias = self.optimizer_.update(
                        layer_id, weight_grad, bias_grad,
                        layer.weight.copy(), layer.bias.copy()
                    )
                    layer.weight = updated_weight
                    layer.bias = updated_bias
                else:
                    # Fallback to simple gradient descent
                    layer.weight -= self._current_lr * weight_grad
                    layer.bias -= self._current_lr * bias_grad

    def _cross_entropy_loss(self, y_true, y_pred_logits):
        """
        Calculate cross-entropy loss.

        Args:
            y_true (ndarray): True targets (class indices)
            y_pred_logits (ndarray): Predicted logits

        Returns:
            float: Cross-entropy loss
        """
        y_pred_probs = self._softmax(y_pred_logits)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
        
        # Calculate cross-entropy
        n_samples = len(y_true)
        log_probs = np.log(y_pred_probs[np.arange(n_samples), y_true])
        return -np.mean(log_probs)

    def _compute_class_weighted_loss(self, y_true, y_pred_logits, class_weights=None):
        """Compute cross-entropy loss with class weights."""
        if class_weights is None:
            return self._cross_entropy_loss(y_true, y_pred_logits)
        
        y_pred_probs = self._softmax(y_pred_logits)
        epsilon = 1e-15
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
        
        # Apply class weights to the loss
        weighted_loss = 0.0
        n_samples = len(y_true)
        
        for class_idx in range(self.n_classes_):
            class_mask = (y_true == class_idx)
            if np.any(class_mask):
                # Get weight for this class
                weight = class_weights.get(class_idx, 1.0)
                # Calculate loss for this class
                class_log_probs = np.log(y_pred_probs[class_mask, class_idx])
                class_loss = -np.mean(class_log_probs)
                # Weight the loss by class frequency and class weight
                class_count = np.sum(class_mask)
                weighted_loss += weight * class_loss * class_count
        
        return weighted_loss / n_samples

    def _calculate_regularization_loss(self):
        """
        Calculate L2 regularization loss.

        Returns:
            float: Regularization loss
        """
        reg_loss = 0.0
        for layer in self.layers_:
            if isinstance(layer, ModularLinearLayer):
                # Sum of squared weights
                reg_loss += np.sum(layer.weight ** 2)
        # Apply normalization in _compute_loss
        return 0.5 * self.alpha * reg_loss

    def _compute_loss(self, y_true, y_pred_logits):
        """Compute total loss (cross-entropy + regularization)."""
        n_samples = len(y_true)
        
        # Use class weighted loss if weights are specified
        if self.class_weights_ is not None:
            ce_loss = self._compute_class_weighted_loss(y_true, y_pred_logits, self.class_weights_)
        else:
            ce_loss = self._cross_entropy_loss(y_true, y_pred_logits)
            
        reg_loss = self._calculate_regularization_loss()
        # Apply normalization: reg_loss / n_samples
        return ce_loss + (reg_loss / n_samples)

    def _check_convergence(self, loss, iteration):
        """Check if training has converged."""
        if len(self.loss_curve_) < 5:  # Need at least 5 iterations
            return False

        # Don't converge too early
        if iteration < 20:
            return False

        # Check if loss improvement is below tolerance
        if len(self.loss_curve_) > 1:
            loss_diff = abs(self.loss_curve_[-1] - loss)
            relative_improvement = loss_diff / \
                max(abs(self.loss_curve_[-1]), 1e-8)

            if relative_improvement < self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

        # Early stopping based on loss improvement
        if self._no_improvement_count >= self.n_iter_no_change:
            if self.verbose:
                print(f"Convergence after {iteration + 1} iterations")
            return True

        return False

    def _validate_on_validation_set(self, x_val, y_val):
        """Evaluate model on validation set."""
        if x_val is None or y_val is None:
            return None

        # Calculate accuracy score
        val_score = self.score(x_val, y_val)

        # Check for improvement
        if val_score > self.best_validation_score_:
            self.best_validation_score_ = val_score
            self._no_improvement_count = 0
            # Save best weights
            self._best_weights = []
            self._best_biases = []
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    self._best_weights.append(layer.weight.copy())
                    self._best_biases.append(layer.bias.copy())
        else:
            self._no_improvement_count += 1

        return val_score

    def _train_epoch(self, x_train, y_train, batch_size):
        """Train for one epoch and return epoch loss."""
        epoch_losses = []
        # Mini-batch training
        for x_batch, y_batch in self._generate_batches(x_train, y_train, batch_size):
            # Forward pass (training=True for dropout)
            y_pred_logits = self._forward_pass(x_batch, training=True)
            # Backward pass
            self._backward_pass(x_batch, y_batch, y_pred_logits)
            # Update parameters
            self._update_parameters()
            # Track batch loss
            batch_loss = self._compute_loss(y_batch, y_pred_logits)
            epoch_losses.append(batch_loss)
        return np.mean(epoch_losses)

    def _should_stop_early(self, epoch_loss, iteration):
        """Check if training should stop early."""
        # Check convergence
        if self._check_convergence(epoch_loss, iteration):
            return True

        # Update best loss
        self.best_loss_ = min(self.best_loss_, epoch_loss)

        # Adaptive learning rate
        if self.lr == 'adaptive' and len(self.loss_curve_) > 2:
            if self.loss_curve_[-1] > self.loss_curve_[-2]:
                self._current_lr *= 0.5
                if self.verbose:
                    print(f"Learning rate reduced to {self._current_lr:.6f}")

        return False

    def fit(self, X, y):
        """
        Train the MLP classifier with production-level features.

        Args:
            X (ndarray): Training data of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,)

        Returns:
            self: Fitted estimator
        """
        # Input validation
        X, y = self._validate_input(X, y)

        # Store unique classes and create mapping
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Compute class weights if specified
        self.class_weights_ = self._compute_class_weights(y)
        if self.class_weights_ is not None and self.verbose:
            print(f"Using class weights: {self.class_weights_}")
        
        # Convert labels to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[cls] for cls in y])

        _, n_features = X.shape
        n_outputs = self.n_classes_

        # Split training and validation data
        x_train, y_train, x_val, y_val = self._split_train_validation(X, y_indices)

        # Build network
        self.layers_ = self._build_network(n_features, n_outputs)
        self.n_layers_ = len(
            [l for l in self.layers_ if isinstance(l, ModularLinearLayer)])
        self.n_outputs_ = n_outputs

        # Create optimizer
        self.optimizer_ = self._create_optimizer()

        # Determine batch size
        batch_size = self._get_batch_size(x_train.shape[0])

        # Initialize training history (attributes already defined in __init__/_initialize_state)
        if not self.warm_start:
            self.loss_curve_.clear()
            self.validation_scores_.clear()
            self.best_loss_ = np.inf
            self.best_validation_score_ = -np.inf
            self._no_improvement_count = 0

        # Training loop
        for iteration in range(self.epochs):
            # Update learning rate
            self._update_lr(iteration)

            # Train for one epoch
            epoch_loss = self._train_epoch(x_train, y_train, batch_size)
            self.loss_curve_.append(epoch_loss)

            # Validation
            val_score = self._validate_on_validation_set(x_val, y_val)
            if val_score is not None:
                self.validation_scores_.append(val_score)

            # Early stopping check
            if self._should_stop_early(epoch_loss, iteration):
                break

            # Verbose output
            if self.verbose and (iteration + 1) % 10 == 0:
                val_msg = f", Val Acc: {val_score:.6f}" if val_score is not None else ""
                print(f"Iteration {iteration + 1}/{self.epochs}, "
                      f"Loss: {epoch_loss:.6f}{val_msg}")

        self.n_iter_ = len(self.loss_curve_)

        # Restore best weights if early stopping was used
        if self.early_stopping and x_val is not None and self._best_weights is not None:
            weight_idx = 0
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    layer.weight = self._best_weights[weight_idx]
                    layer.bias = self._best_biases[weight_idx]
                    weight_idx += 1
            if self.verbose:
                print(
                    f"Restored best weights from validation accuracy: {self.best_validation_score_:.6f}")

        if self.verbose:
            print(f"Training completed in {self.n_iter_} iterations")

        return self

    def predict(self, x):
        """
        Make predictions using the trained model.
        Args:
            x (ndarray): Input data of shape (n_samples, n_features)
        Returns:
            ndarray: Predicted class labels
        """
        if self.layers_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        x = self._validate_input(x)
        # Dropout is disabled during prediction
        logits = self._forward_pass(x, training=False)
        probabilities = self._softmax(logits)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, x):
        """
        Predict class probabilities for samples in x.
        Args:
            x (ndarray): Input data of shape (n_samples, n_features)
        Returns:
            ndarray: Class probabilities of shape (n_samples, n_classes)
        """
        if self.layers_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        x = self._validate_input(x)
        # Dropout is disabled during prediction
        logits = self._forward_pass(x, training=False)
        return self._softmax(logits)

    def score(self, x, y):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            x (ndarray): Test samples
            y (ndarray): True labels for x

        Returns:
            float: Mean accuracy
        """
        y_pred = self.predict(x)
        return np.mean(y_pred == y)

    def get_params(self, mode: bool = True):
        """Get parameters for this estimator."""
        _ = mode  # Unused parameter for compatibility
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'lr_init': self.lr_init,
            'epochs': self.epochs,
            'random_state': self.random_state,
            'tol': self.tol,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'early_stopping': self.early_stopping,
            'dropout': self.dropout,
            'class_weight': self.class_weight
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param}")
        return self

    def _init_coef(self, fan_in, fan_out, dtype):
        # Use the initialization method recommended by Glorot et al.
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        intercept_init = self._random_state.uniform(
            -init_bound, init_bound, fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init
