"""
Advanced optimizers for neural networks.
"""
import numpy as np


class SGDOptimizer:
    """
    Stochastic Gradient Descent optimizer with momentum and Nesterov acceleration.
    """

    def __init__(self, lr=0.01, momentum=0.9, nesterov=True):
        """
        Initialize SGD optimizer.

        Args:
            lr (float): Learning rate
            momentum (float): Momentum factor
            nesterov (bool): Whether to apply Nesterov momentum
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def update(self, layer_id, weight_grad, bias_grad, weights, bias):
        """
        Update weights using SGD with momentum.

        Args:
            layer_id (str): Unique identifier for the layer
            weight_grad (ndarray): Weight gradients
            bias_grad (ndarray): Bias gradients
            weights (ndarray): Current weights
            bias (ndarray): Current bias

        Returns:
            tuple: Updated weights and bias
        """
        # Initialize velocities if not exists
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'weight_velocity': np.zeros_like(weights),
                'bias_velocity': np.zeros_like(bias)
            }

        v_weight = self.velocities[layer_id]['weight_velocity']
        v_bias = self.velocities[layer_id]['bias_velocity']

        # Update velocities
        v_weight = self.momentum * v_weight - self.lr * weight_grad
        v_bias = self.momentum * v_bias - self.lr * bias_grad

        if self.nesterov:
            # Nesterov momentum
            weights += self.momentum * v_weight - self.lr * weight_grad
            bias += self.momentum * v_bias - self.lr * bias_grad
        else:
            # Standard momentum
            weights += v_weight
            bias += v_bias

        # Store updated velocities
        self.velocities[layer_id]['weight_velocity'] = v_weight
        self.velocities[layer_id]['bias_velocity'] = v_bias

        return weights, bias


class AdamOptimizer:
    """
    Adam optimizer implementation.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.

        Args:
            lr (float): Learning rate
            beta1 (float): Exponential decay rate for first moment
            beta2 (float): Exponential decay rate for second moment
            epsilon (float): Small constant for numerical stability
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}

    def update(self, layer_id, weight_grad, bias_grad, weights, bias):
        """
        Update weights using Adam optimization.

        Args:
            layer_id (str): Unique identifier for the layer
            weight_grad (ndarray): Weight gradients
            bias_grad (ndarray): Bias gradients
            weights (ndarray): Current weights
            bias (ndarray): Current bias

        Returns:
            tuple: Updated weights and bias
        """
        # Initialize moments if not exists
        if layer_id not in self.moments:
            self.moments[layer_id] = {
                'm_weight': np.zeros_like(weights),
                'v_weight': np.zeros_like(weights),
                'm_bias': np.zeros_like(bias),
                'v_bias': np.zeros_like(bias),
                't': 0  # Time step per layer
            }

        moments = self.moments[layer_id]
        moments['t'] += 1
        t = moments['t']

        # Update biased first moment estimate
        moments['m_weight'] = self.beta1 * \
            moments['m_weight'] + (1 - self.beta1) * weight_grad
        moments['m_bias'] = self.beta1 * \
            moments['m_bias'] + (1 - self.beta1) * bias_grad

        # Update biased second raw moment estimate
        moments['v_weight'] = self.beta2 * moments['v_weight'] + \
            (1 - self.beta2) * (weight_grad ** 2)
        moments['v_bias'] = self.beta2 * moments['v_bias'] + \
            (1 - self.beta2) * (bias_grad ** 2)

        # Compute bias-corrected first moment estimate
        m_weight_corrected = moments['m_weight'] / (1 - self.beta1 ** t)
        m_bias_corrected = moments['m_bias'] / (1 - self.beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        v_weight_corrected = moments['v_weight'] / (1 - self.beta2 ** t)
        v_bias_corrected = moments['v_bias'] / (1 - self.beta2 ** t)

        # Update parameters
        weights -= self.lr * m_weight_corrected / \
            (np.sqrt(v_weight_corrected) + self.epsilon)
        bias -= self.lr * m_bias_corrected / \
            (np.sqrt(v_bias_corrected) + self.epsilon)

        return weights, bias


class LBFGSOptimizer:
    """
    Limited-memory BFGS optimizer (simplified implementation).
    """

    def __init__(self, lr=1.0, max_iter=20):
        """
        Initialize L-BFGS optimizer.

        Args:
            lr (float): Learning rate
            max_iter (int): Maximum iterations
        """
        self.lr = lr
        self.max_iter = max_iter
        self.history = []

    def update(self, _layer_id, weight_grad, bias_grad, weights, bias):
        """
        Update weights using L-BFGS approximation.

        Args:
            _layer_id (str): Unique identifier for the layer (unused in this simplified implementation)
            weight_grad (ndarray): Weight gradients
            bias_grad (ndarray): Bias gradients
            weights (ndarray): Current weights
            bias (ndarray): Current bias

        Returns:
            tuple: Updated weights and bias
        """
        # Simple L-BFGS approximation - using scaled gradient descent
        # In a full implementation, this would maintain curvature information
        weights -= self.lr * weight_grad
        bias -= self.lr * bias_grad

        return weights, bias


def get_optimizer(solver='adam', **kwargs):
    """
    Factory function to get optimizer instances.

    Args:
        solver (str): Optimizer type ('sgd', 'adam', 'lbfgs')
        **kwargs: Optimizer-specific parameters

    Returns:
        Optimizer instance
    """
    if solver == 'sgd':
        return SGDOptimizer(**kwargs)
    elif solver == 'adam':
        return AdamOptimizer(**kwargs)
    elif solver == 'lbfgs':
        return LBFGSOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")
