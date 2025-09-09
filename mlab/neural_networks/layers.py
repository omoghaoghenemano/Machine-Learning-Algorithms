"""
Neural network layers implementation.
"""
import numpy as np

class Layer:
    """Base class for all neural network layers."""

    def forward(self, input_data):
        """Forward pass through the layer."""
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """Backward pass through the layer."""
        raise NotImplementedError


# Optimized activation functions )
def inplace_relu(x):
    """Compute the rectified linear unit function inplace."""
    np.maximum(x, 0, out=x)


def inplace_tanh(x):
    """Compute the hyperbolic tan function inplace."""
    np.tanh(x, out=x)


def inplace_relu_derivative(z, delta):
    """Apply the derivative of the relu function inplace."""
    delta[z == 0] = 0


def inplace_tanh_derivative(z, delta):
    """Apply the derivative of the hyperbolic tanh function inplace."""
    delta *= 1 - z**2


class ModularLinearLayer:
    """
    Modular linear layer of a neural network with L2 regularization term and bias.
    """

    def __init__(self, in_features, out_features, rng=None):
        """
        Initialize the linear layer with Xavier/Glorot initialization.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            rng (np.random.Generator, optional): Random number generator
        """
        self.in_features = in_features
        self.out_features = out_features

        if rng is None:
            rng = np.random.default_rng(42)

        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = rng.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros(out_features)

        # Gradients (set during backward pass)
        self._weight_grad = None
        self._bias_grad = None
        self._prev_input = None

    @property
    def weight_grad(self):
        """Get weight gradient."""
        return self._weight_grad

    @property
    def bias_grad(self):
        """Get bias gradient."""
        return self._bias_grad

    def __call__(self, x):
        """
        Forward pass: x.dot(weight) + bias
        
        Args:
            x (ndarray): Input data of shape (batch_size, in_features)
            
        Returns:
            ndarray: Output of shape (batch_size, out_features)
        """
        self._prev_input = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, upstream_grad, alpha=None):
        """
        Backward pass with L2 regularization.
        
        Args:
            upstream_grad (ndarray): Gradient from the next layer
            alpha (float, optional): L2 regularization parameter
            
        Returns:
            ndarray: Gradient with respect to input
        """
        # Gradient with respect to weights
        self._weight_grad = np.dot(self._prev_input.T, upstream_grad)

        # Add L2 regularization term if alpha is provided
        if alpha is not None:
            self._weight_grad += alpha * self.weight
        # Gradient with respect to bias
        self._bias_grad = np.sum(upstream_grad, axis=0)
        # Gradient with respect to input
        input_grad = np.dot(upstream_grad, self.weight.T)

        return input_grad

    def update(self, lr):
        """
        Update parameters using gradients.
        
        Args:
            lr (float): Learning rate
        """
        self.weight -= lr * self._weight_grad
        self.bias -= lr * self._bias_grad

    def __repr__(self):
        """String representation of the layer."""
        return (f"ModularLinearLayer(in_features={self.in_features}, "
                f"out_features={self.out_features})")


class SigmoidLayer:
    """Sigmoid activation layer."""

    def __init__(self):
        """Initialize the sigmoid layer."""
        self._prev_result = None

    def __call__(self, x):
        """
        Forward pass: sigmoid(x)

        Args:
            x (ndarray): Input data

        Returns:
            ndarray: Sigmoid activation output
        """
        # Clip input to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        self._prev_result = 1 / (1 + np.exp(-x_clipped))
        return self._prev_result

    def backward(self, upstream_grad):
        """
        Backward pass: sigmoid derivative

        Args:
            upstream_grad (ndarray): Gradient from the next layer

        Returns:
            ndarray: Gradient with respect to input
        """
        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = self._prev_result * (1 - self._prev_result)
        return upstream_grad * sigmoid_grad


class TanhLayer:
    """Tanh activation layer with-compatible efficiency."""

    def __init__(self):
        """Initialize the tanh layer."""
        self._prev_result = None

    def __call__(self, x):
        """
        Forward pass: tanh(x) with-compatible efficiency

        Args:
            x (ndarray): Input data

        Returns:
            ndarray: Tanh activation output
        """
        # Use efficient in-place operation
        output = x.copy()
        np.tanh(output, out=output)
        self._prev_result = output
        return output

    def backward(self, upstream_grad):
        """
        Backward pass: tanh derivative -compatible)

        Args:
            upstream_grad (ndarray): Gradient from the next layer

        Returns:
            ndarray: Gradient with respect to input
        """
        # Tanh derivative: 1 - tanh²(x), using in-place operations
        grad_input = upstream_grad.copy()
        grad_input *= (1 - self._prev_result ** 2)
        return grad_input


class ReLULayer:
    """ReLU activation layer with-compatible efficiency."""

    def __init__(self):
        """Initialize the ReLU layer."""
        self._prev_input = None

    def __call__(self, x):
        """
        Forward pass: ReLU(x) = max(0, x)

        Args:
            x (ndarray): Input data

        Returns:
            ndarray: ReLU activation output
        """
        self._prev_input = x.copy()  # Store for backward pass
        # Use efficient in-place operation
        output = x.copy()
        np.maximum(output, 0, out=output)
        return output

    def backward(self, upstream_grad):
        """
        Backward pass: ReLU derivative -compatible)

        Args:
            upstream_grad (ndarray): Gradient from the next layer

        Returns:
            ndarray: Gradient with respect to input
        """
        # ReLU derivative: gradient is 0 where input was <= 0
        grad_input = upstream_grad.copy()
        grad_input[self._prev_input <= 0] = 0
        return grad_input


class SoftmaxLayer:
    """Softmax activation layer."""

    def __init__(self):
        """Initialize the softmax layer."""
        self._prev_result = None

    def __call__(self, x):
        """
        Forward pass: softmax(x)

        Args:
            x (ndarray): Input data of shape (batch_size, num_classes)

        Returns:
            ndarray: Softmax activation output
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self._prev_result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self._prev_result

    def backward(self, upstream_grad):
        """
        Backward pass: softmax derivative

        Args:
            upstream_grad (ndarray): Gradient from the next layer

        Returns:
            ndarray: Gradient with respect to input
        """
        # For softmax, the gradient computation is more complex
        # This implementation assumes cross-entropy loss (common case)
        return upstream_grad


class DropoutLayer:
    """Dropout layer for regularization during training."""

    def __init__(self, p=0.5):
        """
        Initialize dropout layer.
        
        Args:
            p (float): Dropout probability (0.0 to 1.0)
        """
        self.p = p
        self.mask = None
        self.training = True

    def __call__(self, x):
        """
        Forward pass with dropout.
        
        Args:
            x (ndarray): Input data
            
        Returns:
            ndarray: Output with dropout applied (if training)
        """
        if not self.training or self.p == 0.0:
            return x
        rng = np.random.default_rng()
        self.mask = rng.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
        return x * self.mask

    def backward(self, grad_output):
        """
        Backward pass with dropout mask.
        
        Args:
            grad_output (ndarray): Gradient from next layer
            
        Returns:
            ndarray: Gradient with dropout mask applied
        """
        if not self.training or self.p == 0.0:
            return grad_output
        return grad_output * self.mask

    def eval(self):
        """Set layer to evaluation mode (no dropout)."""
        self.training = False

    def train(self):
        """Set layer to training mode (dropout enabled)."""
        self.training = True
