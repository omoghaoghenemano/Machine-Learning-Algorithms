"""
Neural networks module for machine learning algorithms.
"""
from .layers import (
    Layer,
    ModularLinearLayer,
    SigmoidLayer,
    TanhLayer,
    ReLULayer,
    SoftmaxLayer
)
from ._mlp import (MLPRegressor, MLPClassifier)
from ._cnn import (
    CNNClassifier,
    DenseLayer
)
from .optimizers import (
    SGDOptimizer,
    AdamOptimizer,
    LBFGSOptimizer,
    get_optimizer
)

__all__ = [
    'Layer',
    'ModularLinearLayer',
    'SigmoidLayer',
    'TanhLayer',
    'ReLULayer',
    'SoftmaxLayer',
    'MLPRegressor',
    'CNNClassifier',
    'SGDOptimizer',
    'AdamOptimizer',
    'LBFGSOptimizer',
    'get_optimizer',
    'DenseLayer'
]
