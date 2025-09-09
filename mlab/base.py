# pylint: disable=missing-docstring
from abc import abstractmethod
from typing import Any, TypeVar
import numpy as np

# Define a type variable that's used correctly
T = TypeVar("T", bound="BaseEstimator")


# pylint: disable=too-many-instance-attributes, invalid-name line-too-long missing-docstring
class BaseEstimator:
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param y: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return:
        """
        raise NotImplementedError

    def get_params(self, mode: str = "all") -> Any:
        """
        Get parameters for this estimator.

        :param mode: Specifies which parameters to return. Options are:
            - "all": Return all parameters.
            - "trainable": Return only trainable parameters (e.g., weights).
            - "non_trainable": Return only non-trainable parameters (e.g., configuration settings).
        :return: Dictionary of parameter names mapped to their values.
        """
        if mode == "all":
            return self.__dict__
        if mode == "trainable":
            return {k: v for k, v in self.__dict__.items() if k == "weights"}
        if mode == "non_trainable":
            return {k: v for k, v in self.__dict__.items() if k != "weights"}

        raise ValueError(
            f"Invalid mode '{mode}'. Choose from 'all', 'trainable', or 'non_trainable'."
        )


class BaseTransformer(BaseEstimator):
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :return: numpy array of shape (N, d') with N being the number of samples and d' being the number of feature dimensions after transformation
        """
        raise NotImplementedError


class BaseClassifier(BaseEstimator):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: np array of shape (N, d)
        :return:
        """
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param y: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return: accuracy
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class BaseRegressor(BaseEstimator):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: np array of shape (N, d)
        :return:
        """
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param y: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return: R2 score
        """
        y_pred = self.predict(X)

        ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))

        if ss_tot == 0:
            return 0.0  # Avoid division by zero

        ss_res: float = float(np.sum((y - y_pred) ** 2))

        return 1.0 - (ss_res / ss_tot)
