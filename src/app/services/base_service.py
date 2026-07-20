from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

class BaseModelService(ABC):
    def __init__(self):
        self.model = BaseEstimator | None = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass