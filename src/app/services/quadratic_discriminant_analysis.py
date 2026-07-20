import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from app.services.base_service import BaseModelService

class QDAService(BaseModelService):
    def __init__(self):
        super().__init__()
        self.model = QuadraticDiscriminantAnalysis()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    