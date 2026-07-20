import numpy as np
from sklearn.svm import SVC
from app.services.base_service import BaseModelService

class SVMService(BaseModelService):
    def __init__(self, kernel: str = 'rbf'):
        super().__init__()
        self.model = SVC(kernel=kernel, probability=True)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)