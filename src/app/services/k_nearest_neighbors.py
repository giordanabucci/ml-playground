import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from app.services.base_service import BaseModelService

class KNNService(BaseModelService):
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)