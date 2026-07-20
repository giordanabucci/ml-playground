import numpy as np
from sklearn.tree import DecisionTreeClassifier
from app.core.config import get_settings
from app.services.base_service import BaseModelService

class DTService(BaseModelService):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(random_state=get_settings().random_state)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)