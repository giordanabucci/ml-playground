import numpy as np

from app.schemas.model import ModelTrainRequest, ModelPredictRequest, ModelPredictResponse
from app.schemas.dataset import DatasetRequest

from app.services.dataset_service import DatasetService
from app.services.decision_tree import DTService
from app.services.k_nearest_neighbors import KNNService
from app.services.linear_discriminat_analysis import LDAService
from app.services.logistic_regression import LogisticRegressionService
from app.services.naive_bayes import NBService
from app.services.quadratic_discriminant_analysis import QDAService
from app.services.support_vector_machines import SVMService


_trained_models = {}

class ModelService:
    def __init__(self):
        self.dataset_service = DatasetService()
        self.model_catalog = {
            'dt': DTService,
            'knn': KNNService,
            'lda': LDAService,
            'lr': LogisticRegressionService,
            'nb': NBService,
            'qda': QDAService,
            'svm': SVMService
        }
    
    def _build_model(self, model_name: str, params: dict):
        if model_name == 'knn':
            return KNNService(n_neighbors=params.get('n_neighbors', 5))
        
        if model_name == 'svm':
            return SVMService(kernel=params.get('kernel', 'rbf'))
        
        if model_name == 'lda':
            return LDAService(solver=params.get('solver', 'svd'))
        
        model_class = self.model_catalog.get(model_name)
        if not model_class:
            raise ValueError(f'Algorithm {model_name} not in catalog')
        
        return model_class()
    
    def train_model(self, request: ModelTrainRequest) -> dict:
        dataset_response = self.dataset_service.get_dataset(request.dataset)
        model_instance = self._build_model(request.model_name, request.params)

        X_train = np.array(dataset_response.X_train)
        y_train = np.array(dataset_response.y_train)

        model_instance.train(X_train, y_train)

        _trained_models[request.model_name] = model_instance

        return {"status": "sucesso", "modelo": request.model_name}
    
    def predict_model(self, model_name: str, request: ModelPredictRequest) -> ModelPredictResponse:
        model_instance = _trained_models.get(model_name)
        if not model_instance:
            raise ValueError(f"Modelo {model_name} sem treino prévio")
        
        X_np = np.array(request.X)
        predictions = model_instance.predict(X_np).tolist()

        try:
            probabilities = model_instance.predict_proba(X_np).tolist()
        except AttributeError:
            probabilities = None

        return ModelPredictResponse(predictions=predictions, probabilities=probabilities)
