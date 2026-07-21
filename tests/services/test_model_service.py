import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.services.model_service import ModelService, _trained_models
from app.schemas.model import ModelTrainRequest, ModelPredictRequest
from app.schemas.dataset import DatasetRequest

@pytest.fixture(autouse=True)
def isolate_trained_models():
    with patch.dict('app.services.model_service._trained_models', clear=True):
        yield

@pytest.fixture
def model_service():
    return ModelService()

@pytest.fixture
def base_dataset_request():
    return DatasetRequest(
        dataset_name="circle",
        n_samples=100,
        noise=0.1,
        test_size=0.2,
        normalization="standard"
    )

MODELS = ['dt', 'knn', 'lda', 'lr', 'nb', 'qda', 'svm']

class TestModelService:
    
    @pytest.mark.parametrize("model_name", MODELS)
    def test_train_and_predict_all_models(self, model_service, base_dataset_request, model_name):
        train_req = ModelTrainRequest(
            model_name=model_name,
            dataset=base_dataset_request,
            params={}
        )
        response_train = model_service.train_model(train_req)
        
        assert response_train["status"] == "sucesso"
        assert response_train["modelo"] == model_name
        assert model_name in _trained_models

        X_test = [[0.5, 0.5], [-0.5, -0.5]]
        predict_req = ModelPredictRequest(X=X_test)
        
        response_predict = model_service.predict_model(model_name=model_name, request=predict_req)
        
        assert hasattr(response_predict, "predictions")
        assert hasattr(response_predict, "probabilities")
        assert len(response_predict.predictions) == 2

    @pytest.mark.parametrize("model_name, custom_params", [
        ("knn", {"n_neighbors": 3}),
        ("svm", {"kernel": "linear"}),
        ("lda", {"solver": "lsqr"})
    ])
    def test_build_model_with_specific_parameters(self, model_service, base_dataset_request, model_name, custom_params):
        train_req = ModelTrainRequest(
            model_name=model_name,
            dataset=base_dataset_request,
            params=custom_params
        )
        model_service.train_model(train_req)
        assert model_name in _trained_models

    def test_train_invalid_model_name(self, model_service, base_dataset_request):
        train_req = ModelTrainRequest(
            model_name="modelo_inexistente",
            dataset=base_dataset_request,
            params={}
        )
        
        with pytest.raises(ValueError, match="not in catalog"):
            model_service.train_model(train_req)

    def test_predict_untrained_model(self, model_service):
        predict_req = ModelPredictRequest(X=[[0.0, 0.0]])
        
        with pytest.raises(ValueError, match="sem treino prévio"):
            model_service.predict_model(model_name="svm", request=predict_req)

    def test_predict_model_without_proba_attribute(self, model_service):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.side_effect = AttributeError()
        
        _trained_models["mock_algo"] = mock_model
        
        predict_req = ModelPredictRequest(X=[[0.1, 0.1], [0.9, 0.9]])
        response = model_service.predict_model(model_name="mock_algo", request=predict_req)
        
        assert response.predictions == [1, 0]
        assert response.probabilities is None