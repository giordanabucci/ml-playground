import pytest
import numpy as np
import pydantic_core
from app.schemas.dataset import DatasetRequest
from app.services.dataset_service import DatasetService

DATASETS = ["circle", "diagonal", "gaussian", "linear", "spiral", "xor"]
NORMALIZATIONS = ["none", "standard", "minmax"]
TEST_SIZES = [0.1, 0.2, 0.5, 0.8]

class TestDatasetService:
    @pytest.fixture
    def service(self):
        return DatasetService()

    @pytest.mark.parametrize("dataset_name", DATASETS)
    @pytest.mark.parametrize("normalization", NORMALIZATIONS)
    @pytest.mark.parametrize("test_size", TEST_SIZES)
    def test_get_dataset_combinations(self, service, dataset_name, normalization, test_size):
        n_samples = 1000
        request = DatasetRequest(
            dataset_name=dataset_name,
            n_samples=n_samples,
            noise=0.1,
            test_size=test_size,
            normalization=normalization
        )

        response = service.get_dataset(request)

        # Verifica os tipos de retorno
        assert isinstance(response.X_train, list)
        assert isinstance(response.X_test, list)
        assert isinstance(response.y_train, list)
        assert isinstance(response.y_test, list)

        # Calcula e verifica o tamanho das matrizes
        expected_test_len = int(n_samples * test_size)
        expected_train_len = n_samples - expected_test_len

        assert len(response.X_train) == expected_train_len
        assert len(response.X_test) == expected_test_len
        assert len(response.y_train) == expected_train_len
        assert len(response.y_test) == expected_test_len

        # Verifica os limites matemáticos da normalização MinMax
        if normalization == "minmax" and expected_train_len > 0:
            X_train_np = np.array(response.X_train)
            
            assert np.min(X_train_np) >= -1.0 - 1e-5
            assert np.max(X_train_np) <= 1.0 + 1e-5

    def test_invalid_dataset_name(self, service):
        request = DatasetRequest(
            dataset_name="nome_invalido",
            n_samples=100,
            noise=0.1,
            test_size=0.2,
            normalization="none"
        )
        
        with pytest.raises(ValueError, match="não encontrado"):
            service.get_dataset(request)
    
    def test_invalid_scaler_name(self, service):
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            request = DatasetRequest(
                dataset_name="svm",
                n_samples=100,
                noise=0.1,
                test_size=0.2,
                normalization=""
            )
            service.get_dataset(request)