import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from app.core.config import get_settings
from app.domain.circle import CircleDataset
from app.domain.diagonal import DiagonalDataset
from app.domain.gaussian import GaussianDataset
from app.domain.linear import LinearDataset
from app.domain.spiral import SpiralDataset
from app.domain.xor import XORDataset
from app.schemas.dataset import DatasetRequest, DatasetResponse

class DatasetService:
    def __init__(self):
        self.generators = {
            "circle": CircleDataset,
            "diagonal": DiagonalDataset,
            "gaussian": GaussianDataset,
            "linear": LinearDataset,
            "spiral": SpiralDataset,
            "xor": XORDataset
        }
        self.scalers = {
            "standard": lambda: StandardScaler(),
            "minmax": lambda: MinMaxScaler(feature_range=(-1, 1))
        }
        self.settings = get_settings()

    def get_dataset(self, request: DatasetRequest) -> DatasetResponse:
        generator_class = self.generators.get(request.dataset_name)
        if not generator_class:
            raise ValueError(f"Dataset {request.dataset_name} não encontrado")
        
        dataset = generator_class(n_samples=request.n_samples, noise=request.noise)

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=request.test_size, random_state=self.settings.random_state  
        )

        if request.normalization != "none":
            scaler_builder = self.scalers.get(request.normalization)
            if not scaler_builder:
                raise ValueError(f"Normalização {request.normalization} não encontrada")

            scaler = scaler_builder()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return DatasetResponse(
            X_train=np.array(X_train).tolist(),
            X_test=np.array(X_test).tolist(),
            y_train=np.array(y_train).tolist(),
            y_test=np.array(y_test).tolist()
        )