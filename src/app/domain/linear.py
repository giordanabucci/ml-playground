import numpy as np
from app.domain.base_dataset import BaseDataset

class LinearDataset(BaseDataset):
    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        X, y = self.generate_linear(n_samples=n_samples, noise=noise)
        super().__init__(X=X, y=y)

    @staticmethod
    def generate_linear(n_samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        x = 10 * np.random.rand(n_samples, 2) - 5
        y = np.sign(x[:, 0])

        padding = 0.3
        x[:, 0] += y * padding

        sampled_noise = np.random.normal(scale=noise, size=x.shape)
        x += sampled_noise

        return x, y