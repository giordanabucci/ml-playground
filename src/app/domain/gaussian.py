import numpy as np
from app.domain.base_dataset import BaseDataset

class GaussianDataset(BaseDataset):
    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        n_samples_pos = n_samples // 2
        n_samples_neg = n_samples - n_samples_pos

        x_p, y_p = self.generate_gaussian(2, 2, 1, n_samples=n_samples_pos, noise=noise)
        x_n, y_n = self.generate_gaussian(-2, -2, -1, n_samples=n_samples_neg, noise=noise)

        X = np.vstack([x_p, x_n])
        y = np.hstack([y_p, y_n])

        super().__init__(X=X, y=y)


    @staticmethod
    def generate_gaussian(cx0: float, cx1: float, label: int, n_samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        x = 0.7 * np.random.randn(n_samples, 2)
        x += np.array([cx0, cx1])

        sampled_noise = np.random.normal(scale=noise, size=x.shape)
        x += sampled_noise

        y = label * np.ones(n_samples)

        return x, y
