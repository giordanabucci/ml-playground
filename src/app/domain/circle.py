import numpy as np
from app.domain.base_dataset import BaseDataset

class CircleDataset(BaseDataset):
    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        n_samples_pos = n_samples // 2
        n_samples_neg = n_samples - n_samples_pos

        x_p, y_p = self.generate_circle(0, 2.5, 1, n_samples=n_samples_pos, noise=noise)
        x_n, y_n = self.generate_circle(3.5, 5, -1, n_samples=n_samples_neg, noise=noise)

        X = np.vstack([x_p, x_n])
        y = np.hstack([y_p, y_n])

        super().__init__(X=X, y=y)
    

    @staticmethod
    def generate_circle(r_min: float, r_max: float, label: int, n_samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        r = (r_max - r_min) * np.random.rand(n_samples) + r_min
        angle = 2 * np.pi * np.random.rand(n_samples)

        x1 = np.cos(angle) * r
        x2 = np.sin(angle) * r
        x = np.stack([x1, x2], axis=1)

        sampled_noise = np.random.normal(scale=noise, size=x.shape)
        x += sampled_noise


        y = label * np.ones_like(x1)

        return x, y