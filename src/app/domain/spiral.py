import numpy as np
from app.domain.base import BaseDataset

class SpiralDataset(BaseDataset):
    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        n_samples_pos = n_samples // 2
        n_samples_neg = n_samples - n_samples_pos
        
        x_p, y_p = self.generate_spiral(0, 1, n_samples=n_samples_pos, noise=noise)
        x_n, y_n = self.generate_spiral(np.pi, -1, n_samples=n_samples_neg, noise=noise)

        X = np.vstack([x_p, x_n])
        y = np.hstack([y_p, y_n])

        super().__init__(X=X, y=y)


    @staticmethod
    def generate_spiral(delta_t: float, label: int, n_samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        r = 5 * np.linspace(0, 1, n_samples)
        t = 0.7 * np.pi * r + delta_t

        x1 = np.cos(t) * r
        x2 = np.sin(t) * r
        x = np.stack([x1, x2], axis=1)

        sampled_noise = np.random.normal(scale=noise, size=x.shape)
        x += sampled_noise

        y = label * np.ones_like(x1)

        return x, y