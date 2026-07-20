from dataclasses import dataclass
import numpy as np

@dataclass
class BaseDataset:
    X: np.ndarray
    y: np.ndarray