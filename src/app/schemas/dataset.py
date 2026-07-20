from typing import Literal
from pydantic import BaseModel, Field

class DatasetRequest(BaseModel):
    dataset_name: str = Field(..., description="Dataset name (eg: circle, spiral)")
    n_samples: int = Field(1000, ge=100, le=5000, description="Number of samples")
    noise: float = Field(0.1, ge=0.0, le=1.0)
    test_size: float = Field(0.2, ge=0.1, le=0.9, description="Data proporion for test")
    normalization: Literal['none', 'minmax', 'standard'] = Field('none', description='Normalization strategy')

class DatasetResponse(BaseModel):
    X_train: list[list[float]]
    X_test: list[list[float]]
    y_train: list[int]
    y_test: list[int]