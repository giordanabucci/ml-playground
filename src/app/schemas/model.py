from pydantic import BaseModel, Field
from app.schemas.dataset import DatasetRequest

class ModelTrainRequest(BaseModel):
    model_name: str = Field(..., description="Algorithm name (ex: knn, svm, dt)")
    dataset: DatasetRequest = Field(..., description="dataset request for training")
    params: dict = Field(default_factory=dict, description="Model hyperparameters")

class ModelPredictRequest(BaseModel):
    X: list[list[float]] = Field(..., description="Data for model prediction")

class ModelPredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]] | None = None