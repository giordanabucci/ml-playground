from pydantic import BaseModel, Field

class ModelTrainRequest(BaseModel):
    model_name: str = Field(..., description="Algorithm name (ex: knn, svm, dt)")
    dataset_name: str = Field(..., description="dataset name for training")
    params: dict = Field(default_factory=dict, description="Model hyperparameters")

class ModelPredictRequest(BaseModel):
    X: list[list[float]] = Field(..., description="Data for model prediction")

class ModelPredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]] | None = None