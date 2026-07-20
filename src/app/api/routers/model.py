from fastapi import APIRouter, Depends, HTTPException
from app.schemas.model import ModelTrainRequest, ModelPredictRequest, ModelPredictResponse
from app.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["Models"])

def get_model_service() -> ModelService:
    return ModelService()

@router.post("/train")
def train_model(
    request: ModelTrainRequest,
    service: ModelService = Depends(get_model_service)
):
    try:
        return service.train_model(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.post("/{model_name}/predict", response_model=ModelPredictResponse)
def predict_model(
    model_name: str,
    request: ModelPredictRequest,
    service: ModelService = Depends(get_model_service)
):
    try:
        return service.predict_model(model_name, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))