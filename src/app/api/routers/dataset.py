from fastapi import APIRouter, Depends, HTTPException
from app.services.dataset_service import DatasetService
from app.schemas.dataset import DatasetRequest, DatasetResponse


router = APIRouter(prefix='/datasets', tags=['Datasets'])

def get_dataset_service() -> DatasetService:
    return DatasetService()

@router.post('/generate', response_model=DatasetResponse)
def generate_dataset(
    request: DatasetRequest,
    service: DatasetService = Depends(get_dataset_service)
):
    try:
        return service.get_dataset(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
