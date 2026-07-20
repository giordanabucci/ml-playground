from fastapi import FastAPI
import uvicorn
from app.api.routers import dataset

app = FastAPI(
    title="ML Playground API",
    description="Backend for training and inference of Machine Learning Playground models",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ML Playground API is running"}

app.include_router(dataset.router)

if __name__=="__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=18000, reload=True)