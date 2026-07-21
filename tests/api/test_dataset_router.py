from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_dataset_endpoint():
    payload = {
        "dataset_name": "circle",
        "n_samples": 500,
        "noise": 0.2,
        "test_size": 0.3,
        "normalization": "minmax"
    }

    response = client.post("/datasets/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "X_train" in data
    assert len(data["X_train"]) == 350