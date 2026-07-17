# ML Playground Backend

Backend API for the Machine Learning Playground. This service provides endpoints for model training and decision boundary generation.

## Architecture

- **Framework:** FastAPI
- **Machine Learning:** scikit-learn, numpy
- **Package Management:** hatchling
- **Execution:** uv

## Project Structure

The codebase resides in the `src/app` directory, structured around Clean Architecture principles:

- `api/`: API routers and endpoints
- `core/`: Global configurations
- `domain/`: Business rules and entities
- `schemas/`: Pydantic models for request validation
- `services/`: Scikit-learn model integration

## Setup and Execution

### Prerequisites

- Python 3.14+
- uv

### Run Locally

Execute the server from the project root.

Using the Python script:
```bash
uv run python src/app/main.py
```

Using Uvicorn directly:
```bash
uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Documentation

Access the API documentation in the browser:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)