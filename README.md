# 🛝 Machine Learning Playground 

An interactive platform designed to visualize and experiment with classical machine learning algorithms on 2D datasets in real-time. Built with a microservices architecture, this application features a FastAPI/scikit-learn backend and a React + D3.js frontend.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Google Cloud Run](https://img.shields.io/badge/Deployed_on-Cloud_Run-4285F4.svg)](https://cloud.google.com/run)

## 🔗 Live Demo (Cloud Run)
- Interactive Dashboard: https://ml-frontend-499412997238.us-central1.run.app/
- API Swagger UI: https://ml-playground-api-499412997238.us-central1.run.app/docs

## ✨ Core Features
Data Science Engine
- Dynamic Synthetic Datasets: Instantly generate classic 2D datasets (Circle, Spiral, Gaussian, XOR, Linear, Diagonal) with adjustable noise levels, sample sizes, and train/test splits.
- Extensive Algorithm Catalog: Train and evaluate multiple models including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, Logistic Regression, Naive Bayes, Linear Discriminant Analysis, and Quadratic Discriminant Analysis.
- Mathematical Reliability: Strict data validation using Pydantic schemas ensures that only mathematically sound parameters (e.g., positive sample sizes, valid scaling methods) reach the scikit-learn processing layer.

High-Performance Visualization & UX
- Hybrid Rendering Pipeline: Uses D3.js (SVG) for pixel-perfect plotting of training and testing data points, combined with HTML5 Canvas to render high-density meshgrids (3600+ points) representing complex decision boundaries.
- Reactive State Management: Seamlessly syncs user hyperparameter selections with backend API calls, preventing UI bottlenecks during model training.

Enterprise-Grade Quality
- Clean Architecture: Strict decoupling of Domain (entities), Services (business logic), and API (routing) layers.
- 95% Test Coverage: Comprehensive unit and integration testing using pytest and pytest-cov, including mocked dependencies for isolated environment validation.

## 🏗️ Architecture
This system utilizes a two-tier microservices architecture to separate machine learning processing from data presentation:
- Backend (FastAPI + scikit-learn): A RESTful API that handles matrix generation, data scaling, and model orchestration. By isolating the heavy ML processing on the server, the application remains highly responsive regardless of the client's hardware.
- Frontend (React + Vite): A modern, component-driven client that manages user inputs and leverages D3.js to paint mathematical decision boundaries based on API predictions.

## 🛠️ Technology Stack
The environment operates with pinned dependencies for maximum reproducibility:
- Backend: fastapi, uvicorn, pydantic, scikit-learn, numpy
- Frontend: react, typescript, vite, d3, axios
- Testing: pytest, pytest-cov, httpx
- Infrastructure: docker, docker-compose, Google Cloud Run

## 🚀 Local Development Setup

To run this architecture locally, ensure you have Docker installed.

### 1. Clone the repository
``` bash
git clone https://github.com/giordanabucci/ml-playground.git
cd ml-playground
```

### 2. Launch the Microservices

Use Docker Compose to build and start both the API and the Frontend containers simultaneously:
``` bash
docker compose up --build -d
```

### 3. Access Local Endpoints

- Frontend UI: http://localhost:80
- API Documentation: http://localhost:18000/docs

## 📁 Project Structure
```
.
├── docker-compose.yaml
├── pyproject.toml
├── Dockerfile                  # API Docker setup
├── src/
│   └── app/
│       ├── api/                # FastAPI routers
│       ├── core/               # Configuration settings
│       ├── domain/             # Dataset classes & mathematical logic
│       ├── schemas/            # Pydantic data validation
│       ├── services/           # ML model training and orchestrators
│       └── main.py             # Uvicorn entry point
├── tests/                      # Pytest unit and integration tests
└── frontend/                   
    ├── Dockerfile              # React/Vite Docker setup
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── api/                # Axios client configurations
        ├── components/         # D3.js/Canvas ScatterPlot
        ├── types/              # TypeScript interfaces mapped to Pydantic
        └── App.tsx             # Main dashboard logic
```