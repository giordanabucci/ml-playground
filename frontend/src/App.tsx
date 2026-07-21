import { useState, useEffect } from 'react';
import { api } from './api/client';
import type { DatasetRequest, DatasetResponse } from './types/api';
import ScatterPlot from './components/ScatterPlot';
import './App.css';

export default function App() {
  const [datasetParams, setDatasetParams] = useState<DatasetRequest>({
    dataset_name: 'circle',
    n_samples: 300,
    noise: 0.1,
    test_size: 0.2,
    normalization: 'standard'
  });

  const [modelName, setModelName] = useState<string>('svm');
  const [datasetData, setDatasetData] = useState<DatasetResponse | null>(null);
  
  // Estados para as fronteiras de decisão
  const [meshGrid, setMeshGrid] = useState<[number, number][] | null>(null);
  const [predictions, setPredictions] = useState<number[] | null>(null);
  const gridSize = 60; // Define a resolução da malha (60x60 = 3600 pontos)

  const [statusText, setStatusText] = useState<string>('Aguardando inicialização');

  const fetchDataset = async () => {
    try {
      setStatusText('Buscando dados...');
      const data = await api.generateDataset(datasetParams);
      setDatasetData(data);
      // Reseta as predições anteriores
      setMeshGrid(null);
      setPredictions(null);
      setStatusText('Dados carregados');
    } catch (error) {
      setStatusText('Erro na requisição dos dados');
    }
  };

  const generateMeshGrid = (data: DatasetResponse, resolution: number): [number, number][] => {
    const allX = [...data.X_train, ...data.X_test];
    let minX = Math.min(...allX.map(d => d[0]));
    let maxX = Math.max(...allX.map(d => d[0]));
    let minY = Math.min(...allX.map(d => d[1]));
    let maxY = Math.max(...allX.map(d => d[1]));

    const xMargin = (maxX - minX) * 0.1 || 0.1;
    const yMargin = (maxY - minY) * 0.1 || 0.1;

    minX -= xMargin;
    maxX += xMargin;
    minY -= yMargin;
    maxY += yMargin;

    const grid: [number, number][] = [];
    const dx = (maxX - minX) / resolution;
    const dy = (maxY - minY) / resolution;

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        grid.push([minX + i * dx, minY + j * dy]);
      }
    }
    return grid;
  };

  const handleTrain = async () => {
    if (!datasetData) return;
    
    try {
      setStatusText('Treino em execução...');
      await api.trainModel({
        model_name: modelName,
        dataset: datasetParams,
        params: {}
      });
      
      setStatusText('Buscando limites de decisão...');
      const grid = generateMeshGrid(datasetData, gridSize);
      setMeshGrid(grid);
      
      const predictRes = await api.predict(modelName, { X: grid });
      setPredictions(predictRes.predictions);
      
      setStatusText('Fronteiras geradas com sucesso');
    } catch (error) {
      setStatusText('Erro no processamento do modelo');
    }
  };

  useEffect(() => {
    fetchDataset();
  }, [datasetParams]);

  return (
    <div className="app-container">
      <aside className="sidebar">
        <h2>ML Playground</h2>
        
        <div className="control-group">
          <label>Dataset</label>
          <select 
            value={datasetParams.dataset_name}
            onChange={(e) => setDatasetParams({...datasetParams, dataset_name: e.target.value})}
          >
            <option value="circle">Circle</option>
            <option value="diagonal">Diagonal</option>
            <option value="gaussian">Gaussian</option>
            <option value="linear">Linear</option>
            <option value="spiral">Spiral</option>
            <option value="xor">XOR</option>
          </select>
        </div>

        <div className="control-group">
          <label>Algoritmo</label>
          <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
            <option value="dt">Decision Tree</option>
            <option value="knn">K-Nearest Neighbors (KNN)</option>
            <option value="lda">Linear Discriminant Analysis (LDA)</option>
            <option value="lr">Logistic Regression</option>
            <option value="nb">Naive Bayes</option>
            <option value="qda">Quadratic Discriminant Analysis (QDA)</option>
            <option value="svm">Support Vector Machine (SVM)</option>
          </select>
        </div>

        <div className="control-group">
          <label>Samples: {datasetParams.n_samples}</label>
          <input 
            type="range" min="100" max="5000" step="1" 
            value={datasetParams.n_samples}
            onChange={(e) => setDatasetParams({...datasetParams, n_samples: parseFloat(e.target.value)})}
          />
        </div>

        <div className="control-group">
          <label>Ruído: {datasetParams.noise}</label>
          <input 
            type="range" min="0" max="1" step="0.05" 
            value={datasetParams.noise}
            onChange={(e) => setDatasetParams({...datasetParams, noise: parseFloat(e.target.value)})}
          />
        </div>

        <div className="control-group">
          <label>Test split: {datasetParams.test_size}</label>
          <input 
            type="range" min="0" max="1" step="0.05" 
            value={datasetParams.test_size}
            onChange={(e) => setDatasetParams({...datasetParams, test_size: parseFloat(e.target.value)})}
          />
        </div>

        <button onClick={handleTrain} className="train-button">
          Treinar e Prever
        </button>

        <div className="status-panel">
          <small>Status: {statusText}</small>
        </div>
      </aside>

      <main className="plot-area">
        {datasetData ? (
          <ScatterPlot 
            data={datasetData} 
            predictions={predictions} 
            meshGrid={meshGrid}
            gridSize={gridSize}
          />
        ) : (
          <div className="placeholder">Carregando visualização...</div>
        )}
      </main>
    </div>
  );
}