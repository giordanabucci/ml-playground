import { useState, useEffect } from 'react';
import { api } from './api/client';
import type { DatasetRequest, DatasetResponse } from './types/api';
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
  const [statusText, setStatusText] = useState<string>('Aguardando inicialização');

  const fetchDataset = async () => {
    try {
      setStatusText('Buscando dados...');
      const data = await api.generateDataset(datasetParams);
      setDatasetData(data);
      setStatusText('Dados carregados com sucesso');
    } catch (error) {
      setStatusText('Erro na requisição dos dados');
    }
  };

  const handleTrain = async () => {
    try {
      setStatusText('Treino em execução...');
      await api.trainModel({
        model_name: modelName,
        dataset: datasetParams,
        params: {} // Espaço para hiperparâmetros futuros
      });
      setStatusText('Treino concluído. Pronto para predição.');
    } catch (error) {
      setStatusText('Erro na etapa de treino');
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
            type="range" min="0.1" max="0.9" step="0.05" 
            value={datasetParams.test_size}
            onChange={(e) => setDatasetParams({...datasetParams, test_size: parseFloat(e.target.value)})}
          />
        </div>

        <button onClick={handleTrain} className="train-button">
          Treinar Modelo
        </button>

        <div className="status-panel">
          <small>Status: {statusText}</small>
        </div>
      </aside>

      <main className="plot-area">
        {/* O componente D3 ocupa este espaço na próxima iteração */}
        <div className="placeholder">
          {datasetData ? `Amostras carregadas: ${datasetData.X_train.length} (Treino) | ${datasetData.X_test.length} (Teste)` : 'Carregando...'}
        </div>
      </main>
    </div>
  );
}