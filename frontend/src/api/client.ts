import axios from 'axios'
import type {
    DatasetRequest,
    DatasetResponse,
    ModelTrainRequest,
    ModelPredictRequest,
    ModelPredictResponse
} from '../types/api';

const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:18000',
    headers: {
        'Content-Type': 'application/json'
    }
});

export const api = {
    generateDataset: async (data: DatasetRequest): Promise<DatasetResponse> => {
        const response = await apiClient.post<DatasetResponse>('/datasets/generate', data);
        return response.data;
    },

    trainModel: async (data: ModelTrainRequest): Promise<{ status: string; modelo: string }> => {
        const response = await apiClient.post('/models/train', data);
        return response.data;
    },

    predict: async (modelName: string, data: ModelPredictRequest): Promise<ModelPredictResponse> => {
        const response = await apiClient.post<ModelPredictResponse>(`/models/${modelName}/predict`, data);
        return response.data
    }
}