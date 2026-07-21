export interface DatasetRequest {
    dataset_name: string;
    n_samples: number;
    noise: number;
    test_size: number;
    normalization: string;
}

export interface DatasetResponse {
    X_train: [number, number][];
    X_test: [number, number][];
    y_train: number[];
    y_test: number[];
}

export interface ModelTrainRequest {
    model_name: string;
    dataset: DatasetRequest;
    params: Record<string, any>;
}

export interface ModelPredictRequest {
    X: [number, number][]
}

export interface ModelPredictResponse {
    predictions: number[];
    probabilities: number[][] | null;
}