import os
class DataPaths:
    '''
    This class contains the paths to the raw data, processed data, model, evaluation results, evaluation plots, mlflow experiment, and mlflow run.
    '''
    raw_data: str = os.path.join('data', 'raw', 'predictive_maintenance_dataset.csv')
    processed_data_path = os.path.join('data', 'processed', 'preprocessed_data.csv')
    preview_data = os.path.join('data', 'raw', 'preview.csv')
    model_path = 'models/model.pkl'
    #evaluation_results_path = 'results/evaluation_results.txt'
    #evaluation_plots_path = 'results/evaluation_plots/'
    mlflow_experiment_name = 'Predictive Maintenance'
    mlflow_run_name = 'Predictive Maintenance Pipeline Run'
    mlflow_uri = "https://dagshub.com/typhonshambo/End-to-End-ML-Pipeline-for-Predictive-Maintenance.mlflow"
    steamlit_uploaded_data = os.path.join('data', 'streamlit', 'uploaded_data.csv')