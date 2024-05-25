import os
class DataPaths:
    '''
    This class contains the paths to the raw data, processed data, model, evaluation results, evaluation plots, mlflow experiment, and mlflow run.
    '''
    raw_data: str = os.path.join('data', 'raw', 'predictive_maintenance_dataset.csv')
    processed_data_path = os.path.join('data', 'processed', 'preprocessed_data.csv')
    xtest_data_path = os.path.join('data', 'processed', 'X_test.csv')
    model_path = 'models/model.pkl'
    #evaluation_results_path = 'results/evaluation_results.txt'
    #evaluation_plots_path = 'results/evaluation_plots/'
    mlflow_experiment_name = 'Predictive Maintenance'
    mlflow_run_name = 'Predictive Maintenance Pipeline Run'
    mlflow_uri = 'http://'