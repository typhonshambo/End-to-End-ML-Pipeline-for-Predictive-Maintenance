import pandas as pd
from data_preprocessing import DataPreprocessor
from train_model import ModelTrainer
from evaluate_model import Evaluator

def main_pipeline(data_path):
    # Step 1: Data Preprocessing
    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.load_data()
    df = preprocessor.process_data()
    df = preprocessor.handle_missing_values()
    df = preprocessor.under_sample_data(target_column='failure')

    
    # Save the preprocessed data
    preprocessed_data_path = 'data/processed/preprocessed_data.csv'
    df.to_csv(preprocessed_data_path, index=False)
    

    # Step 2: Model Training
    trainer = ModelTrainer(df)
    X_train, X_test, y_train, y_test = trainer.prepare_data(target_column='failure')
    model = trainer.train_model()
    
    # Step 3: Model Evaluation
    evaluator = Evaluator(model, X_test, y_test)
    metrics = evaluator.compute_metrics()
    print("Model Performance Metrics:")
    print(metrics)
    evaluator.print_classification_report()
    evaluator.plot_confusion_matrix()

if __name__ == "__main__":
    data_path = '/Users/shambo/Documents/ML projects/End-to-End ML Pipeline for Predictive Maintenance/data/raw/predictive_maintenance_dataset.csv'  # Adjust the path to your raw data
    main_pipeline(data_path)
