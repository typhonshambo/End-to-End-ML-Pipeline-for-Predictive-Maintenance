# File: src/pipeline.py

import pandas as pd
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from train_model import ModelTrainer
from evaluate_model import Evaluator

def main_pipeline(data_path):
    # Step 1: Data Preprocessing
    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.load_data()
    df = preprocessor.handle_missing_values()
    numeric_cols = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
    df = preprocessor.scale_features(numeric_cols)
    
    # Save the preprocessed data
    preprocessed_data_path = 'data/processed/preprocessed_data.csv'
    df.to_csv(preprocessed_data_path, index=False)
    
    # Step 2: Feature Engineering
    engineer = FeatureEngineer(df)
    df = engineer.add_rolling_features(window=3)
    df = engineer.add_categorical_feature_counts()
    
    # Save the feature engineered data
    feature_engineered_data_path = 'data/processed/feature_engineered_data.csv'
    df.to_csv(feature_engineered_data_path, index=False)
    
    # Step 3: Model Training
    df.drop(columns=['date', 'device'], inplace=True)
    trainer = ModelTrainer(df)
    X_train, X_test, y_train, y_test = trainer.prepare_data(target_column='failure')
    model = trainer.train_model()
    
    # Step 4: Model Evaluation
    evaluator = Evaluator(model, X_test, y_test)
    metrics = evaluator.compute_metrics()
    print("Model Performance Metrics:")
    print(metrics)
    evaluator.print_classification_report()
    evaluator.plot_confusion_matrix()

if __name__ == "__main__":
    data_path = 'data/raw/predictive_maintenance_dataset.csv'  # Adjust the path to your raw data
    main_pipeline(data_path)
