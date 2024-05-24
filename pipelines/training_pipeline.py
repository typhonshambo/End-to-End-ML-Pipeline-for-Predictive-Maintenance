from mlflow.models import infer_signature
import mlflow.sklearn
import logging
from .config import DataPaths
from src.data_preprocessing import DataPreprocessor
from src.train_model import ModelTrainer
from src.evaluate_model import Evaluator


def training_pipe(data_path : str = DataPaths.raw_data):
    print(data_path)
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
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # Step 3: Model Evaluation
    evaluator = Evaluator(model, X_test, y_test)
    metrics = evaluator.compute_metrics()
    print("Model Performance Metrics:")
    print(metrics)
    evaluator.print_classification_report()
    evaluator.plot_confusion_matrix()