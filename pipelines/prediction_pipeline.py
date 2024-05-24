from mlflow.models import infer_signature
import mlflow.sklearn
import logging

import pandas as pd
from .config import DataPaths
from src.train_model import ModelTrainer
import logging
def prediction_pipeline(
    test_data_path: str = DataPaths.xtest_data_path
):
    prediction_data = pd.read_csv(test_data_path)
    #model = mlflow.sklearn.load_model('models/model')
    model_class = ModelTrainer()
    prediction = model_class.predict(prediction_data)
    if prediction == 1:
        print(f"Maintenance required : Yes")
    elif prediction == 0:
        print(f"Maintenance required : No")
    return prediction

