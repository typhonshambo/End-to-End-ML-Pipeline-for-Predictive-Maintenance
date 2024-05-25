from mlflow.models import infer_signature
import mlflow.sklearn
import logging

import pandas as pd
from .config import DataPaths
from src.train_model import ModelTrainer
from utils.utils import load_object
import logging

def prediction_pipeline(
    test_data_path: str = DataPaths.xtest_data_path
):
    print(test_data_path)
    model = load_object(DataPaths.model_path)
    prediction_data = pd.read_csv(test_data_path)
    #model = mlflow.sklearn.load_model('models/model')
    prediction = model.predict(prediction_data)
    print(prediction)
    for i in prediction:
        if i == 1:
            print(f"Maintenance required : Yes")
        elif i == 0:
            print(f"Maintenance required : No")
    return prediction


if __name__ == "__main__":
    prediction_pipeline()