from pipelines.training_pipeline import training_pipe
from pipelines.prediction_pipeline import prediction_pipeline
import dagshub
import mlflow

if __name__ == "__main__":


    dagshub.init("End-to-End-ML-Pipeline-for-Predictive-Maintenance", "typhonshambo", mlflow=True)
    mlflow.start_run()
    training_pipe()
    prediction_pipeline()

