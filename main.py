from pipelines.training_pipeline import training_pipe
from pipelines.prediction_pipeline import prediction_pipeline
import dagshub
import mlflow
from pipelines.config import DataPaths

if __name__ == "__main__":
    dagshub.init(repo_owner='typhonshambo', repo_name='End-to-End-ML-Pipeline-for-Predictive-Maintenance', mlflow=True)
    with mlflow.start_run():
        remote_server_uri=DataPaths.mlflow_uri
        mlflow.set_tracking_uri(remote_server_uri)
        training_pipe()
        prediction_pipeline()

