import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple
class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, target_column) -> Tuple[
        pd.DataFrame, 
        pd.DataFrame, 
        pd.Series, 
        pd.Series
    ]:
        """Split the dataset into features and target.
        Args:
            target_column (str): The name of the target column.
        Returns:
            x_train (DataFrame): Training features.
            x_test (DataFrame): Testing features.
            y_train (Series): Training target.
            y_test (Series): Testing target.
        """
        x = self.data.drop(columns=target_column)
        y = self.data[target_column]
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the RandomForest model."""
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate_model(self):
        """Evaluate the trained model."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=1)
        recall = recall_score(self.y_test, y_pred, zero_division=1)
        f1 = f1_score(self.y_test, y_pred, zero_division=1)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

#Example usage:
# if __name__ == "__main__":
#     df = pd.read_csv('data/processed/preprocessed_data.csv')
#     trainer = ModelTrainer(df)
#     X_train, X_test, y_train, y_test = trainer.prepare_data(target_column='failure')
#     model = trainer.train_model()
#     metrics = trainer.evaluate_model()
#     print(metrics)

