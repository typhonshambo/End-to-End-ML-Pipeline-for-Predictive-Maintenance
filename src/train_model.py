# File: src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(random_state=0)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, target_column, test_size=0.2):
        """Prepare data for training and testing."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=0)
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
if __name__ == "__main__":
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    trainer = ModelTrainer(df)
    X_train, X_test, y_train, y_test = trainer.prepare_data(target_column='failure')
    model = trainer.train_model()
    metrics = trainer.evaluate_model()
    print(metrics)

