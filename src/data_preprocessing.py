# File: src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load dataset from file."""
        self.data = pd.read_csv(self.file_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        return self.data

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        if self.data is not None:
            self.data.fillna(0, inplace=True)  # Example: filling missing values with 0
        return self.data

    def scale_features(self, feature_columns):
        """Scale numeric features."""
        if self.data is not None:
            features_to_scale = [col for col in feature_columns if col != 'date']
            self.data[features_to_scale] = self.scaler.fit_transform(self.data[features_to_scale])
        return self.data


# Example usage:
if __name__ == "__main__":
    preprocessor = DataPreprocessor('data/raw/predictive_maintenance_dataset.csv')
    df = preprocessor.load_data()
    df = preprocessor.handle_missing_values()
    numeric_cols = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
    df = preprocessor.scale_features(numeric_cols)
    print(df.head())
