import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from imblearn.under_sampling import RandomUnderSampler

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load dataset from file."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        if self.data is not None:
            self.data.fillna(0, inplace=True)  
        return self.data

    def under_sample_data(self, target_column):
        """Under sample the dataset to handle class imbalance."""
        if self.data is not None:
            X = self.data.drop(columns=target_column)
            y = self.data[target_column]
            sampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            self.data = pd.concat([X_resampled, y_resampled], axis=1)
        return self.data


    
# Example usage:
# if __name__ == "__main__":
    # preprocessor = DataPreprocessor('data/raw/predictive_maintenance_dataset.csv')
    # df = preprocessor.load_data()
    # df = preprocessor.handle_missing_values()
    # numeric_cols = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
    # df = preprocessor.scale_features(numeric_cols)
    # print(df.head())
