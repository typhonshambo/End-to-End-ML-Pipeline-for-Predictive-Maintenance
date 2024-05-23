# File: src/feature_engineering.py

import pandas as pd

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def add_rolling_features(self, window=3):
        """Add rolling mean and standard deviation for metrics."""
        metrics = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
        for metric in metrics:
            self.data[f'{metric}_rolling_mean'] = self.data[metric].rolling(window=window).mean()
            self.data[f'{metric}_rolling_std'] = self.data[metric].rolling(window=window).std()
        return self.data


    def add_categorical_feature_counts(self):
        """Add count features for categorical variables."""
        self.data['device_count'] = self.data.groupby('device')['device'].transform('count')
        return self.data

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    engineer = FeatureEngineer(df)
    df = engineer.add_rolling_features(window=3)
    df = engineer.add_datetime_features()
    df = engineer.add_categorical_feature_counts()
    print(df.head())
