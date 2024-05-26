import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import logging

class DataPreprocessor:
    '''
    This class is used to load, process, and preprocess the dataset.
    Methods:
    - load_data: Load dataset from file.
    - process_data: Process the dataset as per EDA.
    - handle_missing_values: Handle missing values in the dataset.
    - under_sample_data: Under sample the dataset to handle class imbalance.
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load dataset from file."""
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except FileNotFoundError:
            logging.error("File not found. Please check the file path.")
            return None

    def process_data(self) -> pd.DataFrame:
        """Process the dataset as per EDA"""
        try:
            self.data['device_model'] = self.data['device'].apply(lambda x : x[:4])
            self.data.drop("device",axis=1,inplace=True)
            
            #handling outliers
            self.data.drop(self.data.loc[self.data["device_model"]=="Z1F2"].index,axis=0,inplace=True)
            self.data.reset_index(drop=True,inplace=True)

            """
            This method processes the 'date' column in the DataFrame to extract and add new temporal features:
            - Converts the 'date' column to datetime objects.
            - Extracts the day of the week and adds it as a new column 'day_of_week'.
            - Extracts the day of the month and adds it as a new column 'day_of_month'.
            - Determines if the date is a weekend and adds it as a new column 'is_weekend'.

            New columns:
            - day_of_week: Indicates the day of the week (0 for Monday, 6 for Sunday).
            - day_of_month: Indicates the day of the month (1 to 31).
            - is_weekend: Indicates whether the day is a weekend (1 for Saturday and Sunday, 0 for weekdays).
            """

            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['day_of_month'] = self.data['date'].dt.day
            self.data['is_weekend'] = self.data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

            # Extract the month number and week number from the 'date' column and drop the 'date' column
            self.data['month'] = self.data['date'].dt.month
            self.data['week'] = self.data['date'].dt.isocalendar().week
            self.data = self.data.drop(['date'], axis=1)
            
            return self.data
    
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return None
        
    def convert_to_dummy(self) -> pd.DataFrame:
        '''
        This method converts the categorical columns in the DataFrame to dummy variables.
        Returns:
        - pd.DataFrame: The DataFrame with dummy variables.
        '''
        try:
            self.data = pd.get_dummies(self.data,drop_first=True)
            return self.data
        except Exception as e:  
            logging.error(f"Error converting to dummy: {e}")
            return None


    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            if self.data is not None:
                self.data.fillna(0, inplace=True)  
            return self.data
        except Exception as e:
            logging.error(f"Error handling missing values: {e}")
            return None

    def under_sample_data(self, target_column) -> pd.DataFrame:
        """Under sample the dataset to handle class imbalance."""
        try:
            if self.data is not None:
                X = self.data.drop(columns=target_column)
                y = self.data[target_column]
                sampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                self.data = pd.concat([X_resampled, y_resampled], axis=1)
            return self.data
        except Exception as e:
            logging.error(f"Error under sampling data: {e}")
            return None

    
# Example usage:
# if __name__ == "__main__":
    # preprocessor = DataPreprocessor('data/raw/predictive_maintenance_dataset.csv')
    # df = preprocessor.load_data()
    # df = preprocessor.handle_missing_values()
    # numeric_cols = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
    # df = preprocessor.scale_features(numeric_cols)
    # print(df.head())
