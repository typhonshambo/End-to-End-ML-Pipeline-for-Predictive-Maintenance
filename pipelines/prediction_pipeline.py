import pandas as pd
from .config import DataPaths
from src.train_model import ModelTrainer
from utils.utils import load_object
import logging
from src.data_preprocessing import DataPreprocessor

def manage_device_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Extract the device model from the 'device' column and add it as a new column 'device_model'.
    Drop the 'device' column from the DataFrame.
    Handle outliers in the 'device_model' column.
    Args:
        - df: pd.DataFrame: The DataFrame to process.
    Returns:
        - pd.DataFrame: The processed DataFrame.
    '''
    try:
        df['device_model_S1F1'] = df['device'].apply(lambda x: True if x[:4] == 'S1F1' else False)
        df['device_model_W1F0'] = df['device'].apply(lambda x: True if x[:4] == 'W1F0' else False)
        df['device_model_W1F1'] = df['device'].apply(lambda x: True if x[:4] == 'W1F1' else False)
        df['device_model_Z1F0'] = df['device'].apply(lambda x: True if x[:4] == 'Z1F0' else False)
        df['device_model_Z1F1'] = df['device'].apply(lambda x: True if x[:4] == 'Z1F1' else False)
        df.drop("device",axis=1,inplace=True)
        df.reset_index(drop=True,inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return None

def prediction_pipeline(
    test_data_path: pd.DataFrame = DataPaths.preview_data
) -> pd.Series:
    '''
    Make predictions using the trained model.
    Args:
        - test_data_path[OPTIONAL]: str: Path to the sample data.
    Returns:
        - pd.Series: The predictions.
    '''
    preprocessor = DataPreprocessor(test_data_path)
    df = preprocessor.load_data()
    df = preprocessor.process_data_date()
    df = preprocessor.handle_missing_values()
    df = manage_device_columns(df)

    # Save the preprocessed data
    preprocessed_data_path = 'data/processed/preview_processed.csv'
    df.to_csv(preprocessed_data_path, index=False)

    # Step 2: Model Training
    model = load_object(DataPaths.model_path)
    # prediction_data = pd.read_csv(test_data_path)
    #model = mlflow.sklearn.load_model('models/model')
    old_df = pd.read_csv(test_data_path)
    prediction = model.predict(df)
    prediction = ['No' if p == 0 else 'Yes' for p in prediction]
    old_df['Failure Possible'] = prediction
    return old_df


if __name__ == "__main__":
    prediction_pipeline()