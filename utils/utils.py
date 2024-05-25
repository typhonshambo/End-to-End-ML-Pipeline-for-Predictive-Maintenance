import os
import sys
import pickle
import logging
from typing import Union

def save_object(obj: object, file_path: str)-> Union[bool | None]:
    '''
    Save an object to a file using pickle.
    Args:
        - file_path: str: Path to the file where the object will be saved.
        - obj: object: The object to save.
    Returns:
        - True: If the object is saved
        - None: If there is an error saving the object
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to file: {file_path}")
        return True

    except Exception as e:
        logging.error(f"Error saving object to file: {e}")
        return None

def load_object(file_path: str) -> Union[object | None]:
    '''
    Load an object from a file using pickle.
    Args:
        - file_path: str: Path to the file where the object is saved.
    Returns:
        - object: The object loaded from the file.
        - None: If there is an error loading the object
    '''
    try:
        with open(file_path, "rb") as file_obj:
            logging.info(f"Loading object from file: {file_path}")
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.error(f"Error loading object from file: {e}")
        return None