import pandas as pd
from src.config import REQUIRED_COLUMNS

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset safely, validates required columns, preserves learner_id exactly, 
    parses signup_date as a date, and does not mutate the source dataframe in-place.
    
    Args:
        filepath (str): Path to the dataset CSV file.
        
    Returns:
        pd.DataFrame: A safe copy of the loaded dataframe.
    """
    try:
        # Load the dataframe, preserving learner_id exactly as string (object) to avoid losing precision or leading zeros if it's numeric-like
        # but in this case, we just read as object
        df = pd.read_csv(filepath, dtype={'learner_id': 'object'})
    except FileNotFoundError:
        raise FileNotFoundError(f"The expected dataset file was not found at: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The dataset file is empty: {filepath}")
        
    # Validate required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")
        
    # Create a copy so we never mutate the source dataframe in-place if passed directly
    df_safe = df.copy()
    
    # Parse signup_date as date
    df_safe['signup_date'] = pd.to_datetime(df_safe['signup_date'], errors='coerce')
    
    return df_safe
