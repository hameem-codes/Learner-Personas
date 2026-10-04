import pytest
import pandas as pd
import tempfile
import os
from src.data_loader import load_data
from src.config import REQUIRED_COLUMNS

def test_load_data_valid(tmp_path):
    """Test that a valid CSV is loaded correctly."""
    # Create a temporary CSV with valid columns
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df.loc[0] = ['id1', '2026-01-01', 'US', 'Web', 2, 'Premium', 15, 3, 20.5, 5, 10, 500, 0.2, 0.5, 1, 0.1, 2, True]
    
    file_path = tmp_path / "valid_data.csv"
    df.to_csv(file_path, index=False)
    
    loaded_df = load_data(str(file_path))
    
    assert len(loaded_df) == 1
    assert loaded_df['learner_id'].iloc[0] == 'id1'
    assert pd.api.types.is_datetime64_any_dtype(loaded_df['signup_date'])
    
def test_load_data_missing_file():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_load_data_missing_columns(tmp_path):
    """Test that a CSV missing required columns raises ValueError."""
    # Create a CSV missing one required column (e.g., 'country')
    columns = [c for c in REQUIRED_COLUMNS if c != 'country']
    df = pd.DataFrame(columns=columns)
    
    file_path = tmp_path / "invalid_data.csv"
    df.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="The dataset is missing required columns"):
        load_data(str(file_path))

def test_load_data_preserves_learner_id(tmp_path):
    """Test that learner_id is preserved exactly as a string (even with leading zeros)."""
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    # Using '00123' to test leading zeros
    df.loc[0] = ['00123', '2026-01-01', 'US', 'Web', 2, 'Premium', 15, 3, 20.5, 5, 10, 500, 0.2, 0.5, 1, 0.1, 2, True]
    
    file_path = tmp_path / "id_data.csv"
    df.to_csv(file_path, index=False)
    
    loaded_df = load_data(str(file_path))
    assert loaded_df['learner_id'].iloc[0] == '00123'
