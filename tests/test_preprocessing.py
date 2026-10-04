import pytest
import pandas as pd
import numpy as np
import os
from src.preprocessing import build_pipelines
from src.config import REQUIRED_COLUMNS, ORIGINAL_DATASET_PATH
from src.data_loader import load_data

def test_real_data_loading():
    # Only test if the file actually exists
    if ORIGINAL_DATASET_PATH.exists():
        df = load_data(ORIGINAL_DATASET_PATH)
        assert len(df) == 6000
        assert len(df.columns) == 18
        assert 'learner_id' in df.columns


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame({
        'learner_id': ['id1', 'id2', 'id3', 'id4', 'id5'] * 1200, # 6000 rows
        'signup_date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05'] * 1200,
        'country': ['US', 'UK', 'CA', 'AU', 'IN'] * 1200,
        'platform': ['Web', 'iOS', 'Android', 'Web', 'iOS'] * 1200,
        'courses_active': [1, 2, 1, 3, 2] * 1200,
        'subscription': ['Premium', 'Free', 'Premium', 'Free', 'Free'] * 1200,
        'days_active_last_30': [10, 20, 5, 25, 0] * 1200,
        'sessions_per_week': [2.5, 5.0, 1.0, 10.0, 0.0] * 1200,
        'avg_session_minutes': [15.0, 30.0, np.nan, 45.0, 5.0] * 1200,
        'longest_streak_days': [5, 10, 2, 30, 0] * 1200,
        'lessons_completed_90d': [50, 100, 20, 500, 0] * 1200,
        'xp_earned_90d': [500, 1000, 200, 5000, 0] * 1200,
        'share_sessions_weekend': [0.2, 0.5, 0.0, 0.8, 0.1] * 1200,
        'share_sessions_morning': [0.5, 0.5, 0.2, 0.1, 0.9] * 1200,
        'leaderboard_weeks_joined': [2, 4, 1, 10, 0] * 1200,
        'share_lessons_new_content': [0.1, 0.2, 0.5, 0.9, 0.0] * 1200,
        'days_since_last_active': [2, 1, 10, 0, 30] * 1200,
        'notifications_enabled': [True, False, True, False, True] * 1200
    }).assign(learner_id=lambda x: x['learner_id'] + x.index.astype(str))

@pytest.fixture(scope="module")
def transformed_dfs(df):
    pipelines, feature_names = build_pipelines(df)
    results = {}
    learner_ids = df['learner_id'].values
    
    for name, pipeline in pipelines.items():
        transformed = pipeline.fit_transform(df)
        
        if name == 'behavior_categorical':
            cat_encoder = pipeline.named_steps['ct'].named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out().tolist()
            cols = feature_names['behavior_standard'] + cat_feature_names
        else:
            cols = feature_names[name]
            
        out_df = pd.DataFrame(transformed, columns=cols)
        out_df.insert(0, 'learner_id', learner_ids)
        results[name] = out_df
    return results

def test_row_count(transformed_dfs):
    for name, df in transformed_dfs.items():
        assert len(df) == 6000, f"Row count changed for {name}"

def test_learner_ids_unique(transformed_dfs):
    for name, df in transformed_dfs.items():
        assert df['learner_id'].nunique() == len(df), f"Learner IDs not unique in {name}"

def test_no_nan_values(transformed_dfs):
    for name, df in transformed_dfs.items():
        assert not df.isna().any().any(), f"NaN values found in {name}"

def test_no_infinite_values(transformed_dfs):
    for name, df in transformed_dfs.items():
        numeric_df = df.drop(columns=['learner_id'])
        assert not np.isinf(numeric_df.values).any(), f"Infinite values found in {name}"

def test_no_object_columns(transformed_dfs):
    for name, df in transformed_dfs.items():
        numeric_df = df.drop(columns=['learner_id'])
        assert not any(numeric_df.dtypes == 'object'), f"Object columns remain in {name}"

def test_deterministic(df):
    pipelines1, _ = build_pipelines(df)
    pipelines2, _ = build_pipelines(df)
    
    res1 = pipelines1['behavior_standard'].fit_transform(df)
    res2 = pipelines2['behavior_standard'].fit_transform(df)
    
    np.testing.assert_array_equal(res1, res2, err_msg="Preprocessing is not deterministic")
