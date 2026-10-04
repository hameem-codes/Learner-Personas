import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.eda import check_logical_ranges, analyze_numeric_features, detect_outliers
from src.config import REQUIRED_COLUMNS
from src.data_loader import load_data

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        'learner_id': ['id1', 'id2', 'id3'],
        'signup_date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'country': ['US', 'UK', 'CA'],
        'platform': ['Web', 'iOS', 'Android'],
        'courses_active': [1, 2, 1],
        'subscription': ['Premium', 'Free', 'Premium'],
        'days_active_last_30': [10, 20, 5],
        'sessions_per_week': [2.5, 5.0, 1.0],
        'avg_session_minutes': [15.0, 30.0, 10.0],
        'longest_streak_days': [5, 10, 2],
        'lessons_completed_90d': [50, 100, 20],
        'xp_earned_90d': [500, 1000, 200],
        'share_sessions_weekend': [0.2, 0.5, 1.5], # 1.5 is suspicious
        'share_sessions_morning': [0.5, 0.5, 0.5],
        'leaderboard_weeks_joined': [2, 4, 1],
        'share_lessons_new_content': [0.1, -0.1, 0.5], # -0.1 is suspicious
        'days_since_last_active': [2, 1, 10],
        'notifications_enabled': [True, False, True]
    })
    return df

def test_check_logical_ranges(sample_df):
    suspicious_df = check_logical_ranges(sample_df)
    assert len(suspicious_df) > 0
    features_flagged = suspicious_df['feature'].tolist()
    assert 'share_sessions_weekend' in features_flagged
    assert 'share_lessons_new_content' in features_flagged

def test_analyze_numeric_features(sample_df):
    summary_df = analyze_numeric_features(sample_df)
    assert not summary_df.empty
    assert 'mean' in summary_df.columns
    assert 'median' in summary_df.columns
    assert 'skewness' in summary_df.columns

def test_detect_outliers(sample_df):
    outliers_df, extreme_learners = detect_outliers(sample_df)
    assert not outliers_df.empty
    assert 'number_of_outliers' in outliers_df.columns
    assert 'total_outlier_flags' in extreme_learners.columns

def test_duplicate_learner_ids(sample_df):
    sample_df.loc[3] = sample_df.loc[0]
    assert sample_df.duplicated(subset=['learner_id']).sum() == 1
