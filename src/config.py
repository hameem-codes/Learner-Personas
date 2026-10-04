import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = DATA_DIR / 'outputs'

# Note: as per discovery, the original dataset is downloaded to this path
ORIGINAL_DATASET_PATH = Path(r"C:\ML_Projects\learner-personas-clustering-datasets\learner-personas-clustering\learners.csv")

REQUIRED_COLUMNS = [
    'learner_id',
    'signup_date',
    'country',
    'platform',
    'courses_active',
    'subscription',
    'days_active_last_30',
    'sessions_per_week',
    'avg_session_minutes',
    'longest_streak_days',
    'lessons_completed_90d',
    'xp_earned_90d',
    'share_sessions_weekend',
    'share_sessions_morning',
    'leaderboard_weeks_joined',
    'share_lessons_new_content',
    'days_since_last_active',
    'notifications_enabled'
]
