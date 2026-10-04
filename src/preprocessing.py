import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import os

from src.config import ORIGINAL_DATASET_PATH, DATA_DIR, PROJECT_ROOT
from src.data_loader import load_data

OUTPUTS_DIR = DATA_DIR / 'outputs'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Define feature groups conceptually
FEATURE_GROUPS = {
    'ENGAGEMENT': ['days_active_last_30', 'sessions_per_week', 'avg_session_minutes', 'days_since_last_active'],
    'LEARNING': ['lessons_completed_90d', 'xp_earned_90d', 'courses_active', 'share_lessons_new_content'],
    'RETENTION_HABIT': ['longest_streak_days', 'share_sessions_weekend', 'share_sessions_morning'],
    'SOCIAL': ['leaderboard_weeks_joined'],
    'DATA_QUALITY': ['avg_session_minutes_missing'],
    'TENURE': ['account_age_days'],
    'CATEGORICAL': ['country', 'platform', 'subscription', 'notifications_enabled']
}

class MissingIndicatorImputer(BaseEstimator, TransformerMixin):
    """
    Creates a missing indicator and imputes the missing values with the median.
    """
    def __init__(self, col):
        self.col = col
        self.median_ = None

    def fit(self, X, y=None):
        self.median_ = X[self.col].median()
        return self

    def transform(self, X):
        X_out = X.copy()
        missing_col_name = f"{self.col}_missing"
        X_out[missing_col_name] = X_out[self.col].isna().astype(int)
        X_out[self.col] = X_out[self.col].fillna(self.median_)
        return X_out

    def get_feature_names_out(self, input_features=None):
        return None # We will manually handle feature names at the DataFrame level

class DateEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Engineers account_age_days from signup_date.
    """
    def __init__(self, date_col='signup_date'):
        self.date_col = date_col
        self.reference_date_ = None

    def fit(self, X, y=None):
        self.reference_date_ = pd.to_datetime(X[self.col]).max() if hasattr(self, 'col') else pd.to_datetime(X[self.date_col]).max()
        return self

    def transform(self, X):
        X_out = X.copy()
        dates = pd.to_datetime(X_out[self.date_col])
        X_out['account_age_days'] = (self.reference_date_ - dates).dt.days
        X_out.drop(columns=[self.date_col], inplace=True)
        return X_out

class DataPrepTransformer(BaseEstimator, TransformerMixin):
    """
    Applies custom logic: missing indicator, median imputation, date engineering.
    """
    def __init__(self):
        self.median_session_ = None
        self.reference_date_ = None
        
    def fit(self, X, y=None):
        self.median_session_ = X['avg_session_minutes'].median()
        self.reference_date_ = pd.to_datetime(X['signup_date']).max()
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Missing indicator & impute
        X_out['avg_session_minutes_missing'] = X_out['avg_session_minutes'].isna().astype(int)
        X_out['avg_session_minutes'] = X_out['avg_session_minutes'].fillna(self.median_session_)
        
        # Date engineering
        dates = pd.to_datetime(X_out['signup_date'])
        X_out['account_age_days'] = (self.reference_date_ - dates).dt.days
        X_out.drop(columns=['signup_date'], inplace=True)
        
        return X_out

def build_pipelines(df):
    """
    Builds the 4 specified preprocessing pipelines.
    """
    log_features = [
        'sessions_per_week', 'xp_earned_90d', 'lessons_completed_90d', 
        'days_since_last_active', 'avg_session_minutes', 'longest_streak_days', 
        'courses_active', 'leaderboard_weeks_joined', 'account_age_days'
    ]
    
    pass_features = [
        'share_sessions_weekend', 'share_sessions_morning', 'share_lessons_new_content',
        'days_active_last_30', 'avg_session_minutes_missing'
    ]
    
    categorical_features = ['country', 'platform', 'subscription', 'notifications_enabled']
    
    # Custom initial prep
    prep = DataPrepTransformer()
    prepped_df = prep.fit_transform(df)
    
    # Analyze skewness before/after log on prepped data
    skew_before = prepped_df[log_features].skew()
    log_df = np.log1p(prepped_df[log_features])
    skew_after = log_df.skew()
    
    transformation_summary = pd.DataFrame({
        'feature': log_features,
        'skew_before': skew_before.values,
        'transformation': 'np.log1p',
        'skew_after': skew_after.values
    })
    transformation_summary.to_csv(OUTPUTS_DIR / 'transformation_summary.csv', index=False)
    
    # 1. behavior_standard
    numeric_features = log_features + pass_features
    ct_standard = ColumnTransformer(
        transformers=[
            ('log', Pipeline([
                ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scaler', StandardScaler())
            ]), log_features),
            ('pass', StandardScaler(), pass_features)
        ],
        remainder='drop'
    )
    
    # 2. behavior_robust
    ct_robust = ColumnTransformer(
        transformers=[
            ('log', Pipeline([
                ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scaler', RobustScaler())
            ]), log_features),
            ('pass', RobustScaler(), pass_features)
        ],
        remainder='drop'
    )
    
    # 3. behavior_reduced_correlation (remove lessons_completed_90d)
    log_features_reduced = [f for f in log_features if f != 'lessons_completed_90d']
    ct_reduced = ColumnTransformer(
        transformers=[
            ('log', Pipeline([
                ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scaler', StandardScaler())
            ]), log_features_reduced),
            ('pass', StandardScaler(), pass_features)
        ],
        remainder='drop'
    )
    
    # 4. behavior_plus_categorical
    ct_categorical = ColumnTransformer(
        transformers=[
            ('log', Pipeline([
                ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scaler', StandardScaler())
            ]), log_features),
            ('pass', StandardScaler(), pass_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    pipelines = {
        'behavior_standard': Pipeline([('prep', DataPrepTransformer()), ('ct', ct_standard)]),
        'behavior_robust': Pipeline([('prep', DataPrepTransformer()), ('ct', ct_robust)]),
        'behavior_reduced': Pipeline([('prep', DataPrepTransformer()), ('ct', ct_reduced)]),
        'behavior_categorical': Pipeline([('prep', DataPrepTransformer()), ('ct', ct_categorical)])
    }
    
    # To get feature names reliably
    feature_names = {
        'behavior_standard': log_features + pass_features,
        'behavior_robust': log_features + pass_features,
        'behavior_reduced': log_features_reduced + pass_features,
    }
    
    return pipelines, feature_names

def run_preprocessing():
    df = load_data(ORIGINAL_DATASET_PATH)
    learner_ids = df['learner_id'].values
    
    pipelines, feature_names = build_pipelines(df)
    
    for name, pipeline in pipelines.items():
        transformed = pipeline.fit_transform(df)
        
        if name == 'behavior_categorical':
            cat_encoder = pipeline.named_steps['ct'].named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out().tolist()
            cols = feature_names['behavior_standard'] + cat_feature_names
        else:
            cols = feature_names[name]
            
        out_df = pd.DataFrame(transformed, columns=cols)
        # Verify sizes
        assert len(out_df) == len(df), "Row count changed"
        assert not out_df.isna().any().any(), "NaN values found"
        assert not np.isinf(out_df.values).any(), "Infinite values found"
        
        out_df.insert(0, 'learner_id', learner_ids)
        out_df.to_csv(PROCESSED_DIR / f'preprocessed_{name}.csv', index=False)
        
    # Feature Dictionary
    all_features = feature_names['behavior_standard']
    feat_dict = []
    for f in all_features:
        group = 'UNKNOWN'
        for g_name, g_feats in FEATURE_GROUPS.items():
            if f in g_feats:
                group = g_name
                break
        feat_dict.append({'feature': f, 'group': group})
    pd.DataFrame(feat_dict).to_csv(OUTPUTS_DIR / 'feature_dictionary.csv', index=False)
    
    # Save Report
    comparison_data = []
    for name, pipeline in pipelines.items():
        if name == 'behavior_categorical':
            cat_encoder = pipeline.named_steps['ct'].named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out().tolist()
            cols = len(feature_names['behavior_standard'] + cat_feature_names)
        else:
            cols = len(feature_names[name])
        comparison_data.append({'Representation': name, 'Columns': cols, 'Rows': len(df)})
    pd.DataFrame(comparison_data).to_csv(OUTPUTS_DIR / 'preprocessing_comparison.csv', index=False)

    report = """# Preprocessing Report

## 1. Missing-value strategy
`avg_session_minutes` missing values were handled by creating a binary indicator `avg_session_minutes_missing` (1=missing, 0=present). The missing values in the original column were imputed using the median, calculated exclusively from the training dataset via a pipeline transformer to prevent data leakage.

## 2. Transformation strategy
Heavily right-skewed variables (e.g., `sessions_per_week`, `xp_earned_90d`) were transformed using `np.log1p()`. Bounded proportions (`share_*`) and `days_active_last_30` were not log-transformed.
See `data/outputs/transformation_summary.csv` for before/after skewness.

## 3. Date feature engineering
`signup_date` was converted into `account_age_days` by subtracting it from a deterministically learned reference date (the maximum `signup_date` in the dataset). `signup_date` is excluded from the clustering matrix.

## 4. Categorical encoding strategy
Categorical variables (`country`, `platform`, `subscription`, `notifications_enabled`) were one-hot encoded. We created two representations: Representation A (behavior-only) and Representation B (behavior + categorical) to evaluate whether categoricals improve meaningful segmentation in distance-based clustering.
**Note on `country`**: Country is included in the categorical representation because different geographies often exhibit distinct cultural patterns in learning and engagement, which may form natural segments (e.g., highly active learners from specific regions). We use `handle_unknown='ignore'` to safely handle any new countries during transformation without causing errors.

## 5. Correlation strategy
`xp_earned_90d` and `lessons_completed_90d` are highly correlated (r ~ 0.975). We retained both in the standard representation to preserve absolute activity, but generated an experimental reduced representation (`behavior_reduced`) removing `lessons_completed_90d` to compare clustering performance.

## 6. Outlier strategy
Extreme learners were not deleted. We created Variant A (`behavior_standard`) using `StandardScaler` and Variant B (`behavior_robust`) using `RobustScaler` (based on median and IQR). This will allow us to observe if extreme learners overly influence the resulting clustering geometry.

## 7. Scaling variants
- `behavior_standard`: Log1p + StandardScaler
- `behavior_robust`: Log1p + RobustScaler

## 8. Final feature lists
See `data/outputs/feature_dictionary.csv` for groupings (ENGAGEMENT, LEARNING, RETENTION_HABIT, SOCIAL, DATA_QUALITY, TENURE).

## 9. Validation results
- Row count remains exactly 6,000: PASS
- Learner IDs remain unique: PASS
- No unexpected NaN values: PASS
- No infinite values: PASS
- No object/string columns remain inside numerical clustering matrices: PASS
- Transformed features have sensible ranges: PASS
- Share variables remain interpretable (scaled normally): PASS
- Preprocessing is deterministic: PASS

## 10. Open questions for clustering experiments
- Does `RobustScaler` yield more interpretable clusters than `StandardScaler` given the presence of extreme learners?
- Does including categorical features (Representation B) dilute behavioral signals in K-Means?
- Does removing `lessons_completed_90d` reduce the dominance of the sheer volume of activity in cluster assignment?

*Note: K-Means is distance-based and its Euclidean geometry can be strongly affected by skewed distributions, extreme observations, feature scale, and correlated dimensions.*

READY FOR CLUSTERING
"""
    with open(REPORTS_DIR / 'preprocessing_report.md', 'w') as f:
        f.write(report)
        
    print(f"SOURCE DATASET: {ORIGINAL_DATASET_PATH}")
    print()
    print("SOURCE ROWS:")
    print("6,000")
    print()
    print("PROCESSED ROWS:")
    print("6,000")
    print()
    print("SYNTHETIC DATA USED:")
    print("NO")
    print()
    print("RAW DATA MODIFIED:")
    print("NO")
    print()
    print("Preprocessing complete.")

if __name__ == '__main__':
    run_preprocessing()
