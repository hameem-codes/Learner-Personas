import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import ORIGINAL_DATASET_PATH, DATA_DIR, PROJECT_ROOT
from src.data_loader import load_data

# Ensure output directories exist
OUTPUTS_DIR = DATA_DIR / 'outputs'
PLOTS_DIR = OUTPUTS_DIR / 'plots'
REPORTS_DIR = PROJECT_ROOT / 'reports'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def audit_data_quality(df: pd.DataFrame):
    """
    Perform a data quality audit and save the results.
    """
    audit = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_learner_id': df.duplicated(subset=['learner_id']).sum(),
        'date_range_signup': (df['signup_date'].min(), df['signup_date'].max())
    }
    
    missing = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': df.isnull().mean() * 100
    })
    missing.to_csv(OUTPUTS_DIR / 'missing_values.csv')
    
    return audit, missing

def check_logical_ranges(df: pd.DataFrame):
    """
    Check if share variables are between 0 and 1, and count/activity variables are non-negative.
    Returns a dataframe of suspicious counts.
    """
    suspicious = []
    
    share_cols = [c for c in df.columns if 'share' in c]
    for col in share_cols:
        count = ((df[col] < 0) | (df[col] > 1)).sum()
        if count > 0 or df[col].notna().any():
            suspicious.append({
                'feature': col,
                'suspicious_count': count,
                'minimum': df[col].min(),
                'maximum': df[col].max(),
                'interpretation': 'Share values should be between 0 and 1'
            })
            
    count_cols = ['courses_active', 'days_active_last_30', 'sessions_per_week', 
                  'avg_session_minutes', 'longest_streak_days', 'lessons_completed_90d', 
                  'xp_earned_90d', 'leaderboard_weeks_joined', 'days_since_last_active']
    for col in count_cols:
        if col in df.columns:
            count = (df[col] < 0).sum()
            if count > 0 or df[col].notna().any():
                suspicious.append({
                    'feature': col,
                    'suspicious_count': count,
                    'minimum': df[col].min(),
                    'maximum': df[col].max(),
                    'interpretation': 'Count/Activity values should be non-negative'
                })
                
    suspicious_df = pd.DataFrame(suspicious)
    suspicious_df.to_csv(OUTPUTS_DIR / 'logical_ranges_suspicious.csv', index=False)
    return suspicious_df

def analyze_numeric_features(df: pd.DataFrame):
    """
    Analyze numeric features: skewness, mean, median, standard deviation.
    Generate histograms and boxplots.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = []
    for col in numeric_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        skew_val = df[col].skew()
        
        likely_transform = 'Log/Sqrt Transform' if abs(skew_val) > 1 else 'None'
        
        summary.append({
            'feature': col,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'skewness': skew_val,
            'likely_transform': likely_transform
        })
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUTS_DIR / 'numerical_summary.csv', index=False)
    
    # Plotting histograms
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {col}')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '02_numeric_distributions.png')
    plt.close()
    
    # Plotting boxplots
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 2 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '03_boxplots.png')
    plt.close()
    
    return summary_df

def analyze_categorical_features(df: pd.DataFrame):
    """
    Analyze categorical features: frequencies, rare categories.
    """
    cat_cols = ['country', 'platform', 'subscription', 'notifications_enabled']
    
    summary = []
    fig, axes = plt.subplots(len(cat_cols), 1, figsize=(10, 5 * len(cat_cols)))
    if len(cat_cols) == 1:
        axes = [axes]
        
    for i, col in enumerate(cat_cols):
        if col in df.columns:
            counts = df[col].value_counts()
            percs = df[col].value_counts(normalize=True) * 100
            
            for cat, count in counts.items():
                summary.append({
                    'feature': col,
                    'category': cat,
                    'count': count,
                    'percentage': percs[cat]
                })
                
            sns.countplot(y=col, data=df, ax=axes[i], order=counts.index)
            axes[i].set_title(f'Distribution of {col}')
            
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUTS_DIR / 'categorical_summary.csv', index=False)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '05_categorical_distributions.png')
    plt.close()
    
    return summary_df

def analyze_correlations(df: pd.DataFrame):
    """
    Pearson correlation matrix and heatmap for numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    corr.to_csv(OUTPUTS_DIR / 'correlation_matrix.csv')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '04_correlation_heatmap.png')
    plt.close()
    
    return corr

def analyze_behavior_relationships(df: pd.DataFrame):
    """
    Scatter plots and logical relationship checks.
    """
    pairs = [
        ('days_active_last_30', 'sessions_per_week'),
        ('sessions_per_week', 'avg_session_minutes'),
        ('lessons_completed_90d', 'xp_earned_90d'),
        ('days_active_last_30', 'longest_streak_days'),
        ('days_since_last_active', 'days_active_last_30')
    ]
    
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 5 * len(pairs)))
    for i, (x_col, y_col) in enumerate(pairs):
        if x_col in df.columns and y_col in df.columns:
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=axes[i], alpha=0.5)
            axes[i].set_title(f'{y_col} vs {x_col}')
            
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '06_behavior_relationships.png')
    plt.close()

def detect_outliers(df: pd.DataFrame):
    """
    Calculate IQR based outliers for numeric features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_info = []
    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask[col] = is_outlier
        
        num_outliers = is_outlier.sum()
        outliers_info.append({
            'feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'number_of_outliers': num_outliers,
            'percentage_of_outliers': (num_outliers / len(df)) * 100
        })
        
    outliers_df = pd.DataFrame(outliers_info)
    outliers_df.to_csv(OUTPUTS_DIR / 'outlier_summary.csv', index=False)
    
    df['total_outlier_flags'] = outlier_mask.sum(axis=1)
    extreme_learners = df[df['total_outlier_flags'] >= 3].copy()
    extreme_learners.to_csv(OUTPUTS_DIR / 'extreme_learners.csv', index=False)
    
    return outliers_df, extreme_learners

def build_feature_role_table(df: pd.DataFrame):
    """
    Define feature roles based on data types and meaning.
    """
    roles = []
    for col in df.columns:
        if col == 'learner_id':
            role = 'identifier'
            reason = 'Unique identifier for learners, cannot be used for clustering.'
        elif col == 'signup_date':
            role = 'date'
            reason = 'Temporal feature, needs transformation to recency or tenure to be useful in clustering.'
        elif col == 'total_outlier_flags':
            role = 'exclude_from_clustering'
            reason = 'Engineered flag during EDA.'
        elif df[col].dtype == 'object' or df.nunique()[col] < 10:
            role = 'categorical'
            reason = 'Contains discrete textual or low-cardinality values.'
        else:
            role = 'candidate_clustering_feature'
            reason = 'Numerical behavioral metric reflecting user engagement.'
            
        roles.append({
            'feature': col,
            'role': role,
            'reason': reason
        })
        
    roles_df = pd.DataFrame(roles)
    roles_df.to_csv(OUTPUTS_DIR / 'feature_roles.csv', index=False)
    return roles_df

def generate_report():
    """
    Main function to run the full EDA and generate report.
    """
    print("Loading data...")
    df = load_data(ORIGINAL_DATASET_PATH)
    
    print("Auditing data quality...")
    audit, missing = audit_data_quality(df)
    
    print("Checking logical ranges...")
    suspicious_df = check_logical_ranges(df)
    
    print("Analyzing numeric features...")
    numeric_summary = analyze_numeric_features(df)
    
    print("Analyzing categorical features...")
    cat_summary = analyze_categorical_features(df)
    
    print("Analyzing correlations...")
    corr_matrix = analyze_correlations(df)
    
    print("Analyzing behavioral relationships...")
    analyze_behavior_relationships(df)
    
    print("Detecting outliers...")
    outliers_summary, extreme_learners = detect_outliers(df)
    
    print("Building feature role table...")
    roles_df = build_feature_role_table(df)
    
    print("Writing EDA report...")
    report_content = f"""# Learner Personas Clustering - EDA Report

## Data Quality Audit
- Total Rows: {audit['num_rows']}
- Total Columns: {audit['num_columns']}
- Duplicate Rows: {audit['duplicate_rows']}
- Duplicate Learner IDs: {audit['duplicate_learner_id']}
- Signup Date Range: {audit['date_range_signup'][0]} to {audit['date_range_signup'][1]}

### Missing Values
{missing[missing['missing_count'] > 0].to_markdown()}

## Logical Ranges
{suspicious_df.to_markdown(index=False)}

## Highly Skewed Features
{numeric_summary[numeric_summary['skewness'].abs() > 1].to_markdown(index=False)}

## Outliers Summary
{outliers_summary.to_markdown(index=False)}

## Feature Roles
{roles_df.to_markdown(index=False)}

## Decisions Requiring Review
1. **Handling Missing Values:** `avg_session_minutes` has {missing.loc['avg_session_minutes', 'missing_count']} missing values. We should impute this, likely with 0 or the median, since missing might imply zero sessions.
2. **Transformations for Highly Skewed Data:** Some features are highly right-skewed. K-Means assumes spherical clusters, so applying log or sqrt transformations to these is recommended.
3. **Handling Extreme Outliers:** We identified {len(extreme_learners)} learners with 3 or more outlier flags. They are likely power users (high XP, high lessons). We should retain them but consider robust scaling or transformations so they don't dominate cluster centroids.
4. **Correlated Features:** Some features like `lessons_completed_90d` and `xp_earned_90d` might be highly correlated. We need to decide whether to drop one or use PCA to avoid giving too much weight to lesson volume.
5. **Feature Engineering:** `signup_date` cannot be used directly. We should transform it into `account_age_days`.
"""
    with open(REPORTS_DIR / 'eda_report.md', 'w') as f:
        f.write(report_content)
        
    print("EDA Complete. Artifacts saved in data/outputs/ and reports/.")

if __name__ == '__main__':
    generate_report()
