# Preprocessing Report

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
