# Learner Personas Clustering - EDA Report

## Data Quality Audit
- Total Rows: 6000
- Total Columns: 18
- Duplicate Rows: 0
- Duplicate Learner IDs: 0
- Signup Date Range: 2023-01-01 00:00:00 to 2025-06-19 00:00:00

### Missing Values
|                     |   missing_count |   missing_percentage |
|:--------------------|----------------:|---------------------:|
| avg_session_minutes |              93 |                 1.55 |

## Logical Ranges
| feature                   |   suspicious_count |   minimum |   maximum | interpretation                               |
|:--------------------------|-------------------:|----------:|----------:|:---------------------------------------------|
| share_sessions_weekend    |                  0 |       0   |       1   | Share values should be between 0 and 1       |
| share_sessions_morning    |                  0 |       0   |       1   | Share values should be between 0 and 1       |
| share_lessons_new_content |                  0 |       0   |       1   | Share values should be between 0 and 1       |
| courses_active            |                  0 |       1   |       3   | Count/Activity values should be non-negative |
| days_active_last_30       |                  0 |       0   |      30   | Count/Activity values should be non-negative |
| sessions_per_week         |                  0 |       0.1 |     232.7 | Count/Activity values should be non-negative |
| avg_session_minutes       |                  0 |       1   |     564   | Count/Activity values should be non-negative |
| longest_streak_days       |                  0 |       1   |    3169   | Count/Activity values should be non-negative |
| lessons_completed_90d     |                  0 |       1   |   10151   | Count/Activity values should be non-negative |
| xp_earned_90d             |                  0 |      14   |  150369   | Count/Activity values should be non-negative |
| leaderboard_weeks_joined  |                  0 |       0   |      13   | Count/Activity values should be non-negative |
| days_since_last_active    |                  0 |       0   |     535   | Count/Activity values should be non-negative |

## Highly Skewed Features
| feature                  |       mean |   median |          std |   skewness | likely_transform   |
|:-------------------------|-----------:|---------:|-------------:|-----------:|:-------------------|
| courses_active           |    1.36483 |      1   |     0.608653 |    1.4504  | Log/Sqrt Transform |
| sessions_per_week        |    8.17843 |      3.7 |    15.4734   |    6.67734 | Log/Sqrt Transform |
| avg_session_minutes      |   24.4643  |     11.2 |    36.4607   |    4.467   | Log/Sqrt Transform |
| longest_streak_days      |  109.71    |     14   |   212.395    |    4.11046 | Log/Sqrt Transform |
| lessons_completed_90d    |  347.376   |    146   |   687.206    |    6.22873 | Log/Sqrt Transform |
| xp_earned_90d            | 6282.68    |   3025   | 11686.8      |    6.36678 | Log/Sqrt Transform |
| leaderboard_weeks_joined |    3.42767 |      2   |     3.87633  |    1.03825 | Log/Sqrt Transform |
| days_since_last_active   |   11.0613  |      4   |    20.773    |    5.94434 | Log/Sqrt Transform |

## Outliers Summary
| feature                   |      Q1 |      Q3 |     IQR |   lower_bound |   upper_bound |   number_of_outliers |   percentage_of_outliers |
|:--------------------------|--------:|--------:|--------:|--------------:|--------------:|---------------------:|-------------------------:|
| courses_active            |    1    |    2    |    1    |        -0.5   |         3.5   |                    0 |                  0       |
| days_active_last_30       |    4    |   24    |   20    |       -26     |        54     |                    0 |                  0       |
| sessions_per_week         |    1.5  |    9.2  |    7.7  |       -10.05  |        20.75  |                  477 |                  7.95    |
| avg_session_minutes       |    4.9  |   29.8  |   24.9  |       -32.45  |        67.15  |                  486 |                  8.1     |
| longest_streak_days       |    4    |  127.25 |  123.25 |      -180.875 |       312.125 |                  628 |                 10.4667  |
| lessons_completed_90d     |   42    |  370    |  328    |      -450     |       862     |                  543 |                  9.05    |
| xp_earned_90d             | 1191.75 | 6695.5  | 5503.75 |     -7063.88  |     14951.1   |                  534 |                  8.9     |
| share_sessions_weekend    |    0.21 |    0.58 |    0.37 |        -0.345 |         1.135 |                    0 |                  0       |
| share_sessions_morning    |    0.25 |    0.61 |    0.36 |        -0.29  |         1.15  |                    0 |                  0       |
| leaderboard_weeks_joined  |    0    |    6    |    6    |        -9     |        15     |                    0 |                  0       |
| share_lessons_new_content |    0.47 |    0.79 |    0.32 |        -0.01  |         1.27  |                    0 |                  0       |
| days_since_last_active    |    0    |   12    |   12    |       -18     |        30     |                  595 |                  9.91667 |

## Feature Roles
| feature                   | role                         | reason                                                                                  |
|:--------------------------|:-----------------------------|:----------------------------------------------------------------------------------------|
| learner_id                | identifier                   | Unique identifier for learners, cannot be used for clustering.                          |
| signup_date               | date                         | Temporal feature, needs transformation to recency or tenure to be useful in clustering. |
| country                   | categorical                  | Contains discrete textual or low-cardinality values.                                    |
| platform                  | categorical                  | Contains discrete textual or low-cardinality values.                                    |
| courses_active            | categorical                  | Contains discrete textual or low-cardinality values.                                    |
| subscription              | categorical                  | Contains discrete textual or low-cardinality values.                                    |
| days_active_last_30       | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| sessions_per_week         | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| avg_session_minutes       | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| longest_streak_days       | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| lessons_completed_90d     | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| xp_earned_90d             | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| share_sessions_weekend    | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| share_sessions_morning    | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| leaderboard_weeks_joined  | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| share_lessons_new_content | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| days_since_last_active    | candidate_clustering_feature | Numerical behavioral metric reflecting user engagement.                                 |
| notifications_enabled     | categorical                  | Contains discrete textual or low-cardinality values.                                    |
| total_outlier_flags       | exclude_from_clustering      | Engineered flag during EDA.                                                             |

## Decisions Requiring Review
1. **Handling Missing Values:** `avg_session_minutes` has 93 missing values. We should impute this, likely with 0 or the median, since missing might imply zero sessions.
2. **Transformations for Highly Skewed Data:** Some features are highly right-skewed. K-Means assumes spherical clusters, so applying log or sqrt transformations to these is recommended.
3. **Handling Extreme Outliers:** We identified 230 learners with 3 or more outlier flags. They are likely power users (high XP, high lessons). We should retain them but consider robust scaling or transformations so they don't dominate cluster centroids.
4. **Correlated Features:** Some features like `lessons_completed_90d` and `xp_earned_90d` might be highly correlated. We need to decide whether to drop one or use PCA to avoid giving too much weight to lesson volume.
5. **Feature Engineering:** `signup_date` cannot be used directly. We should transform it into `account_age_days`.
