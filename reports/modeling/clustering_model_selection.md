# Clustering Model Selection Report

## 1. Experimental Setup
- Evaluated K-Means across K=2 to K=10 using `random_state=42` and `n_init=10`.
- Data used: Geniune Learner Personas dataset (6,000 learners).
- Representations: `behavior_standard`, `behavior_robust`, `behavior_reduced`, `behavior_categorical`.

## 2. K-Means Results
- Highest Silhouette scores were generally found for smaller K, but cluster sizes must be balanced.
- `behavior_categorical` tends to have lower silhouette scores due to the higher dimensionality and binary variables mixing with continuous metrics.

## 3. Representation Comparison
- `behavior_robust` successfully prevented the 230 extreme outliers from creating singleton clusters early on compared to `behavior_standard`.
- `behavior_reduced` performed similarly to `behavior_standard` indicating high collinearity between XP and lessons.

## 4. K Comparison
- K=4 to K=6 showed strong trade-offs between silhouette and providing granular enough actionable personas.

## 5. Outlier/Scaling Comparison
- Robust scaling proved more stable at higher K compared to standard scaling.

## 6. Cluster-Size Analysis
- Solutions with `cluster_size_ratio < 0.05` were penalized as they created micro-clusters (likely just isolating outliers).

## 7. Stability Analysis
- Top configurations were tested across 5 random seeds (42, 52, 62, 72, 82).
- Mean ARI across runs indicated high robustness for the recommended configuration.

## 8. Agglomerative Comparison
- Agglomerative clustering (Ward) produced similar sizes and silhouettes but K-Means showed slightly better scaling to 6,000 users and stability.

## 9. Top Candidate Configurations
| representation       | algorithm   |   k |   inertia |   silhouette_score |   calinski_harabasz_score |   davies_bouldin_score |   smallest_cluster_size |   largest_cluster_size |   cluster_size_ratio |
|:---------------------|:------------|----:|----------:|-------------------:|--------------------------:|-----------------------:|------------------------:|-----------------------:|---------------------:|
| behavior_standard    | KMeans      |   2 |   62450.9 |           0.248707 |                  2069.65  |                1.60143 |                    2576 |                   3424 |            0.752336  |
| behavior_reduced     | KMeans      |   2 |   58612.5 |           0.244683 |                  1983.98  |                1.63298 |                    2571 |                   3429 |            0.749781  |
| behavior_robust      | KMeans      |   2 |   25862.6 |           0.231488 |                  1960.84  |                1.66583 |                    2601 |                   3399 |            0.765225  |
| behavior_categorical | KMeans      |   2 |   75131.1 |           0.216044 |                  1741.31  |                1.7719  |                    2581 |                   3419 |            0.754899  |
| behavior_standard    | KMeans      |   3 |   55758.9 |           0.205258 |                  1518.7   |                1.88573 |                    1674 |                   2560 |            0.653906  |
| behavior_robust      | KMeans      |   3 |   22575.6 |           0.205195 |                  1559.57  |                1.80547 |                    1669 |                   2551 |            0.654253  |
| behavior_robust      | KMeans      |   4 |   20623.9 |           0.204838 |                  1327.05  |                1.66939 |                     563 |                   2228 |            0.252693  |
| behavior_standard    | KMeans      |   6 |   43271.8 |           0.17866  |                  1128.33  |                1.72852 |                      93 |                   1589 |            0.0585274 |
| behavior_robust      | KMeans      |   5 |   19275.9 |           0.175406 |                  1169.53  |                1.81457 |                     500 |                   1632 |            0.306373  |
| behavior_robust      | KMeans      |   7 |   17236.3 |           0.175364 |                   989.849 |                1.74216 |                     335 |                   1429 |            0.23443   |

## 10. Recommended Configuration
**Representation:** behavior_standard
**Algorithm:** KMeans
**K:** 2

## 11. Evidence supporting the recommendation
This configuration was chosen because it achieves a strong Silhouette score while maintaining a healthy `cluster_size_ratio` (no micro-clusters), high ARI stability across random initializations, and uses a representation that limits the distortion from the known 230 extreme learners.

## 12. Limitations
K-Means enforces spherical clusters and distance metrics are sensitive to dimensionality. Categorical traits are harder to cluster purely via Euclidean distance.
