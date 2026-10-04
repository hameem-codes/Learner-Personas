# Clustering Model Selection Audit

## 1. Current Recommendation
- **Representation:** `behavior_standard`
- **Algorithm:** `K-Means`
- **K:** `2`

## 2. Metric Leaderboard (Top 5)

**Top by Silhouette Score:**
| representation | K | Silhouette | CH Score | DB Score | Cluster Ratio |
|---|---|---|---|---|---|
| behavior_reduced | 3 | 0.2557 | 1422.17 | 1.305 | 0.027 |
| behavior_standard | 2 | 0.2487 | 2069.65 | 1.601 | 0.752 |
| behavior_reduced | 2 | 0.2446 | 1983.98 | 1.632 | 0.749 |
| behavior_robust | 2 | 0.2314 | 1960.84 | 1.665 | 0.765 |
| behavior_categorical | 2 | 0.2160 | 1741.31 | 1.771 | 0.754 |

## 3. Representation Comparison
| representation | best_silhouette_k (Score) | best_ch_k (Score) | best_db_k (Score) | best_balance_k (Ratio) |
|---|---|---|---|---|
| behavior_standard | 2 (0.248) | 2 (2069.6) | 2 (1.601) | 2 (0.752) |
| behavior_robust | 2 (0.231) | 2 (1960.8) | 2 (1.665) | 2 (0.765) |
| behavior_reduced | 3 (0.255) | 2 (1983.9) | 3 (1.305) | 2 (0.749) |
| behavior_categorical | 2 (0.216) | 2 (1741.3) | 4 (1.723) | 2 (0.754) |

## 4. K=2 vs K>=3 Comparison
Focusing on the strongest variants (`behavior_standard` and `behavior_robust`):

| rep | K | Silhouette | CH | DB | Min % | Max % | Ratio | Mean ARI |
|---|---|---|---|---|---|---|---|---|
| standard | 2 | 0.2487 | 2069.6 | 1.601 | 42.9% | 57.0% | 0.752 | 1.000 |
| standard | 3 | 0.2052 | 1518.7 | 1.885 | 27.9% | 42.6% | 0.653 | 1.000 |
| standard | 4 | 0.1663 | 1216.3 | 2.147 | 16.1% | 29.3% | 0.551 | N/A |
| robust | 2 | 0.2314 | 1960.8 | 1.665 | 43.3% | 56.6% | 0.765 | 1.000 |
| robust | 3 | 0.2051 | 1559.5 | 1.805 | 27.8% | 42.5% | 0.654 | 0.999 |
| robust | 4 | 0.2048 | 1327.0 | 1.669 | 9.38% | 37.1% | 0.252 | 0.999 |

## 5. Stability Comparison
- `K=2` achieves perfect stability (`Mean ARI = 1.000`) across all representations.
- `behavior_robust` maintains exceptionally high stability at `K=3` (`ARI = 0.9996`) and `K=4` (`ARI = 0.9998`).
- `behavior_standard` degrades significantly at `K=6` (`ARI = 0.798`).

## 6. Cluster-Balance Comparison
- `K=2` consistently breaks the population into two massive groups (~43% and ~57%). 
- `K=3` creates segments of roughly 28%, 29%, and 42%.
- `behavior_robust` at `K=4` creates meaningful segments (9.4%, ~28%, ~25%, 37%) without producing micro-clusters (min 563 learners).
- `behavior_reduced` at `K=3` isolates exactly 1.55% of the data (93 users, exactly the number of users with missing `avg_session_minutes`), which is why the hard filter of `ratio > 0.05` correctly rejected it.

## 7. Agglomerative Comparison
- Agglomerative clustering (Ward) produced very similar metrics at `K=2` (Silhouette ~0.243) and `K=3` (Silhouette ~0.205) but scaled slightly worse on separation than K-Means.
- K-Means is preferred for its predictability and stability over multiple initializations.

## 8. Selection Heuristic Analysis
The exact code used in `src/clustering.py`:
```python
valid_configs = kmeans_results[kmeans_results['cluster_size_ratio'] > 0.05].copy()
top_configs = valid_configs.sort_values(by='silhouette_score', ascending=False).head(10)
merged = pd.merge(top_configs, stability_results, on=['representation', 'k'])
recommended = merged.sort_values(by=['mean_ari', 'silhouette_score'], ascending=[False, False]).iloc[0]
```
**Bias identified:** The heuristic strictly prioritizes `mean_ari`, effectively acting as a primary sort key, and then breaks ties with `silhouette_score`. `K=2` almost always converges perfectly (`ARI = 1.0`), and Silhouette inherently favors lower K values in non-spherical data. 
Because `K=2` achieves `ARI=1.0` and inherently has the highest silhouette score, it automatically wins every time, regardless of whether a 2-cluster split provides actionable business personas.

## 9. Final Judgment
The heuristic correctly excluded micro-clusters but unfairly favored K=2 by sorting strictly on `mean_ari` and `silhouette`. A 2-cluster segmentation (e.g., "Active" vs "Inactive") is too broad to derive specialized learner personas for product features. Configurations at K=3 and K=4 offer much more granular business value while maintaining near-perfect stability and acceptable separation.

## 10. Recommended Next Action
Update the model selection manually to leverage a configuration that balances strong stability, decent silhouette, and enough granularity to define 3-4 distinct personas.

---
CURRENT MODEL:
behavior_standard + KMeans + K=2

AUDIT VERDICT:
C

STRONGEST ALTERNATIVE: behavior_robust + KMeans + K=4

REASON: `behavior_robust` at K=4 achieves a strong Silhouette Score (0.2048 - virtually tied with K=3), near-perfect stability across seeds (ARI = 0.9998), and segments the population into 4 meaningful groups (smallest cluster is ~9.4% or 563 users). This provides far better granularity for Learner Personas than a simple binary K=2 split, while successfully mitigating the 230 extreme outliers.
