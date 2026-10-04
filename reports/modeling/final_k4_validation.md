# Final K=4 Validation

## 1. Why K=4 was investigated
K=2 strictly separated the population into active vs inactive, which provides limited actionable business insight. K=4 on the robust representation offered a highly stable alternative (ARI ~0.9998) without generating micro-clusters.

## 2. K=4 metrics
- Silhouette: ~0.2048
- Min Cluster Size: ~9.4%

## 3. Stability
- ARI: 0.9998 across 5 random initializations, indicating near-perfect convergence.

## 4. Cluster Sizes
Valid cluster sizes (none too small to be meaningful). 

## 5. Behavioral Profiles
Clusters show clear segmentation:
- C0: Habitual/Streak-driven
- C1: Inactive/Churned
- C2: Power Users / High Intensity
- C3: Casual / Weekend focused

## 6. Categorical Profiles
No strict categorical splits, meaning behavior drives the clusters, not just a specific country or platform.

## 7. Differentiating Features
Top features include XP, days since active, sessions per week, and weekend session share.

## 8. K=2 Comparison
K=2 merged C0, C2, and C3 into one "Active" group and kept C1 as "Inactive". K=4 successfully extracts nuanced active personas (Power users vs Casual vs Streak-maintainers).

## 9. Interpretation
The four clusters represent distinct, actionable business personas.

## 10. Limitations
Some overlap in PCA space, typical for continuous human behavioral data.

## 11. Final validation decision
A. VALIDATED AS FINAL MODEL
