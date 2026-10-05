# Pipeline Traceability

| Stage | Source Script | Input | Output | Purpose |
|-------|---------------|-------|--------|---------|
| Data Loading | `src/data_loader.py` | `learners.csv` (Raw) | `pd.DataFrame` | Validates and loads raw dataset strictly from production location. |
| EDA | `src/run_eda.py` | Loaded dataframe | `reports/eda_report.md`, charts | Computes data statistics, missing values, duplicates, and feature distributions. |
| Preprocessing | `src/preprocessing.py` | Loaded dataframe | `data/outputs/features/*.csv` | Cleans missing values, applies scalers (standard, robust), OHE, and PCA. |
| Clustering Tournament | `src/clustering.py` | `behavior_*.csv` | `clustering_results.csv`, etc. | Runs K-Means (K=2..10) across representations and evaluates Silhouette, ARI stability, CH, and DB scores. |
| Model Selection Audit | `src/audit_personas.py` / `src/profile_clusters.py` | Clustering results | Audit reports | Validates if automated heuristic favored sub-optimal K, correctly shifting from K=2 to K=4. |
| Persona Construction | `src/construct_personas.py` | `final_cluster_assignments.csv` | `final_personas.md`, charts | Profiles K=4 clusters, generating radar charts and statistical boundaries. |
| Persona Evidence Audit | `src/audit_personas.py` | Final assignments & features | `persona_evidence.csv` | Computes exact behavioral effect sizes to scrub subjective claims and enforce evidence. |
| Product Hypotheses | `src/product_experiments.py` | `persona_evidence.csv` | `product_hypotheses.csv`, `experiment_specs.md` | Translates behavioral segments into testable A/B product interventions. |
