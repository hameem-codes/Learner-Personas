# Repository Inventory

## Required Source Files
- `src/config.py`: Core paths and constants.
- `src/data_loader.py`: Safe data ingestion.
- `src/preprocessing.py`: Feature engineering pipelines.
- `src/clustering.py`: Unsupervised model training.
- `src/profile_clusters.py`: Statistical profiling.
- `src/construct_personas.py`: Persona asset generation.
- `src/audit_personas.py`: Evidence extraction and claim auditing.
- `src/product_experiments.py`: Product hypothesis mappings.

## Tests
- `tests/test_data_loader.py`
- `tests/test_eda.py`
- `tests/test_preprocessing.py`
- `tests/test_clustering.py`
- `tests/test_personas.py`

## Reports & Artifacts
- `reports/eda_report.md`
- `reports/modeling/*`
- `reports/personas/*`
- `reports/product/*`
- `reports/final/*`

## Verdict
The repository is well-structured, separating `src`, `tests`, `reports`, and `data/outputs` successfully. No dangling temporary scripts found.
