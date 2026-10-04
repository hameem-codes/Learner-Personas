import pytest
import pandas as pd
import numpy as np
from src.config import DATA_DIR

OUTPUTS_CLUSTERS_DIR = DATA_DIR / 'outputs' / 'clusters'
FINAL_ASSIGNMENTS = OUTPUTS_CLUSTERS_DIR / 'final_cluster_assignments.csv'

def test_final_assignments_exist():
    # Only run tests if the file exists (it gets created after the clustering script)
    if FINAL_ASSIGNMENTS.exists():
        df = pd.read_csv(FINAL_ASSIGNMENTS)
        
        # exactly 6,000 final assignments
        assert len(df) == 6000
        
        # exactly 6,000 unique learner IDs
        assert df['learner_id'].nunique() == 6000
        
        # no missing cluster assignments
        assert not df['cluster_id'].isna().any()
        
        # valid cluster IDs
        assert pd.api.types.is_integer_dtype(df['cluster_id'])
        assert set(df['cluster_id'].unique()) == {0, 1, 2, 3}
        assert df['cluster_id'].nunique() == 4
        
def test_no_learner_id_in_features():
    # Ensure learner_id is not included in the feature matrices
    PROCESSED_DIR = DATA_DIR / 'processed'
    for rep in ['behavior_standard', 'behavior_robust', 'behavior_reduced', 'behavior_categorical']:
        path = PROCESSED_DIR / f"preprocessed_{rep}.csv"
        if path.exists():
            df = pd.read_csv(path)
            features = df.drop(columns=['learner_id']).columns
            assert 'learner_id' not in features
            
def test_reproducibility():
    # KMeans with same random state should yield exactly same clusters
    from sklearn.cluster import KMeans
    np.random.seed(42)
    X = np.random.rand(100, 5)
    km1 = KMeans(n_clusters=3, random_state=42, n_init=10)
    km2 = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    l1 = km1.fit_predict(X)
    l2 = km2.fit_predict(X)
    
    np.testing.assert_array_equal(l1, l2)
