import pytest
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, PROJECT_ROOT

def test_persona_assignments():
    file_path = DATA_DIR / 'outputs' / 'clusters' / 'final_cluster_assignments.csv'
    assert file_path.exists()
    df = pd.read_csv(file_path)
    
    # Exactly 4 personas (0,1,2,3)
    assert set(df['cluster_id'].unique()) == {0, 1, 2, 3}
    
    # All 6,000 learners assigned
    assert len(df) == 6000
    
    # No learner appears twice
    assert df['learner_id'].nunique() == 6000

def test_persona_reports_exist():
    reports_dir = PROJECT_ROOT / 'reports' / 'personas'
    assert (reports_dir / 'persona_quality_audit.md').exists()
    assert (reports_dir / 'final_personas.md').exists()

    with open(reports_dir / 'final_personas.md', 'r') as f:
        content = f.read()
        assert "Evidence" in content
        assert "Dormant Learners" in content
        assert "Streak Maintainers" in content
        assert "Weekend Warriors" in content
        assert "Power Learners" in content
