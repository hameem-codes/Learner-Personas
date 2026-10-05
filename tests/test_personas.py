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
    
    # Persona populations check
    pcts = df['cluster_id'].value_counts(normalize=True) * 100
    assert abs(pcts.sum() - 100.0) < 1e-5

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
        
        # no unsupported demographic/psychological claims
        assert "lazy" not in content.lower()
        assert "addicted" not in content.lower()
        assert "motivated" not in content.lower()
        assert "lazy" not in content.lower()
        
def test_persona_evidence_exists():
    evidence_path = DATA_DIR / 'outputs' / 'profiles' / 'persona_evidence.csv'
    assert evidence_path.exists()
    df = pd.read_csv(evidence_path)
    assert not df.empty
