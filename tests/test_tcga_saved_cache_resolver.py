
import importlib.util
from pathlib import Path
import sys
import pandas as pd

MODULE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tcga_brca_rnaseq_external_validation.py"
)
spec = importlib.util.spec_from_file_location("rna_v2", MODULE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

class MethodStore:
    def sample_metadata(self, ids):
        return pd.DataFrame(
            {"GSM_ID": ids, "Label": [1, 0]}
        )

class PropertyStore:
    metadata = pd.DataFrame(
        {
            "GSM_ID": ["A", "B"],
            "Label": ["cancer", "normal"],
        }
    )

def test_method_metadata():
    labels = module._resolve_sample_labels(
        MethodStore(), ["A", "B"]
    )
    assert labels.to_dict() == {"A": 1, "B": 0}

def test_property_metadata():
    labels = module._resolve_sample_labels(
        PropertyStore(), ["A", "B"]
    )
    assert labels.to_dict() == {"A": 1, "B": 0}
