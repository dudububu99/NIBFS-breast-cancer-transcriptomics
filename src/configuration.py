from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["project", "preprocessing", "ppi", "feature_selection", "models", "outputs"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    if cfg["project"].get("analysis_scope") != "raw_nibfs_plus_frozen_kan_bridge":
        raise ValueError("This clean package only accepts analysis_scope=raw_nibfs_plus_frozen_kan_bridge")
    if not bool(cfg["outputs"].get("export_kan_bridge", False)):
        raise ValueError("The agreed design requires export_kan_bridge=true; KAN itself is not trained here.")
    if cfg["preprocessing"].get("mode") != "post_harmonization_split":
        raise ValueError("preprocessing.mode must be post_harmonization_split")
    if cfg["preprocessing"].get("batch_correction") != "neuroCombat":
        raise ValueError("preprocessing.batch_correction must be neuroCombat")
    final_k = int(cfg["project"]["final_k"])
    ks = [int(k) for k in cfg["project"]["sensitivity_k"]]
    if final_k <= 0 or final_k not in ks:
        raise ValueError("final_k must be positive and included in sensitivity_k")
    if int(cfg["project"]["cv_folds"]) < 2:
        raise ValueError("cv_folds must be at least 2")
    if not 0 < float(cfg["project"]["test_size"]) < 1:
        raise ValueError("test_size must be between 0 and 1")
    score = int(cfg["ppi"]["required_score"])
    if not 0 <= score <= 1000:
        raise ValueError("STRING required_score must be in [0,1000]")
