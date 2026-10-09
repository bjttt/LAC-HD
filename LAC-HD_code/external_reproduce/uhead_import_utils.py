"""Import helpers for local UQ Head reproduction without global installation.

Adds cloned repos to sys.path and supplies tiny compatibility stubs for the small
lm_polygraph classes imported by `luh.__init__`. The benchmark runner uses a
teacher-forcing path and does not rely on lm-polygraph's generation wrapper.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_repo_paths(root: str | Path | None = None) -> None:
    root = Path(root or Path(__file__).resolve().parents[1])
    paths = [
        root / "external_reproduce" / "vendor",
        root / "external_reproduce" / "llm-uncertainty-head",
        root / "external_reproduce" / "lm-polygraph" / "src",
    ]
    for p in paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def ensure_lm_polygraph_stubs() -> None:
    """Create minimal stubs if importing full lm_polygraph is impossible.

    This is enough for importing `luh` and using `AutoUncertaintyHead` directly.
    It is NOT a replacement for official lm-polygraph evaluation.
    """
    try:
        import lm_polygraph  # noqa: F401
        return
    except Exception:
        pass

    lm_polygraph = types.ModuleType("lm_polygraph")
    estimators = types.ModuleType("lm_polygraph.estimators")
    estimator_mod = types.ModuleType("lm_polygraph.estimators.estimator")
    stat_calculators = types.ModuleType("lm_polygraph.stat_calculators")
    stat_calculator_mod = types.ModuleType("lm_polygraph.stat_calculators.stat_calculator")
    utils = types.ModuleType("lm_polygraph.utils")
    model_mod = types.ModuleType("lm_polygraph.utils.model")

    class Estimator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class StatCalculator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Model:  # pragma: no cover - marker stub only
        pass

    estimator_mod.Estimator = Estimator
    stat_calculator_mod.StatCalculator = StatCalculator
    model_mod.Model = Model

    sys.modules.setdefault("lm_polygraph", lm_polygraph)
    sys.modules.setdefault("lm_polygraph.estimators", estimators)
    sys.modules.setdefault("lm_polygraph.estimators.estimator", estimator_mod)
    sys.modules.setdefault("lm_polygraph.stat_calculators", stat_calculators)
    sys.modules.setdefault("lm_polygraph.stat_calculators.stat_calculator", stat_calculator_mod)
    sys.modules.setdefault("lm_polygraph.utils", utils)
    sys.modules.setdefault("lm_polygraph.utils.model", model_mod)


def prepare_uhead_imports(root: str | Path | None = None) -> None:
    ensure_repo_paths(root)
    ensure_lm_polygraph_stubs()
