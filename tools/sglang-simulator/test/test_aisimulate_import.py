"""The modern namespace has no ownership ambiguity with AIConfigurator."""

import builtins
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "missing", ["aisimulate", "aisimulate.sdk", "missing_dependency"]
)
def test_install_hint_does_not_mask_transitive_import_errors(monkeypatch, missing):
    original = builtins.__import__

    def intercepted(name, *args, **kwargs):
        if name == "aisimulate.sdk":
            raise ModuleNotFoundError(f"No module named {missing}", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercepted)
    path = (
        Path(__file__).parents[1] / "src/sglang_simulator/time_predictor/aisimulate.py"
    )
    with pytest.raises(ModuleNotFoundError) as caught:
        runpy.run_path(str(path))
    if missing == "missing_dependency":
        assert caught.value.name == missing
        assert "pip install" not in str(caught.value)
    else:
        assert "tools/sglang-simulator[aisimulate]" in str(caught.value)
