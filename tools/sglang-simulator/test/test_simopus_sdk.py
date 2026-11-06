"""The unmerged provider requires the SimOpus SDK exclusively."""

from types import SimpleNamespace

import pytest
from sglang_simulator.time_predictor import simopus_sdk


def test_loads_only_simopus_sdk(monkeypatch):
    calls = []
    package, sdk = SimpleNamespace(), SimpleNamespace()

    def load(name):
        calls.append(name)
        return sdk if name.endswith(".sdk") else package

    monkeypatch.setattr(simopus_sdk, "import_module", load)
    assert simopus_sdk.load_sdk() == (package, sdk)
    assert calls == ["simopus", "simopus.sdk"]


@pytest.mark.parametrize("missing", ["simopus", "yaml", "simopus.sdk"])
def test_missing_or_broken_sdk_is_not_replaced(monkeypatch, missing):
    calls = []

    def load(name):
        calls.append(name)
        if name == "simopus" and missing == "simopus.sdk":
            return SimpleNamespace()
        raise ModuleNotFoundError("broken canonical install", name=missing)

    monkeypatch.setattr(simopus_sdk, "import_module", load)
    with pytest.raises(ModuleNotFoundError) as error:
        simopus_sdk.load_sdk()
    assert error.value.name == missing
    assert all(name in {"simopus", "simopus.sdk"} for name in calls)
