"""Canonical SDK selection and explicit compatibility boundaries."""

from types import SimpleNamespace

import pytest
from sglang_simulator.time_predictor import simopus_sdk


@pytest.mark.parametrize("legacy", [False, True])
def test_canonical_sdk_preferred_with_missing_package_only_fallback(
    monkeypatch, legacy
):
    calls = []
    package, sdk = SimpleNamespace(), SimpleNamespace()

    def load(name):
        calls.append(name)
        if name == "simopus" and legacy:
            raise ModuleNotFoundError("missing canonical package", name="simopus")
        return sdk if name.endswith(".sdk") else package

    monkeypatch.setattr(simopus_sdk, "import_module", load)
    assert simopus_sdk.load_sdk() == (package, sdk)
    assert calls == (
        ["simopus", "infercast", "infercast.sdk"]
        if legacy
        else ["simopus", "simopus.sdk"]
    )


@pytest.mark.parametrize("missing", ["yaml", "simopus.sdk"])
def test_broken_canonical_install_never_tries_legacy(monkeypatch, missing):
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
    assert all(not name.startswith("infercast") for name in calls)
