"""AISimulate estimator-SDK provenance tests for the predictor adapter.

AISimulate 0.12.1 — the pinned release — ships the estimator SDK under the
``aiconfigurator.sdk`` compatibility namespace *inside* the ``aisimulate``
distribution; ``aisimulate.sdk`` only exists in later prereleases and is not
imported. These tests pin that contract:

- the installed ``aiconfigurator`` import namespace must be provided by the
  ``aisimulate`` distribution, and the legacy standalone ``aiconfigurator``
  distribution must not be installed;
- an environment with only the legacy distribution must fail the adapter
  import with an actionable error instead of silently running;
- a missing SDK produces an install hint, while a missing dependency or
  symbol inside an installed SDK surfaces unchanged.

The subprocess cases run with ``-S`` and control the whole import environment
through stub packages and fake distribution metadata, so the assertions do
not depend on which distributions the parent test environment happens to
have. They exercise import selection and provenance only; predictor behavior
is covered by ``test_aisimulate_predictor.py`` and the simulation tests.
"""

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = "sglang_simulator.time_predictor.aisimulate"
SRC_PATH = str(Path(__file__).parents[1] / "src")

AISIMULATE_FAMILIES = {"aisimulate", "aisimulate_core"}
# AISimulate 0.12.1 provides the estimator SDK through the compatibility
# namespace family: the SDK re-exports some modules through ``*_core``
# packages, so both names appear within one family group.
AICONFIGURATOR_COMPAT_FAMILIES = {"aiconfigurator", "aiconfigurator_core"}
SDK_NAMESPACE_FAMILIES = AISIMULATE_FAMILIES | AICONFIGURATOR_COMPAT_FAMILIES

SYMBOLS = (
    "models",
    "get_backend",
    "get_database",
    "get_systems_paths",
    "ModelConfig",
    "RuntimeConfig",
    "InferenceSession",
    "CommQuantMode",
    "DatabaseMode",
    "FMHAQuantMode",
    "GEMMQuantMode",
    "KVCacheQuantMode",
    "MoEQuantMode",
)

ISOLATED_CODE = (
    "import sys\n"
    "import importlib\n"
    f"sys.path.insert(0, {{stub_root!r}})\n"
    f"sys.path.insert(1, {SRC_PATH!r})\n"
    "try:\n"
    f"    importlib.import_module({MODULE!r})\n"
    "except BaseException as exc:\n"
    "    print(type(exc).__name__, ':', exc)\n"
    "    raise\n"
)


def _family(obj) -> str:
    name = getattr(obj, "__module__", None) or obj.__name__
    return name.split(".")[0]


def _aisimulate_installed() -> bool:
    try:
        importlib.metadata.version("aisimulate")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _legacy_distribution_installed() -> bool:
    try:
        importlib.metadata.version("aiconfigurator")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _write_stubs(stub_root: Path, stub_files: dict):
    for rel_path, content in stub_files.items():
        path = stub_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _run_isolated(tmp_path, stub_files: dict) -> subprocess.CompletedProcess:
    """Import the adapter in a ``-S`` subprocess with a controlled environment.

    ``-S`` keeps the parent environment's site-packages — including any real
    aisimulate installation — off the import path, so each case controls
    exactly which namespace and distribution metadata exist. numpy and joblib
    stubs keep the simulator import chain working up to the SDK guard.
    """
    stub_root = tmp_path / "stubs"
    stub_files = {
        "numpy.py": "def mean(*args, **kwargs):\n    return 0\n",
        "joblib.py": "def load(*args, **kwargs):\n    raise NotImplementedError\n",
        **stub_files,
    }
    _write_stubs(stub_root, stub_files)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-S", "-c", ISOLATED_CODE.format(stub_root=str(stub_root))],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _sdk_stubs() -> dict:
    """Stubs providing every submodule the adapter imports."""
    enum_stub = (
        "class _Mode:\n"
        "    def __getattr__(self, name):\n"
        "        return _Mode()\n"
        "\n"
        "    def __or__(self, other):\n"
        "        return self\n"
        "\n"
        "    def __ror__(self, other):\n"
        "        return self\n"
        "\n\n"
    )
    return {
        "aiconfigurator/__init__.py": "",
        "aiconfigurator/sdk/__init__.py": "",
        "aiconfigurator/sdk/models.py": (
            "class BaseModel:\n    pass\n\n\nMARKER = 'stub'\n"
        ),
        "aiconfigurator/sdk/backends/__init__.py": "",
        "aiconfigurator/sdk/backends/factory.py": (
            "def get_backend(*args, **kwargs):\n    raise NotImplementedError\n"
        ),
        "aiconfigurator/sdk/common.py": (
            enum_stub + "CommQuantMode = _Mode()\n"
            "DatabaseMode = _Mode()\n"
            "FMHAQuantMode = _Mode()\n"
            "GEMMQuantMode = _Mode()\n"
            "KVCacheQuantMode = _Mode()\n"
            "MoEQuantMode = _Mode()\n"
        ),
        "aiconfigurator/sdk/config.py": (
            "class ModelConfig:\n    pass\n\n\nclass RuntimeConfig:\n    pass\n"
        ),
        "aiconfigurator/sdk/inference_session.py": "class InferenceSession:\n    pass\n",
        "aiconfigurator/sdk/perf_database.py": (
            "def get_database(*args, **kwargs):\n"
            "    raise NotImplementedError\n\n\n"
            "def get_systems_paths(*args, **kwargs):\n"
            "    raise NotImplementedError\n"
        ),
    }


def _legacy_distribution_metadata(version: str = "0.10.0") -> dict:
    """Fake legacy standalone distribution metadata."""
    return {
        f"aiconfigurator-{version}.dist-info/METADATA": (
            f"Metadata-Version: 2.1\nName: aiconfigurator\nVersion: {version}\n"
        )
    }


def test_installed_sdk_provenance_is_aisimulate():
    if not _aisimulate_installed():
        pytest.fail(
            "aisimulate==0.12.1 must be installed for the simulator CPU suite; "
            "the AISimulate cases are not skippable"
        )
    assert not _legacy_distribution_installed(), (
        "the legacy standalone 'aiconfigurator' distribution must not be "
        "installed in the AISimulate environment"
    )

    providers = importlib.metadata.packages_distributions().get("aiconfigurator", [])
    assert "aisimulate" in providers, providers

    mod = importlib.import_module(MODULE)

    families = set()
    for symbol in SYMBOLS:
        family = _family(getattr(mod, symbol))
        assert family in SDK_NAMESPACE_FAMILIES, symbol
        families.add(family)
    # All symbols must come from one namespace family group.
    assert (
        families <= AISIMULATE_FAMILIES or families <= AICONFIGURATOR_COMPAT_FAMILIES
    ), families


def test_adapter_rejects_legacy_distribution_only(tmp_path):
    """A legacy-aiconfigurator-only environment must fail, not silently run.

    The stub provides a complete ``aiconfigurator.sdk`` namespace plus legacy
    distribution metadata; the adapter must reject it with an actionable
    error.
    """
    stub_files = _sdk_stubs()
    stub_files.update(_legacy_distribution_metadata())

    result = _run_isolated(tmp_path, stub_files)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, result.stdout
    assert "legacy standalone 'aiconfigurator' distribution" in combined, combined
    assert "tools/sglang-simulator[aisimulate]" in combined, combined


def test_namespace_without_aisimulate_provenance_is_rejected(tmp_path):
    """An SDK namespace not owned by the aisimulate distribution is rejected.

    No distribution metadata is present at all, so both provenance guards see
    an unowned namespace; the provider check must fire.
    """
    result = _run_isolated(tmp_path, _sdk_stubs())
    combined = result.stdout + result.stderr

    assert result.returncode != 0, result.stdout
    assert "not provided by the" in combined, combined
    assert "'aisimulate' distribution" in combined, combined


def test_missing_sdk_reports_install_hint(tmp_path):
    """Without any estimator SDK the adapter fails with an install hint."""
    result = _run_isolated(tmp_path, {})
    combined = result.stdout + result.stderr

    assert result.returncode != 0, result.stdout
    assert "ModuleNotFoundError" in combined
    assert "tools/sglang-simulator[aisimulate]" in combined, combined
    assert "aiconfigurator.sdk" in combined, combined


def test_missing_dependency_inside_sdk_propagates(tmp_path):
    """A missing dependency inside the SDK surfaces unchanged, not as a hint."""
    stub_files = _sdk_stubs()
    stub_files["aiconfigurator/sdk/models.py"] = (
        "import aiconfigurator_missing_dependency_xyz\n"
    )

    result = _run_isolated(tmp_path, stub_files)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, result.stdout
    assert "ModuleNotFoundError" in combined
    assert "aiconfigurator_missing_dependency_xyz" in combined, combined
    assert "[aisimulate]" not in combined, combined


def test_missing_symbol_inside_sdk_propagates(tmp_path):
    """A missing symbol inside the SDK surfaces as a plain ImportError."""
    stub_files = _sdk_stubs()
    stub_files["aiconfigurator/sdk/backends/factory.py"] = (
        "def other_backend():\n    pass\n"
    )

    result = _run_isolated(tmp_path, stub_files)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, result.stdout
    assert "ImportError" in combined
    assert "cannot import name 'get_backend'" in combined, combined
