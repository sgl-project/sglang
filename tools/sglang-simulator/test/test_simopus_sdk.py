"""The optional provider requires the SimOpus SDK exclusively."""

import ast
from pathlib import Path
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


def test_the_read_scope_of_the_sdk_opens_the_database():
    opened = []

    def scope(system, **arguments):
        opened.append((system, arguments))
        return "database"

    sdk = SimpleNamespace(PerfDB=scope)

    assert (
        simopus_sdk.open_database(
            sdk, "mi355x", "0.5.22", perfdb_dir="/data", database_mode="SILICON"
        )
        == "database"
    )
    assert opened == [
        (
            "mi355x",
            {
                "perfdb_dir": "/data",
                "backend": "sglang",
                "version": "0.5.22",
                "database_mode": "SILICON",
            },
        )
    ]


def test_simopus_runtime_imports_use_the_public_sdk():
    """Simulator integrations depend on the facade, not SDK implementation paths."""
    runtime = Path(simopus_sdk.__file__).resolve().parents[1]
    for path in runtime.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "simopus"
            ):
                assert node.module == "simopus.sdk", (path, node.lineno, node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("simopus"):
                        assert alias.name in {"simopus", "simopus.sdk"}, (
                            path,
                            node.lineno,
                            alias.name,
                        )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("simopus")
            ):
                assert node.args[0].value in {"simopus", "simopus.sdk"}, (
                    path,
                    node.lineno,
                )
