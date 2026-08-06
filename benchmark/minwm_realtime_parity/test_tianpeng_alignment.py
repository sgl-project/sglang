import builtins
import importlib.util
import sys
from pathlib import Path


def test_contract_helpers_do_not_import_websocket_dependencies(monkeypatch):
    module_dir = Path(__file__).resolve().parent
    module_path = module_dir / "tianpeng_alignment.py"
    real_import = builtins.__import__

    def reject_websocket_dependencies(name, *args, **kwargs):
        if name == "run_sglang_api" or name.startswith("msgspec"):
            raise AssertionError(f"unexpected websocket dependency import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_websocket_dependencies)
    monkeypatch.syspath_prepend(str(module_dir))
    spec = importlib.util.spec_from_file_location(
        "tianpeng_alignment_without_websocket_dependencies",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
