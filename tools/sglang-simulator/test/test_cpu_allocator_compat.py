"""The simulator uses the installed SGLang page counter without a Torch import."""

import ast
import builtins
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest


def counter_loader():
    # Execute the actual small import boundary separately from Torch allocators.
    path = (
        Path(__file__).resolve().parents[1]
        / "src/sglang_simulator/simulation/sglang/mem_cache_allocator.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_native_page_counter"
    )
    namespace = {"lru_cache": lru_cache}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["_native_page_counter"]


@pytest.mark.parametrize("legacy", [None, ModuleNotFoundError, ImportError])
def test_installed_counter_identity_and_cached_import(monkeypatch, legacy):
    selected = object()
    calls = []
    original = builtins.__import__

    def load(name, *args, **kwargs):
        if name in ("sglang.srt.mem_cache.utils", "sglang.srt.utils"):
            calls.append(name)
            if name == "sglang.srt.mem_cache.utils" and legacy:
                raise legacy(name, name=name)
            return SimpleNamespace(get_num_new_pages=selected)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", load)
    loader = counter_loader()
    assert loader() is selected
    observed = tuple(calls)
    assert loader() is selected
    assert tuple(calls) == observed
    assert calls == ["sglang.srt.mem_cache.utils"] + (
        ["sglang.srt.utils"] if legacy else []
    )


def test_dependency_import_failure_is_not_replaced(monkeypatch):
    original = builtins.__import__
    cause = ModuleNotFoundError("broken installed dependency", name="torch")

    def load(name, *args, **kwargs):
        if name == "sglang.srt.mem_cache.utils":
            raise cause
        if name == "sglang.srt.utils":
            pytest.fail("must not replace a broken dependency with another module")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", load)
    with pytest.raises(ModuleNotFoundError) as raised:
        counter_loader()()
    assert raised.value is cause
