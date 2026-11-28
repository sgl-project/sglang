"""Bounded logical-capacity inversion through the existing native allocator."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def profile_budget(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "src/sglang_simulator/simulation/sglang/model_runner.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "C_KVCacheConfiguratorHook"
    )
    hook = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "hook"
    )
    profile = next(
        node
        for node in hook.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "override_profile_available_bytes"
    )
    # Load the actual method's local functions without physical execution
    # imports. The factory below supplies the native layout under test.
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"ConfigManager": Mock()}
    exec(
        compile(
            ast.Module(body=[*helpers, profile], type_ignores=[]),
            str(source),
            "exec",
        ),
        namespace,
    )
    module_name = "sglang.srt.model_executor.pool_configurator"
    native = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, native)

    def run(pool, tokens, page_size=4):
        native.create_memory_pool_configurator = lambda _: pool
        configurator = SimpleNamespace(
            server_args=SimpleNamespace(max_total_tokens=tokens), page_size=page_size
        )
        return namespace["override_profile_available_bytes"](configurator, 0)

    return run


class LinearPool:
    def __init__(self):
        self.calls = 0

    def calculate_pool_sizes(self, budget_bytes, page_size):
        self.calls += 1
        if budget_bytes < 11:
            raise ValueError("cannot hold even one request")
        tokens = (budget_bytes - 11) // 3
        return SimpleNamespace(max_total_num_tokens=tokens // page_size * page_size)


def test_explicit_capacity_uses_native_pool_calculation(profile_budget):
    assert profile_budget(LinearPool(), 16) == 59


def test_non_aligned_target_keeps_existing_at_least_capacity_rule(profile_budget):
    assert profile_budget(LinearPool(), 13) == 59


def test_zero_cell_native_rounding_stops_unreachable_capacity(profile_budget):
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_executor/pool_configurator.py"
    )
    owner = next(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DefaultPoolConfigurator"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "calculate_pool_sizes"
    )
    namespace = {"MemoryPoolConfig": SimpleNamespace}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    native_calculate = namespace["calculate_pool_sizes"]

    class ZeroCellPool:
        _fixed_overhead_bytes = 0
        _cell_size = 0
        _zero_kv_max_tokens = 13
        calls = 0

        def calculate_pool_sizes(self, budget_bytes, page_size):
            self.calls += 1
            assert self.calls <= 64, "capacity probing did not terminate"
            return native_calculate(self, budget_bytes, page_size)

    pool = ZeroCellPool()
    assert native_calculate(pool, 1, 4).max_total_num_tokens == 12
    with pytest.raises(ValueError, match="Native KV pool cannot represent"):
        profile_budget(pool, 13)
    assert pool.calls == 64


def test_undersized_runtime_errors_keep_existing_probe_behavior(profile_budget):
    class Pool(LinearPool):
        def calculate_pool_sizes(self, budget_bytes, page_size):
            if budget_bytes < 11:
                raise RuntimeError("undersized probe")
            return super().calculate_pool_sizes(budget_bytes, page_size)

    assert profile_budget(Pool(), 13) == 59


def test_other_value_errors_are_not_treated_as_small_budgets(profile_budget):
    class InvalidPool:
        def calculate_pool_sizes(self, budget_bytes, page_size):
            raise ValueError("invalid layout")

    with pytest.raises(ValueError, match="invalid layout"):
        profile_budget(InvalidPool(), 16)
