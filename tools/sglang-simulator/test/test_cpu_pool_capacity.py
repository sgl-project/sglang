"""Logical CPU capacity mapping, independent of optional SDK budget selection."""

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def model_runner(monkeypatch):
    """Exercise the actual pool hooks with no GPU operations or SDK imports."""
    name = "sglang_simulator.simulation.sglang.model_runner"
    if importlib.util.find_spec("torch") is None:
        # The module only needs nn.Module at definition time. Restore both
        # modules afterwards so later native-runtime tests never inherit a stub.
        torch = ModuleType("torch")
        torch.nn = SimpleNamespace(Module=object)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module
    return importlib.import_module(name)


class LinearPool:
    def __init__(self, *, cell_bytes=3, fixed_bytes=11, alignment=None):
        self.cell_bytes = cell_bytes
        self.fixed_bytes = fixed_bytes
        self.alignment = alignment
        self.calls = 0

    def calculate_pool_sizes(self, budget_bytes, page_size):
        self.calls += 1
        if budget_bytes < self.fixed_bytes:
            raise ValueError("cannot hold even one request")
        alignment = self.alignment or page_size
        tokens = (budget_bytes - self.fixed_bytes) // self.cell_bytes
        return SimpleNamespace(max_total_num_tokens=tokens // alignment * alignment)


def test_native_pool_budget_inverts_exact_aligned_capacity(model_runner):
    pool = LinearPool()
    budget = model_runner._native_pool_budget(pool, 16, 4, require_exact=True)
    assert budget == 59  # Fixed 11 B + sixteen native rows at 3 B each.
    assert pool.calculate_pool_sizes(budget, 4).max_total_num_tokens == 16
    assert pool.calculate_pool_sizes(budget - 1, 4).max_total_num_tokens == 12


def test_native_pool_budget_rejects_zero_cell_constant_capacity(model_runner):
    class ZeroCellPool:
        calls = 0

        def calculate_pool_sizes(self, budget_bytes, page_size):
            self.calls += 1
            # Native zero-cell layouts can return context_len for every budget.
            return SimpleNamespace(max_total_num_tokens=8)

    pool = ZeroCellPool()
    with pytest.raises(ValueError, match="cannot represent"):
        model_runner._native_pool_budget(pool, 16, 4, require_exact=True)
    assert pool.calls == 64


def test_native_pool_budget_rejects_unreachable_rounded_target(model_runner):
    pool = LinearPool(fixed_bytes=0, alignment=8)
    with pytest.raises(ValueError, match="exact SimOpus capacity"):
        model_runner._native_pool_budget(pool, 12, 4, require_exact=True)


@pytest.mark.parametrize("tokens", [True, 0, 1, 6, 8.0])
def test_native_pool_budget_rejects_invalid_exact_capacity(model_runner, tokens):
    pool = LinearPool()
    with pytest.raises(ValueError, match="page-aligned"):
        model_runner._native_pool_budget(pool, tokens, 4, require_exact=True)
    assert pool.calls == 0


def test_native_pool_budget_preserves_legacy_undersized_errors(model_runner):
    class Pool(LinearPool):
        def calculate_pool_sizes(self, budget_bytes, page_size):
            if budget_bytes < self.fixed_bytes:
                raise RuntimeError("undersized probe")
            return super().calculate_pool_sizes(budget_bytes, page_size)

    assert model_runner._native_pool_budget(Pool(), 13, 4) == 59
    with pytest.raises(RuntimeError, match="undersized probe"):
        model_runner._native_pool_budget(Pool(), 16, 4, require_exact=True)
