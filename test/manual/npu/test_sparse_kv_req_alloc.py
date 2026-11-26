"""CPU checks for fused request-slot reset without loading native operators."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def make_manager(pool):
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/fused_manager.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    manager = next(node for node in module.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "_install_req_alloc_hook"
    )
    module.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        method,
    ]
    scope = {}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    manager = SimpleNamespace(reset_requests=Mock())
    scope["_install_req_alloc_hook"](manager, pool)
    return manager


@pytest.mark.parametrize(
    "holds_kv,indices,expected",
    [([False], [2], [2]), ([True], [2], []), ([True, False], [2, 3], [3])],
)
def test_only_new_request_rows_are_reset(holds_kv, indices, expected):
    requests = [SimpleNamespace(kv=SimpleNamespace(holds_kv=held)) for held in holds_kv]

    def allocate(reqs):
        for req in reqs:
            req.kv.holds_kv = True
        return indices

    original = Mock(side_effect=allocate)
    pool = SimpleNamespace(alloc=original)
    manager = make_manager(pool)

    assert pool.alloc(requests) == indices
    original.assert_called_once_with(requests)
    manager.reset_requests.assert_called_once_with(expected)


def test_failed_allocation_preserves_residency():
    pool = SimpleNamespace(alloc=Mock(return_value=None))
    manager = make_manager(pool)
    request = SimpleNamespace(kv=SimpleNamespace(holds_kv=False))

    assert pool.alloc([request]) is None
    manager.reset_requests.assert_not_called()
