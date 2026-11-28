"""CPU checks for fused request-slot reset without loading native operators."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def load_class(path, class_name, *methods):
    source = Path(__file__).resolve().parents[3] / "python/sglang/srt" / path
    module = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    module.body = ast.parse("from __future__ import annotations").body + [cls]
    scope = {}
    exec(compile(module, str(source), "exec"), scope)
    return scope[class_name]


class ResidentRows:
    def __init__(self):
        self.dirty_rows = {2, 3}

    def reset_rows(self, rows):
        self.dirty_rows.difference_update(rows)


@pytest.fixture(
    params=[
        ("mem_cache/memory_pool.py", "ReqToTokenPool", ("alloc_rows",)),
        ("disaggregation/decode.py", "DecodeReqToTokenPool", ()),
    ],
    ids=["req-to-token", "decode-req-to-token"],
)
def request_pool(request):
    path, class_name, extra_methods = request.param
    pool_type = load_class(
        path, class_name, "alloc", "register_on_alloc_rows", *extra_methods
    )
    pool = pool_type()
    pool._on_alloc_rows = None
    pool.req_generation = MagicMock()

    manager_type = load_class(
        "hardware_backend/npu/sparsity_driven_kv_offload/fused_manager.py",
        "FusedSparseKVCacheManager",
        "_install_req_alloc_hook",
    )
    manager = manager_type()
    manager.resident_pool = ResidentRows()
    original_alloc = pool.alloc
    manager._install_req_alloc_hook(pool)
    assert pool.alloc == original_alloc
    assert pool._on_alloc_rows == manager.resident_pool.reset_rows
    return pool, manager.resident_pool


@pytest.mark.parametrize(
    "existing_rows,indices,new_rows",
    [([None], [2], [2]), ([2], [2], []), ([2, None], [2, 3], [3])],
)
def test_only_new_request_rows_are_reset(
    request_pool, existing_rows, indices, new_rows
):
    pool, resident = request_pool
    pool.free_slots = new_rows.copy()
    requests = [
        SimpleNamespace(
            kv=SimpleNamespace(
                req_pool_idx=row,
                holds_kv=row is not None,
                kv_allocated_len=0 if row is None else 1,
            )
        )
        for row in existing_rows
    ]

    assert pool.alloc(requests) == indices
    assert not pool.free_slots
    assert resident.dirty_rows == {2, 3}.difference(new_rows)


def test_failed_allocation_preserves_residency(request_pool):
    pool, resident = request_pool
    pool.free_slots = []
    request = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=None, holds_kv=False))
    assert pool.alloc([request]) is None
    assert request.kv.req_pool_idx is None
    assert resident.dirty_rows == {2, 3}
