import ast
import inspect
import pathlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.tolist())


class _FakeTreeCache:
    def __init__(self):
        self.protected_size_ = 0


def _make_manager(page_size: int, allocator: _FakeAllocator):
    # __init__ builds host pools and a HiCacheController, neither of which a
    # release-path test can stand up; the fields it would set are supplied here.
    manager = object.__new__(DecodeKVCacheOffloadManager)
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    manager.req_to_token_pool = SimpleNamespace(
        req_to_token=req_to_token, free=lambda req: None
    )
    manager.token_to_kv_pool_allocator = allocator
    manager.page_size = page_size
    manager.tree_cache = _FakeTreeCache()
    manager.offloaded_state = {}
    return manager


def _make_req(committed: int, allocated: int, origin_len: int):
    return SimpleNamespace(
        rid="req-a",
        req_pool_idx=0,
        kv_committed_len=committed,
        kv=SimpleNamespace(kv_allocated_len=allocated),
        origin_input_ids=list(range(origin_len)),
        prefix_indices=[],
        effective_kv_committed_len=lambda: committed,
    )


def test_release_emits_one_prefill_free_and_one_merged_free():
    """The prefill segment and the merged [start_offset, kv_allocated_len) are the only two frees."""
    allocator = _FakeAllocator(page_size=4)
    manager = _make_manager(page_size=4, allocator=allocator)
    manager.offloaded_state["req-a"] = OffloadedState(prefill_len=16, inc_len=8)

    req = _make_req(committed=40, allocated=44, origin_len=18)
    manager._release_finished_req(req, start_offset=24)

    assert allocator.freed == [list(range(0, 16)), list(range(24, 44))]


def test_release_covers_the_spec_v2_over_allocation():
    """Spec v2 pushes allocated past ceil(committed); the merged range must still reach it."""
    allocator = _FakeAllocator(page_size=4)
    manager = _make_manager(page_size=4, allocator=allocator)

    # committed=34 -> ceil = 36, but the allocator handed out 44.
    req = _make_req(committed=34, allocated=44, origin_len=18)
    manager._release_finished_req(req, start_offset=16)

    assert allocator.freed == [list(range(16, 44))]


def test_release_covers_the_kv_stripped_by_strip_thinking_cache():
    """strip_thinking_cache reports a tiny effective committed len; the free must not shrink with it."""
    allocator = _FakeAllocator(page_size=4)
    manager = _make_manager(page_size=4, allocator=allocator)

    req = _make_req(committed=40, allocated=44, origin_len=18)
    # strip_thinking_cache parks thinking+answer above the effective committed
    # len and leaves them to be reclaimed by the release path.
    req.effective_kv_committed_len = lambda: 18
    manager._release_finished_req(req, start_offset=16)

    assert allocator.freed == [list(range(16, 44))]


def test_release_rejects_an_unaligned_start_offset():
    """start_offset is page-aligned by construction; an unaligned one must fail loudly."""
    allocator = _FakeAllocator(page_size=4)
    manager = _make_manager(page_size=4, allocator=allocator)

    req = _make_req(committed=40, allocated=44, origin_len=18)
    with pytest.raises(AssertionError):
        manager._release_finished_req(req, start_offset=18)


def test_release_rejects_an_unaligned_prefill_len():
    """prefill_len is floored to the page at offload time; an unaligned one must fail loudly."""
    allocator = _FakeAllocator(page_size=4)
    manager = _make_manager(page_size=4, allocator=allocator)
    manager.offloaded_state["req-a"] = OffloadedState(prefill_len=18, inc_len=0)

    req = _make_req(committed=40, allocated=44, origin_len=18)
    with pytest.raises(AssertionError):
        manager._release_finished_req(req, start_offset=24)


def test_release_is_skipped_for_an_already_released_req():
    """ReqToTokenPool.free clears req_pool_idx; a second release must not double-free."""
    allocator = _FakeAllocator(page_size=4)
    manager = _make_manager(page_size=4, allocator=allocator)

    req = _make_req(committed=40, allocated=44, origin_len=18)
    req.req_pool_idx = None
    manager._release_finished_req(req, start_offset=16)

    assert allocator.freed == []


def test_offload_manager_never_reads_the_declared_page_size():
    """Every page in this module derives from self.page_size, which must come from
    the allocator: server_args.page_size is the declared page, and under DCP the
    allocator's page is a multiple of it. Reading the declared one would silently
    offload and free on the wrong page. __init__ builds host pools and a
    HiCacheController, so the assignment is pinned at the source level."""
    source = pathlib.Path(
        inspect.getsourcefile(DecodeKVCacheOffloadManager)
    ).read_text()
    tree = ast.parse(source)

    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "page_size"
        and isinstance(node.value, ast.Name)
        and node.value.id == "server_args"
    ]

    assert reads == []


def test_offload_manager_takes_its_page_from_the_allocator():
    """The one assignment feeding this module's whole alignment chain must read the
    allocator's page. Pinned at the source level for the same reason as above."""
    source = pathlib.Path(
        inspect.getsourcefile(DecodeKVCacheOffloadManager)
    ).read_text()
    tree = ast.parse(source)

    assigned_from = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "self.page_size"
    ]

    assert assigned_from == ["token_to_kv_pool_allocator.page_size"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
