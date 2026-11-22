"""PD CPU retraction must carry draft KV that lives outside the target pool."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import backup_kv_cache, restore_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SIZE = 64


def _mha_pool(layers, base, cls=MHATokenToKVPool):
    pool = object.__new__(cls)
    pool.use_hnd = False
    pool.layer_num = layers
    pool.cpu_offloading_chunk_size = 5
    pool.k_buffer = [
        (torch.arange(SIZE, dtype=torch.float32) + base + 100 * layer).view(SIZE, 1, 1)
        for layer in range(layers)
    ]
    pool.v_buffer = [buf + 50 for buf in pool.k_buffer]
    return pool


def _mla_pool(layers, base):
    pool = object.__new__(MLATokenToKVPool)
    pool.dcp_replicated = True
    pool.layer_num = layers
    pool.cpu_offloading_chunk_size = 5
    pool.kv_buffer = [
        (torch.arange(SIZE, dtype=torch.float32) + base + 100 * layer).view(SIZE, 1, 1)
        for layer in range(layers)
    ]
    return pool


def _buffers(pool):
    if isinstance(pool, MLATokenToKVPool):
        return tuple(pool.kv_buffer)
    return (*pool.k_buffer, *pool.v_buffer)


def _runner(pool, fused=False):
    return SimpleNamespace(
        token_to_kv_pool=pool, runner_ids=SimpleNamespace(is_fused_draft=fused)
    )


def _spec_worker(target_pool, draft_runners, cls=BaseSpecWorker):
    worker = object.__new__(cls)
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(token_to_kv_pool=target_pool)
    )
    worker._draft_model_runners = lambda: tuple(draft_runners)
    return worker


def _retract_and_resume(register, mla_draft=False):
    """Back up a request, let another request overwrite its slots, restore it
    on relocated slots, and return (target, drafts, expected, new slots)."""
    target = _mla_pool(2, base=1000) if mla_draft else _mha_pool(2, base=1000)
    drafts = (
        [_mla_pool(1, base=3000)]
        if mla_draft
        else [_mha_pool(1, base=3000), _mha_pool(1, base=5000)]
    )
    allocator = TokenToKVPoolAllocator(
        SIZE, torch.float32, "cpu", target, need_sort=False
    )
    if register:
        allocator.register_retraction_draft_pools(tuple(drafts))

    req = object.__new__(Req)
    req.kv = ReqKvInfo(req_pool_idx=0)
    req.origin_input_ids = [1] * 9
    req.output_ids = [2] * 4
    old = torch.cat((torch.arange(4, 10), torch.arange(30, 36)))
    req_pool = SimpleNamespace(
        req_to_token=old.unsqueeze(0).clone(), holds_state_slots=False
    )
    expected = {
        "target": [buf[old].clone() for buf in _buffers(target)],
        "draft": [buf[old].clone() for pool in drafts for buf in _buffers(pool)],
    }

    with get_parallel().override(attn_dcp_rank=0, attn_dcp_size=1):
        assert backup_kv_cache(req, None, req_pool, allocator, "cpu_tensor")
        for pool in (target, *drafts):
            for buf in _buffers(pool):
                buf.fill_(-1)
        new = old + 12
        req_pool.req_to_token = new.unsqueeze(0)
        restore_kv_cache(req, None, req_pool, allocator, "cpu_tensor")
    assert req.kv.retraction_backup is None
    return target, drafts, expected, new


@pytest.mark.parametrize("mla_draft", [False, True])
def test_round_trip_restores_every_draft_pool_on_relocated_slots(mla_draft):
    """Reused slots must not leak into a resumed request's draft KV."""
    target, drafts, expected, new = _retract_and_resume(
        register=True, mla_draft=mla_draft
    )
    for got, want in zip((buf[new] for buf in _buffers(target)), expected["target"]):
        torch.testing.assert_close(got, want)
    got = [buf[new] for pool in drafts for buf in _buffers(pool)]
    for g, want in zip(got, expected["draft"]):
        torch.testing.assert_close(g, want)


def test_unregistered_draft_pool_resumes_on_another_requests_kv():
    """Without registration only the target KV comes back."""
    _, drafts, _, new = _retract_and_resume(register=False)
    assert all((buf[new] == -1).all() for pool in drafts for buf in _buffers(pool))


def test_supported_mha_and_mla_draft_pools_are_selected():
    target = _mha_pool(1, base=0)
    full = _mha_pool(1, base=10)
    mla = object.__new__(MLATokenToKVPool)
    fused = _mha_pool(1, base=30)
    worker = _spec_worker(
        target,
        [
            _runner(full),
            _runner(full),  # a shared pool is backed up once
            _runner(target),  # a draft writing into the target pool
            _runner(fused, fused=True),
            _runner(mla),
        ],
    )
    assert worker.retraction_draft_pools() == (full, mla)
    assert worker.supports_cpu_retraction_backup()


@pytest.mark.parametrize("bad_cls", [PageMajorMHATokenToKVPool, MLATokenToKVPoolFP4])
def test_unsupported_draft_pool_declines_cpu_retraction_backup(bad_cls):
    target = _mha_pool(1, base=0)
    unsupported = object.__new__(bad_cls)
    worker = _spec_worker(target, [_runner(unsupported)])
    assert worker.retraction_draft_pools() == ()
    assert not worker.supports_cpu_retraction_backup()

    allocator = TokenToKVPoolAllocator(
        SIZE, torch.float32, "cpu", target, need_sort=False
    )
    allocator.register_retraction_draft_pools(
        worker.retraction_draft_pools(),
        supported=worker.supports_cpu_retraction_backup(),
    )
    req = object.__new__(Req)
    req.kv = ReqKvInfo(req_pool_idx=0)
    req_pool = SimpleNamespace(
        req_to_token=torch.zeros((1, 4), dtype=torch.int64), holds_state_slots=False
    )
    assert not backup_kv_cache(req, None, req_pool, allocator, "cpu_tensor")


@pytest.mark.parametrize("compact", [False, True])
def test_dflash_compact_and_non_compact_draft_pools_are_backed_up(compact):
    target = _mha_pool(1, base=0)
    draft = _mha_pool(1, base=10)
    worker = _spec_worker(target, [_runner(draft)], cls=DFlashWorkerV2)
    worker.use_compact_draft_cache = compact
    assert worker.retraction_draft_pools() == (draft,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
