"""Phase 0 baseline for #30093: real decode-KV-sync slot-zero reproduction.

Drives the real `event_loop_overlap_mlx` chained-decode loop against a real
(fake-weights) `MlxModelRunner` / `ContiguousAttentionKVCache` /
`MlxAttentionKVPool` stack -- the same objects and the same
`_sync_decode_kv_to_pool` call site the issue's own repro script and the
serving-level measurement on the #30093 thread exercised -- and tallies, per
generated position, which shared-pool slot the sync actually resolves. This
mirrors the issue thread's serving-level framing ("a 220 token generation
resolved 164 then 56 slot resolutions, of which 163 and 56 were slot 0") at
unit scale: exact counts differ, the mechanism does not.

`test_committed_position_sync_never_resolves_padding_slot_zero` is the
dedicated slot-0 assert requested in the #30093 accounting-patch task: no
position the request has actually generated may have its KV synced to
padding slot 0. Before the fix in `scheduler_mixin.py` (`_launch_chained`
never calling `prepare_for_decode()`), chained positions are unwritten
`req_to_token` cells -- zero-initialized, i.e. slot 0 -- so this is red on
unpatched main. After the fix, every chained position gets a real allocation
from `alloc_for_decode()`, which never hands out slot 0 (reserved as
padding, see `kv_cache/attention_kv_pool.py`), so it is green.

This module deliberately drives the real `_sync_decode_kv_to_pool`
(`model_runner.py`, untouched by this patch) rather than only inspecting
`req_to_token` directly (as `test_chained_decode_accounting.py` does), to
pin the exact mechanism the issue and the serving-level measurement used.
"""

from __future__ import annotations

import importlib.util
import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

_N_KV_HEADS = 2
_HEAD_DIM = 4


class _DummyKVCache(KVCache):
    """Scheduler-facing KV cache sized only to back a real allocator free-list.

    Mirrors `hardware_backend/mlx/model_runner_stub.py`'s `_DummyKVCache`: the
    MLX backend stores real attention KV itself (in `MlxAttentionKVPool`), so
    the core scheduler's pool object only needs to size the slot free-list.
    """

    def __init__(self, size, dtype, device):
        self.size = size
        self.page_size = 1
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def get_key_buffer(self, layer_id):
        raise RuntimeError("_DummyKVCache has no key buffer")

    def get_value_buffer(self, layer_id):
        raise RuntimeError("_DummyKVCache has no value buffer")

    def get_kv_buffer(self, layer_id):
        raise RuntimeError("_DummyKVCache has no kv buffer")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        raise RuntimeError("_DummyKVCache cannot set kv buffer")

    def get_kv_size_bytes(self):
        return 0, 0


def _install_server_args(test_case, **fields):
    override = get_context().override_server_args(**fields)
    override.install()
    test_case.addCleanup(override.restore)


def _make_pools(pool_size=64, req_slots=8, max_context_len=128):
    rtp = ReqToTokenPool(
        size=req_slots,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )
    dummy_kv = _DummyKVCache(pool_size, torch.float32, "cpu")
    allocator = TokenToKVPoolAllocator(
        size=pool_size,
        dtype=torch.float32,
        device="cpu",
        kvcache=dummy_kv,
        need_sort=False,
    )
    tree_cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=rtp,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    return rtp, allocator, tree_cache


def _make_req(rid, prompt_len, max_new_tokens=64):
    sp = SamplingParams()
    sp.max_new_tokens = max_new_tokens
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", range(prompt_len)),
        sampling_params=sp,
    )


def _admit_prefill(req, rtp, allocator):
    prompt_len = len(req.origin_input_ids)
    req_pool_idx = rtp.alloc([req])[0]
    prefix_slots = allocator.alloc(prompt_len)
    assert prefix_slots is not None
    rtp.write((req_pool_idx, slice(0, prompt_len)), prefix_slots.to(torch.int32))
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = prompt_len
    req.kv = ReqKvInfo(kv_allocated_len=prompt_len, swa_evicted_seqlen=0)
    req.decode_batch_idx = 0
    return req_pool_idx


def _make_decode_batch(reqs, req_pool_indices, rtp, allocator, tree_cache):
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    model_config = SimpleNamespace(is_encoder_decoder=False, vocab_size=100)
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=rtp,
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree_cache,
        model_config=model_config,
        enable_overlap=False,
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    seq_lens = [r.kv_committed_len for r in reqs]
    batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64)
    batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    return batch


def _make_runner(rtp, pool_size=64):
    """A real, partially wired `MlxModelRunner` -- same pattern as the issue's
    own deterministic repro script: `__new__` bypasses `__init__` (no model
    weights needed), then only the attributes `_sync_decode_kv_to_pool`
    actually reads are set, using the real `MlxModelCacheLayout` /
    `MlxAttentionKVPool` classes, not stand-ins.
    """
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
        MlxAttentionKVPool,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner.disable_radix_cache = False
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
        layers=["attn"], attention_attrs=["self_attn"]
    )
    runner._attention_kv_pool = MlxAttentionKVPool(
        pool_size=pool_size,
        num_layers=1,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=mx.float32,
    )
    runner._req_to_token_pool = rtp  # same object identity the scheduler writes
    runner._req_caches = {}
    runner._req_pool_idx = {}
    runner._req_synced_offset = {}
    return runner


def _make_fake_attention_cache():
    import mlx.core as mx

    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
        ContiguousAttentionKVCache,
    )

    return [
        ContiguousAttentionKVCache(
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            max_seq_len=64,
            dtype=mx.float32,
        )
    ]


def _advance_cache_one_token(cache, value):
    """Simulate one MLX forward step writing this token's KV into the
    per-request private cache -- exactly what `decode_batch_start` /
    `decode_batch_start_chained` do internally, advancing `cache.offset` by
    one regardless of whether the scheduler committed the position.
    """
    import mlx.core as mx

    k = mx.full((1, _N_KV_HEADS, 1, _HEAD_DIM), value, dtype=mx.float32)
    cache[0].write_token(k, k)


def _read_slot_resolutions(runner, req_id, req_pool_idx):
    """Recompute the exact read `_sync_decode_kv_to_pool` performs, without
    consuming it (so the real call afterwards still has work to do): for
    every position between the last synced offset and the current MLX cache
    offset, what `req_to_token` slot id does it resolve to.
    """
    cache = runner._req_caches[req_id]
    current_offset = runner._first_attention_cache(cache).offset
    synced_offset = runner._req_synced_offset.get(req_id, 0)
    return (
        runner._req_to_token_pool.req_to_token[
            req_pool_idx, synced_offset:current_offset
        ]
        .to(dtype=int)
        .tolist()
    )


class _StopLoop(Exception):
    pass


def _run_chained_decode(rtp, allocator, tree_cache, req, req_pool_idx, n_iters):
    """Drive the real `event_loop_overlap_mlx` for `n_iters` recv cycles
    (one fresh launch + several chained continuations under steady decode,
    no concurrent arrivals -- exactly the `can_chain` precondition), while a
    real `MlxModelRunner` + `ContiguousAttentionKVCache` advance in lockstep
    with each MLX-level step, mirroring what a real forward does.

    Returns the runner, so callers can inspect/sync pool state afterwards.
    """
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
        SchedulerMlxOverlapMixin,
    )

    batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

    runner = _make_runner(rtp)
    fake_cache = _make_fake_attention_cache()
    runner._req_caches[req.rid] = fake_cache
    runner._req_pool_idx[req.rid] = req_pool_idx
    runner._req_synced_offset[req.rid] = 0
    token_value = [100.0]

    def _step_mlx_forward(*_args, **_kwargs):
        _advance_cache_one_token(fake_cache, token_value[0])
        token_value[0] += 1.0
        return MagicMock(), [], [], MagicMock(), "decode"

    scheduler = MagicMock()
    scheduler.forward_ct = 0
    scheduler.running_batch = batch
    scheduler.last_batch = None
    scheduler.gracefully_exit = False
    scheduler._engine_paused = False
    scheduler.waiting_queue = []
    scheduler.result_queue = deque()
    scheduler.request_receiver.recv_requests.side_effect = [
        [] for _ in range(n_iters)
    ] + [_StopLoop()]
    scheduler._prepare_mlx_launch.side_effect = lambda b: (
        SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, b)
    )
    result = MagicMock()
    result.next_token_ids = None
    scheduler.tp_worker.finalize_mlx_result.return_value = result

    plan = SimpleNamespace(batch_to_run=batch, running_batch=batch)
    scheduler.get_next_batch_to_run.return_value = plan
    scheduler.tp_worker.async_forward_batch_generation_mlx.side_effect = (
        _step_mlx_forward
    )
    scheduler.tp_worker.async_chained_decode_mlx.side_effect = (
        lambda _decode: _step_mlx_forward()
    )

    try:
        SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)
    except _StopLoop:
        pass

    n_chained_calls = scheduler.tp_worker.async_chained_decode_mlx.call_count
    return runner, n_chained_calls


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestChainedDecodeSlotZeroBaseline(CustomTestCase):
    """Real `_sync_decode_kv_to_pool` slot-resolution reproduction (#30093)."""

    def setUp(self):
        _install_server_args(self)

    def test_chain_scatters_kv_to_padding_slot_zero(self):
        """Phase 0.1: reproduce the scatter at unit scale, same mechanism as
        the issue's own repro and the serving-level measurement -- drive a
        chain, then read what shared-pool slot each generated position's KV
        actually resolves to via the real sync.
        """
        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=3)
        req_pool_idx = _admit_prefill(req, rtp, allocator)

        runner, n_chained_calls = _run_chained_decode(
            rtp, allocator, tree_cache, req, req_pool_idx, n_iters=6
        )
        self.assertGreater(n_chained_calls, 0)

        resolutions = _read_slot_resolutions(runner, req.rid, req_pool_idx)
        n_slot_zero = sum(1 for slot_id in resolutions if slot_id == 0)
        print(
            f"[#30093 baseline] {len(resolutions)} slot resolutions, "
            f"{n_slot_zero} were padding slot 0 "
            f"(kv_committed_len={req.kv_committed_len}, chained_calls={n_chained_calls})"
        )

        # Drive the real production sync too: it must not raise, and on the
        # unpatched base it writes garbage into the pool's reserved slot 0
        # (mirrors the issue repro's "decode KV written to reserved padding
        # slot 0" observation).
        runner._sync_decode_kv_to_pool(req.rid)

        self.assertEqual(
            n_slot_zero,
            0,
            f"{n_slot_zero} of {len(resolutions)} slot resolutions were "
            "padding slot 0 -- chained positions never received a real "
            "allocation before their KV was synced",
        )

    def test_committed_position_sync_never_resolves_padding_slot_zero(self):
        """Phase 0.2: the dedicated slot-0 assert, as a test. No position the
        request has actually generated -- committed by definition once this
        patch lands -- may have its KV sync read or write padding slot 0.
        Longer chain, includes the loop's normal steady-state shape.
        """
        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=5)
        req_pool_idx = _admit_prefill(req, rtp, allocator)

        runner, n_chained_calls = _run_chained_decode(
            rtp, allocator, tree_cache, req, req_pool_idx, n_iters=10
        )
        self.assertGreater(n_chained_calls, 0)

        resolutions = _read_slot_resolutions(runner, req.rid, req_pool_idx)
        self.assertNotIn(
            0,
            resolutions,
            "a generated position's decode KV sync resolved padding slot 0",
        )

        runner._sync_decode_kv_to_pool(req.rid)
        # The real sync must have advanced past every position the MLX cache
        # produced -- not just up to whatever kv_committed_len says today.
        cache_offset = runner._first_attention_cache(runner._req_caches[req.rid]).offset
        self.assertEqual(runner._req_synced_offset[req.rid], cache_offset)


if __name__ == "__main__":
    unittest.main()
