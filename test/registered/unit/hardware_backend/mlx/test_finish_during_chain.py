"""Finish during chain regression guards for the MLX overlap loop (PR #33874).

A request that finishes at step N while _launch_chained has prepared step N+1
ends with kv_committed_len one past len(output_ids). These tests pin the
consequences: the phantom position never enters the radix key, since
cache_finished_req bounds by the token list rather than kv_committed_len; a
reused row does not expose the old owner's entries; and the phantom slot
currently leaks, documented below and tracked separately.
"""

from __future__ import annotations

import importlib.util
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

EOS_ID = 999


class _DummyKVCache(KVCache):
    """Scheduler-facing KV cache that allocates no real GPU/Metal memory.

    Mirrors `hardware_backend/mlx/model_runner_stub.py`'s `_DummyKVCache`:
    the MLX backend manages real attention KV itself, so the core scheduler
    only needs this to size `TokenToKVPoolAllocator`'s slot free-list.
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


def _make_pools(pool_size=64, req_slots=4, max_context_len=128):
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


def _make_req(rid, prompt_len, max_new_tokens=64, id_base=0):
    # id_base keeps each request's prompt token IDs in a disjoint range so
    # RadixCache's content-based prefix matching can't spuriously match one
    # request's prompt against another's cached output (both would
    # otherwise start from token 0).
    sp = SamplingParams()
    sp.max_new_tokens = max_new_tokens
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", range(id_base, id_base + prompt_len)),
        sampling_params=sp,
        vocab_size=1_000_000,
    )
    req.eos_token_ids = {EOS_ID}
    return req


def _admit_prefill(req, rtp, allocator, tree_cache):
    """Simulate the prefill admission that would run before decode: allocate
    the row + real prefix slots and write them into req_to_token, exactly
    like `alloc_for_extend` does. Returns req_pool_idx.
    """
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
    model_config = SimpleNamespace(
        is_encoder_decoder=False, vocab_size=2000, think_end_ids=None
    )
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
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        filter_batch=lambda *a, **k: None,
    )
    batch.hisparse_coordinator = None
    return batch


def _make_result_processor(rtp, allocator, tree_cache, model_config):
    """A real SchedulerBatchResultProcessor: this is what turns a decoded
    token into update_finish_state() + release_kv_cache(), i.e. the exact
    scheduler-side logic the P1 claim is about. Nothing here is mocked.
    """
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=True,
        server_args=None,
        model_config=model_config,
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree_cache,
        hisparse_coordinator=None,
        req_to_token_pool=rtp,
        decode_offload_manager=None,
        metrics_collector=MagicMock(),
        metrics_reporter=MagicMock(),
        draft_worker=None,
        model_worker=None,
        logprob_result_processor=MagicMock(),
        output_streamer=MagicMock(),
        abort_request=MagicMock(),
    )


class _StopLoop(Exception):
    """Sentinel to break out of event_loop_overlap_mlx's `while True`."""


def _make_scheduler(batch, processor, *, n_iters, eos_at_call, eos_rid):
    """A MagicMock scheduler whose scheduling primitives are wired to the
    real implementations (`_prepare_mlx_launch`, `_finalize_mlx_pending_job`,
    `process_batch_result` -> `process_batch_result_decode`,
    `get_next_batch_to_run` -> filter_batch + prepare_for_decode, mirroring
    `update_running_batch`). Only the MLX model-forward boundary
    (`tp_worker.*`) is a bare mock: nothing about scheduler-side accounting,
    finish detection, or KV release is faked.
    """
    from collections import deque

    from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
        SchedulerMlxOverlapMixin,
    )

    scheduler = MagicMock()
    scheduler.forward_ct = 0
    scheduler.running_batch = batch
    scheduler.last_batch = None
    scheduler.gracefully_exit = False
    scheduler._engine_paused = False
    scheduler.waiting_queue = []
    scheduler.result_queue = deque()
    scheduler.batch_result_processor = processor

    scheduler._prepare_mlx_launch.side_effect = lambda b: (
        SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, b)
    )
    scheduler._finalize_mlx_pending_job.side_effect = lambda p: (
        SchedulerMlxOverlapMixin._finalize_mlx_pending_job(scheduler, p)
    )
    scheduler.process_batch_result.side_effect = (
        lambda b, r: processor.process_batch_result_decode(b, r)
    )

    def get_next_batch_to_run(**_kwargs):
        # Mirrors update_running_batch(): filter finished reqs, then commit
        # this step's real allocation via prepare_for_decode() -- exactly
        # what the real scheduler does before handing back batch_to_run.
        batch.filter_batch()
        if not batch.is_empty() and batch.check_decode_mem():
            batch.prepare_for_decode()
        return SimpleNamespace(batch_to_run=batch, running_batch=batch)

    scheduler.get_next_batch_to_run.side_effect = get_next_batch_to_run

    call_count = {"n": 0}

    def finalize_mlx_result(prefills, extends, decode, mode, reqs):
        call_count["n"] += 1
        n = call_count["n"]
        result = MagicMock()
        result.copy_done = None
        result.routed_experts_output = None
        result.indexer_topk_output = None
        result.can_run_cuda_graph = False
        result.speculative_num_draft_tokens = None
        result.num_correct_drafts = None
        result.num_block_accept_tokens = None
        result.num_cap_tokens = None
        logits_output = MagicMock()
        logits_output.hidden_states = None
        result.logits_output = logits_output
        tokens = [
            EOS_ID if (n == eos_at_call and r.rid == eos_rid) else (100 + n)
            for r in reqs
        ]
        result.next_token_ids = torch.tensor(tokens, dtype=torch.long)
        return result

    scheduler.tp_worker.finalize_mlx_result.side_effect = finalize_mlx_result
    scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
        MagicMock(),
        [],
        [],
        MagicMock(),
        "decode",
    )
    scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
        MagicMock(),
        [],
        [],
        MagicMock(),
        "decode",
    )

    recv_count = {"n": 0}

    def recv_requests():
        recv_count["n"] += 1
        if recv_count["n"] > n_iters:
            raise _StopLoop()
        return []

    scheduler.request_receiver.recv_requests.side_effect = recv_requests
    return scheduler


def _run_loop(scheduler):
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
        SchedulerMlxOverlapMixin,
    )

    try:
        SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)
    except _StopLoop:
        pass


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestFinishDuringChainRegressionGuard(CustomTestCase):
    """Drives the real event_loop_overlap_mlx through a request that EOSes
    while the chain has already committed one more position underneath it,
    with a second live request in the pool so a freed-and-reused row is
    observable. Mocks only tp_worker (the MLX model-forward boundary).
    """

    def setUp(self):
        _install_server_args(self)

    def _drive_finish_during_chain(self, *, req_slots=4, pool_size=64):
        rtp, allocator, tree_cache = _make_pools(
            pool_size=pool_size, req_slots=req_slots
        )
        req_a = _make_req("a", prompt_len=3, id_base=10_000)  # EOSes mid-chain
        req_b = _make_req("b", prompt_len=3, id_base=20_000)  # stays alive
        idx_a = _admit_prefill(req_a, rtp, allocator, tree_cache)
        idx_b = _admit_prefill(req_b, rtp, allocator, tree_cache)
        batch = _make_decode_batch(
            [req_a, req_b], [idx_a, idx_b], rtp, allocator, tree_cache
        )
        processor = _make_result_processor(
            rtp, allocator, tree_cache, batch.model_config
        )
        # Call 1: fresh-launch finalize (both get an ordinary token).
        # Call 2: pending_curr finalize -- req_a's real last token is EOS,
        #         but _launch_chained already committed step N+1 for req_a
        #         earlier in this same iteration (the race under test).
        # Call 3: the chain-break finalize of the already-launched N+1 step;
        #         req_a is already finished() by then, so
        #         process_batch_result_decode must skip it (line ~847-852).
        scheduler = _make_scheduler(
            batch, processor, n_iters=3, eos_at_call=2, eos_rid="a"
        )
        _run_loop(scheduler)
        return rtp, allocator, tree_cache, req_a, req_b, idx_a, idx_b

    def test_no_phantom_committed_position_is_radix_inserted(self):
        """The Codex P1 claim: does the over-committed chain step leak into
        the radix cache as if it were real, forwarded KV? It must not.
        """
        rtp, allocator, tree_cache, req_a, req_b, idx_a, idx_b = (
            self._drive_finish_during_chain()
        )

        self.assertTrue(req_a.finished())
        correct_len = len(req_a.origin_input_ids) + len(req_a.output_ids)

        # C1: _launch_chained's prepare_for_decode() really did commit one
        # position past what req_a's real (EOS) token authorizes.
        self.assertGreater(req_a.kv_committed_len, correct_len)
        self.assertEqual(req_a.kv_committed_len, correct_len + 1)

        # C2 (the actual P1 claim): the radix insert must be bounded by
        # what was really produced (origin_input_ids + output_ids), not by
        # the over-committed kv_committed_len. A match on req_a's true
        # sequence must land exactly on correct_len -- not correct_len + 1.
        full_ids = req_a.origin_input_ids + req_a.output_ids
        match = tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids, None)))
        self.assertEqual(len(match.device_indices), correct_len)

        # And the specific over-committed slot must not appear among the
        # values radix now serves for req_a.
        phantom_slot = int(rtp.req_to_token[idx_a, correct_len].item())
        inserted = set(int(x) for x in match.device_indices.tolist())
        self.assertNotIn(phantom_slot, inserted)

    def test_row_reuse_does_not_corrupt_the_new_owner_at_the_scheduler_level(
        self,
    ):
        """The freed row's new owner sees only its own committed range: req_a's
        leftover slot ids and radix entries do not reach req_c. The deferred
        MLX KV sync half of the row reuse concern is out of scope here,
        tracked in #30147.
        """
        rtp, allocator, tree_cache, req_a, req_b, idx_a, idx_b = (
            self._drive_finish_during_chain(req_slots=2)
        )
        self.assertIsNone(req_a.req_pool_idx)  # freed

        req_c = _make_req("c", prompt_len=2, id_base=30_000)
        idx_c = _admit_prefill(req_c, rtp, allocator, tree_cache)
        self.assertEqual(idx_c, idx_a, "test setup expects the freed row to be reused")

        full_ids_c = req_c.origin_input_ids
        match_c = tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(full_ids_c, None))
        )
        # req_c has no cached prefix of its own yet (fresh prompt): the
        # reused row must not let req_a's old radix entries satisfy req_c.
        self.assertEqual(len(match_c.device_indices), 0)

        row_c_committed = rtp.req_to_token[idx_c, : req_c.kv_committed_len].tolist()
        row_b_committed = rtp.req_to_token[idx_b, : req_b.kv_committed_len].tolist()
        self.assertEqual(len(set(row_c_committed) & set(row_b_committed)), 0)

    def test_documents_leaked_kv_slot_on_finish_during_chain(self):
        """Documents the leak, separate from the P1 claim: the over committed
        slot is excluded from the radix insert and from
        _release_overallocated_kv_indices, because kv_allocated_len advanced
        in lockstep with kv_committed_len, so start_p == end_p and nothing
        frees it. Flip the final assert to assertIn once the leak is fixed.
        """
        rtp, allocator, tree_cache, req_a, req_b, idx_a, idx_b = (
            self._drive_finish_during_chain()
        )
        correct_len = len(req_a.origin_input_ids) + len(req_a.output_ids)
        phantom_slot = int(rtp.req_to_token[idx_a, correct_len].item())

        free_pages = (
            set(int(x) for x in allocator.free_pages.tolist())
            if allocator.free_pages is not None
            else set()
        )
        release_pages = (
            set(int(x) for x in allocator.release_pages.tolist())
            if getattr(allocator, "release_pages", None) is not None
            else set()
        )
        # Documents current behavior (the leak): flip this to assertIn(...)
        # once the leak is closed (mainline-shared fix; see report).
        self.assertNotIn(phantom_slot, free_pages | release_pages)


if __name__ == "__main__":
    unittest.main()
