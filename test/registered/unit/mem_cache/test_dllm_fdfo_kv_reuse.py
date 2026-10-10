"""Tests dLLM KV reuse and committed-token boundaries."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, base=1000, page_size=1):
        self.base = base
        self.page_size = page_size
        self.alloc_calls = []
        self.extend_calls = []

    def available_size(self):
        return 1 << 30

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
        return torch.arange(self.base, self.base + need_size, dtype=torch.int64)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        **kwargs,
    ):
        self.extend_calls.append(
            {
                "extend_num_tokens": extend_num_tokens,
                "seq_lens_cpu": seq_lens_cpu.tolist(),
            }
        )
        return torch.arange(self.base, self.base + extend_num_tokens, dtype=torch.int64)


class _FakeTreeCache:
    def __init__(self, allocator):
        self.page_size = allocator.page_size
        self.token_to_kv_pool_allocator = allocator

    def supports_prefix_sharing(self):
        return False

    def maybe_hand_to_session(self, req):
        pass

    def prefix_device_indices(self, req):
        return req.tree_prefix


def _make_req(rid, prefix, block_size, *, req_pool_idx=None, reuse=False):
    return SimpleNamespace(
        rid=rid,
        tree_prefix=torch.tensor(prefix, dtype=torch.int32),
        prefix_len=len(prefix),
        dllm_incomplete_ids=array("q", range(block_size)) if reuse else array("q"),
        inflight_middle_chunks=1 if req_pool_idx is not None else 0,
        kv=ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=len(prefix) if req_pool_idx is not None else 0,
            kv_allocated_len=(
                len(prefix) + block_size if req_pool_idx is not None else 0
            ),
        ),
    )


def _remove_allocated_req_slots(pool, *reqs):
    for req in reqs:
        if req.kv.req_pool_idx in pool.free_slots:
            pool.free_slots.remove(req.kv.req_pool_idx)


def _make_batch(pool, allocator, reqs, extend_lens):
    seq_lens_cpu = torch.tensor(
        [req.prefix_len + extend_len for req, extend_len in zip(reqs, extend_lens)],
        dtype=torch.int64,
    )
    return SimpleNamespace(
        device="cpu",
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=_FakeTreeCache(allocator),
        prefix_lens=[req.prefix_len for req in reqs],
        extend_lens=extend_lens,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=sum(extend_lens),
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: True,
    )


def _seed_retained_block(pool, req, values):
    prefix_len = req.prefix_len
    pool.req_to_token[req.kv.req_pool_idx, :prefix_len] = req.tree_prefix
    pool.req_to_token[req.kv.req_pool_idx, prefix_len : prefix_len + len(values)] = (
        torch.tensor(values, dtype=torch.int32)
    )


class TestDllmFdfoKvReuse(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        override = get_context().override_server_args(
            attention_backend="torch_native", dcp_size=1
        )
        override.install()
        self.addCleanup(override.restore)

    def test_alloc_for_extend_mixed_reuse_allocates_only_fresh_and_writes_rows(self):
        allocator = _FakeAllocator(base=200)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        fresh = _make_req("fresh", [20, 21, 22, 23], self.block_size)
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused, fresh], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(allocator.alloc_calls, [4])
        # Allocation order is not semantically meaningful (ReqToTokenPool.alloc
        # picks whichever free slot is cheapest to pop), so only pin the
        # reused row's index and that the fresh row got a different, real slot.
        self.assertEqual(req_pool_indices_cpu[0].item(), 1)
        fresh_idx = req_pool_indices_cpu[1].item()
        self.assertNotEqual(fresh_idx, 1)
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 200, 201, 202, 203])
        self.assertEqual(self.pool.req_to_token[1, 4:8].tolist(), [100, 101, 102, 103])
        self.assertEqual(
            self.pool.req_to_token[fresh_idx, 4:8].tolist(), [200, 201, 202, 203]
        )
        self.assertEqual(reused.kv.kv_allocated_len, 8)
        self.assertEqual(fresh.kv.kv_allocated_len, 8)

    def test_alloc_for_extend_all_reuse_allocates_nothing(self):
        allocator = _FakeAllocator(base=900)
        req0 = _make_req(
            "r0", [1, 2, 3, 4], self.block_size, req_pool_idx=1, reuse=True
        )
        req1 = _make_req(
            "r1", [5, 6, 7, 8], self.block_size, req_pool_idx=2, reuse=True
        )
        _remove_allocated_req_slots(self.pool, req0, req1)
        _seed_retained_block(self.pool, req0, [300, 301, 302, 303])
        _seed_retained_block(self.pool, req1, [400, 401, 402, 403])

        batch = _make_batch(self.pool, allocator, [req0, req1], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(allocator.alloc_calls, [])
        self.assertEqual(req_pool_indices_cpu.tolist(), [1, 2])
        self.assertEqual(out.tolist(), [300, 301, 302, 303, 400, 401, 402, 403])

    def test_alloc_for_extend_paged_mixed_reuse_skips_reused_rows(self):
        allocator = _FakeAllocator(base=500, page_size=4)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        fresh = _make_req("fresh", [20, 21, 22, 23], self.block_size)
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused, fresh], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        # See test_alloc_for_extend_mixed_reuse_allocates_only_fresh_and_writes_rows:
        # allocation order is not semantically meaningful.
        self.assertEqual(req_pool_indices_cpu[0].item(), 1)
        self.assertNotEqual(req_pool_indices_cpu[1].item(), 1)
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 500, 501, 502, 503])
        self.assertEqual(
            allocator.extend_calls,
            [{"extend_num_tokens": 4, "seq_lens_cpu": [4, 8]}],
        )

    def test_alloc_for_extend_rejects_partial_retained_block_reuse(self):
        allocator = _FakeAllocator(base=700)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused], [2])
        with self.assertRaisesRegex(RuntimeError, "full block"):
            alloc_for_extend(batch)

    def test_dllm_manager_pop_aborted_reqs_removes_waiting_and_staging(self):
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        waiting = _make_req("abort-waiting", [1], self.block_size)
        staging = _make_req("abort-staging", [2], self.block_size)
        keep = _make_req("keep", [3], self.block_size)
        manager.waiting_queue = [waiting, keep]
        manager.staging_queue = [staging, waiting]

        aborted = manager.pop_aborted_reqs(False, "abort")

        self.assertEqual(
            [req.rid for req in aborted], ["abort-waiting", "abort-staging"]
        )
        self.assertEqual(manager.waiting_queue, [keep])
        self.assertEqual(manager.staging_queue, [])


class TestDllmFdfoResolvedBlockKeepsRow(unittest.TestCase):
    def test_resolved_block_keeps_row_until_next_block(self):
        """A resolved FDFO block used to hand its row back to the pool while the
        request kept running. An abort before the next block then skipped
        release_kv_cache (it only runs for row holders), leaking the request's
        tree lock and any KV the tree does not own."""
        pool = ReqToTokenPool(
            size=4, max_context_len=16, device="cpu", enable_memory_saver=False
        )
        req = SimpleNamespace(
            dllm_incomplete_ids=array("q"),
            is_dllm_prefill=lambda: False,
            kv=ReqKvInfo(kv_allocated_len=8, kv_committed_len=8),
        )
        pool.alloc([req])
        row = req.kv.req_pool_idx
        scheduler = SimpleNamespace(
            dllm_config=SimpleNamespace(
                first_done_first_out_mode=True,
                requires_separate_context_encoding=False,
            ),
            req_to_token_pool=pool,
            stash_chunked_request=Mock(),
        )

        SchedulerDllmMixin.finish_dllm_forward(scheduler, req)

        scheduler.stash_chunked_request.assert_called_once_with(req)
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(req.kv.req_pool_idx, row)
        self.assertNotIn(row, pool.free_slots)


class TestDllmCommittedTokens(unittest.TestCase):
    @staticmethod
    def make_req(prompt, output=()):
        req = Req(
            rid="committed-tokens",
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(max_new_tokens=16, temperature=0),
            vocab_size=128,
            dllm_config=DllmConfig("LowConfidence", {}, 4, 99, 4),
        )
        req.output_ids.extend(output)
        return req

    def test_cache_lookup_stops_before_partial_prompt_and_synthetic_masks(self):
        # A different request legitimately cached these literal mask IDs. Only
        # the first complete block belongs to this request's committed prefix.
        cache = RadixCache.create_simulated(page_size=4)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4, 5, 99, 99, 99])),
                value=torch.arange(8),
            )
        )
        req = self.make_req([1, 2, 3, 4, 5])
        req.init_next_round_input(cache)
        self.assertEqual(req.prefix_len, 4)
        self.assertEqual(req.dllm_block_offset, 4)
        self.assertFalse(req.is_dllm_prefill())
        self.assertEqual(list(req.full_untruncated_fill_ids[4:8]), [5, 99, 99, 99])

    def test_retraction_reencodes_committed_outputs_before_denoising(self):
        req = self.make_req([1, 99, 2], [7, 99, 8, 9, 10])
        req.reset_for_retract()
        req.init_next_round_input()
        self.assertEqual(list(req.output_ids), [7, 99, 8, 9, 10])
        self.assertTrue(req.is_dllm_prefill())
        # The second complete block consists of already-emitted output tokens.
        req.prefix_len = 4
        req.determine_dllm_phase()
        self.assertTrue(req.is_dllm_prefill())
        req.prefix_len = 8
        req.determine_dllm_phase()
        self.assertFalse(req.is_dllm_prefill())

    def test_repeated_denoise_passes_do_not_subtract_cache_hits(self):
        req = self.make_req([1, 2, 3, 4, 5])
        req._init_fill_ids_for_dllm()
        req.prefix_len = 4
        req.extend_end = 8
        req.already_computed = 8
        req.cached_tokens = 4
        req._cache_breakdown_computed = True
        batch = ScheduleBatch(
            reqs=[req],
            device="cpu",
            dllm_config=req.dllm_config,
            model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=128),
        )
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=False),
            mamba=SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        # Allocation and sampling are collaborators; exercise the scheduler's
        # real cache accounting on two revisits of the same fixed-size block.
        with (
            patch(
                "sglang.srt.managers.schedule_batch.get_exec", return_value=execution
            ),
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_extend",
                return_value=(torch.arange(4), torch.tensor([0]), torch.tensor([0])),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch"
            ),
        ):
            for _ in range(2):
                batch.prepare_for_extend()
                self.assertEqual(req.cached_tokens, 4)

    def test_literal_prompt_masks_are_preserved_and_not_emitted(self):
        class Runner:
            def forward(self, batch, **kwargs):
                logits = torch.zeros(4, 16)
                logits[:, 7] = 20
                return SimpleNamespace(
                    logits_output=SimpleNamespace(full_logits=logits),
                    can_run_graph=False,
                )

        for cls, vectorized in (
            (LowConfidence, False),
            (JointThreshold, False),
            (JointThreshold, True),
        ):
            for fdfo in (False, True):
                with self.subTest(
                    algorithm=cls.__name__, vectorized=vectorized, fdfo=fdfo
                ):
                    algorithm = cls(
                        DllmConfig(
                            cls.__name__,
                            {"vectorized_decoding": vectorized},
                            4,
                            15,
                            1,
                            fdfo,
                        )
                    )
                    batch = SimpleNamespace(
                        batch_size=1,
                        input_ids=torch.tensor([15, 2, 15, 15]),
                        dllm_prompt_mask=torch.tensor([[True, True, False, False]]),
                    )
                    states = None
                    for _ in range(32):
                        _, output, accepted, states, _ = algorithm.run(
                            Runner(), batch, states
                        )
                        self.assertEqual(batch.input_ids[:2].tolist(), [15, 2])
                        if not fdfo or accepted == [4]:
                            break
                    else:
                        self.fail("denoising did not finish")
                    self.assertEqual(batch.input_ids.tolist(), [15, 2, 7, 7])
                    if not fdfo:
                        self.assertEqual(output[0].tolist(), [7, 7])


if __name__ == "__main__":
    unittest.main()
