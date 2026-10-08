"""Tests dLLM FDFO KV slot reuse in alloc_for_extend."""

import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
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

    def is_chunk_cache(self):
        return True


def _make_req(rid, prefix, block_size, *, req_pool_idx=None, reuse=False):
    return SimpleNamespace(
        rid=rid,
        prefix_indices=torch.tensor(prefix, dtype=torch.int32),
        dllm_incomplete_ids=array("q", range(block_size)) if reuse else array("q"),
        dllm_block_done=False,
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
        [
            len(req.prefix_indices) + extend_len
            for req, extend_len in zip(reqs, extend_lens)
        ],
        dtype=torch.int64,
    )
    return SimpleNamespace(
        device="cpu",
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=_FakeTreeCache(allocator),
        prefix_lens=[len(req.prefix_indices) for req in reqs],
        extend_lens=extend_lens,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=sum(extend_lens),
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: True,
    )


def _seed_retained_block(pool, req, values):
    prefix_len = len(req.prefix_indices)
    pool.req_to_token[req.kv.req_pool_idx, :prefix_len] = req.prefix_indices
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

    def _check_delayed_prompt_result(self, scheduler, req):
        fill = list(req.full_untruncated_fill_ids)
        done = req.dllm_block_done
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=lambda: None, free_group_end=lambda: None
        )
        scheduler.metrics_reporter = SimpleNamespace(
            report_prefill_stats=lambda **kwargs: None
        )
        scheduler.output_streamer = SimpleNamespace(stream_output=lambda *args: None)
        scheduler.process_batch_result_dllm(
            SimpleNamespace(
                reqs=[req], return_logprob=False,
                prefill_stats=None, dp_cooperation_info=None,
            ),
            GenerationBatchResult(
                dllm_block_ids=(7,),
                next_token_ids=torch.tensor([[2, 3, 4, 5]]),
                dllm_block_done=torch.tensor([True]),
                dllm_algo_state=None,
            ),
        )
        self.assertEqual(list(req.full_untruncated_fill_ids), fill)
        self.assertEqual(req.dllm_block_done, done)
        self.assertEqual(list(req.output_ids), [])

    def test_completed_or_prompt_block_keeps_kv_and_opens_only_when_prepared(self):
        for fdfo, page_size in ((True, 1), (True, 4)):
            for done, prompt in (
                (True, [2, 3]),
                (True, list(range(2, 12))),
                (False, list(range(2, 12))),
                (False, [2, 3, 4, 5]),
                (False, [2, 3]),
            ):
                with self.subTest(
                    fdfo=fdfo, page_size=page_size, done=done, prompt=prompt
                ):
                    self.pool.clear()
                    allocator = _FakeAllocator(base=500, page_size=page_size)
                    allocator.device = torch.device("cpu")
                    config = SimpleNamespace(
                        block_size=4,
                        mask_id=0,
                        max_running_requests=2,
                        first_done_first_out_mode=fdfo,
                    )
                    req = Req(
                        rid="completed",
                        origin_input_text="test",
                        origin_input_ids=array("q", prompt),
                        sampling_params=SamplingParams(max_new_tokens=32),
                        dllm_config=config,
                    )
                    req.init_next_round_input()
                    req.dllm_block_id = 7
                    req.dllm_block_done = done
                    req.set_extend_range(0, 4)
                    if done and len(prompt) < 4:
                        req.output_ids = array("q", [4, 5])
                    req.kv = ReqKvInfo(
                        req_pool_idx=1, kv_allocated_len=4, kv_committed_len=4
                    )
                    _remove_allocated_req_slots(self.pool, req)
                    _seed_retained_block(self.pool, req, [100, 101, 102, 103])
                    manager = DllmManager(config)
                    manager.waiting_queue = [req]
                    self.assertEqual(manager.staging_queue, [])
                    scheduler = Scheduler.__new__(Scheduler)
                    scheduler.dllm_config = config
                    scheduler.enable_overlap = True
                    scheduler.req_to_token_pool = self.pool
                    scheduler.tree_cache = ChunkCache(
                        SimpleNamespace(
                            req_to_token_pool=self.pool,
                            token_to_kv_pool_allocator=allocator,
                            page_size=page_size,
                        )
                    )
                    scheduler.future_map = SimpleNamespace(
                        dllm_block_tokens_buf=torch.full((9, 4), 9)
                    )
                    scheduler.forward_stream_ctx = nullcontext()
                    budget = [0]
                    adder = SimpleNamespace(
                        dllm_config=config,
                        can_run_list=[],
                        _get_dllm_remain_tokens=lambda: budget[0],
                        _update_prefill_budget=lambda *args, **kwargs: None,
                        _mamba_gap_budget_for_req=lambda req: 0,
                    )
                    adder.add_dllm_staging_req = (
                        lambda item: PrefillAdder.add_dllm_staging_req(adder, item)
                    )

                    scheduler.process_dllm_staging_reqs(adder, manager.waiting_queue)
                    self.assertEqual(adder.can_run_list, [])
                    if not done and len(prompt) < 4:
                        budget[0] = 4
                        scheduler.process_dllm_staging_reqs(adder, manager.waiting_queue)
                        self.assertEqual(adder.can_run_list, [req])
                        self.assertFalse(req.dllm_block_done)
                        self.assertEqual(req.dllm_block_id, 7)
                        self.assertEqual(req.dllm_block_offset, 0)
                        self.assertEqual(req.prefix_indices.tolist(), [])
                        self.assertEqual(req.kv.req_pool_idx, 1)
                        self.assertEqual(req.get_fill_ids().tolist(), [2, 3, 0, 0])
                        self.assertEqual(
                            scheduler.future_map.dllm_block_tokens_buf[1].tolist(),
                            [9] * 4,
                        )
                        continue
                    self.assertEqual(req.dllm_block_id, 7)
                    self.assertTrue(req.dllm_block_done)
                    self.assertIsNone(req.kv.req_pool_idx)
                    self.assertEqual(
                        req.prefix_indices.tolist(), [100, 101, 102, 103]
                    )
                    self.assertEqual(
                        scheduler.future_map.dllm_block_tokens_buf[1].tolist(),
                        [-1] * 4 if fdfo else [9] * 4,
                    )
                    if not done:
                        self._check_delayed_prompt_result(scheduler, req)
                    prepared_ids = list(req.full_untruncated_fill_ids)
                    block_offset = req.dllm_block_offset

                    budget[0] = 4
                    scheduler.process_dllm_staging_reqs(adder, manager.waiting_queue)
                    self.assertEqual(adder.can_run_list, [req])
                    self.assertEqual(list(req.full_untruncated_fill_ids), prepared_ids)
                    self.assertEqual(req.dllm_block_offset, block_offset)
                    self.assertEqual(req.dllm_block_id, 7)
                    self.assertTrue(req.dllm_block_done)
                    self.assertEqual(
                        (req.extend_range.start, req.extend_range.end), (4, 8)
                    )

                    class AllocationComplete(Exception):
                        pass

                    outputs = []

                    def allocate(batch):
                        outputs.append(alloc_for_extend(batch))
                        raise AllocationComplete

                    for _ in range(2):
                        batch = _make_batch(self.pool, allocator, [req], [4])
                        with patch(
                            "sglang.srt.managers.schedule_batch.alloc_for_extend",
                            side_effect=allocate,
                        ), self.assertRaises(AllocationComplete):
                            ScheduleBatch.prepare_for_extend(batch)
                        self.assertEqual(req.dllm_block_id, 8)
                        self.assertFalse(req.dllm_block_done)
                        if not done:
                            self._check_delayed_prompt_result(scheduler, req)
                        self.assertEqual(batch.prefix_lens, [4])
                        self.assertEqual(batch.seq_lens_cpu.tolist(), [8])
                        expected_ids = [0] * 4 if len(prompt) <= 4 else [6, 7, 8, 9]
                        self.assertEqual(list(req.get_fill_ids()[4:]), expected_ids)
                        slot = req.kv.req_pool_idx
                        self.assertEqual(
                            self.pool.req_to_token[slot, :8].tolist(),
                            [100, 101, 102, 103, 500, 501, 502, 503],
                        )
                    self.assertEqual(outputs[0][0].tolist(), [500, 501, 502, 503])
                    self.assertEqual(outputs[1][0].tolist(), [500, 501, 502, 503])
                    allocation_count = len(allocator.alloc_calls) + len(
                        allocator.extend_calls
                    )
                    self.assertEqual(allocation_count, 1)

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


if __name__ == "__main__":
    unittest.main()
