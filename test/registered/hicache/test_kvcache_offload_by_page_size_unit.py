from __future__ import annotations

import types
import unittest
from dataclasses import dataclass
from typing import Any, List, Optional, Union
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")


@dataclass
class _TimeStats:
    forward_entry_time: float = 0.0
    completion_time: float = 0.0


class _DummyReq(Req):
    def __init__(self, *, rid: str, req_pool_idx: int, origin_input_ids: list[int]):
        self.rid = rid
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = origin_input_ids
        self.output_ids: list[int] = []
        self.prefix_indices = [1, 2, 3]
        self.time_stats = _TimeStats()
        self._committed_kv_cache = 0
        self._overalloc = (0, 0)
        self.is_retracted = False

    def finished(self) -> bool:
        return True

    def pop_committed_kv_cache(self) -> int:
        return self._committed_kv_cache

    def pop_overallocated_kv_cache(self):
        return self._overalloc


class _DummyReqToTokenPool(ReqToTokenPool):
    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token
        self.freed_reqs: list[str] = []
        self.idx_to_rid = {}

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, int):
            rid = self.idx_to_rid.get(free_index, str(free_index))
            self.freed_reqs.append(rid)
        else:
            for idx in free_index:
                rid = self.idx_to_rid.get(idx, str(idx))
                self.freed_reqs.append(rid)


class _DummyAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self):
        self.freed: list[torch.Tensor] = []
        self.is_not_in_free_group = True

    def free(self, indices: torch.Tensor):
        self.freed.append(indices.detach().cpu().clone())

    def clear(self):
        pass

    def alloc(self, need_size: int):
        return None


class _DummyTreeCache(BasePrefixCache):
    def __init__(self):
        self.protected_size_ = 0

    def reset(self):
        pass

    def match_prefix(self, key: Any, **kwargs):
        return None

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        pass

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        pass


class TestFinalizeReleaseOnFinish(unittest.TestCase):
    def _make_manager(self, *, page_size: int):
        req_to_token = torch.arange(0, 64, dtype=torch.int64).reshape(1, 64)
        req_to_token_pool = _DummyReqToTokenPool(req_to_token=req_to_token)

        allocator = _DummyAllocator()
        tree_cache = _DummyTreeCache()

        mgr = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        mgr.req_to_token_pool = req_to_token_pool
        mgr.token_to_kv_pool_allocator = allocator
        mgr.page_size = page_size
        mgr.tree_cache = tree_cache
        mgr.offloaded_state = {}
        return mgr, req_to_token_pool, allocator, tree_cache

    def test_finalize_frees_prefill_aligned_when_no_inc_offload(self):
        mgr, pool, allocator, tree_cache = self._make_manager(page_size=4)
        tree_cache.protected_size_ = 3

        req = _DummyReq(rid="r1", req_pool_idx=0, origin_input_ids=list(range(10)))
        req._committed_kv_cache = 10
        pool.idx_to_rid[0] = "r1"

        mgr.finalize_release_on_finish(req)

        self.assertEqual(len(allocator.freed), 2)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(0, 8)))
        self.assertTrue(torch.equal(allocator.freed[1], torch.arange(8, 10)))
        self.assertEqual(pool.freed_reqs, ["r1"])
        self.assertEqual(tree_cache.protected_size_, 0)

    def test_finalize_does_not_double_free_prefill_when_inc_offloaded(self):
        mgr, pool, allocator, _ = self._make_manager(page_size=4)
        mgr.offloaded_state["r2"] = OffloadedState(prefill_len=8, inc_len=4)

        req = _DummyReq(rid="r2", req_pool_idx=0, origin_input_ids=list(range(10)))
        req._committed_kv_cache = 14
        pool.idx_to_rid[0] = "r2"

        mgr.finalize_release_on_finish(req)

        self.assertEqual(len(allocator.freed), 1)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(12, 14)))
        self.assertEqual(pool.freed_reqs, ["r2"])
        self.assertNotIn("r2", mgr.offloaded_state)

    def test_finalize_frees_prefill_when_state_exists_but_inc_never_offloaded(self):
        mgr, pool, allocator, _ = self._make_manager(page_size=4)
        mgr.offloaded_state["r3"] = OffloadedState(prefill_len=8, inc_len=0)

        req = _DummyReq(rid="r3", req_pool_idx=0, origin_input_ids=list(range(10)))
        req._committed_kv_cache = 10
        pool.idx_to_rid[0] = "r3"

        mgr.finalize_release_on_finish(req)

        self.assertEqual(len(allocator.freed), 2)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(0, 8)))
        self.assertTrue(torch.equal(allocator.freed[1], torch.arange(8, 10)))
        self.assertEqual(pool.freed_reqs, ["r3"])
        self.assertNotIn("r3", mgr.offloaded_state)

    def test_finalize_frees_overallocated_kv(self):
        mgr, pool, allocator, _ = self._make_manager(page_size=4)

        req = _DummyReq(rid="r4", req_pool_idx=0, origin_input_ids=list(range(10)))
        req._committed_kv_cache = 10
        req._overalloc = (10, 14)
        pool.idx_to_rid[0] = "r4"

        mgr.finalize_release_on_finish(req)

        # 1. prefill-aligned: [0, 8]
        # 2. remaining: [8, 10]
        # 3. overalloc: [12, 14] (10 is aligned up to 12)
        self.assertEqual(len(allocator.freed), 3)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(0, 8)))
        self.assertTrue(torch.equal(allocator.freed[1], torch.arange(8, 10)))
        self.assertTrue(torch.equal(allocator.freed[2], torch.arange(12, 14)))
        self.assertEqual(pool.freed_reqs, ["r4"])


class TestSchedulerFinalizeReleaseCall(unittest.TestCase):
    def test_decode_finished_offload_failure_triggers_finalize_release(self):
        class _Allocator:
            def free_group_begin(self):
                pass

            def free_group_end(self):
                pass

        class _Batch(ScheduleBatch):
            def __init__(self, reqs):
                self.reqs = reqs
                self.return_logprob = False
                self.spec_algorithm = MagicMock()
                self.spec_algorithm.is_none.return_value = True
                self.enable_overlap = False

            def batch_size(self):
                return len(self.reqs)

        class _Result(LogitsProcessorOutput):
            def __init__(self):
                self.copy_done = None
                self.logits_output = None
                self.next_token_ids = torch.tensor([7], dtype=torch.int64)
                self.can_run_cuda_graph = False

        class _Req(Req):
            def __init__(self):
                self.rid = "r1"
                self._finished = False
                self.output_ids = []
                self.origin_input_ids = [1, 2, 3]
                self.is_retracted = False
                self.to_finish = None
                self.time_stats = _TimeStats()
                self.return_logprob = False
                self.return_hidden_states = False
                self.grammar = None
                self.prefix_indices = []
                self.req_pool_idx = 0

            def finished(self):
                return self._finished

            def check_finished(self, *_args, **_kwargs):
                self._finished = True

        class _DecodeOffloadMgr:
            def __init__(self):
                self.finalize_calls = 0
                self.last_req = None

            def offload_kv_cache(self, _req):
                return False

            def finalize_release_on_finish(self, req):
                self.finalize_calls += 1
                self.last_req = req

        class _Processor(SchedulerOutputProcessorMixin):
            def __init__(self):
                self.server_args = types.SimpleNamespace(
                    disaggregation_decode_enable_offload_kvcache=True,
                    decode_log_interval=1000000,
                )
                self.enable_overlap = False
                self.enable_metrics = False
                self.current_scheduler_metrics_enabled = False
                self.num_generated_tokens = 0
                self.forward_ct_decode = 0
                self.token_to_kv_pool_allocator = _Allocator()
                self.decode_offload_manager = _DecodeOffloadMgr()

            def _mamba_prefix_cache_update(self, *_args, **_kwargs):
                return None

            def maybe_collect_routed_experts(self, *_args, **_kwargs):
                return None

            def maybe_collect_customized_info(self, *_args, **_kwargs):
                return None

            def stream_output(self, *_args, **_kwargs):
                return None

        p = _Processor()
        req = _Req()
        batch = _Batch([req])
        result = _Result()

        p.process_batch_result_decode(batch, result)

        self.assertEqual(p.decode_offload_manager.finalize_calls, 1)
        self.assertIs(p.decode_offload_manager.last_req, req)


if __name__ == "__main__":
    unittest.main()
