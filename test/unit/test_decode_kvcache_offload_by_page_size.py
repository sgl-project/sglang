"""
Unit tests for decode-side KV cache offload finalize-release logic.

Usage:
    python -m pytest test/unit/test_decode_kvcache_offload_by_page_size.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass

import torch


def _ensure_pkg(name: str) -> types.ModuleType:
    if name in sys.modules:
        m = sys.modules[name]
        if not hasattr(m, "__path__"):
            m.__path__ = []
        return m
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


def _install_stub_module(fullname: str, **attrs) -> None:
    parts = fullname.split(".")
    for i in range(1, len(parts)):
        _ensure_pkg(".".join(parts[:i]))
    m = types.ModuleType(fullname)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[fullname] = m


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = m
    spec.loader.exec_module(m)
    return m


def _bootstrap_sglang_namespace(python_dir: str) -> None:
    sglang_dir = os.path.join(python_dir, "sglang")
    sg = types.ModuleType("sglang")
    sg.__path__ = [sglang_dir]
    sys.modules["sglang"] = sg
    _ensure_pkg("sglang.srt")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def _load_decode_kvcache_offload_manager():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    python_dir = os.path.join(repo_dir, "python")
    _bootstrap_sglang_namespace(python_dir)

    class HiCacheController:  # pragma: no cover
        pass

    class BaseTokenToKVPoolAllocator:  # pragma: no cover
        pass

    class BasePrefixCache:  # pragma: no cover
        pass

    class MHATokenToKVPool:  # pragma: no cover
        pass

    class MLATokenToKVPool:  # pragma: no cover
        pass

    class ReqToTokenPool:  # pragma: no cover
        pass

    class MHATokenToKVPoolHost:  # pragma: no cover
        pass

    class MLATokenToKVPoolHost:  # pragma: no cover
        pass

    class ServerArgs:  # pragma: no cover
        pass

    _install_stub_module(
        "sglang.srt.managers.cache_controller",
        HiCacheController=HiCacheController,
    )
    _install_stub_module(
        "sglang.srt.mem_cache.allocator",
        BaseTokenToKVPoolAllocator=BaseTokenToKVPoolAllocator,
    )
    _install_stub_module(
        "sglang.srt.mem_cache.base_prefix_cache",
        BasePrefixCache=BasePrefixCache,
    )
    _install_stub_module(
        "sglang.srt.mem_cache.memory_pool",
        MHATokenToKVPool=MHATokenToKVPool,
        MLATokenToKVPool=MLATokenToKVPool,
        ReqToTokenPool=ReqToTokenPool,
    )
    _install_stub_module(
        "sglang.srt.mem_cache.memory_pool_host",
        MHATokenToKVPoolHost=MHATokenToKVPoolHost,
        MLATokenToKVPoolHost=MLATokenToKVPoolHost,
    )
    _install_stub_module(
        "sglang.srt.server_args",
        ServerArgs=ServerArgs,
    )

    decode_mgr_path = os.path.join(
        python_dir, "sglang", "srt", "disaggregation", "decode_kvcache_offload_manager.py"
    )
    return _load_module_from_path("_decode_kvcache_offload_manager_for_test", decode_mgr_path)


@dataclass
class _TimeStats:
    forward_entry_time: float = 0.0
    completion_time: float = 0.0


class _DummyReq:
    def __init__(self, *, rid: str, req_pool_idx: int, origin_input_ids: list[int]):
        self.rid = rid
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = origin_input_ids
        self.output_ids: list[int] = []
        self.prefix_indices = [1, 2, 3]
        self.time_stats = _TimeStats()
        self._committed_kv_cache = 0
        self._overalloc = (0, 0)

    def finished(self) -> bool:
        return True

    def pop_committed_kv_cache(self) -> int:
        return self._committed_kv_cache

    def pop_overallocated_kv_cache(self):
        return self._overalloc


class _DummyReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token
        self.freed_reqs: list[str] = []

    def free(self, req: _DummyReq):
        self.freed_reqs.append(req.rid)


class _DummyAllocator:
    def __init__(self):
        self.freed: list[torch.Tensor] = []

    def free(self, indices: torch.Tensor):
        self.freed.append(indices.detach().cpu().clone())


class _DummyTreeCache:
    def __init__(self):
        self.protected_size_ = 0


class TestFinalizeReleaseOnFinish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._decode_mgr_mod = _load_decode_kvcache_offload_manager()
        cls.DecodeKVCacheOffloadManager = cls._decode_mgr_mod.DecodeKVCacheOffloadManager

    def _make_manager(self, *, page_size: int):
        req_to_token = torch.arange(0, 64, dtype=torch.int64).reshape(1, 64)
        req_to_token_pool = _DummyReqToTokenPool(req_to_token=req_to_token)
        allocator = _DummyAllocator()
        tree_cache = _DummyTreeCache()

        mgr = self.DecodeKVCacheOffloadManager.__new__(self.DecodeKVCacheOffloadManager)
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

        mgr.finalize_release_on_finish(req)

        self.assertEqual(len(allocator.freed), 2)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(0, 8)))
        self.assertTrue(torch.equal(allocator.freed[1], torch.arange(8, 10)))
        self.assertEqual(pool.freed_reqs, ["r1"])
        self.assertEqual(tree_cache.protected_size_, 0)

    def test_finalize_does_not_double_free_prefill_when_inc_offloaded(self):
        mgr, pool, allocator, _ = self._make_manager(page_size=4)
        mgr.offloaded_state["r2"] = {"prefill_len": 8, "inc_len": 4}

        req = _DummyReq(rid="r2", req_pool_idx=0, origin_input_ids=list(range(10)))
        req._committed_kv_cache = 14

        mgr.finalize_release_on_finish(req)

        self.assertEqual(len(allocator.freed), 1)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(12, 14)))
        self.assertEqual(pool.freed_reqs, ["r2"])
        self.assertNotIn("r2", mgr.offloaded_state)

    def test_finalize_frees_prefill_when_state_exists_but_inc_never_offloaded(self):
        mgr, pool, allocator, _ = self._make_manager(page_size=4)
        mgr.offloaded_state["r3"] = {"prefill_len": 8, "inc_len": 0}

        req = _DummyReq(rid="r3", req_pool_idx=0, origin_input_ids=list(range(10)))
        req._committed_kv_cache = 10

        mgr.finalize_release_on_finish(req)

        self.assertEqual(len(allocator.freed), 2)
        self.assertTrue(torch.equal(allocator.freed[0], torch.arange(0, 8)))
        self.assertTrue(torch.equal(allocator.freed[1], torch.arange(8, 10)))
        self.assertEqual(pool.freed_reqs, ["r3"])
        self.assertNotIn("r3", mgr.offloaded_state)


class TestSchedulerFinalizeReleaseCall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        python_dir = os.path.join(repo_dir, "python")
        _bootstrap_sglang_namespace(python_dir)

        class _EnvVar:
            def __init__(self, v):
                self._v = v

            def get(self):
                return self._v

        class _Envs:
            SGLANG_FORWARD_TIMEOUT_MS = _EnvVar(0)
            SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = _EnvVar(0)

        class DisaggregationMode:
            DECODE = "decode"

        _install_stub_module("sglang.srt.disaggregation.utils", DisaggregationMode=DisaggregationMode)
        _install_stub_module("sglang.srt.environ", envs=_Envs())
        _install_stub_module("sglang.srt.layers.logits_processor", LogitsProcessorOutput=object)
        _install_stub_module(
            "sglang.srt.layers.moe.routed_experts_capturer",
            get_global_experts_capturer=lambda: types.SimpleNamespace(
                get_routed_experts=lambda **_: []
            ),
        )
        _install_stub_module(
            "sglang.srt.managers.io_struct",
            AbortReq=object,
            BatchEmbeddingOutput=object,
            BatchTokenIDOutput=object,
        )
        _install_stub_module(
            "sglang.srt.managers.schedule_batch",
            FINISH_ABORT=lambda *_args, **_kwargs: None,
            BaseFinishReason=object,
            Req=object,
            RequestStage=types.SimpleNamespace(),
            ScheduleBatch=object,
        )
        _install_stub_module("sglang.srt.mem_cache.common", release_kv_cache=lambda *_: None)
        _install_stub_module("sglang.srt.server_args", get_global_server_args=lambda: types.SimpleNamespace())
        _install_stub_module(
            "sglang.srt.tracing.trace",
            trace_slice=lambda *_args, **_kwargs: None,
            trace_slice_batch=lambda *_args, **_kwargs: None,
            trace_slice_end=lambda *_args, **_kwargs: None,
        )

        sched_path = os.path.join(
            python_dir, "sglang", "srt", "managers", "scheduler_output_processor_mixin.py"
        )
        cls._sched_mod = _load_module_from_path(
            "_scheduler_output_processor_mixin_for_test", sched_path
        )
        cls.SchedulerOutputProcessorMixin = cls._sched_mod.SchedulerOutputProcessorMixin

    def test_decode_finished_offload_failure_triggers_finalize_release(self):
        class _Allocator:
            def free_group_begin(self):
                pass

            def free_group_end(self):
                pass

        class _SpecAlgorithm:
            def is_none(self):
                return True

        class _Batch:
            def __init__(self, reqs):
                self.reqs = reqs
                self.return_logprob = False
                self.spec_algorithm = _SpecAlgorithm()
                self.is_spec_v2 = False

            def batch_size(self):
                return len(self.reqs)

        class _Result:
            def __init__(self):
                self.copy_done = None
                self.logits_output = None
                self.next_token_ids = torch.tensor([7], dtype=torch.int64)
                self.can_run_cuda_graph = False

        class _Req:
            def __init__(self):
                self._finished = False
                self.output_ids = []
                self.origin_input_ids = [1, 2, 3]
                self.is_retracted = False
                self.to_finish = None
                self.time_stats = _TimeStats()
                self.return_logprob = False
                self.return_hidden_states = False
                self.grammar = None

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

        class _Processor(self.SchedulerOutputProcessorMixin):
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
