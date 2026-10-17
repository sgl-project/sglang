"""Logical DCP capacities must agree across validation, admission and telemetry."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PHYSICAL = 690240
CONTEXT = 1048576


def make_configurator(*, is_hybrid_swa=False, is_draft_worker=False):
    configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
    configurator.is_hybrid_swa = is_hybrid_swa
    configurator.is_draft_worker = is_draft_worker
    return configurator


def make_unified_mamba_allocator(*, n_full_tokens, n_mamba_slots):
    full = MLASubPoolSpec(
        name="full",
        layer_num=3,
        kv_lora_rank=6,
        qk_rope_head_dim=2,
        store_dtype=torch.bfloat16,
        grow_direction="down",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=2,
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
        grow_direction="up",
    )
    pool = UnifiedKVPool(
        total_bytes=full.entry_bytes() * n_full_tokens
        + mamba.entry_bytes() * n_mamba_slots,
        sub_pool_specs=[full, mamba],
        device="cpu",
        enable_memory_saver=False,
        page_size=1,
    )
    kvcache = NS(
        full_kv_pool=NS(buf=torch.empty(pool.max_slots("full"))),
        mamba_pool=NS(buf=torch.empty(pool.max_slots("mamba"))),
    )
    return UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=pool, kvcache=kvcache, device="cpu", page_size=1
    )


def make_worker(dcp_size):
    kv = NS(size=PHYSICAL, mem_usage=8.89)
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = NS(dcp_size=dcp_size)
    runner.kv_cache_configurator = make_configurator()
    runner.is_hybrid_swa = False
    runner.max_total_num_tokens = PHYSICAL
    runner.max_running_requests = 64
    runner.token_to_kv_pool = kv
    runner.token_to_kv_pool_allocator = None
    runner.req_to_token_pool = ReqToTokenPool.__new__(ReqToTokenPool)
    runner.req_to_token_pool.size = 96
    runner.req_to_token_pool.max_context_len = CONTEXT
    runner.req_to_token_pool._aux_cache = None
    runner.forward_stream = None
    return NS(
        model_runner=runner,
        model_config=NS(context_len=CONTEXT),
        random_seed=0,
        device="cpu",
        dllm_algorithm=None,
    )


class TestDcpLogicalCapacity(CustomTestCase):
    def make_worker(self, dcp_size):
        enter_scope(self, published_topology(tp_size=dcp_size, dcp_size=dcp_size))
        return make_worker(dcp_size)

    def setUp(self):
        for config in (
            patch(
                "sglang.srt.managers.tp_worker.get_schedule",
                return_value=NS(max_prefill_tokens=16384, max_queued_requests=None),
            ),
        ):
            config.start()
            self.addCleanup(config.stop)

    def test_sharded_worker_request_limits(self):
        worker = self.make_worker(1)
        runner = worker.model_runner
        runner.max_total_num_tokens = 64
        allocator = PageInterleavePoolAllocator.__new__(PageInterleavePoolAllocator)
        allocator.shard_size = 3
        allocator.size = 64 * 3
        runner.token_to_kv_pool_allocator = allocator

        info = TpModelWorker.get_worker_info(worker)

        self.assertEqual(info[0], 64 * 3)
        self.assertEqual(info[4], 64 * 3 - 1)
        self.assertEqual(info[5], 64 * 3 - 6)

    def test_configurator_to_worker_converts_capacity_once(self):
        for dcp_size, shard_size in ((1, 1), (4, 1), (1, 4)):
            with self.subTest(dcp_size=dcp_size, shard_size=shard_size):
                worker = self.make_worker(dcp_size)
                runner = worker.model_runner
                cfg = runner.kv_cache_configurator
                allocator = None
                if shard_size > 1:
                    allocator = PageInterleavePoolAllocator(
                        size=64,
                        physical_page_size=4,
                        shard_size=shard_size,
                        dtype=torch.int64,
                        device="cpu",
                        kvcache=None,
                        need_sort=False,
                    )
                sizes = NS(
                    max_total_num_tokens=64,
                    max_running_requests=8,
                    full_max_total_num_tokens=None,
                    swa_max_total_num_tokens=None,
                )
                pools = NS(
                    token_to_kv_pool_allocator=allocator,
                    token_to_kv_pool=runner.token_to_kv_pool,
                    req_to_token_pool=runner.req_to_token_pool,
                    unified_memory_pool=None,
                )
                cfg.kv_cache_dtype = torch.float32
                cfg.device, cfg.gpu_id = "cpu", 0
                cfg.spec_algorithm = NS(is_none=lambda: True)
                cfg.req_to_token_pool = runner.req_to_token_pool
                cfg.token_to_kv_pool_allocator = allocator
                with (
                    patch.object(
                        KVCacheConfigurator,
                        "_resolve_memory_pool_config",
                        return_value=sizes,
                    ),
                    patch.object(
                        KVCacheConfigurator, "_derive_pool_sizes", return_value=sizes
                    ),
                    patch.object(
                        KVCacheConfigurator, "_init_pools", return_value=pools
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
                        return_value=0,
                    ),
                ):
                    result = cfg.configure(pre_model_load_memory=0)
                self.assertEqual(result.max_total_num_tokens, 64)
                runner.max_total_num_tokens = result.max_total_num_tokens
                runner.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
                logical = 64 * dcp_size * shard_size
                self.assertEqual(runner.logical_max_total_num_tokens, logical)
                info = TpModelWorker.get_worker_info(worker)
                self.assertEqual(info[0], logical)
                self.assertEqual(info[4], logical - 1)

    def test_scheduler_consumers_receive_logical_capacity(self):
        class SchedulerState(NS):
            def __getattr__(self, name):
                return None

        state = SchedulerState(max_total_num_tokens=256, kv_shard_widening=4)
        for method, consumer in (
            (Scheduler.init_pool_stats_observer, "SchedulerPoolStatsObserver"),
            (Scheduler.init_invariant_checker, "SchedulerInvariantChecker"),
            (Scheduler.init_load_inquirer, "SchedulerLoadInquirer"),
        ):
            with (
                self.subTest(consumer=consumer),
                patch("sglang.srt.managers.scheduler." + consumer) as constructor,
            ):
                method(state)
                self.assertEqual(
                    constructor.call_args.kwargs["max_total_num_tokens"], 256
                )

    def test_logical_capacity(self):
        for dcp_size in (1, 2, 8):
            with self.subTest(dcp_size=dcp_size):
                info = TpModelWorker.get_worker_info(self.make_worker(dcp_size))
                self.assertEqual(info[0], PHYSICAL * dcp_size)
                self.assertEqual(info[4], min(CONTEXT, PHYSICAL * dcp_size) - 1)

        # Unified Mamba's allocator.size also counts Mamba state bytes, so
        # capacity must come from the configured rows.
        runner = self.make_worker(8).model_runner
        runner.max_total_num_tokens = 64
        runner.token_to_kv_pool_allocator = make_unified_mamba_allocator(
            n_full_tokens=64, n_mamba_slots=8
        )
        self.assertGreater(runner.token_to_kv_pool_allocator.size, 64 * 8)
        self.assertEqual(runner.logical_max_total_num_tokens, 64 * 8)

        # Draft sizes already carry loc_space_scale; SWA never widens.
        runner.kv_cache_configurator = make_configurator(is_draft_worker=True)
        self.assertEqual(runner.logical_max_total_num_tokens, 64)
        runner.is_hybrid_swa = True
        runner.kv_cache_configurator = make_configurator(is_hybrid_swa=True)
        runner.full_max_total_num_tokens = 32
        runner.swa_max_total_num_tokens = 16
        self.assertEqual(runner.logical_max_total_num_tokens, 64)
        self.assertEqual(runner.effective_logical_max_total_num_tokens, 32)


if __name__ == "__main__":
    unittest.main()
