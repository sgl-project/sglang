"""Logical DCP capacities must agree across validation, admission and telemetry."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
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
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.srt.runtime_context import get_parallel
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
    allocator = PagedTokenToKVPoolAllocator(
        PHYSICAL * dcp_size, 64 * dcp_size, torch.uint8, "cpu", kv, False
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = NS(dcp_size=dcp_size)
    runner.kv_cache_configurator = make_configurator()
    runner.is_hybrid_swa = False
    runner.max_total_num_tokens = PHYSICAL
    runner.max_running_requests = 64
    runner.token_to_kv_pool_allocator = allocator
    runner.token_to_kv_pool = kv
    runner.req_to_token_pool = ReqToTokenPool.__new__(ReqToTokenPool)
    runner.req_to_token_pool.size = 96
    runner.req_to_token_pool.max_context_len = CONTEXT
    runner.req_to_token_pool._aux_cache = None
    runner.forward_stream = None
    runner.weight_load_mem_usage = 0
    return NS(
        model_runner=runner,
        model_config=NS(context_len=CONTEXT),
        server_args=NS(max_prefill_tokens=16384, max_queued_requests=None),
        random_seed=0,
        device="cpu",
        graph_memory_usage={},
    )


def make_scheduler(worker):
    info = TpModelWorker.get_worker_info(worker)
    runner = worker.model_runner
    scheduler = NS(
        tp_worker=worker,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        req_to_token_pool=runner.req_to_token_pool,
        tree_cache=NS(evictable_size=lambda: 0),
        session_controller=None,
        hisparse_coordinator=None,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        enable_hisparse=False,
        sliding_window_size=None,
        chunked_prefill_size=16384,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=info[0],
        max_running_requests=info[2],
        max_req_len=info[4],
        max_req_input_len=info[5],
        max_new_tokens_limit=None,
        page_size=64,
        server_args=NS(dcp_size=runner.server_args.dcp_size, enable_lora=False),
        running_batch=NS(reqs=[]),
        last_batch=None,
        waiting_queue=[],
        chunked_req=None,
        disaggregation_mode=DisaggregationMode.DECODE,
        disagg_decode_prealloc_queue=NS(queue=[], retracted_queue=[]),
        disagg_decode_transfer_queue=NS(queue=[]),
        spec_algorithm=NS(is_none=lambda: True),
        metrics_reporter=NS(stats=SchedulerStats()),
        metrics_collector=Mock(),
        draft_worker=None,
        startup_available_gpu_memory_gb=None,
        model_config=worker.model_config,
    )
    Scheduler.init_pool_stats_observer(scheduler)
    Scheduler.init_load_inquirer(scheduler)
    return scheduler


class TestDcpLogicalCapacity(CustomTestCase):
    def make_worker(self, dcp_size, **server_args_fields):
        enter_scope(
            self,
            published_topology(
                tp_size=dcp_size, dcp_size=dcp_size, **server_args_fields
            ),
        )
        return make_worker(dcp_size)

    def setUp(self):
        for config in (
            patch(
                "sglang.srt.managers.tp_worker.get_schedule",
                return_value=NS(max_prefill_tokens=16384, max_queued_requests=None),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.load_inquirer.get_lora",
                return_value=NS(enable_lora=False),
            ),
        ):
            config.start()
            self.addCleanup(config.stop)

    def test_worker_capacity_uses_effective_dcp(self):
        for dcp_size in (1, 2, 8):
            with self.subTest(dcp_size=dcp_size):
                worker = self.make_worker(dcp_size)
                # The runner's configured DCP is not the published attention group.
                worker.model_runner.server_args.dcp_size = 8
                info = TpModelWorker.get_worker_info(worker)
                capacity = PHYSICAL * dcp_size
                self.assertEqual(info[0], capacity)
                self.assertEqual(info[4], min(CONTEXT, capacity) - 1)
                self.assertEqual(info[5], info[4] - 5)

    def test_capacity_is_rows_times_dcp_not_allocator_size(self):
        """The unified Mamba allocator's size also counts Mamba state bytes as
        tokens, so capacity must come from the configured rows, not the allocator."""
        runner = self.make_worker(8).model_runner
        runner.max_total_num_tokens = 64
        runner.token_to_kv_pool_allocator = make_unified_mamba_allocator(
            n_full_tokens=64, n_mamba_slots=8
        )
        self.assertGreater(runner.token_to_kv_pool_allocator.size, 64 * 8)
        self.assertEqual(runner.logical_max_total_num_tokens, 64 * 8)
        # Draft sizes are already widened by loc_space_scale; SWA never widens.
        for configurator in (
            make_configurator(is_draft_worker=True),
            make_configurator(is_hybrid_swa=True),
        ):
            runner.kv_cache_configurator = configurator
            self.assertEqual(runner.logical_max_total_num_tokens, 64)

    def test_hybrid_swa_bounds_do_not_widen_under_dcp(self):
        runner = self.make_worker(1).model_runner
        runner.is_hybrid_swa = True
        runner.kv_cache_configurator = make_configurator(is_hybrid_swa=True)
        runner.swa_max_total_num_tokens = PHYSICAL // 4
        for dcp_size, full_capacity, expected in (
            (1, PHYSICAL // 2, PHYSICAL // 2),
            (1, 0, PHYSICAL // 4),
            (8, PHYSICAL // 2, PHYSICAL // 2),
        ):
            with self.subTest(dcp_size=dcp_size, full_capacity=full_capacity):
                enter_scope(self, get_parallel().override(attn_dcp_size=dcp_size))
                runner.full_max_total_num_tokens = full_capacity
                self.assertEqual(
                    runner.effective_logical_max_total_num_tokens, expected
                )
                self.assertEqual(runner.logical_max_total_num_tokens, PHYSICAL)

    def test_output_budget_does_not_multiply_logical_capacity_again(self):
        worker = self.make_worker(8)
        worker.model_config.context_len = PHYSICAL * 16
        scheduler = make_scheduler(worker)
        req = NS(
            rid="near-capacity",
            origin_input_ids=range(scheduler.max_total_num_tokens - 1024),
            sampling_params=NS(max_new_tokens=600, min_new_tokens=0),
        )
        Scheduler.init_req_max_new_tokens(scheduler, req)
        self.assertEqual(req.sampling_params.max_new_tokens, 511)

    def test_one_million_token_context_is_not_clipped_to_per_rank_rows(self):
        scheduler = make_scheduler(self.make_worker(8))
        self.assertEqual(scheduler.max_req_input_len, CONTEXT - 6)
        req = NS(
            rid="dcp-long",
            origin_input_ids=range(1_000_000),
            output_ids=[],
            sampling_params=NS(max_new_tokens=64, min_new_tokens=0),
        )
        Scheduler.init_req_max_new_tokens(scheduler, req)
        self.assertEqual(req.sampling_params.max_new_tokens, 64)
        self.assertFalse(
            PrefillBootstrapQueue._check_if_req_exceed_kv_capacity(
                NS(
                    max_total_num_tokens=scheduler.tp_worker.model_runner.effective_logical_max_total_num_tokens
                ),
                req,
            )
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.max_total_num_tokens = scheduler.max_total_num_tokens
        queue.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        queue.token_to_kv_pool = scheduler.tp_worker.model_runner.token_to_kv_pool
        queue.num_reserved_decode_tokens = 0
        queue.scheduler = scheduler
        scheduler.output_streamer = Mock()
        self.assertFalse(
            DecodePreallocQueue._check_if_req_exceed_kv_capacity(queue, req)
        )
        # Context safety still applies independently of the larger pool.
        req.origin_input_ids = range(CONTEXT - 32)
        Scheduler.init_req_max_new_tokens(scheduler, req)
        self.assertEqual(req.sampling_params.max_new_tokens, 30)

        req.origin_input_ids = range(scheduler.max_total_num_tokens + 1)
        req.return_logprob = False
        with patch("sglang.srt.disaggregation.decode.prepare_abort") as abort:
            self.assertTrue(
                DecodePreallocQueue._check_if_req_exceed_kv_capacity(queue, req)
            )
        abort.assert_called_once()
        queue.scheduler.output_streamer.stream_output.assert_called_once()

    def test_load_usage_uses_logical_capacity_exactly_once(self):
        for dcp_size in (1, 8):
            with self.subTest(dcp_size=dcp_size):
                scheduler = make_scheduler(
                    self.make_worker(dcp_size, enable_metrics=True)
                )
                allocator = scheduler.token_to_kv_pool_allocator
                # Keep half the real allocator's pages available; no CUDA calls.
                allocator.free_pages = allocator.free_pages[
                    : len(allocator.free_pages) // 2
                ]
                loads = scheduler.load_inquirer.get_loads()
                self.assertEqual(loads.max_total_num_tokens, allocator.size)
                self.assertEqual(loads.memory.token_capacity, allocator.size)
                used = allocator.size - (allocator.num_pages // 2) * allocator.page_size
                self.assertEqual(loads.num_used_tokens, used)
                self.assertEqual(loads.token_usage, 0.5)
                self.assertEqual(
                    scheduler.pool_stats_observer.max_total_num_tokens, allocator.size
                )
                Scheduler.emit_metrics_constants(scheduler)
                constants = scheduler.metrics_collector.emit_constants.call_args
                self.assertEqual(constants.kwargs["num_pages"], PHYSICAL // 64)


if __name__ == "__main__":
    unittest.main()
