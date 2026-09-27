"""Logical DCP capacities must agree across validation, admission and telemetry."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
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
