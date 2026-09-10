import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPagedDSparkWithEncoderReplay(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=6,
            speculative_dspark_block_size=5,
            page_size=256,
            max_running_requests=2,
            chunked_prefill_size=256,
        )
        override.install()
        self.addCleanup(override.restore)

    def make_pool(self, *, draft):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=2,
            num_req_slots=3,
            swa_size=1024,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.bfloat16,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 0, 0],
            online_mtp_max_draft_tokens=6,
            full_size=2048,
            is_draft_worker=draft,
        )

    def test_target_window_and_draft_paged_storage_share_allocator_mapping(self):
        target = self.make_pool(draft=False)
        draft = self.make_pool(draft=True)
        self.assertIsNotNone(target.request_window)
        self.assertIsNone(target.swa_kv_pool)
        self.assertIsNone(draft.request_window)
        self.assertEqual(len(draft.swa_kv_pool.kv_buffer), 3)
        self.assertTrue(target.needs_paged_swa_allocator)
        self.assertTrue(draft.needs_paged_swa_allocator)
        allocator = SWATokenToKVPoolAllocator(
            2048, 1024, 256, torch.float8_e4m3fn, "cpu", target, False
        )
        draft.register_mapping(allocator.full_to_swa_index_mapping)
        mapping = allocator.full_to_swa_index_mapping
        mapping[256:512] = torch.arange(768, 1024)
        full = torch.tensor([256, 300, 511])
        self.assertEqual(
            draft.translate_loc_from_full_to_swa(full).tolist(), [768, 812, 1023]
        )
        self.assertEqual(
            draft.translate_loc_from_full_to_swa(torch.tensor([-1])).item(), -1
        )
        # Page reuse updates the single shared mapping, without rebuilding draft state.
        mapping[256:512] = torch.arange(256, 512)
        self.assertEqual(
            draft.translate_loc_from_full_to_swa(full).tolist(), [256, 300, 511]
        )

    def test_budget_reserves_target_window_and_real_draft_layer_count(self):
        cfg = SimpleNamespace(
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
            context_len=131072,
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
            window_size=128,
            hf_config=SimpleNamespace(kv_source_layer_ids=[2, 8, 14, 20]),
        )
        spec = SimpleNamespace(is_dspark=lambda: True, is_none=lambda: False)
        kvc = SimpleNamespace(
            kv_cache_dtype_str="fp8_e4m3",
            model_config=cfg,
            layer_info=SimpleNamespace(start_layer=0, end_layer=40),
            ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
            sliding_window_size=128,
            page_size=256,
            spec_algorithm=spec,
            spec_aux_config=SimpleNamespace(dflash_draft_num_layers=3),
        )
        planner = DSV4PoolConfigurator(kvc)
        self.assertEqual(planner.bytes_per_swa_token, 3 * 584)
        self.assertGreater(planner.swa_cap_tokens, 0)
        self.assertEqual(
            planner._get_swa_fixed_bytes(),
            planner.request_window_bytes + planner.swa_cap_tokens * 3 * 584,
        )
        budget = 256 * 1024 * 1024
        sizes = planner.calculate_pool_sizes(budget, 256)
        self.assertEqual(sizes.swa_max_total_num_tokens, planner.swa_cap_tokens)
        self.assertLessEqual(
            sizes.full_max_total_num_tokens * planner.bytes_per_full_token
            + planner._get_swa_fixed_bytes(),
            budget,
        )


if __name__ == "__main__":
    unittest.main()
