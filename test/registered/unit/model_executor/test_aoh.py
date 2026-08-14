"""CPU unit tests for AoH sidecars, eviction, and KV-pool sizing."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.common import (
    free_aoh_out_of_window_slots,
    get_aoh_evictable_end,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType, SWAComponent
from sglang.srt.model_executor.aoh import (
    AoHConfig,
    build_aoh_page_plan,
    get_aoh_cacheable_prefix_len,
    get_aoh_kv_group,
    get_aoh_kv_groups,
    get_aoh_max_kv_pages,
    normalize_aoh_window,
)
from sglang.srt.model_executor.pool_configurator import AoHPoolConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free_swa(self, slots):
        self.freed.append(slots.clone())


class TestAoHConfig(CustomTestCase):
    def _sidecar(self, payload):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "aoh.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(tmp.cleanup)
        return path

    def test_load_and_route_kv_groups(self):
        config = AoHConfig.load(
            self._sidecar(
                {
                    "version": 1,
                    "layers": {"3": ["streaming", "retrieval"]},
                }
            )
        )
        self.assertEqual(config.mode_for(3, 0), "streaming")
        self.assertEqual(config.mode_for(3, 1), "retrieval")
        with self.assertRaisesRegex(ValueError, "KV-group modes"):
            config.mode_for(3, 2)

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "invalid modes"):
            AoHConfig.load(
                self._sidecar({"version": 1, "layers": {"3": ["streaming", "dense"]}})
            )

    def test_rejects_non_string_modes(self):
        with self.assertRaisesRegex(ValueError, "must all be strings"):
            AoHConfig.load(
                self._sidecar({"version": 1, "layers": {"3": ["streaming", 1]}})
            )

    def test_rejects_duplicate_normalized_layer_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate normalized layer id"):
            AoHConfig.load(
                self._sidecar(
                    {
                        "version": 1,
                        "layers": {
                            "3": ["streaming"],
                            "03": ["retrieval"],
                        },
                    }
                )
            )

    def test_replicated_gqa_routes_one_whole_kv_group_per_rank(self):
        groups = [
            get_aoh_kv_group(
                total_kv_heads=2,
                kv_tp_size=8,
                kv_tp_rank=rank,
                local_kv_heads=1,
            )
            for rank in range(8)
        ]
        self.assertEqual(groups, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_sharded_model_routes_all_local_kv_groups(self):
        self.assertEqual(
            get_aoh_kv_groups(
                total_kv_heads=8,
                kv_tp_size=2,
                kv_tp_rank=1,
                local_kv_heads=4,
            ),
            (4, 5, 6, 7),
        )

    def test_rejects_incomplete_kv_group_partition(self):
        with self.assertRaisesRegex(ValueError, "complete, non-overlapping"):
            get_aoh_kv_groups(
                total_kv_heads=10,
                kv_tp_size=3,
                kv_tp_rank=0,
                local_kv_heads=3,
            )

    def test_routes_replicated_and_sharded_kv_groups_across_tp_sizes(self):
        for tp_size in (2, 4, 8):
            groups = [
                get_aoh_kv_groups(
                    total_kv_heads=2,
                    kv_tp_size=tp_size,
                    kv_tp_rank=rank,
                    local_kv_heads=1,
                )[0]
                for rank in range(tp_size)
            ]
            self.assertEqual(groups.count(0), tp_size // 2)
            self.assertEqual(groups.count(1), tp_size // 2)

        for tp_size in (1, 2, 4, 8):
            local_kv_heads = 8 // tp_size
            groups = tuple(
                group
                for rank in range(tp_size)
                for group in get_aoh_kv_groups(
                    total_kv_heads=8,
                    kv_tp_size=tp_size,
                    kv_tp_rank=rank,
                    local_kv_heads=local_kv_heads,
                )
            )
            self.assertEqual(groups, tuple(range(8)))

    def test_window_sizes_are_normalized_without_changing_context_semantics(self):
        self.assertEqual(normalize_aoh_window(128, 256, 4096), (128, 256))
        self.assertEqual(normalize_aoh_window(3000, 3000, 4096), (3000, 1096))
        self.assertEqual(normalize_aoh_window(8192, 256, 4096), (4096, 1))

    def test_radix_cache_only_shares_the_permanent_anchor(self):
        self.assertEqual(get_aoh_cacheable_prefix_len(64, 128), 64)
        self.assertEqual(get_aoh_cacheable_prefix_len(128, 128), 128)
        self.assertEqual(get_aoh_cacheable_prefix_len(4096, 128), 128)
        self.assertEqual(get_aoh_cacheable_prefix_len(4096, 130, 128), 256)

    def test_radix_cache_anchor_ignores_the_request_swa_frontier(self):
        component = SWAComponent.__new__(SWAComponent)
        component.cache = SimpleNamespace(aoh_radix_anchor_only=True)
        insert_params = SimpleNamespace()
        req = SimpleNamespace(kv=SimpleNamespace(swa_evicted_seqlen=1024))

        component.prepare_for_caching_req(
            req, insert_params, token_ids_len=128, is_finished=True
        )
        self.assertEqual(insert_params.swa_evicted_seqlen, 0)

        validator = component.create_match_validator()
        cached_anchor = SimpleNamespace(
            component_data={ComponentType.SWA: SimpleNamespace(value=torch.tensor([1]))}
        )
        self.assertTrue(validator(cached_anchor))


class TestAoHEviction(CustomTestCase):
    def test_frontier_is_based_on_the_tail_not_the_combined_window(self):
        self.assertEqual(
            get_aoh_evictable_end(1024, sink_size=128, recent_size=256, page_size=128),
            768,
        )

    def test_keeps_anchors_and_moves_tail_frontier(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).view(1, -1)
        )
        allocator = _RecordingAllocator()

        free_aoh_out_of_window_slots(
            req,
            20,
            sink_size=4,
            recent_size=8,
            page_size=4,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )
        free_aoh_out_of_window_slots(
            req,
            28,
            sink_size=4,
            recent_size=8,
            page_size=4,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(
            [x.tolist() for x in allocator.freed],
            [list(range(4, 12)), list(range(12, 20))],
        )
        self.assertEqual(req.kv.swa_evicted_seqlen, 20)

    def test_only_frees_complete_middle_pages(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).view(1, -1)
        )
        allocator = _RecordingAllocator()

        free_aoh_out_of_window_slots(
            req,
            31,
            sink_size=5,
            recent_size=9,
            page_size=4,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(allocator.freed[0].tolist(), list(range(8, 20)))
        self.assertEqual(req.kv.swa_evicted_seqlen, 20)

    def test_does_not_free_a_cache_protected_page(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=12,
            kv=SimpleNamespace(swa_evicted_seqlen=0),
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int64).view(1, -1)
        )
        allocator = _RecordingAllocator()

        free_aoh_out_of_window_slots(
            req,
            31,
            sink_size=5,
            recent_size=9,
            page_size=4,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(allocator.freed[0].tolist(), list(range(12, 20)))


class TestAoHPagePlan(CustomTestCase):
    def test_decode_buffer_width_uses_runtime_sizes(self):
        self.assertEqual(get_aoh_max_kv_pages(128, 256, 128), 4)
        self.assertEqual(get_aoh_max_kv_pages(5, 9, 4), 5)

    def test_prefill_compacts_anchor_tail_and_current_chunk(self):
        plan = build_aoh_page_plan(
            total_kv_len=48,
            query_len=16,
            sink_size=8,
            recent_size=12,
            page_size=4,
        )
        self.assertEqual(plan.page_starts, (0, 4, 20, 24, 28, 32, 36, 40, 44))
        self.assertEqual(plan.actual_kv_len, 36)
        self.assertEqual(plan.tail_start, 21)
        self.assertFalse(plan.can_use_causal_template)

    def test_every_prefill_query_has_only_anchor_and_a_bounded_recent_tail(self):
        plan = build_aoh_page_plan(
            total_kv_len=48,
            query_len=16,
            sink_size=8,
            recent_size=12,
            page_size=4,
        )
        selected_positions = {
            position
            for page_start in plan.page_starts
            for position in range(page_start, min(page_start + 4, plan.total_kv_len))
        }
        for query_position in range(plan.query_start, plan.total_kv_len):
            tail_start = max(8, query_position - 12 + 1)
            visible = set(range(min(8, query_position + 1))) | set(
                range(tail_start, query_position + 1)
            )
            self.assertTrue(visible.issubset(selected_positions))
            self.assertLessEqual(len(visible - set(range(8))), 12)

    def test_non_aligned_boundaries_require_custom_mask(self):
        plan = build_aoh_page_plan(
            total_kv_len=31,
            query_len=1,
            sink_size=5,
            recent_size=9,
            page_size=4,
        )
        self.assertEqual(plan.page_starts, (0, 4, 20, 24, 28))
        self.assertEqual(plan.actual_kv_len, 19)
        self.assertEqual(plan.anchor_end, 5)
        self.assertEqual(plan.tail_start, 22)
        self.assertFalse(plan.can_use_causal_template)

    def test_short_sequence_uses_contiguous_pages(self):
        plan = build_aoh_page_plan(
            total_kv_len=7,
            query_len=7,
            sink_size=9,
            recent_size=3,
            page_size=4,
        )
        self.assertEqual(plan.page_starts, (0, 4))
        self.assertEqual(plan.actual_kv_len, 7)
        self.assertTrue(plan.can_use_causal_template)

    def test_no_query_counts_only_selected_anchor_pages(self):
        plan = build_aoh_page_plan(
            total_kv_len=32,
            query_len=0,
            sink_size=5,
            recent_size=9,
            page_size=4,
        )
        self.assertEqual(plan.page_starts, (0, 4))
        self.assertEqual(plan.actual_kv_len, 8)

    def test_decode_plan_is_bounded_for_runtime_parameter_values(self):
        for page_size in (1, 3, 16, 128):
            for sink_size in (1, page_size + 1, 3 * page_size):
                for recent_size in (1, page_size + 2, 5 * page_size):
                    max_pages = get_aoh_max_kv_pages(sink_size, recent_size, page_size)
                    for total_kv_len in (
                        0,
                        1,
                        sink_size,
                        sink_size + recent_size,
                        sink_size + recent_size + 2 * page_size + 1,
                    ):
                        plan = build_aoh_page_plan(
                            total_kv_len=total_kv_len,
                            query_len=1 if total_kv_len else 0,
                            sink_size=sink_size,
                            recent_size=recent_size,
                            page_size=page_size,
                        )
                        self.assertLessEqual(len(plan.page_starts), max_pages)
                        self.assertTrue(
                            all(start % page_size == 0 for start in plan.page_starts)
                        )
                        self.assertLessEqual(plan.actual_kv_len, max_pages * page_size)


class TestAoHPoolConfigurator(CustomTestCase):
    def _make_configurator(
        self,
        *,
        chunked_prefill_size=256,
        context_len=4096,
        max_running_requests=8,
        attn_dp_size=1,
        retrieval_layers=5,
        streaming_layers=5,
        max_prefill_tokens=4096,
    ):
        model_config = SimpleNamespace(
            context_len=context_len,
            head_dim=256,
            v_head_dim=256,
            get_num_kv_heads=lambda tp_size: 1,
        )
        server_args = SimpleNamespace(
            max_running_requests=max_running_requests,
            chunked_prefill_size=chunked_prefill_size,
            max_prefill_tokens=max_prefill_tokens,
            disable_overlap_schedule=True,
            aoh_sink_size=128,
            aoh_recent_size=256,
        )
        kvc = SimpleNamespace(
            is_aoh=True,
            server_args=server_args,
            page_size=128,
            aoh_retrieval_layer_ids=list(range(retrieval_layers)),
            aoh_streaming_layer_ids=list(range(streaming_layers)),
            kv_cache_dtype=torch.bfloat16,
            model_config=model_config,
            ps=SimpleNamespace(attn_dp_size=attn_dp_size),
        )
        with patch(
            "sglang.srt.model_executor.pool_configurator.envs."
            "SGLANG_SWA_EVICTION_INTERVAL.get",
            return_value=128,
        ):
            return AoHPoolConfigurator(kvc)

    def test_stream_pool_is_fixed_and_remaining_memory_grows_retrieval_pool(self):
        with get_parallel().override(attn_tp_size=2):
            configurator = self._make_configurator()
            # Per request includes the 128-token decode eviction interval.
            self.assertEqual(configurator._stream_capacity, 5504)
            config = configurator.calculate_pool_sizes(
                available_bytes=(5504 * 5 + 10000 * 5) * 1024,
                page_size=128,
            )

        self.assertEqual(config.swa_max_total_num_tokens, 5504)
        self.assertEqual(config.full_max_total_num_tokens, 9984)
        self.assertEqual(config.max_total_num_tokens, 9984)

    def test_disabled_chunked_prefill_reserves_the_full_context(self):
        with get_parallel().override(attn_tp_size=2):
            configurator = self._make_configurator(
                chunked_prefill_size=-1, context_len=2048
            )
            # The unchunked batch can consume max_prefill_tokens across requests.
            # stream capacity: 640 * 8 + 4096 + 128 = 9344.
            self.assertEqual(configurator._stream_capacity, 9344)

    def test_request_reserve_rounds_up_across_attention_dp_workers(self):
        with get_parallel().override(attn_tp_size=2):
            configurator = self._make_configurator(
                max_running_requests=9, attn_dp_size=2
            )
            # Five requests per worker: 640 * 5 + 256 + 128.
            self.assertEqual(configurator._stream_capacity, 3584)

    def test_rank_with_only_streaming_groups_uses_a_logical_full_address_space(self):
        with get_parallel().override(attn_tp_size=2):
            configurator = self._make_configurator(retrieval_layers=0)
            config = configurator.calculate_pool_sizes(
                available_bytes=5504 * 5 * 1024,
                page_size=128,
            )

        self.assertEqual(config.full_max_total_num_tokens, 5504)
        self.assertEqual(config.swa_max_total_num_tokens, 5504)
        self.assertEqual(config.max_total_num_tokens, 5504)


if __name__ == "__main__":
    unittest.main()
