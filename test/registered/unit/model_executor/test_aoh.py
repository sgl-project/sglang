"""CPU unit tests for AoH sidecars, eviction, and KV-pool sizing."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
    _validate_per_record_graph_update,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
    _build_aoh_graph_update_inputs,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.mem_cache.common import (
    free_aoh_out_of_window_slots,
    get_aoh_evictable_end,
)
from sglang.srt.mem_cache.kv_cache_configurator import _get_hybrid_swa_pool_class
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    MambaComponent,
    SWAComponent,
)
from sglang.srt.model_executor.aoh import (
    AoHConfig,
    build_aoh_decode_mask,
    build_aoh_page_plan,
    get_aoh_cacheable_prefix_len,
    get_aoh_decode_mask_periodic_start,
    get_aoh_kv_group,
    get_aoh_kv_groups,
    get_aoh_max_kv_pages,
    get_aoh_prefill_chunk_size,
    get_aoh_speculative_query_len,
    normalize_aoh_window,
)
from sglang.srt.model_executor.model_runner import ModelRunner
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

    def test_runner_keeps_normalized_window_outside_read_only_server_args(self):
        class ReadOnlyArgs:
            aoh_sink_size = 8192
            aoh_recent_size = 256

            def __setattr__(self, name, value):
                raise AttributeError(f"read-only: {name}")

        runner = ModelRunner.__new__(ModelRunner)
        runner.server_args = ReadOnlyArgs()
        runner.aoh_sink_size = runner.server_args.aoh_sink_size
        runner.aoh_recent_size = runner.server_args.aoh_recent_size
        runner.model_config = SimpleNamespace(context_len=4096)

        runner._configure_aoh_window()

        self.assertEqual((runner.aoh_sink_size, runner.aoh_recent_size), (4096, 1))
        self.assertEqual(
            (runner.server_args.aoh_sink_size, runner.server_args.aoh_recent_size),
            (8192, 256),
        )

    def test_runner_accepts_model_configs_without_hybrid_layer_metadata(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model_config = SimpleNamespace()

        self.assertIsNone(runner._get_aoh_full_attention_layer_ids())

    def test_runner_rejects_mla_attention(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.server_args = SimpleNamespace(aoh_config="aoh.json")
        runner.use_mla_backend = True
        runner.is_draft_worker = False

        with self.assertRaisesRegex(ValueError, "MHA/GQA"):
            runner.configure_aoh()

    def test_draft_runner_ignores_target_aoh_sidecar(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.server_args = SimpleNamespace(aoh_config="aoh.json")
        runner.is_draft_worker = True

        runner.configure_aoh()

    def test_runner_accepts_linear_eagle_speculation(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_eagle=lambda: True,
        )
        runner.server_args = SimpleNamespace(speculative_eagle_topk=1)

        runner._validate_aoh_speculative_compatibility()

    def test_runner_rejects_branched_eagle_speculation(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_eagle=lambda: True,
        )
        runner.server_args = SimpleNamespace(speculative_eagle_topk=2)

        with self.assertRaisesRegex(ValueError, "topk 1"):
            runner._validate_aoh_speculative_compatibility()

    def test_runner_rejects_non_eagle_speculation(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_eagle=lambda: False,
        )
        runner.server_args = SimpleNamespace(speculative_eagle_topk=1)

        with self.assertRaisesRegex(ValueError, "EAGLE/NEXTN"):
            runner._validate_aoh_speculative_compatibility()

    def test_runner_rejects_paged_mamba_without_extra_buffer(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.page_size = 128
        runner.server_args = SimpleNamespace(
            uses_mamba_radix_cache=True,
            enable_mamba_extra_buffer=lambda: False,
        )

        with self.assertRaisesRegex(ValueError, "extra_buffer"):
            runner._validate_aoh_mamba_cache_compatibility()

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

    def test_eagle_anchor_accounts_for_bigram_boundary_token(self):
        self.assertEqual(
            get_aoh_cacheable_prefix_len(4096, 128, 128, is_eagle=True), 129
        )
        self.assertEqual(
            get_aoh_cacheable_prefix_len(128, 128, 128, is_eagle=True), 128
        )

    def test_npu_uses_its_paged_mha_pool_in_generic_hybrid_paths(self):
        with patch("sglang.srt.mem_cache.kv_cache_configurator._is_npu", True):
            self.assertIs(
                _get_hybrid_swa_pool_class(MHATokenToKVPool),
                NPUMHATokenToKVPool,
            )
            self.assertIs(
                _get_hybrid_swa_pool_class(PageMajorMHATokenToKVPool),
                NPUMHATokenToKVPool,
            )

        with patch("sglang.srt.mem_cache.kv_cache_configurator._is_npu", False):
            self.assertIs(
                _get_hybrid_swa_pool_class(MHATokenToKVPool), MHATokenToKVPool
            )
            self.assertIs(
                _get_hybrid_swa_pool_class(PageMajorMHATokenToKVPool),
                PageMajorMHATokenToKVPool,
            )

    def test_radix_rejects_mamba_state_past_the_aoh_anchor(self):
        component = MambaComponent.__new__(MambaComponent)
        component.cache = SimpleNamespace(
            aoh_radix_anchor_only=True,
            enable_mamba_extra_buffer=True,
        )
        req = SimpleNamespace(mamba_last_track_seqlen=256)
        insert_params = SimpleNamespace(mamba_value=None)

        self.assertEqual(
            component.prepare_for_caching_req(
                req, insert_params, token_ids_len=129, is_finished=True
            ),
            0,
        )
        self.assertIsNone(insert_params.mamba_value)

        component.cache.enable_mamba_extra_buffer = False
        req = SimpleNamespace(get_fill_ids=lambda: list(range(256)))
        self.assertEqual(
            component.prepare_for_caching_req(
                req, insert_params, token_ids_len=129, is_finished=False
            ),
            0,
        )

    def test_npu_paged_pool_moves_token_rows_without_triton_copy(self):
        pool = NPUMHATokenToKVPool.__new__(NPUMHATokenToKVPool)
        pool.layer_num = 2
        pool.size = 8
        pool.page_size = 4
        pool.use_fia = False
        pool.use_native_move_kv_cache = True
        pool.k_buffer = torch.arange(24).view(2, 3, 4, 1, 1).clone()
        pool.v_buffer = (torch.arange(24) + 100).view(2, 3, 4, 1, 1).clone()

        expected_k = pool.k_buffer[:, 0, 1].clone()
        expected_v = pool.v_buffer[:, 0, 1].clone()
        pool.move_kv_cache(torch.tensor([10]), torch.tensor([1]))

        torch.testing.assert_close(pool.k_buffer[:, 2, 2], expected_k)
        torch.testing.assert_close(pool.v_buffer[:, 2, 2], expected_v)

        pool.use_fia = True
        pool.k_buffer = [torch.arange(12).view(12, 1, 1, 1) for _ in range(2)]
        pool.v_buffer = [(torch.arange(12) + 100).view(12, 1, 1, 1) for _ in range(2)]
        pool.move_kv_cache(torch.tensor([10]), torch.tensor([1]))

        for k_buffer, v_buffer in zip(pool.k_buffer, pool.v_buffer):
            torch.testing.assert_close(k_buffer[10], torch.tensor([[[1]]]))
            torch.testing.assert_close(v_buffer[10], torch.tensor([[[101]]]))


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
    def test_npu_mtp_query_uses_uniform_request_lengths(self):
        self.assertEqual(
            get_aoh_speculative_query_len(num_query_tokens=8, batch_size=2), 4
        )

    def test_npu_mtp_query_rejects_nonuniform_request_lengths(self):
        with self.assertRaisesRegex(RuntimeError, "equal number of query tokens"):
            get_aoh_speculative_query_len(num_query_tokens=7, batch_size=2)

    def test_decode_buffer_width_uses_runtime_sizes(self):
        self.assertEqual(get_aoh_max_kv_pages(128, 256, 128), 4)
        self.assertEqual(get_aoh_max_kv_pages(5, 9, 4), 5)

    def test_verify_buffer_width_accounts_for_all_query_tokens(self):
        page_size = 4
        query_len = 13
        max_pages = get_aoh_max_kv_pages(5, 9, page_size, query_len)

        for total_kv_len in range(query_len, 128):
            plan = build_aoh_page_plan(
                total_kv_len=total_kv_len,
                query_len=query_len,
                sink_size=5,
                recent_size=9,
                page_size=page_size,
            )
            self.assertLessEqual(len(plan.page_starts), max_pages)

    def test_long_decode_uses_the_compact_kv_length(self):
        plan = build_aoh_page_plan(
            total_kv_len=884,
            query_len=1,
            sink_size=128,
            recent_size=256,
            page_size=128,
        )

        self.assertEqual(plan.actual_kv_len, 500)
        self.assertLessEqual(plan.actual_kv_len, 4 * 128)

    def test_npu_graph_keeps_empty_updates_for_streaming_events(self):
        update_inputs = _build_aoh_graph_update_inputs(
            ["retrieval", "streaming", "retrieval"],
            full_kv_lens=[884, 256],
        )

        self.assertEqual(
            update_inputs,
            [
                {"actual_seq_lengths_kv": [884, 256]},
                {},
                {"actual_seq_lengths_kv": [884, 256]},
            ],
        )

    def test_npu_graph_forwards_empty_updates_to_streaming_records(self):
        class Graph:
            def update(self, *, cpu_update_input):
                self.cpu_update_input = cpu_update_input

            def replay(self):
                pass

        graph = Graph()
        backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
        backend._graphs = {"shape": graph}
        backend._outputs = {"shape": "output"}
        backend._device_id = 0
        backend._device_module = SimpleNamespace(set_device=lambda _: None)

        update_inputs = [
            {"actual_seq_lengths_kv": [884]},
            {},
            {"actual_seq_lengths_kv": [884]},
        ]
        self.assertEqual(
            backend.replay_with_input_update(
                "shape", seq_lens=None, cpu_update_input=update_inputs
            ),
            "output",
        )
        self.assertEqual(graph.cpu_update_input, update_inputs)

    def test_npu_graph_rejects_mismatched_per_record_updates(self):
        graph = SimpleNamespace(
            graph_dispatch_mode=SimpleNamespace(
                graph_dispatch_records=[object(), object()]
            )
        )

        with self.assertRaisesRegex(RuntimeError, "captured 2 FIA records"):
            _validate_per_record_graph_update(
                graph,
                [
                    {"actual_seq_lengths_kv": [884]},
                    {},
                    {"actual_seq_lengths_kv": [884]},
                ],
            )

    def test_decode_mask_is_page_periodic_after_anchor_and_recent_separate(self):
        for sink_size, recent_size, page_size in (
            (128, 256, 128),
            (5, 9, 4),
            (7, 3, 8),
        ):
            periodic_start = get_aoh_decode_mask_periodic_start(
                sink_size, recent_size, page_size
            )
            max_pages = get_aoh_max_kv_pages(sink_size, recent_size, page_size)
            templates = []
            for offset in range(page_size):
                plan = build_aoh_page_plan(
                    total_kv_len=periodic_start + offset,
                    query_len=1,
                    sink_size=sink_size,
                    recent_size=recent_size,
                    page_size=page_size,
                )
                templates.append(build_aoh_decode_mask(plan, page_size, max_pages))

            for seq_len in range(periodic_start, periodic_start + 4 * page_size):
                plan = build_aoh_page_plan(
                    total_kv_len=seq_len,
                    query_len=1,
                    sink_size=sink_size,
                    recent_size=recent_size,
                    page_size=page_size,
                )
                template_idx = (seq_len - periodic_start) % page_size
                self.assertEqual(
                    build_aoh_decode_mask(plan, page_size, max_pages),
                    templates[template_idx],
                )

    def test_decode_mask_keeps_only_anchor_and_recent_tokens(self):
        page_size = 4
        plan = build_aoh_page_plan(
            total_kv_len=31,
            query_len=1,
            sink_size=5,
            recent_size=9,
            page_size=page_size,
        )
        mask = build_aoh_decode_mask(
            plan,
            page_size,
            get_aoh_max_kv_pages(5, 9, page_size),
        )

        visible_positions = []
        for compact_idx, is_masked in enumerate(mask):
            page_idx, offset = divmod(compact_idx, page_size)
            if not is_masked:
                visible_positions.append(plan.page_starts[page_idx] + offset)

        self.assertEqual(visible_positions, list(range(5)) + list(range(22, 31)))

    def test_npu_graph_retrieval_only_rank_does_not_need_compact_lengths(self):
        self.assertEqual(
            _build_aoh_graph_update_inputs(
                ["retrieval", "retrieval"],
                full_kv_lens=[884, 256],
            ),
            [
                {"actual_seq_lengths_kv": [884, 256]},
                {"actual_seq_lengths_kv": [884, 256]},
            ],
        )

    def test_npu_graph_rejects_invalid_attention_mode(self):
        with self.assertRaisesRegex(RuntimeError, "invalid attention mode"):
            _build_aoh_graph_update_inputs(
                ["invalid"],
                full_kv_lens=[884, 256],
            )

    def test_npu_target_verify_uses_fia_v1_length_name(self):
        update_inputs = _build_aoh_graph_update_inputs(
            ["retrieval", "streaming"],
            full_kv_lens=[884],
        )

        self.assertEqual(
            update_inputs,
            [{"actual_seq_lengths_kv": [884]}, {}],
        )

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

    def test_prefill_crossing_window_boundary_requires_custom_mask(self):
        plan = build_aoh_page_plan(
            total_kv_len=512,
            query_len=256,
            sink_size=128,
            recent_size=256,
            page_size=128,
        )
        self.assertFalse(plan.can_use_causal_template)

    def test_prefill_chunk_uses_mask_budget_instead_of_recent_window(self):
        self.assertEqual(
            get_aoh_prefill_chunk_size(
                query_start=0,
                remaining_query_len=1536,
                sink_size=128,
                recent_size=256,
                page_size=128,
                max_mask_elements=2048 * 2048,
            ),
            1536,
        )
        chunk_size = get_aoh_prefill_chunk_size(
            query_start=4096,
            remaining_query_len=8192,
            sink_size=128,
            recent_size=256,
            page_size=128,
            max_mask_elements=2048 * 2048,
        )
        plan = build_aoh_page_plan(
            total_kv_len=4096 + chunk_size,
            query_len=chunk_size,
            sink_size=128,
            recent_size=256,
            page_size=128,
        )
        self.assertGreater(chunk_size, 256)
        self.assertEqual(chunk_size % 128, 0)
        self.assertLessEqual(chunk_size * len(plan.page_starts) * 128, 2048 * 2048)

        with self.assertRaisesRegex(ValueError, "cannot fit one query row"):
            get_aoh_prefill_chunk_size(
                query_start=4096,
                remaining_query_len=1,
                sink_size=128,
                recent_size=256,
                page_size=128,
                max_mask_elements=128,
            )

        for mask_budget_mb, expected_chunks in ((4, 5), (16, 3), (64, 1)):
            query_offset = 0
            chunk_count = 0
            while query_offset < 8192:
                query_offset += get_aoh_prefill_chunk_size(
                    query_start=query_offset,
                    remaining_query_len=8192 - query_offset,
                    sink_size=128,
                    recent_size=256,
                    page_size=128,
                    max_mask_elements=mask_budget_mb * 1024 * 1024,
                )
                chunk_count += 1
            self.assertEqual(chunk_count, expected_chunks)

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
        speculative_num_draft_tokens=None,
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
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        kvc = SimpleNamespace(
            is_aoh=True,
            aoh_sink_size=128,
            aoh_recent_size=256,
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

    def test_speculative_verify_reserves_uncommitted_streaming_slots(self):
        with get_parallel().override(attn_tp_size=2):
            configurator = self._make_configurator(speculative_num_draft_tokens=4)

        self.assertEqual(configurator._stream_capacity, 6528)

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
