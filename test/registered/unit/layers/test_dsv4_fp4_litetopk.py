import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4 import fp4_litetopk_hip
from sglang.kernels.ops.attention.dsv4.fp4_litetopk_hip import (
    FP4LiteTopKScratch,
    aiter_fp4_litetopk,
    aiter_fp4_litetopk_supports_topk,
    can_use_fp4_litetopk,
    fp4_litetopk_ineligible_reason,
    fp4_litetopk_must_dispatch,
    is_fp4_litetopk_prefill_workspace_compatible,
    max_c4_context_from_seq_lens,
    prepare_fp4_litetopk_scratch,
    prepare_fp4_litetopk_scratch_for_dispatch,
    run_aiter_fp4_litetopk_chunks,
    validate_fp4_litetopk_static_configuration,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"
_METADATA = "sglang.srt.layers.attention.dsv4.metadata"


class TestDSV4FP4LiteTopK(CustomTestCase):
    def _eligible(self, **overrides):
        values = {
            "enabled": True,
            "is_hip": True,
            "arch": "gfx950",
            "is_extend": True,
            "batch_size": 1,
            "query_rows": 64,
            "heads": 64,
            "packed_head_dim": 64,
            "topk": 512,
            "page_size": 64,
            "max_context": 65_536,
            "attn_cp_size": 1,
            "use_sgl_topk": True,
            "use_prefill_graph": False,
            "capturing": False,
            "in_piecewise_graph": False,
            "in_breakable_graph": False,
            "has_tbo_parent": False,
            "has_tbo_children": False,
            "has_spec_info": False,
            "has_spec_algorithm": False,
            "enable_multi_stream": False,
            "skip_compressor": False,
            "has_hisparse": False,
            "has_prefill_workspace": True,
            "aiter_available": True,
        }
        values.update(overrides)
        return can_use_fp4_litetopk(**values)

    def test_feature_is_default_off(self):
        self.assertFalse(envs.SGLANG_DSV4_FP4_LITETOPK.default)
        self.assertFalse(envs.SGLANG_DSV4_FP4_LITETOPK_REQUIRED.default)

    def test_v4_flash_and_pro_topk_are_supported(self):
        self.assertTrue(self._eligible(topk=512))
        self.assertTrue(self._eligible(topk=1024))
        self.assertTrue(self._eligible(max_context=196_608))

    def test_c4_context_boundaries_use_floor_division(self):
        self.assertFalse(self._eligible(max_context=65_535))
        self.assertTrue(self._eligible(max_context=65_536))
        self.assertTrue(self._eligible(max_context=196_608))
        self.assertFalse(self._eligible(max_context=196_609))
        self.assertEqual(max_c4_context_from_seq_lens([262_143]), 65_535)
        self.assertEqual(max_c4_context_from_seq_lens([262_144]), 65_536)
        self.assertEqual(max_c4_context_from_seq_lens([786_435]), 196_608)
        self.assertEqual(max_c4_context_from_seq_lens([786_436]), 196_609)

    def test_ineligible_configurations_fail_closed(self):
        cases = (
            {"enabled": False},
            {"is_hip": False},
            {"arch": "gfx942"},
            {"is_extend": False},
            {"batch_size": 2},
            {"query_rows": 0},
            {"heads": 32},
            {"packed_head_dim": 128},
            {"topk": 2048},
            {"page_size": 1},
            {"max_context": 65_535},
            {"max_context": 196_609},
            {"attn_cp_size": 2},
            {"use_sgl_topk": False},
            {"use_prefill_graph": True},
            {"capturing": True},
            {"in_piecewise_graph": True},
            {"in_breakable_graph": True},
            {"has_tbo_parent": True},
            {"has_tbo_children": True},
            {"has_spec_info": True},
            {"has_spec_algorithm": True},
            {"enable_multi_stream": True},
            {"skip_compressor": True},
            {"has_hisparse": True},
            {"has_prefill_workspace": False},
            {"aiter_available": False},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(self._eligible(**case))

    def test_ineligible_reason_names_the_failed_requirement(self):
        values = {
            "enabled": True,
            "is_hip": True,
            "arch": "gfx950",
            "is_extend": True,
            "batch_size": 1,
            "query_rows": 64,
            "heads": 64,
            "packed_head_dim": 64,
            "topk": 512,
            "page_size": 64,
            "max_context": 65_536,
            "attn_cp_size": 1,
            "use_sgl_topk": True,
            "use_prefill_graph": False,
            "capturing": False,
            "in_piecewise_graph": False,
            "in_breakable_graph": False,
            "has_tbo_parent": False,
            "has_tbo_children": False,
            "has_spec_info": False,
            "has_spec_algorithm": False,
            "enable_multi_stream": False,
            "skip_compressor": False,
            "has_hisparse": False,
            "has_prefill_workspace": True,
            "aiter_available": True,
        }
        self.assertIsNone(fp4_litetopk_ineligible_reason(**values))
        values["attn_cp_size"] = 2
        self.assertEqual(
            fp4_litetopk_ineligible_reason(**values),
            "attention context parallelism is active",
        )

    def test_required_static_configuration_fails_early(self):
        validate_fp4_litetopk_static_configuration(
            required=False,
            is_hip_platform=False,
            use_fp4_indexer=False,
        )
        with self.assertRaisesRegex(RuntimeError, "requires a HIP platform"):
            validate_fp4_litetopk_static_configuration(
                required=True,
                is_hip_platform=False,
                use_fp4_indexer=True,
            )
        with self.assertRaisesRegex(RuntimeError, "requires the AITER FP4 indexer"):
            validate_fp4_litetopk_static_configuration(
                required=True,
                is_hip_platform=True,
                use_fp4_indexer=False,
            )
        with self.assertRaisesRegex(RuntimeError, "requires gfx950"):
            validate_fp4_litetopk_static_configuration(
                required=True,
                is_hip_platform=True,
                use_fp4_indexer=True,
                arch="gfx942",
            )
        with self.assertRaisesRegex(RuntimeError, "requires H=64"):
            validate_fp4_litetopk_static_configuration(
                required=True,
                is_hip_platform=True,
                use_fp4_indexer=True,
                heads=32,
            )
        validate_fp4_litetopk_static_configuration(
            required=True,
            is_hip_platform=True,
            use_fp4_indexer=True,
            topk=1024,
        )
        with self.assertRaisesRegex(RuntimeError, "requires the SGL top-k"):
            validate_fp4_litetopk_static_configuration(
                required=True,
                is_hip_platform=True,
                use_fp4_indexer=True,
                use_sgl_topk=False,
            )
        with self.assertRaisesRegex(RuntimeError, "complete AITER LiteTopK API"):
            validate_fp4_litetopk_static_configuration(
                required=True,
                is_hip_platform=True,
                use_fp4_indexer=True,
                aiter_available=False,
            )

    def test_required_applies_only_to_qualified_prefill_phase(self):
        self.assertFalse(
            fp4_litetopk_must_dispatch(
                required=True,
                is_plain_extend=True,
                max_context=65_535,
            )
        )
        self.assertFalse(
            fp4_litetopk_must_dispatch(
                required=True,
                is_plain_extend=False,
                max_context=65_536,
            )
        )
        self.assertTrue(
            fp4_litetopk_must_dispatch(
                required=True,
                is_plain_extend=True,
                max_context=65_536,
            )
        )
        self.assertTrue(
            fp4_litetopk_must_dispatch(
                required=True,
                is_plain_extend=True,
                max_context=196_609,
            )
        )

    def test_aiter_api_requires_status_aware_contract(self):
        class LegacyWorkspace:
            _fields = ("status",)

        class CurrentWorkspace:
            _fields = ("status", "status_ok")

        def legacy_run():
            pass

        def current_run(*, enforce_status=True):
            pass

        self.assertFalse(
            fp4_litetopk_hip._is_status_aware_aiter_api(current_run, LegacyWorkspace)
        )
        self.assertFalse(
            fp4_litetopk_hip._is_status_aware_aiter_api(legacy_run, CurrentWorkspace)
        )
        self.assertTrue(
            fp4_litetopk_hip._is_status_aware_aiter_api(current_run, CurrentWorkspace)
        )

    def test_aiter_topk_capability_is_explicit(self):
        api = (MagicMock(), MagicMock(), MagicMock(), object, object, {512, 1024})
        with patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api):
            self.assertTrue(aiter_fp4_litetopk_supports_topk(512))
            self.assertTrue(aiter_fp4_litetopk_supports_topk(1024))
            self.assertFalse(aiter_fp4_litetopk_supports_topk(2048))
        with patch.object(
            fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=None
        ):
            self.assertFalse(aiter_fp4_litetopk_supports_topk(1024))

    def test_prefill_workspace_compatibility(self):
        workspace = SimpleNamespace(
            guarded_page_table=torch.empty((4, 1028), dtype=torch.int32),
            row_to_batch=torch.empty(4, dtype=torch.int32),
            local_starts=torch.empty(4, dtype=torch.int32),
            max_seq_len=65_536,
        )
        kwargs = {
            "rows": 4,
            "max_seq_len": 65_536,
            "device": torch.device("cpu"),
        }
        self.assertTrue(
            is_fp4_litetopk_prefill_workspace_compatible(workspace, **kwargs)
        )
        self.assertFalse(
            is_fp4_litetopk_prefill_workspace_compatible(
                SimpleNamespace(
                    **{
                        **vars(workspace),
                        "guarded_page_table": workspace.guarded_page_table[:3],
                    }
                ),
                **kwargs,
            )
        )
        self.assertFalse(
            is_fp4_litetopk_prefill_workspace_compatible(
                workspace, **{**kwargs, "rows": 5}
            )
        )
        self.assertFalse(
            is_fp4_litetopk_prefill_workspace_compatible(
                SimpleNamespace(**{**vars(workspace), "max_seq_len": 65_535}),
                **kwargs,
            )
        )
        self.assertFalse(is_fp4_litetopk_prefill_workspace_compatible(None, **kwargs))

    def test_workspace_preflight_rejects_before_allocation(self):
        allocator = MagicMock()
        api = (
            MagicMock(),
            allocator,
            MagicMock(return_value=1024),
            object,
            object,
            {512, 1024},
        )
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.mem_get_info", return_value=(1024, 4096)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=0),
            self.assertRaises(torch.OutOfMemoryError),
        ):
            prepare_fp4_litetopk_scratch(
                rows=1, topk=1024, device=torch.device("cuda:0")
            )
        allocator.assert_not_called()

    def test_compatible_workspace_is_reused_without_preflight(self):
        workspace = object()
        scratch = FP4LiteTopKScratch(
            workspace=workspace, rows=4, topk=1024, device_index=0, stream_id=17
        )
        api = (MagicMock(), MagicMock(), MagicMock(), object, object, {512, 1024})
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.mem_get_info") as mem_get_info,
        ):
            actual = prepare_fp4_litetopk_scratch(
                rows=3,
                topk=1024,
                device=torch.device("cuda:0"),
                scratch=scratch,
            )
        self.assertIs(actual, scratch)
        mem_get_info.assert_not_called()

    def test_workspace_on_another_device_is_not_reused(self):
        workspace = object()
        scratch = FP4LiteTopKScratch(
            workspace=workspace, rows=4, topk=1024, device_index=1, stream_id=17
        )
        allocated = object()
        allocator = MagicMock(return_value=allocated)
        api = (
            MagicMock(),
            allocator,
            MagicMock(return_value=1024),
            object,
            object,
            {512, 1024},
        )
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.mem_get_info", return_value=(1 << 30, 1 << 30)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            actual = prepare_fp4_litetopk_scratch(
                rows=3,
                topk=1024,
                device=torch.device("cuda:0"),
                scratch=scratch,
            )
        self.assertEqual(actual.device_index, 0)
        self.assertEqual(actual.topk, 1024)
        self.assertIs(actual.workspace, allocated)
        allocator.assert_called_once_with(
            3, torch.device("cuda:0"), topk=1024, stream=stream
        )

    def test_workspace_with_another_topk_is_not_reused(self):
        scratch = FP4LiteTopKScratch(object(), 4, 512, 0, 17)
        allocated = object()
        allocator = MagicMock(return_value=allocated)
        workspace_size = MagicMock(return_value=1024)
        api = (MagicMock(), allocator, workspace_size, object, object, {512, 1024})
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.mem_get_info", return_value=(1 << 30, 1 << 30)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            actual = prepare_fp4_litetopk_scratch(
                rows=3,
                topk=1024,
                device=torch.device("cuda:0"),
                scratch=scratch,
            )
        self.assertEqual(actual.topk, 1024)
        workspace_size.assert_called_once_with(3, topk=1024)
        allocator.assert_called_once_with(
            3, torch.device("cuda:0"), topk=1024, stream=stream
        )

    def test_workspace_preflight_counts_allocator_cache(self):
        allocated = object()
        allocator = MagicMock(return_value=allocated)
        required_bytes = 1024
        cached_bytes = fp4_litetopk_hip._ALLOCATION_HEADROOM_BYTES + required_bytes
        api = (
            MagicMock(),
            allocator,
            MagicMock(return_value=required_bytes),
            object,
            object,
            {512, 1024},
        )
        stream = SimpleNamespace(cuda_stream=17)
        with (
            patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.mem_get_info", return_value=(0, 1 << 30)),
            patch("torch.cuda.memory_reserved", return_value=cached_bytes),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            actual = prepare_fp4_litetopk_scratch(
                rows=1,
                topk=1024,
                device=torch.device("cuda:0"),
            )

        self.assertIs(actual.workspace, allocated)
        allocator.assert_called_once()

    def test_dispatch_workspace_failure_policy(self):
        failures = (
            torch.OutOfMemoryError("test allocation failure"),
            RuntimeError("test workspace setup failure"),
        )
        for setup_error in failures:
            with (
                self.subTest(error=type(setup_error).__name__),
                patch.object(
                    fp4_litetopk_hip,
                    "prepare_fp4_litetopk_scratch",
                    side_effect=setup_error,
                ),
            ):
                self.assertIsNone(
                    prepare_fp4_litetopk_scratch_for_dispatch(
                        rows=1,
                        topk=1024,
                        device=torch.device("cuda:0"),
                        scratch=None,
                        required=False,
                    )
                )
                with self.assertRaisesRegex(
                    RuntimeError, "workspace setup failed in required mode"
                ) as error:
                    prepare_fp4_litetopk_scratch_for_dispatch(
                        rows=1,
                        topk=1024,
                        device=torch.device("cuda:0"),
                        scratch=None,
                        required=True,
                    )
                self.assertIs(error.exception.__cause__, setup_error)

    def test_adapter_explicitly_enforces_status(self):
        rows, topk = 2, 1024
        workspace = SimpleNamespace(
            selected_values=torch.empty((rows, topk), dtype=torch.float32),
            output_counts=torch.empty(rows, dtype=torch.int32),
            candidate_counts=torch.empty(rows, dtype=torch.int32),
            status=torch.empty(rows, dtype=torch.int32),
        )
        scratch = FP4LiteTopKScratch(workspace, rows, topk, 0, 17)
        run_litetopk = MagicMock()
        result_type = MagicMock(return_value=object())
        api = (
            run_litetopk,
            MagicMock(),
            MagicMock(),
            object,
            result_type,
            {512, 1024},
        )
        with (
            patch.object(fp4_litetopk_hip, "get_aiter_fp4_litetopk", return_value=api),
            patch.object(
                fp4_litetopk_hip,
                "prepare_fp4_litetopk_scratch",
                return_value=scratch,
            ),
        ):
            actual = aiter_fp4_litetopk(
                q_fp4=torch.empty((rows, 1, 1), dtype=torch.uint8),
                q_scale=torch.empty((rows, 1), dtype=torch.uint8),
                k_payload=torch.empty(1, dtype=torch.uint8),
                k_scale=torch.empty(1, dtype=torch.uint8),
                weights=torch.empty((rows, 1)),
                guarded_page_table=torch.empty((1, 1), dtype=torch.int32),
                row_to_batch=torch.zeros(rows, dtype=torch.int32),
                row_starts=torch.zeros(rows, dtype=torch.int32),
                c4_seq_lens=torch.ones(rows, dtype=torch.int32),
                max_seq_len=1,
                topk=topk,
                weight_scale=1.0,
                out_page_indices=torch.empty((rows, topk), dtype=torch.int32),
                out_raw_indices=torch.empty((rows, topk), dtype=torch.int32),
                scratch=scratch,
            )

        self.assertIs(actual, scratch)
        self.assertTrue(run_litetopk.call_args.kwargs["enforce_status"])
        self.assertEqual(run_litetopk.call_args.kwargs["topk"], topk)

    def test_chunk_boundary_preserves_global_row_metadata(self):
        query_rows, topk = 1025, 1024
        scratch = FP4LiteTopKScratch(
            workspace=object(), rows=1024, topk=topk, device_index=0, stream_id=17
        )
        final_scratch = FP4LiteTopKScratch(
            workspace=object(), rows=1024, topk=topk, device_index=0, stream_id=17
        )
        row_ids = torch.arange(query_rows, dtype=torch.int32)
        out_page_indices = torch.empty(query_rows, topk, dtype=torch.int32)
        out_raw_indices = torch.empty_like(out_page_indices)
        run = MagicMock(side_effect=(scratch, final_scratch))
        with patch.object(fp4_litetopk_hip, "aiter_fp4_litetopk", run):
            actual = run_aiter_fp4_litetopk_chunks(
                q_fp4=row_ids.reshape(-1, 1, 1),
                q_scale=row_ids.reshape(-1, 1),
                k_payload=torch.empty(1),
                k_scale=torch.empty(1),
                weights=row_ids.reshape(-1, 1),
                guarded_page_table=torch.empty(1),
                row_to_batch=row_ids,
                row_starts=row_ids + 10_000,
                c4_seq_lens=row_ids + 65_536,
                max_seq_len=196_608,
                topk=topk,
                weight_scale=1.0,
                out_page_indices=out_page_indices,
                out_raw_indices=out_raw_indices,
                rows_per_chunk=1024,
                scratch=scratch,
            )

        self.assertIs(actual, final_scratch)
        self.assertEqual(run.call_count, 2)
        first = run.call_args_list[0].kwargs
        second = run.call_args_list[1].kwargs
        self.assertEqual(first["row_to_batch"].tolist(), list(range(1024)))
        self.assertEqual(second["row_to_batch"].tolist(), [1024])
        self.assertEqual(second["row_starts"].tolist(), [11_024])
        self.assertEqual(second["c4_seq_lens"].tolist(), [66_560])
        self.assertEqual(second["q_fp4"].reshape(-1).tolist(), [1024])
        self.assertEqual(second["out_page_indices"].shape, (1, topk))
        self.assertEqual(second["out_raw_indices"].shape, (1, topk))
        self.assertEqual(second["topk"], topk)
        self.assertIs(second["scratch"], scratch)

    def test_forward_dispatches_litetopk_publishes_and_captures(self):
        rows, topk, c4_len, layer_id = 1, 1024, 65_536, 7
        page_table = torch.arange(c4_len // 64, dtype=torch.int32).reshape(1, -1)
        c4_seq_lens = torch.tensor([c4_len], dtype=torch.int32)
        with patch(f"{_METADATA}.is_hip", return_value=True):
            indexer_metadata = PagedIndexerMetadata(
                page_size=256,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                use_topk_v2=False,
                use_prefill_cuda_graph=False,
            )

        physical = torch.full((rows, topk), -1, dtype=torch.int32)
        core = SimpleNamespace(
            positions=torch.zeros(rows, dtype=torch.int32),
            c4_sparse_page_indices=physical,
            c4_sparse_raw_indices=None,
        )
        prefill = SimpleNamespace(
            guarded_page_table=page_table,
            row_to_batch=torch.zeros(rows, dtype=torch.int32),
            local_starts=torch.zeros(rows, dtype=torch.int32),
            max_seq_len=c4_len,
        )
        metadata = SimpleNamespace(
            indexer_metadata=indexer_metadata,
            core_metadata=core,
            fp4_prefill_workspace=prefill,
            fp4_decode_workspace=None,
            fp4_q_positions=None,
        )
        payload = torch.empty(1, dtype=torch.uint8)
        scale = torch.empty(1)
        pool = SimpleNamespace(
            get_index_k_fp4_payload_buffer=MagicMock(return_value=payload),
            get_index_k_fp4_scale_buffer=MagicMock(return_value=scale),
            layer_mapping={layer_id: SimpleNamespace(compress_layer_id=3)},
        )
        backend = C4IndexerBackendMixin()
        backend.token_to_kv_pool = pool
        backend.forward_metadata = metadata
        backend.hisparse_coordinator = None
        backend.spec_algorithm = None
        backend.fp4_litetopk_scratch = None
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            batch_size=1,
            seq_lens_cpu=[4 * c4_len],
            tbo_parent_token_range=None,
            tbo_children=None,
            spec_info=None,
        )
        c4_indexer = SimpleNamespace(
            use_fp4_indexer=True,
            layer_id=layer_id,
            index_topk=topk,
            weight_scale=0.125,
        )
        q_fp4 = torch.empty((rows, 64, 64), dtype=torch.uint8)
        q_scale = torch.empty((rows, 1, 4, 16, 4), dtype=torch.uint8)
        weights = torch.empty((rows, 64), dtype=torch.float32)
        prepared = FP4LiteTopKScratch(object(), rows, topk, 0, 17)
        retained = FP4LiteTopKScratch(object(), rows, topk, 0, 17)
        expected_raw = torch.arange(topk, dtype=torch.int32).reshape(rows, topk)
        expected_physical = expected_raw + 4096
        events = []

        def fake_run(**kwargs):
            events.append("run")
            kwargs["out_raw_indices"].copy_(expected_raw)
            kwargs["out_page_indices"].copy_(expected_physical)
            return retained

        def fake_capture(compress_layer_id, raw):
            events.append("capture")
            self.assertEqual(compress_layer_id, 3)
            self.assertIs(backend.fp4_litetopk_scratch, retained)
            self.assertIs(core.c4_sparse_raw_indices, raw)

        run = MagicMock(side_effect=fake_run)
        dense = MagicMock(side_effect=AssertionError("dense scorer was reached"))
        capturer = SimpleNamespace(capture=MagicMock(side_effect=fake_capture))
        with (
            envs.SGLANG_DSV4_FP4_LITETOPK.override(True),
            envs.SGLANG_DSV4_FP4_LITETOPK_REQUIRED.override(False),
            get_parallel().override(attn_cp_size=1),
            patch(f"{_INDEXER}.is_hip", return_value=True),
            patch(f"{_INDEXER}.aiter_fp4_litetopk_supports_topk", return_value=True),
            patch(f"{_INDEXER}.get_global_indexer_capturer", return_value=capturer),
            patch(f"{_INDEXER}.is_in_tc_piecewise_cuda_graph", return_value=False),
            patch(f"{_INDEXER}.is_in_breakable_cuda_graph", return_value=False),
            patch(
                f"{_INDEXER}.torch.cuda.is_current_stream_capturing",
                return_value=False,
            ),
            patch(
                f"{_INDEXER}.torch.cuda.get_device_properties",
                return_value=SimpleNamespace(gcnArchName="gfx950:sramecc+"),
            ),
            patch(f"{_INDEXER}.torch.cuda.current_device", return_value=0),
            patch(
                f"{_INDEXER}.prepare_fp4_litetopk_scratch_for_dispatch",
                return_value=prepared,
            ),
            patch(f"{_INDEXER}.run_aiter_fp4_litetopk_chunks", run),
            patch(f"{_INDEXER}.aiter_fp4_paged_mqa_logits", dense),
            patch.object(
                C4IndexerBackendMixin,
                "_forward_prepare_normal",
                return_value=((q_fp4, q_scale), weights),
            ),
        ):
            result = backend.forward_c4_indexer(
                torch.empty((rows, 1)),
                torch.empty((rows, 1)),
                c4_indexer,
                batch,
            )
            events.append("return")

        self.assertIsNone(result)
        run.assert_called_once()
        dense.assert_not_called()
        self.assertIs(backend.fp4_litetopk_scratch, retained)
        self.assertEqual(run.call_args.kwargs["topk"], topk)
        self.assertIs(run.call_args.kwargs["out_page_indices"], physical)
        self.assertIs(
            run.call_args.kwargs["out_raw_indices"], core.c4_sparse_raw_indices
        )
        torch.testing.assert_close(physical, expected_physical)
        torch.testing.assert_close(core.c4_sparse_raw_indices, expected_raw)
        capturer.capture.assert_called_once_with(3, core.c4_sparse_raw_indices)
        self.assertEqual(events, ["run", "capture", "return"])


if __name__ == "__main__":
    unittest.main()
