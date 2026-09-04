"""Unit tests for HiSparse hierarchical sparse KV cache system.

Tests cover:
- CUDA kernel correctness (swap_in_selected_pages vs naive_load_topk oracle)
- Memory allocator lifecycle (alloc / free / available_size)
- Request lifecycle (staging path, direct-to-host path)
- Batch multi-request correctness
"""

import ast
import os
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Test configuration (small-scale for fast CI runs)
# ---------------------------------------------------------------------------
SIZE = 8192  # includes native MTP staging for multi-request coverage
PAGE_SIZE = 64  # page size (must be 64 for CUDA, 1 for ROCm)
TOP_K = 256  # top-k selection count
DEVICE_BUFFER_SIZE = 512  # device buffer per request
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576  # MLA dim (DeepSeek-style)
LAYER_NUM = 2
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 2048


def _make_req(rid="test-req-0", origin_input_ids=None, output_ids=None):
    """Create a minimal mock Req object with the fields HiSparseCoordinator uses."""
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    if output_ids is None:
        output_ids = []
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        fill_ids=origin_input_ids + output_ids,
        seqlen=len(origin_input_ids) + len(output_ids),
        kv=ReqKvInfo(),
        finished_reason=None,
        hisparse_staging=False,
        staging=False,
        inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return req


class TestHiSparseMTPDemandSelector(unittest.TestCase):
    def test_hisparse_config_parses_mtp_demand_buffer_as_opt_in(self):
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        disabled = parse_hisparse_config(SimpleNamespace(hisparse_config=None))
        enabled = parse_hisparse_config(
            SimpleNamespace(hisparse_config='{"mtp_demand_buffer": true}')
        )

        self.assertFalse(disabled.mtp_demand_buffer)
        self.assertTrue(enabled.mtp_demand_buffer)

        with self.assertRaisesRegex(ValueError, "mtp_demand_buffer must be a boolean"):
            parse_hisparse_config(
                SimpleNamespace(hisparse_config='{"mtp_demand_buffer": 1}')
            )

    def test_generic_free_clears_mapping_without_cpu_scalar_copy(self):
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )

        class MappingWithAsyncClear:
            def __init__(self):
                self.tensor = torch.tensor([0, 11, 12, 13], dtype=torch.int64)
                self.index_fill_calls = []

            def __getitem__(self, key):
                return self.tensor[key]

            def __setitem__(self, key, value):
                if isinstance(value, int) and value == 0:
                    raise AssertionError("mapping clear must not copy a CPU scalar")
                self.tensor[key] = value

            def index_fill_(self, dim, index, value):
                self.index_fill_calls.append((dim, index.clone(), value))
                return self.tensor.index_fill_(dim, index, value)

        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator._kvcache = SimpleNamespace(
            _translate_loc_to_hisparse_device=lambda indices: torch.tensor(
                [21, 22], dtype=torch.int64
            )
        )
        allocator.full_to_hisparse_device_index_mapping = MappingWithAsyncClear()
        freed = []
        allocator.free_hisparse_indices = lambda indices: freed.append(indices.clone())
        free_indices = torch.tensor([1, 3], dtype=torch.int32)

        allocator.free_hisparse(free_indices)

        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], torch.tensor([21, 22])))
        calls = allocator.full_to_hisparse_device_index_mapping.index_fill_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 0)
        self.assertEqual(calls[0][1].dtype, torch.int64)
        self.assertTrue(torch.equal(calls[0][1], free_indices.to(torch.int64)))
        self.assertEqual(calls[0][2], 0)
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping.tensor,
                torch.tensor([0, 0, 12, 0], dtype=torch.int64),
            )
        )

    def test_mtp_demand_indexer_only_publishes_logical_and_host_row_outputs(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        raw_indices = torch.empty(8, 2048, dtype=torch.int32)
        host_locs = torch.empty_like(raw_indices)
        req_to_host = torch.empty(4, 4096, dtype=torch.int64)
        forward_metadata = SimpleNamespace(
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )
        backend = SimpleNamespace(
            use_fused_topk=True,
            hisparse_coordinator=SimpleNamespace(
                mtp_demand_buffer_enabled=True,
                mem_pool_device=SimpleNamespace(start_layer=0),
                get_mtp_demand_topk_output_buffers=lambda _rows: SimpleNamespace(
                    raw_indices=raw_indices,
                    physical_indices=host_locs,
                    direct_transform_table=req_to_host,
                    direct_transform_rows=torch.tensor(
                        [1, 1, 1, 1, 3, 3, 3, 3], dtype=torch.int64
                    ),
                ),
            ),
            forward_metadata=forward_metadata,
            dsa_topk_backend=SimpleNamespace(should_use_topk_v2=lambda: True),
            get_topk_transform_method=lambda _mode: "paged",
        )
        target_verify = SimpleNamespace(
            is_decode_or_idle=lambda: False,
            is_target_verify=lambda: True,
        )
        req_pool_indices = torch.tensor([1, 3], dtype=torch.int64)

        metadata = DeepseekSparseAttnBackend.get_indexer_metadata(
            backend,
            layer_id=0,
            forward_batch=SimpleNamespace(
                forward_mode=target_verify,
                batch_size=2,
                req_pool_indices=req_pool_indices,
                spec_info=SimpleNamespace(num_tokens_per_req=4),
            ),
        )

        self.assertFalse(metadata.force_unfused_topk)
        self.assertIsNotNone(metadata.topk_output_buffers)
        outputs = metadata.topk_output_buffers
        self.assertEqual(outputs.raw_indices.shape, (8, 2048))
        self.assertEqual(outputs.raw_indices.data_ptr(), raw_indices.data_ptr())
        self.assertEqual(outputs.physical_indices.shape, (8, 2048))
        self.assertEqual(outputs.physical_indices.data_ptr(), host_locs.data_ptr())
        self.assertIs(outputs.direct_transform_table, req_to_host)
        self.assertTrue(
            torch.equal(
                outputs.direct_transform_rows,
                torch.tensor([1, 1, 1, 1, 3, 3, 3, 3], dtype=torch.int64),
            )
        )
        self.assertFalse(hasattr(metadata, "demand_source_plan"))
        self.assertFalse(hasattr(metadata, "demand_cache_tags"))

    def test_eagle_draft_input_snapshots_staged_request_state(self):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        draft_input = EagleDraftInput(
            topk_p=torch.tensor([[0.1], [0.2]]),
            topk_index=torch.tensor([[11], [22]]),
            hidden_states=torch.tensor([[1.0], [2.0]]),
            bonus_tokens=torch.tensor([101, 202]),
            future_indices=torch.tensor([3, 7]),
            future_dsa_topk_indices_available=True,
        )

        staged = draft_input.slice_single(1)

        self.assertTrue(torch.equal(staged.future_indices, torch.tensor([7])))
        self.assertTrue(torch.equal(staged.topk_p, torch.tensor([[0.2]])))
        self.assertTrue(torch.equal(staged.topk_index, torch.tensor([[22]])))
        self.assertTrue(torch.equal(staged.bonus_tokens, torch.tensor([202])))
        self.assertTrue(staged.future_dsa_topk_indices_available)

    def test_prefill_result_processor_stashes_hisparse_spec_state(self):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        batch = SimpleNamespace(
            spec_info=EagleDraftInput(
                topk_p=torch.tensor([[0.25], [0.75]]),
                topk_index=torch.tensor([[31], [47]]),
                hidden_states=torch.tensor([[3.0], [4.0]]),
                bonus_tokens=torch.tensor([301, 401]),
                future_indices=torch.tensor([5, 9]),
            )
        )
        req = SimpleNamespace(hisparse_spec_info=None)

        processor._stash_hisparse_spec_info(batch, 1, req)

        self.assertTrue(
            torch.equal(req.hisparse_spec_info.future_indices, torch.tensor([9]))
        )
        self.assertTrue(
            torch.equal(req.hisparse_spec_info.topk_index, torch.tensor([[47]]))
        )

    def test_flashmla_wrapper_exposes_one_mtp_demand_adapter(self):
        import sglang

        wrapper = (
            Path(sglang.__file__).parent / "kernels/aot/python/sgl_kernel/flash_mla.py"
        )
        module = ast.parse(wrapper.read_text())
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "flash_mla_with_kvcache"
        )
        parameters = {argument.arg for argument in function.args.args}
        self.assertIn("hisparse_demand", parameters)
        for name in (
            "hisparse_host_kv",
            "hisparse_host_locs",
            "hisparse_device_locs",
            "hisparse_cache_tags",
            "hisparse_decode_calls",
            "hisparse_num_real_reqs",
            "hisparse_req_pool_indices",
            "hisparse_seq_lens",
            "hisparse_mtp_committed_lens",
            "hisparse_cache_rows",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, parameters)
        self.assertNotIn("hisparse_source_plan", parameters)
        self.assertNotIn("hisparse_cache_stats", parameters)

    def test_glm52_w4a_four_token_mtp_route_requires_explicit_opt_in(self):
        from sglang.srt.model_executor import model_runner

        selector = getattr(
            model_runner, "should_enable_hisparse_mtp_demand_buffer", None
        )
        self.assertIsNotNone(selector, "MTP Demand selector is missing")
        if selector is None:
            return

        server_args = SimpleNamespace(
            enable_hisparse=True,
            device="cuda",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="flashmla_kv",
            dsa_topk_backend="sgl-kernel",
            tp_size=8,
            dp_size=8,
            enable_dp_attention=True,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            speculative_attention_mode="prefill",
            pp_size=1,
            attn_cp_size=1,
            enable_pdmux=False,
            disaggregation_mode="null",
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"],
                index_share_for_mtp_iteration=True,
            )
        )

        self.assertTrue(
            selector(
                server_args,
                model_config,
                device_capability=(9, 0),
                mtp_demand_buffer=True,
            )
        )
        self.assertFalse(
            selector(
                server_args,
                model_config,
                device_capability=(9, 0),
                mtp_demand_buffer=False,
            )
        )
        unresolved_device = SimpleNamespace(**vars(server_args))
        unresolved_device.device = None
        self.assertTrue(
            selector(
                unresolved_device,
                model_config,
                device_capability=(9, 0),
                mtp_demand_buffer=True,
            )
        )
        for field, value in (
            ("speculative_num_steps", 1),
            ("speculative_num_draft_tokens", 2),
            ("speculative_eagle_topk", 2),
            ("dsa_decode_backend", "flashmla_sparse"),
            ("dsa_topk_backend", "torch"),
            ("tp_size", 4),
            ("dp_size", 4),
            ("enable_dp_attention", False),
            ("enable_pdmux", True),
            ("disaggregation_mode", "decode"),
        ):
            candidate = SimpleNamespace(**vars(server_args))
            setattr(candidate, field, value)
            with self.subTest(field=field, value=value):
                self.assertFalse(
                    selector(
                        candidate,
                        model_config,
                        device_capability=(9, 0),
                        mtp_demand_buffer=True,
                    )
                )

        from sglang.srt.environ import envs

        for env_var in (envs.SGLANG_DSA_FUSE_TOPK, envs.SGLANG_OPT_USE_TOPK_V2):
            with self.subTest(env_var=env_var), env_var.override(False):
                self.assertFalse(
                    selector(
                        server_args,
                        model_config,
                        device_capability=(9, 0),
                        mtp_demand_buffer=True,
                    )
                )

        pd_decode = SimpleNamespace(**vars(server_args))
        pd_decode.disaggregation_mode = "decode"
        no_iteration_share = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"],
                index_share_for_mtp_iteration=False,
            )
        )
        self.assertTrue(
            selector(
                pd_decode,
                no_iteration_share,
                device_capability=(9, 0),
                mtp_demand_buffer=True,
            )
        )

    def test_mtp_keeps_draft_kv_device_resident(self):
        from sglang.srt.model_executor import model_runner

        resolve = getattr(model_runner, "resolve_hisparse_for_runner", None)
        self.assertIsNotNone(resolve, "per-runner HiSparse resolver is missing")
        if resolve is None:
            return

        server_args = SimpleNamespace(
            enable_hisparse=True,
            hisparse_config='{"mtp_demand_buffer": true}',
            device="cuda",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="flashmla_kv",
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            speculative_attention_mode="prefill",
            pp_size=1,
            attn_cp_size=1,
            enable_pdmux=False,
            disaggregation_mode="null",
        )

        self.assertTrue(resolve(server_args, is_draft_worker=False))
        self.assertFalse(resolve(server_args, is_draft_worker=True))

        native = SimpleNamespace(**vars(server_args))
        native.hisparse_config = None
        self.assertFalse(resolve(native, is_draft_worker=True))

        unsupported_mtp = SimpleNamespace(**vars(server_args))
        unsupported_mtp.speculative_num_steps = 2
        self.assertFalse(resolve(unsupported_mtp, is_draft_worker=True))

    def test_hisparse_model_runner_rebinds_mapping_to_active_pool(self):
        from sglang.srt.model_executor import model_runner

        bind = getattr(model_runner, "_bind_hisparse_mapping_to_active_pool", None)
        self.assertIsNotNone(bind, "active HiSparse pool mapping binder is missing")
        if bind is None:
            return

        seen = []
        mapping = torch.arange(8, dtype=torch.int64)
        pool = SimpleNamespace(register_mapping=lambda value: seen.append(value))
        allocator = SimpleNamespace(full_to_hisparse_device_index_mapping=mapping)

        bind(pool, allocator)

        self.assertEqual(seen, [mapping])

    def test_target_verify_keeps_logical_topk_and_builds_four_row_demand_bundle(self):
        from sglang.srt.layers.attention import dsa_backend

        prepare = getattr(dsa_backend, "_prepare_hisparse_mtp_demand", None)
        self.assertIsNotNone(prepare, "MTP Demand DSA route is missing")
        if prepare is None:
            return

        calls = []
        coordinator = SimpleNamespace(
            get_mtp_demand_attention_inputs=lambda **kwargs: (
                calls.append(kwargs) or {"marker": "demand"}
            ),
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
            seq_lens=torch.tensor([128, 256], dtype=torch.int64),
        )
        metadata = SimpleNamespace(
            dsa_seqlens_expanded=torch.tensor(
                [129, 130, 131, 132, 257, 258, 259, 260], dtype=torch.int32
            )
        )
        logical_topk = torch.arange(8 * 2048, dtype=torch.int32).view(8, 2048)
        page_table, bundle = prepare(
            coordinator=coordinator,
            metadata=metadata,
            logical_topk=logical_topk,
            layer_id=12,
        )

        self.assertIs(page_table, logical_topk)
        self.assertEqual(bundle, {"marker": "demand"})
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["layer_id"], 12)

    def test_target_verify_native_hisparse_runs_multistep_swap(self):
        from sglang.srt.layers.attention import dsa_backend

        prepare = getattr(dsa_backend, "_prepare_hisparse_mtp_native", None)
        self.assertIsNotNone(prepare, "native MTP HiSparse route is missing")
        if prepare is None:
            return

        logical_topk = torch.arange(8 * 16, dtype=torch.int32).view(8, 16)
        physical_storage = torch.arange(2 * 4 * 32, dtype=torch.int32).view(2, 4, 32)
        physical_topk_3d = physical_storage[..., ::2]
        self.assertFalse(physical_topk_3d.is_contiguous())
        physical_topk = physical_topk_3d.reshape(8, 16)
        swap = MagicMock(return_value=physical_topk_3d)
        coordinator = SimpleNamespace(swap_in_selected_pages_mtp=swap)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
        )
        metadata = SimpleNamespace(
            dsa_seqlens_expanded=torch.tensor(
                [129, 130, 131, 132, 257, 258, 259, 260],
                dtype=torch.int32,
            )
        )

        actual = prepare(
            coordinator=coordinator,
            forward_batch=forward_batch,
            metadata=metadata,
            relative_topk=logical_topk,
            layer_id=12,
            num_steps=4,
        )

        self.assertTrue(torch.equal(actual, physical_topk))
        swap.assert_called_once()
        call = swap.call_args.kwargs
        self.assertIs(call["req_pool_indices"], forward_batch.req_pool_indices)
        self.assertIs(call["seq_lens"], metadata.dsa_seqlens_expanded)
        self.assertEqual(call["top_k_result"].shape, (2, 4, 16))
        self.assertEqual(call["layer_id"], 12)

    def test_mtp_demand_prepares_expanded_batch_metadata_once(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = True
        coordinator.mtp_demand_expanded_req_pool_indices = torch.full(
            (12,), -1, dtype=torch.int64
        )
        coordinator.mtp_demand_expanded_committed_lens = torch.full(
            (12,), -1, dtype=torch.int32
        )
        coordinator.mtp_demand_num_real_query_rows = torch.zeros(1, dtype=torch.int32)

        coordinator.prepare_mtp_demand_batch(
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
            committed_lens=torch.tensor([128, 256], dtype=torch.int64),
            num_query_rows=8,
        )

        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_expanded_req_pool_indices[:8],
                torch.tensor([3, 3, 3, 3, 7, 7, 7, 7]),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_expanded_committed_lens[:8],
                torch.tensor(
                    [128, 128, 128, 128, 256, 256, 256, 256], dtype=torch.int32
                ),
            )
        )
        self.assertEqual(int(coordinator.mtp_demand_num_real_query_rows.item()), 8)

    def test_mtp_direct_demand_passes_no_precomputed_source_plan(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = True
        coordinator.mtp_demand_cache_rows = 4096
        coordinator.top_k_host_locs_buffer = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
        )
        original_host_locs = coordinator.top_k_host_locs_buffer.clone()
        coordinator.mtp_demand_cache_tags = torch.zeros((1, 2, 4096), dtype=torch.int64)
        coordinator.mtp_demand_device_locs = torch.zeros((2, 4102), dtype=torch.int64)
        coordinator.mtp_demand_decode_calls = torch.ones(2, dtype=torch.int32)
        coordinator.mtp_demand_host_kv = torch.empty((1, 32, 656), dtype=torch.uint8)
        coordinator.mtp_demand_num_real_query_rows = torch.tensor(
            [2], dtype=torch.int32
        )
        coordinator.mtp_demand_expanded_req_pool_indices = torch.tensor(
            [0, 1], dtype=torch.int64
        )
        coordinator.mtp_demand_expanded_committed_lens = torch.tensor(
            [128, 256], dtype=torch.int32
        )
        coordinator.mem_pool_device = SimpleNamespace(start_layer=12, layer_num=1)

        inputs = coordinator.get_mtp_demand_attention_inputs(
            layer_id=12,
            seq_lens=torch.tensor([132, 260], dtype=torch.int32),
        )

        self.assertFalse(hasattr(inputs, "source_plan"))
        self.assertTrue(
            torch.equal(coordinator.top_k_host_locs_buffer, original_host_locs)
        )

    def test_coordinator_exposes_mtp_demand_metadata_adapter(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        self.assertTrue(
            hasattr(HiSparseCoordinator, "get_mtp_demand_attention_inputs"),
            "MTP Demand coordinator adapter is missing",
        )

    def test_mtp_verify_overlay_binds_resolved_four_rows_across_hot_boundary(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        class MappingWithAsyncPublish:
            def __init__(self):
                self.tensor = torch.zeros(64, dtype=torch.int64)
                self.index_copy_calls = []

            def __getitem__(self, key):
                return self.tensor[key]

            def __setitem__(self, key, value):
                if isinstance(key, torch.Tensor) and key.dtype == torch.int32:
                    raise AssertionError("mapping publish must not use int32 indexing")
                self.tensor[key] = value

            def index_copy_(self, dim, index, source):
                self.index_copy_calls.append((dim, index.clone(), source.clone()))
                return self.tensor.index_copy_(dim, index, source)

        class TokenTableWithAsyncClear:
            class Slice:
                def __init__(self, tensor, calls):
                    self.tensor = tensor
                    self.calls = calls

                def index_fill_(self, dim, index, value):
                    self.calls.append((dim, index.clone(), value))
                    return self.tensor.index_fill_(dim, index, value)

            def __init__(self):
                self.tensor = torch.full((1, 2, 4160), -1, dtype=torch.int32)
                self.index_fill_calls = []

            def __getitem__(self, key):
                value = self.tensor[key]
                if (
                    isinstance(key, tuple)
                    and len(key) == 3
                    and key[0] == slice(None)
                    and key[1] == slice(None)
                    and key[2] == slice(4097, 4160)
                ):
                    return self.Slice(value, self.index_fill_calls)
                return value

            def __setitem__(self, key, value):
                if isinstance(value, int) and value == -1:
                    raise AssertionError("slot clear must not copy a CPU scalar")
                self.tensor[key] = value

        for committed_len in (128, 4094, 4096):
            with self.subTest(committed_len=committed_len):
                coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
                coordinator.mtp_demand_buffer_enabled = True
                coordinator.mtp_demand_cache_rows = 4096
                coordinator.is_dsv4_hisparse = False
                coordinator.device_buffer_size = 4096
                coordinator.padded_buffer_size = 4160
                coordinator.req_to_device_buffer = torch.arange(
                    2 * 4160, dtype=torch.int64
                ).view(2, 4160)
                coordinator.req_device_buffer_tokens = TokenTableWithAsyncClear()
                coordinator.mtp_demand_device_locs = torch.zeros(
                    (2, 4102), dtype=torch.int64
                )
                coordinator.req_to_host_pool = torch.full(
                    (2, 8192), -1, dtype=torch.int64
                )
                coordinator.req_to_host_pool_allocated_len = torch.zeros(
                    2, dtype=torch.int64
                )
                coordinator.mem_pool_host = SimpleNamespace(
                    alloc_paged_token_slots=lambda *args: torch.arange(
                        4, dtype=torch.int64
                    )
                )
                full_to_device = MappingWithAsyncPublish()
                coordinator.token_to_kv_pool_allocator = SimpleNamespace(
                    full_to_hisparse_device_index_mapping=full_to_device
                )
                ensure_calls = []
                coordinator._ensure_padded_buffer = ensure_calls.append

                req_pool_indices = torch.tensor([1], dtype=torch.int64)
                req_pool_indices_cpu = req_pool_indices.clone()
                verify_cache_locs = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
                coordinator.prepare_verify_slots_spec_v2(
                    req_pool_indices=req_pool_indices,
                    req_pool_indices_cpu=req_pool_indices_cpu,
                    verify_cache_locs=verify_cache_locs,
                    num_tokens_per_req=4,
                    start_positions=torch.tensor([committed_len]),
                    host_reserve_end_positions_cpu=[committed_len + 8],
                )
                self.assertIs(ensure_calls[0], req_pool_indices_cpu)

                token_positions = torch.arange(committed_len, committed_len + 4)
                columns = torch.where(
                    token_positions < 4096,
                    token_positions,
                    torch.arange(4097, 4101),
                )
                expected = coordinator.req_to_device_buffer[1, columns]
                self.assertTrue(
                    torch.equal(
                        coordinator.mtp_demand_device_locs[1, 4098:4102],
                        expected,
                    )
                )
                self.assertTrue(
                    torch.equal(full_to_device[verify_cache_locs], expected)
                )
                self.assertEqual(len(full_to_device.index_copy_calls), 1)
                self.assertEqual(full_to_device.index_copy_calls[0][0], 0)
                self.assertEqual(
                    full_to_device.index_copy_calls[0][1].dtype, torch.int64
                )
                self.assertTrue(
                    torch.equal(
                        full_to_device.index_copy_calls[0][1],
                        verify_cache_locs.to(torch.int64),
                    )
                )
                self.assertTrue(
                    torch.equal(full_to_device.index_copy_calls[0][2], expected)
                )
                clear_calls = coordinator.req_device_buffer_tokens.index_fill_calls
                self.assertEqual(len(clear_calls), 1)
                self.assertEqual(clear_calls[0][0], 1)
                self.assertTrue(torch.equal(clear_calls[0][1], req_pool_indices))
                self.assertEqual(clear_calls[0][2], -1)

    def test_eagle_verify_preparation_delegates_four_row_overlay_to_hisparse(self):
        from sglang.srt.speculative import eagle_utils

        prepare = getattr(eagle_utils, "_prepare_hisparse_mtp_verify_slots", None)
        self.assertIsNotNone(prepare, "EAGLE HiSparse verify adapter is missing")
        if prepare is None:
            return

        calls = []
        coordinator = SimpleNamespace(
            prepare_verify_slots_spec_v2=lambda **kwargs: calls.append(kwargs)
        )
        seq_lens = torch.tensor([128, 4094], dtype=torch.int64)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            hisparse_coordinator=coordinator,
            reqs=[
                SimpleNamespace(kv_committed_len=128),
                SimpleNamespace(kv_committed_len=4094),
            ],
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([3, 7], dtype=torch.int64),
            out_cache_loc=torch.tensor(
                [101, 102, 103, 104, 201, 202, 203, 204], dtype=torch.int64
            ),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
        )

        prepare(batch, draft_token_num=4)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["req_pool_indices"], batch.req_pool_indices)
        self.assertIs(call["req_pool_indices_cpu"], batch.req_pool_indices_cpu)
        self.assertIs(call["verify_cache_locs"], batch.out_cache_loc)
        self.assertEqual(call["num_tokens_per_req"], 4)
        self.assertIs(call["start_positions"], batch.seq_lens)
        self.assertEqual(call["host_reserve_end_positions_cpu"], [136, 4102])

    def test_eagle_verify_preparation_keeps_gpu_only_seq_lens(self):
        from sglang.srt.speculative import eagle_utils

        calls = []
        coordinator = SimpleNamespace(
            prepare_verify_slots_spec_v2=lambda **kwargs: calls.append(kwargs)
        )
        seq_lens = torch.tensor([128, 4094], dtype=torch.int64)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            hisparse_coordinator=coordinator,
            reqs=[
                SimpleNamespace(kv_committed_len=124),
                SimpleNamespace(kv_committed_len=4090),
            ],
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([3, 7], dtype=torch.int64),
            out_cache_loc=torch.tensor(
                [101, 102, 103, 104, 201, 202, 203, 204], dtype=torch.int64
            ),
            seq_lens=seq_lens,
            seq_lens_cpu=None,
        )

        eagle_utils._prepare_hisparse_mtp_verify_slots(batch, draft_token_num=4)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["start_positions"], seq_lens)
        self.assertEqual(call["host_reserve_end_positions_cpu"], [132, 4098])

    def test_eagle_verify_finalizes_native_hisparse_slots(self):
        from sglang.srt.speculative import eagle_worker_common

        finalize = getattr(eagle_worker_common, "_finalize_hisparse_mtp_verify", None)
        self.assertIsNotNone(finalize, "native HiSparse verify finalizer is missing")
        if finalize is None:
            return

        calls = []
        coordinator = SimpleNamespace(
            mtp_demand_buffer_enabled=False,
            supports_hisparse_draft_slots=lambda: True,
            finalize_accepted_tokens_spec_v2=lambda **kwargs: calls.append(kwargs),
        )
        batch = SimpleNamespace(
            hisparse_coordinator=coordinator,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
            seq_lens=torch.tensor([5000], dtype=torch.int64),
            out_cache_loc=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        )
        accept_index = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32)

        finalize(batch, accept_index)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["req_pool_indices"], batch.req_pool_indices)
        self.assertIs(call["seq_lens"], batch.seq_lens)
        self.assertIs(call["verify_cache_locs"], batch.out_cache_loc)
        self.assertIs(call["accept_index"], accept_index)

    def test_mtp_demand_residency_alloc_and_free_are_request_local(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        freed = []
        allocator = SimpleNamespace(
            free_hisparse_indices=lambda rows: freed.append(rows.clone())
        )
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = True
        coordinator.mtp_demand_cache_rows = 4096
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.mtp_demand_device_locs = torch.zeros((2, 4102), dtype=torch.int64)
        coordinator.mtp_demand_cache_tags = torch.zeros((2, 2, 4096), dtype=torch.int64)
        coordinator.mtp_demand_decode_calls = torch.zeros(2, dtype=torch.int32)

        coordinator._bind_mtp_demand_buffer(
            1, torch.arange(101, 4197, dtype=torch.int64)
        )
        coordinator._bind_mtp_demand_overlay(
            torch.tensor([1]), torch.tensor([9001, 9002, 9003, 9004])
        )

        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_device_locs[1, :4096],
                torch.arange(101, 4197, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_device_locs[1, 4098:4102],
                torch.tensor([9001, 9002, 9003, 9004]),
            )
        )
        self.assertEqual(
            coordinator.mtp_demand_device_locs[0].count_nonzero().item(),
            0,
        )

        coordinator._free_mtp_demand_buffer(1)

        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], torch.arange(101, 4197)))
        self.assertEqual(
            coordinator.mtp_demand_device_locs[1].count_nonzero().item(),
            0,
        )

    def test_mtp_demand_side_reserve_is_claimed_with_hot_buffer(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = False
        coordinator.device_buffer_size = 4096
        coordinator.padded_buffer_size = 4160
        coordinator.mtp_demand_buffer_enabled = True
        coordinator.mtp_demand_cache_rows = 4096
        coordinator.mtp_side_reserve_size = 4096
        coordinator.mtp_staging_size = 0
        events = []
        coordinator.mtp_demand_device_locs = torch.zeros((2, 4102), dtype=torch.int64)
        coordinator.mtp_demand_cache_tags = torch.zeros((1, 2, 4096), dtype=torch.int64)
        coordinator.mtp_demand_decode_calls = torch.zeros(2, dtype=torch.int32)
        coordinator.mem_pool_device = SimpleNamespace(
            page_size=64,
            translate_loc_from_full_to_compressed=lambda value: value,
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(256, dtype=torch.int64).view(2, 128)
        )

        def alloc_with_reserve(indices, size, reserve_size):
            events.append((size, reserve_size))
            return (
                torch.arange(1000, 1000 + size, dtype=torch.int64),
                torch.arange(5000, 5000 + reserve_size, dtype=torch.int64),
            )

        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            alloc_device_buffer_with_reserve=alloc_with_reserve
        )
        coordinator.req_to_device_buffer = torch.zeros((2, 4160), dtype=torch.int64)
        coordinator.req_device_buffer_size = torch.zeros(2, dtype=torch.int64)
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 2, 4096), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (1, 2, 4160), -1, dtype=torch.int32
        )
        coordinator._device_buffer_arange_i32 = torch.arange(4096, dtype=torch.int32)
        req = SimpleNamespace(
            req_pool_idx=1,
            rid="capacity-order",
            kv=SimpleNamespace(kv_allocated_len=128),
        )

        coordinator.alloc_device_buffer(req)

        self.assertEqual(events, [(128, 4096)])
        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_device_locs[1, :4096],
                torch.arange(5000, 9096, dtype=torch.int64),
            )
        )

    def test_padded_mtp_arange_does_not_overfill_native_hot_slice(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = False
        coordinator.mtp_demand_buffer_enabled = False
        coordinator.mtp_side_reserve_size = 0
        coordinator.mtp_staging_size = 0
        coordinator.device_buffer_size = 4096
        coordinator.padded_buffer_size = 4160
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(128, dtype=torch.int64).view(1, 128)
        )
        coordinator.mem_pool_device = SimpleNamespace(
            page_size=64,
            translate_loc_from_full_to_compressed=lambda value: value,
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            alloc_device_buffer_with_reserve=lambda logical, size, reserve_size: (
                torch.arange(1, size + 1, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        )
        coordinator.req_to_device_buffer = torch.zeros((1, 4160), dtype=torch.int64)
        coordinator.req_to_mtp_staging = torch.zeros((1, 0), dtype=torch.int64)
        coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int64)
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 1, 4160), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (1, 1, 4160), -1, dtype=torch.int64
        )
        coordinator._device_buffer_arange_i32 = torch.arange(4160, dtype=torch.int32)
        req = SimpleNamespace(
            req_pool_idx=0,
            rid="padded-arange",
            kv=SimpleNamespace(kv_allocated_len=128),
        )

        coordinator.alloc_device_buffer(req)

        self.assertTrue(
            torch.equal(
                coordinator.req_device_buffer_tokens[0, 0, :4096],
                torch.arange(4096, dtype=torch.int32),
            )
        )

    def test_mtp_demand_request_reset_is_request_local(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = True
        coordinator.mtp_demand_cache_tags = torch.full((2, 3, 8), 77, dtype=torch.int64)
        coordinator.mtp_demand_decode_calls = torch.tensor(
            [11, 22, 33], dtype=torch.int32
        )

        coordinator._reset_mtp_demand_request_state(1)

        self.assertEqual(
            coordinator.mtp_demand_cache_tags[:, 1].count_nonzero().item(), 0
        )
        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_cache_tags[:, 0],
                torch.full((2, 8), 77, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_cache_tags[:, 2],
                torch.full((2, 8), 77, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_decode_calls,
                torch.tensor([11, 0, 33], dtype=torch.int32),
            )
        )

    def test_mtp_demand_epoch_advances_once_per_request(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = True
        coordinator.mtp_demand_decode_calls = torch.tensor(
            [10, 20, 30, 40], dtype=torch.int32
        )

        coordinator.advance_mtp_demand_epoch(torch.tensor([3, 1]))

        self.assertTrue(
            torch.equal(
                coordinator.mtp_demand_decode_calls,
                torch.tensor([10, 21, 30, 41], dtype=torch.int32),
            )
        )

    def test_mtp_demand_commit_waits_for_draft_extend_then_retires_verify_window(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        class MappingWithAsyncClear:
            def __init__(self, tensor):
                self.tensor = tensor
                self.index_fill_calls = []

            def __getitem__(self, key):
                return self.tensor[key]

            def __setitem__(self, key, value):
                if isinstance(value, int) and value == 0:
                    raise AssertionError("mapping clear must not copy a CPU scalar")
                self.tensor[key] = value

            def index_fill_(self, dim, index, value):
                self.index_fill_calls.append((dim, index.clone(), value))
                return self.tensor.index_fill_(dim, index, value)

        backups = []
        transfers = []
        mapping_tensor = torch.zeros(32, dtype=torch.int64)
        mapping_tensor[10:14] = torch.tensor([100, 101, 102, 103])
        mapping = MappingWithAsyncClear(mapping_tensor)
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = True
        coordinator._pending_mtp_demand_commit = None
        coordinator.device_buffer_size = 4096
        coordinator.req_to_host_pool = torch.full((2, 6000), -1, dtype=torch.int64)
        coordinator.req_to_host_pool[1, 5000:5004] = torch.tensor([200, 201, 202, 203])
        coordinator.req_to_device_buffer = torch.zeros((2, 4160), dtype=torch.int64)
        coordinator.req_to_device_buffer[1, 4096] = 700
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 2, 4160), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 2, 4160), -1, dtype=torch.int32
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_hisparse_device_index_mapping=mapping
        )
        coordinator.mem_pool_device = SimpleNamespace(
            transfer_values_on_device=lambda **kwargs: transfers.append(kwargs)
        )
        coordinator._backup_device_locs_to_host = (
            lambda host_locs, device_locs, accept_index: backups.append(
                (host_locs.clone(), device_locs.clone(), accept_index.clone())
            )
        )

        coordinator.finalize_accepted_tokens_spec_v2(
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([5000]),
            verify_cache_locs=torch.tensor([10, 11, 12, 13]),
            accept_index=torch.tensor([[0, 1, 2, -1]]),
        )

        self.assertEqual(backups, [])
        self.assertIsNotNone(coordinator._pending_mtp_demand_commit)

        coordinator.finish_pending_mtp_demand_commit()

        self.assertEqual(len(backups), 1)
        self.assertTrue(torch.equal(backups[0][0], torch.tensor([200, 201, 202, 203])))
        self.assertTrue(torch.equal(backups[0][1], torch.tensor([100, 101, 102, 103])))
        self.assertTrue(
            torch.equal(backups[0][2], torch.tensor([0, 1, 2, -1], dtype=torch.int32))
        )
        self.assertEqual(len(transfers), 1)
        self.assertTrue(torch.equal(transfers[0]["dst_indices"], torch.tensor([700])))
        self.assertTrue(torch.equal(transfers[0]["src_indices"], torch.tensor([102])))
        self.assertEqual(len(mapping.index_fill_calls), 1)
        self.assertEqual(mapping.index_fill_calls[0][0], 0)
        self.assertTrue(
            torch.equal(mapping.index_fill_calls[0][1], torch.tensor([10, 11, 12, 13]))
        )
        self.assertEqual(mapping.index_fill_calls[0][2], 0)
        self.assertTrue(torch.equal(mapping[10:14], torch.tensor([0, 0, 700, 0])))
        self.assertTrue(
            torch.equal(
                coordinator.req_device_buffer_tokens[:, 1, 4096],
                torch.tensor([5002, 5002], dtype=torch.int32),
            )
        )
        self.assertIsNone(coordinator._pending_mtp_demand_commit)

    def test_native_mtp_finalize_commits_accepted_verify_rows(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = False
        calls = []
        coordinator.finalize_accepted_tokens = lambda **kwargs: calls.append(kwargs)

        req_pool_indices = torch.tensor([2, 5], dtype=torch.int64)
        seq_lens = torch.tensor([5000, 7000], dtype=torch.int64)
        verify_cache_locs = torch.tensor(
            [10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int64
        )
        accept_index = torch.tensor(
            [[0, 1, -1, -1], [4, -1, -1, -1]], dtype=torch.int32
        )

        coordinator.finalize_accepted_tokens_spec_v2(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            verify_cache_locs=verify_cache_locs,
            accept_index=accept_index,
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["req_pool_indices"], req_pool_indices)
        self.assertTrue(
            torch.equal(call["accepted_cache_locs"], torch.tensor([10, 11, 20]))
        )
        self.assertIs(call["draft_cache_locs"], verify_cache_locs)
        self.assertTrue(torch.equal(call["num_correct_drafts"], torch.tensor([1, 0])))
        self.assertTrue(
            torch.equal(call["num_correct_drafts_cpu"], torch.tensor([1, 0]))
        )
        self.assertTrue(
            torch.equal(
                call["accepted_token_positions"],
                torch.tensor([5000, 5001, 7000]),
            )
        )


class TestHiSparseMTPNative(unittest.TestCase):
    def test_union_target_verify_selects_raw_topk_route(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        coordinator = SimpleNamespace(mtp_union_enabled=True)
        self.assertTrue(
            dsa_backend._use_hisparse_mtp_union(
                coordinator=coordinator,
                forward_mode=ForwardMode.TARGET_VERIFY,
                dsa_impl="flashmla_kv",
            )
        )
        self.assertFalse(
            dsa_backend._use_hisparse_mtp_union(
                coordinator=coordinator,
                forward_mode=ForwardMode.DECODE,
                dsa_impl="flashmla_kv",
            )
        )
        self.assertFalse(
            dsa_backend._use_hisparse_mtp_union(
                coordinator=coordinator,
                forward_mode=ForwardMode.TARGET_VERIFY,
                dsa_impl="tilelang",
            )
        )

    def test_native_target_verify_keeps_request_relative_topk(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
            DSATopKBackend,
            TopkTransformMethod,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_fused_topk = True
        backend.hisparse_coordinator = SimpleNamespace(mtp_demand_buffer_enabled=False)
        backend.forward_metadata = SimpleNamespace(
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend.get_topk_transform_method = MagicMock(
            return_value=TopkTransformMethod.PAGED
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        metadata = backend.get_indexer_metadata(layer_id=0, forward_batch=forward_batch)

        self.assertTrue(metadata.force_unfused_topk)

    def test_mtp_keeps_draft_kv_device_resident(self):
        from sglang.srt.model_executor import model_runner

        server_args = SimpleNamespace(
            enable_hisparse=True,
            device="cuda",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="flashmla_kv",
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            speculative_attention_mode="prefill",
            pp_size=1,
            attn_cp_size=1,
            enable_pdmux=False,
            disaggregation_mode="null",
        )

        self.assertTrue(
            model_runner.resolve_hisparse_for_runner(server_args, is_draft_worker=False)
        )
        self.assertFalse(
            model_runner.resolve_hisparse_for_runner(server_args, is_draft_worker=True)
        )

    def test_target_verify_runs_native_multistep_swap(self):
        from sglang.srt.layers.attention import dsa_backend

        logical_topk = torch.arange(8 * 16, dtype=torch.int32).view(8, 16)
        physical_storage = torch.arange(2 * 4 * 32, dtype=torch.int32).view(2, 4, 32)
        physical_topk_3d = physical_storage[..., ::2]
        swap = MagicMock(return_value=physical_topk_3d)
        coordinator = SimpleNamespace(swap_in_selected_pages_mtp=swap)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64)
        )
        metadata = SimpleNamespace(
            dsa_seqlens_expanded=torch.tensor(
                [129, 130, 131, 132, 257, 258, 259, 260], dtype=torch.int32
            )
        )

        actual = dsa_backend._prepare_hisparse_mtp_native(
            coordinator=coordinator,
            forward_batch=forward_batch,
            metadata=metadata,
            relative_topk=logical_topk,
            layer_id=12,
            num_steps=4,
        )

        self.assertTrue(torch.equal(actual, physical_topk_3d.reshape(8, 16)))
        call = swap.call_args.kwargs
        self.assertEqual(call["top_k_result"].shape, (2, 4, 16))
        self.assertEqual(call["layer_id"], 12)

    def test_verify_preparation_delegates_four_rows(self):
        from sglang.srt.speculative import eagle_utils

        calls = []
        seq_lens = torch.tensor([128, 4094], dtype=torch.int64)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            hisparse_coordinator=SimpleNamespace(
                prepare_verify_slots_spec_v2=lambda **kwargs: calls.append(kwargs)
            ),
            reqs=[
                SimpleNamespace(kv_committed_len=128),
                SimpleNamespace(kv_committed_len=4094),
            ],
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([3, 7], dtype=torch.int64),
            out_cache_loc=torch.tensor(
                [101, 102, 103, 104, 201, 202, 203, 204], dtype=torch.int64
            ),
            seq_lens=seq_lens,
        )

        eagle_utils._prepare_hisparse_mtp_verify_slots(batch, draft_token_num=4)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIs(call["verify_cache_locs"], batch.out_cache_loc)
        self.assertEqual(call["num_tokens_per_req"], 4)
        self.assertIs(call["start_positions"], seq_lens)

    def test_native_finalize_commits_accepted_rows(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.mtp_demand_buffer_enabled = False
        calls = []
        coordinator.finalize_accepted_tokens = lambda **kwargs: calls.append(kwargs)
        verify_cache_locs = torch.tensor(
            [10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int64
        )

        coordinator.finalize_accepted_tokens_spec_v2(
            req_pool_indices=torch.tensor([2, 5], dtype=torch.int64),
            seq_lens=torch.tensor([5000, 7000], dtype=torch.int64),
            verify_cache_locs=verify_cache_locs,
            accept_index=torch.tensor(
                [[0, 1, -1, -1], [4, -1, -1, -1]], dtype=torch.int32
            ),
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertTrue(
            torch.equal(call["accepted_cache_locs"], torch.tensor([10, 11, 20]))
        )
        self.assertTrue(
            torch.equal(
                call["accepted_token_positions"], torch.tensor([5000, 5001, 7000])
            )
        )


class TestHiSparseUnit(unittest.TestCase):
    """Test class that builds a minimal HiSparse component stack."""

    # ==================================================================
    # Fixture
    # ==================================================================

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiSparse tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        if is_hip():
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            global_page_size = 64 if aiter_can_use_preshuffle_paged_mqa() else 1
        else:
            global_page_size = PAGE_SIZE

        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        cls.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )

        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        cls.page_size = global_page_size
        cls.coordinator = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            mtp_num_rows=4,
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Reset shared allocator / coordinator state so tests are isolated.

        Without this, a mid-test assertion failure skips cleanup and leaks
        resources, causing unrelated failures in later tests.
        """
        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.coordinator.mem_pool_host.clear()
        # Reset per-request coordinator bookkeeping
        self.coordinator.req_to_device_buffer.zero_()
        self.coordinator.req_to_mtp_staging.zero_()
        self.coordinator.req_device_buffer_size.zero_()
        self.coordinator.req_to_host_pool.fill_(-1)
        self.coordinator.req_to_host_pool_allocated_len.zero_()
        self.coordinator.req_device_buffer_tokens.fill_(-1)
        self.coordinator.req_device_buffer_token_locs.fill_(-1)
        self.coordinator.lru_slots[:] = self.coordinator._lru_init.view(1, 1, -1)
        self.coordinator.ack_staging_queue.clear()
        self.coordinator._has_pending_backup = False
        for i in range(len(self.coordinator._skip_first_backup)):
            self.coordinator._skip_first_backup[i] = False

    # ==================================================================
    # Low-level helpers
    # ==================================================================

    def _alloc_req_slot(self, req):
        """Allocate a req_pool_idx for the request."""
        indices = self.req_to_token_pool.alloc([req])
        self.assertIsNotNone(indices, "Failed to allocate req pool slot")
        return req.kv.req_pool_idx

    def _free_req_slot(self, req):
        """Free the req_pool_idx."""
        if req.kv.req_pool_idx is not None:
            self.req_to_token_pool.free(req)

    def _alloc_kv(self, req, fill_len, *, logical_only=False):
        """Allocate KV indices, write req_to_token_pool, update req fields.
        If logical_only=True, uses alloc_logical_only (PD-separated path).
        Returns kv_loc tensor."""
        device = self.allocator.device
        alloc_fn = (
            self.allocator.alloc_logical_only
            if logical_only
            else self.allocator.alloc_extend
        )
        kv_loc = alloc_fn(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc, "KV alloc failed")
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, len(kv_loc))), kv_loc
        )
        req.kv.kv_allocated_len = fill_len
        req.kv.kv_committed_len = fill_len
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.extend_range = Range(0, fill_len)
        return kv_loc

    # ==================================================================
    # Mid-level helpers
    # ==================================================================

    @staticmethod
    def _kv_pattern(layer_id, token_id):
        """Deterministic KV value for (layer, token) — used by write & verify."""
        v = (layer_id * 10000 + token_id + 1) * 0.001
        return float(torch.tensor(v, dtype=torch.bfloat16))

    def _write_device_patterns(self, kv_loc, fill_len):
        """Write distinguishable patterns into device KV buffer for all layers.

        kv_loc contains *logical* indices; we must translate them to hisparse
        device indices before indexing kv_buffer (which is sized for the
        hisparse pool, not the larger logical space).
        """
        hisparse_locs = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                self.device_pool.kv_buffer[lid][hisparse_locs[i]] = self._kv_pattern(
                    lid, i
                )

    def _populate_host_pool(self, req, fill_len):
        """Allocate host slots, write known patterns, register in coordinator.
        Returns host_indices (cuda tensor)."""
        host_pool = self.coordinator.mem_pool_host
        host_indices = host_pool.alloc(fill_len)
        self.assertIsNotNone(host_indices, "Host alloc failed")
        host_indices = host_indices.to(device="cuda")
        self.coordinator.req_to_host_pool[req.kv.req_pool_idx, :fill_len] = host_indices
        self.coordinator.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = fill_len
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                host_pool.kv_buffer[lid][host_indices[i]] = self._kv_pattern(lid, i)
        return host_indices

    def _build_topk_tokens(self, fill_len, *, include_newest=False):
        """Build a 1-D [TOP_K] int32 cuda tensor of token positions.

        If include_newest=True, fill_len-1 is guaranteed as the last valid slot.
        Pads with -1 when fill_len (or fill_len-1) < TOP_K.

        For long-sequence tests (fill_len > DEVICE_BUFFER_SIZE) where the
        "newest token" reserved slot is not populated (it requires an actual
        decode step + map_last_loc_to_buffer), callers should pass
        ``fill_len - 1`` as the effective pool size so position fill_len-1 is
        never randomly selected.
        """
        n = min(fill_len, TOP_K)
        if include_newest and n > 1:
            tokens = torch.randperm(fill_len - 1, device="cuda")[: n - 1].to(
                torch.int32
            )
            tokens = torch.cat(
                [tokens, torch.tensor([fill_len - 1], dtype=torch.int32, device="cuda")]
            )
        else:
            tokens = torch.randperm(fill_len, device="cuda")[:n].to(torch.int32)
        if n < TOP_K:
            pad = torch.full((TOP_K - n,), -1, dtype=torch.int32, device="cuda")
            tokens = torch.cat([tokens, pad])
        return tokens

    def _make_batch_tensors(self, reqs, fill_lens):
        """Build (req_pool_indices [int64], seq_lens [int32]) on cuda."""
        rpi = torch.tensor(
            [r.kv.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        sls = torch.tensor(fill_lens, dtype=torch.int32, device="cuda")
        return rpi, sls

    def _assert_kv_correct(self, locs_row, tokens_row, layer_id, count, msg=""):
        """Assert device KV data at *locs_row[:count]* matches the written
        pattern for the corresponding *tokens_row[:count]* positions."""
        for i in range(count):
            tok = int(tokens_row[i].item())
            if tok < 0:
                continue
            expected = self._kv_pattern(layer_id, tok)
            actual = self.device_pool.kv_buffer[layer_id][locs_row[i].long()]
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    torch.full_like(actual.float(), expected),
                    atol=1e-2,
                ),
                f"{msg}layer {layer_id}, token {tok}: KV data mismatch",
            )

    def _assert_matches_naive(self, rpi, sls, batch, kernel_locs, layer_id, msg=""):
        """Assert kernel swap_in KV data matches naive_load_topk KV data."""
        naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, layer_id)
        for b in range(batch.shape[0]):
            for i in range(TOP_K):
                if batch[b, i] < 0:
                    continue
                naive_data = self.device_pool.kv_buffer[layer_id][
                    naive_locs[b, i].long()
                ]
                kernel_data = self.device_pool.kv_buffer[layer_id][
                    kernel_locs[b, i].long()
                ]
                self.assertTrue(
                    torch.allclose(naive_data.float(), kernel_data.float(), atol=1e-2),
                    f"{msg}layer {layer_id}, b{b} idx {i}: naive != kernel",
                )

    def _swap_in_selected_pages(
        self,
        rpi: torch.Tensor,
        sls: torch.Tensor,
        batch: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Wrapper that sets num_real_reqs before calling swap_in_selected_pages.

        In production, model_runner sets num_real_reqs before each forward
        pass.  Tests must replicate that to get correct kernel behaviour.
        """
        self.coordinator.num_real_reqs[0] = rpi.shape[0]
        return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id)

    def _cleanup_req(self, req, kv_loc, *, logical_only=False):
        """request_finished -> free KV -> free req slot."""
        self.coordinator.request_finished(req)
        if logical_only:
            self.allocator.logical_attn_allocator.free(kv_loc)
        else:
            self.allocator.free(kv_loc)
        self._free_req_slot(req)

    def _get_initial_sizes(self):
        """Snapshot allocator available sizes."""
        return (
            self.allocator.logical_attn_allocator.available_size(),
            self.allocator.hisparse_attn_allocator.available_size(),
            self.coordinator.mem_pool_host.available_size(),
        )

    def _assert_sizes_restored(self, initial_sizes, msg=""):
        """Assert allocator sizes match the snapshot."""
        logical, hisparse, host = self._get_initial_sizes()
        self.assertEqual(logical, initial_sizes[0], f"Logical leak {msg}")
        self.assertEqual(hisparse, initial_sizes[1], f"HiSparse leak {msg}")
        self.assertEqual(host, initial_sizes[2], f"Host leak {msg}")

    # ==================================================================
    # Test: Kernel correctness — short sequence (fast path)
    # ==================================================================
    def test_kernel_correctness_short_seq(self):
        """Short seq (len <= device_buffer_size): kernel fast path returns
        device buffer locs, matching naive_load_topk."""
        initial = self._get_initial_sizes()
        req = _make_req("short-seq", list(range(self.page_size)))
        self._alloc_req_slot(req)

        fill_len = self.page_size
        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.alloc_device_buffer(req)

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            valid = batch[0] >= 0
            self.assertTrue(
                torch.equal(naive_locs[0][valid].cpu(), kernel_locs[0][valid].cpu()),
                f"Layer {lid}: kernel locs != naive oracle",
            )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "short_seq")

    # ==================================================================
    # Test: Kernel correctness — long sequence (cache miss + host DMA)
    # ==================================================================
    def test_kernel_correctness_long_seq(self):
        """Long seq (len > device_buffer_size): kernel loads from host,
        matching naive_load_topk for data correctness."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("long-seq", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        # Pass fill_len-1 so position fill_len-1 ("newest token") is never
        # randomly selected — its reserved device-buffer slot is only valid
        # after map_last_loc_to_buffer in a real decode step.
        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            self.assertTrue(torch.all(naive_locs[0, :TOP_K] >= 0))
            self.assertTrue(torch.all(kernel_locs[0, :TOP_K] >= 0))
            # Verify both return correct KV data independently
            self._assert_kv_correct(naive_locs[0], tokens, lid, TOP_K, msg="Naive: ")
            self._assert_kv_correct(kernel_locs[0], tokens, lid, TOP_K, msg="Kernel: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "long_seq")

    # ==================================================================
    # Test: Kernel LRU replacement across multiple decode steps
    # ==================================================================
    def test_kernel_lru_replacement(self):
        """Multi-step swap-in: second call hits cached tokens, only
        evicts/loads new misses."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("lru-test", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        rpi, sls = self._make_batch_tensors([req], [fill_len])

        # Step 1: load the first TOP_K positions from host (no newest token —
        # the reserved slot is only valid after map_last_loc_to_buffer which is
        # called during an actual decode step, not modelled here).
        tokens_s1 = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        locs1 = self._swap_in_selected_pages(
            rpi, sls, tokens_s1.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs1[0, :TOP_K] >= 0))

        # Step 2: half overlap (hit) + half new (miss).
        # Choose new tokens from a range safely below fill_len.
        half = TOP_K // 2
        new_start = TOP_K  # first position not in step-1
        tokens_s2 = torch.cat(
            [
                tokens_s1[:half],  # hits
                torch.arange(
                    new_start, new_start + half, dtype=torch.int32, device="cuda"
                ),  # misses
            ]
        )
        locs2 = self._swap_in_selected_pages(
            rpi, sls, tokens_s2.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs2[0, :TOP_K] >= 0))

        # Verify repeated (hit) tokens still have correct KV data
        self._assert_kv_correct(
            locs2[0], tokens_s2, layer_id=0, count=half, msg="LRU hit: "
        )
        # Also verify new (miss) tokens loaded correctly
        self._assert_kv_correct(
            locs2[0, half:],
            tokens_s2[half:],
            layer_id=0,
            count=half,
            msg="LRU miss: ",
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "lru_replacement")

    def test_native_mtp_multistep_swap_preserves_cross_step_rows(self):
        """Later MTP rows must not overwrite earlier returned page tables."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 8
        req = _make_req("native-mtp-swap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        # Four rows share 128 tokens and contribute 128 unique tokens each:
        # 640 distinct rows in total, which exceeds the 512-row native hot
        # slice. Returning four physical page tables is only correct if later
        # steps cannot overwrite rows still referenced by earlier steps.
        common = torch.arange(128, dtype=torch.int32, device="cuda")
        rows = []
        for step in range(4):
            unique_start = 128 + step * 128
            rows.append(
                torch.cat(
                    (
                        common,
                        torch.arange(
                            unique_start,
                            unique_start + 128,
                            dtype=torch.int32,
                            device="cuda",
                        ),
                    )
                )
            )
        topk = torch.stack(rows).unsqueeze(0)
        req_pool_indices = torch.tensor(
            [req.kv.req_pool_idx], dtype=torch.int64, device="cuda"
        )
        seq_lens = torch.full((4,), fill_len, dtype=torch.int32, device="cuda")
        self.coordinator.num_real_reqs.fill_(1)

        locs = self.coordinator.swap_in_selected_pages_mtp(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            top_k_result=topk,
            layer_id=0,
        )

        self.assertEqual(locs.shape, topk.shape)
        self.assertTrue(torch.all(locs >= 0))
        for step in range(4):
            self._assert_kv_correct(
                locs[0, step],
                topk[0, step],
                layer_id=0,
                count=TOP_K,
                msg=f"native MTP step {step}: ",
            )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "native_mtp_multistep")

    # ==================================================================
    # Test: Allocator alloc/free lifecycle
    # ==================================================================
    def test_allocator_alloc_free_cycle(self):
        """alloc_extend / alloc_device_buffer / free restores available_size."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size * 2

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), fill_len)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        need_size = min(
            ((fill_len + self.page_size - 1) // self.page_size) * self.page_size,
            DEVICE_BUFFER_SIZE,
        )
        buf_idx = self.allocator.alloc_device_buffer(kv_loc, need_size)
        self.assertIsNotNone(buf_idx)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")

        self.allocator.free_hisparse_indices(buf_idx)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "alloc_free_cycle")

    def test_allocator_reserve_failure_is_transactional(self):
        """A failed side-reserve claim must not publish a partial transition."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size * 2
        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)

        mapping_before = self.allocator.full_to_hisparse_device_index_mapping[
            kv_loc
        ].clone()
        available_before = self.allocator.hisparse_attn_allocator.available_size()
        claimed = self.allocator.alloc_device_buffer_with_reserve(
            kv_loc,
            need_size=self.page_size,
            reserve_size=available_before + 2 * self.page_size,
        )

        self.assertIsNone(claimed)
        self.assertTrue(
            torch.equal(
                mapping_before,
                self.allocator.full_to_hisparse_device_index_mapping[kv_loc],
            )
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(), available_before
        )

        claimed = self.allocator.alloc_device_buffer_with_reserve(
            kv_loc,
            need_size=self.page_size,
            reserve_size=self.page_size,
        )
        self.assertIsNotNone(claimed)
        hot, reserve = claimed
        self.assertTrue(
            torch.all(self.allocator.full_to_hisparse_device_index_mapping[kv_loc] == 0)
        )

        self.allocator.free_hisparse_indices(torch.cat([hot, reserve]))
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "transactional_reserve")

    def test_allocator_page_size_one_alloc_free_cycle(self):
        """alloc() maps logical to hisparse indices for ROCm page_size=1."""
        if self.page_size != 1:
            self.skipTest("page_size=1 alloc path is ROCm-specific")

        initial = self._get_initial_sizes()
        need_size = 16

        kv_loc = self.allocator.alloc(need_size)
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), need_size)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        self.allocator.free(kv_loc)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")
        self._assert_sizes_restored(initial, "page_size_one_alloc_free_cycle")

    def test_decode_remap_frees_stale_page_size_one_mapping(self):
        """map_last_loc_to_buffer frees the temporary alloc() hisparse slot."""
        if self.page_size != 1:
            self.skipTest("page_size=1 decode remap path is ROCm-specific")

        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = 2
        req = _make_req("decode-remap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        self.coordinator._skip_first_backup[req.kv.req_pool_idx] = True

        out_loc = self.allocator.alloc(1)
        self.assertIsNotNone(out_loc)
        stale_loc = self.allocator.full_to_hisparse_device_index_mapping[
            out_loc
        ].clone()
        self.assertTrue(torch.all(stale_loc > 0), "Temporary mapping should exist")

        seq_len = fill_len + 1
        self.req_to_token_pool.write((req.kv.req_pool_idx, fill_len), out_loc)
        req.kv.kv_allocated_len = seq_len
        req.kv.kv_committed_len = seq_len

        self.coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            out_cache_loc=out_loc,
            req_pool_indices=torch.tensor(
                [req.kv.req_pool_idx], dtype=torch.int64, device=device
            ),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([req.kv.req_pool_idx], dtype=torch.int64),
        )

        remapped_loc = self.allocator.full_to_hisparse_device_index_mapping[out_loc]
        self.assertTrue(torch.all(remapped_loc > 0), "Remapped loc should exist")
        self.assertFalse(
            torch.equal(stale_loc, remapped_loc),
            "Decode loc should move from temporary mapping to device buffer",
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - seq_len,
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(torch.cat([kv_loc, out_loc]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "decode_remap")

    # ==================================================================
    # Test: Staging (PD Colocate) path
    # ==================================================================
    def test_request_lifecycle_staging_path(self):
        """prefill -> staging DMA -> collect_ready -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("staging-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        self.assertTrue(req.hisparse_staging)

        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(len(ready), 1)
        self.assertFalse(req.hisparse_staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.kv.req_pool_idx])

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        valid_n = min(fill_len, TOP_K)
        self.assertTrue(torch.all(locs[0, :valid_n] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=valid_n, msg="Staging: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Staging: ")

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "staging_path")

    # ==================================================================
    # Test: Single-node staging host page allocation
    # ==================================================================
    def test_single_node_staging_allocates_paged_host_slots(self):
        """Single-node staging should allocate host slots at page granularity."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        req = _make_req("single-node-staging-pages", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(ready, [req])

        host_row = self.coordinator.req_to_host_pool[req.kv.req_pool_idx, :rounded_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(torch.unique(host_row).numel(), rounded_len)
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.kv.req_pool_idx]),
            rounded_len,
        )

        available_size = self.coordinator.mem_pool_host.available_size()
        next_host_index = self.coordinator.mem_pool_host.alloc_paged_token_slots(
            self.coordinator.req_to_host_pool,
            self.coordinator.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            fill_len,
            1,
        )
        # With page_size>1 the rounded-up staging allocation provides headroom,
        # so no new pages are needed.  With page_size=1 there is no headroom and
        # exactly one new page is allocated for the next token.
        expected_new_pages = 0 if fill_len < rounded_len else 1
        self.assertEqual(
            self.coordinator.mem_pool_host.available_size(),
            available_size - expected_new_pages,
        )
        self.assertTrue(torch.all(next_host_index >= 0))

        expected_total = rounded_len + expected_new_pages * self.page_size
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.kv.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.kv.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), expected_total)

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "single_node_staging_pages")

    # ==================================================================
    # Test: Direct-to-host (PD separated) path
    # ==================================================================
    def test_request_lifecycle_direct_path(self):
        """alloc_logical_only -> host write -> admit_direct -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("direct-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        self.assertFalse(req.staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.kv.req_pool_idx])
        buf_tokens = self.coordinator.req_device_buffer_tokens[
            :, req.kv.req_pool_idx, :DEVICE_BUFFER_SIZE
        ]
        self.assertTrue(torch.all(buf_tokens == -1))

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=TOP_K, msg="Direct: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Direct: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "direct_path")

    # ==================================================================
    # Test: PD decode prealloc host page allocation
    # ==================================================================
    def test_pd_decode_prealloc_hisparse_host_slots(self):
        """PD decode prealloc should allocate RDMA targets through the host pool."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        req = _make_req("pd-decode-prealloc", list(range(fill_len)))

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = self.req_to_token_pool
        queue.token_to_kv_pool_allocator = self.allocator
        queue.token_to_kv_pool = self.allocator.get_kvcache()
        queue.tree_cache = SimpleNamespace(
            evictable_size=lambda: 0,
            protected_size=lambda: 0,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=self.coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )

        host_indices = queue._pre_alloc(req)
        self.assertEqual(host_indices.numel(), fill_len)
        self.assertTrue(torch.all(host_indices >= 0))
        self.assertTrue(
            torch.equal(
                host_indices,
                self.coordinator.req_to_host_pool[req.kv.req_pool_idx, :fill_len],
            )
        )
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)

        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.kv.req_pool_idx]),
            rounded_len,
        )
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.kv.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.kv.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), rounded_len)

        kv_loc = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.kv.kv_allocated_len
        ].clone()
        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "pd_decode_prealloc_hisparse")

    # ==================================================================
    # Test: Batch multiple requests
    # ==================================================================
    def test_batch_multiple_requests(self):
        """Mix of short & long requests in batch: kernel correct + no leaks."""
        initial = self._get_initial_sizes()

        configs = [
            ("batch-short-0", self.page_size),
            ("batch-short-1", self.page_size),
            ("batch-long-0", DEVICE_BUFFER_SIZE + self.page_size),
            ("batch-long-1", DEVICE_BUFFER_SIZE + self.page_size * 2),
        ]

        reqs, kv_locs = [], []
        for rid, fl in configs:
            req = _make_req(rid, list(range(fl)))
            self._alloc_req_slot(req)
            is_long = fl > DEVICE_BUFFER_SIZE
            kv_loc = self._alloc_kv(req, fl, logical_only=is_long)
            if is_long:
                self._populate_host_pool(req, fl)
                self.coordinator.admit_request_direct(req)
            else:
                self._write_device_patterns(kv_loc, fl)
                self.coordinator.alloc_device_buffer(req)
            reqs.append(req)
            kv_locs.append(kv_loc)

        rpi, sls = self._make_batch_tensors(reqs, [c[1] for c in configs])
        top_k_batch = torch.stack(
            [
                # For long sequences pass fl-1 to exclude the "newest token" position
                # whose reserved device-buffer slot is not populated in unit tests.
                self._build_topk_tokens(fl - 1 if fl > DEVICE_BUFFER_SIZE else fl)
                for _, fl in configs
            ]
        )

        for lid in range(LAYER_NUM):
            locs = self._swap_in_selected_pages(rpi, sls, top_k_batch, lid)
            for i, (rid, fl) in enumerate(configs):
                vn = min(fl, TOP_K)
                self.assertTrue(
                    torch.all(locs[i, :vn] >= 0),
                    f"Req {rid}, layer {lid}: negative locs",
                )
                self._assert_kv_correct(
                    locs[i], top_k_batch[i], lid, vn, msg=f"{rid}: "
                )

        for i, req in enumerate(reqs):
            is_long = configs[i][1] > DEVICE_BUFFER_SIZE
            self._cleanup_req(req, kv_locs[i], logical_only=is_long)

        self._assert_sizes_restored(initial, "batch_multiple")


if __name__ == "__main__":
    unittest.main()
