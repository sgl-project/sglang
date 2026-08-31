import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.utils as srt_utils

fake_aiter_for_import = ModuleType("aiter")
fake_aiter_for_import.__path__ = []
fake_aiter_ops = ModuleType("aiter.ops")
fake_aiter_ops.__path__ = []
fake_aiter_triton = ModuleType("aiter.ops.triton")
fake_aiter_triton.__path__ = []
fake_aiter_quant = ModuleType("aiter.ops.triton.quant")
fake_aiter_quant.dynamic_mxfp4_quant = Mock()

fake_aiter_modules = {
    "aiter": fake_aiter_for_import,
    "aiter.ops": fake_aiter_ops,
    "aiter.ops.triton": fake_aiter_triton,
    "aiter.ops.triton.quant": fake_aiter_quant,
}
missing_module = object()
previous_aiter_modules = {
    name: sys.modules.get(name, missing_module) for name in fake_aiter_modules
}
sys.modules.update(fake_aiter_modules)
try:
    with (
        patch.dict(os.environ, {"SGLANG_USE_AITER": "0"}),
        patch.object(srt_utils, "is_hip", return_value=False),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(gcnArchName="gfx950", major=9, minor=5),
        ),
    ):
        import sglang.kernels.ops.attention.dsv4.aiter_fp4_indexer as aiter_fp4_indexer
        import sglang.kernels.ops.attention.dsv4.topk as topk_module
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.indexer import (
            C4IndexerBackendMixin,
            topk_transform_512_pytorch_vectorized,
        )
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.test.ci.ci_register import register_cpu_ci
finally:
    for module_name, previous_module in previous_aiter_modules.items():
        if previous_module is missing_module:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fake_flydsl():
    flydsl = ModuleType("aiter.ops.flydsl")
    flydsl.flydsl_pa_mqa_logits_fp4 = Mock()
    flydsl.flydsl_pa_mqa_logits_fp4_prefill = Mock()
    aiter = ModuleType("aiter")
    aiter.__path__ = []
    ops = ModuleType("aiter.ops")
    ops.__path__ = []
    aiter.ops = ops
    ops.flydsl = flydsl
    return flydsl, {
        "aiter": aiter,
        "aiter.ops": ops,
        "aiter.ops.flydsl": flydsl,
    }


class TestAITERFP4IndexerLogits(unittest.TestCase):
    def test_topk_wrapper_materializes_strided_logits(self):
        jit_module = SimpleNamespace(topk_transform=Mock())
        scores = torch.randn((2, 768), dtype=torch.float32)[:, :576]
        seq_lens = torch.tensor([576, 513], dtype=torch.int32)
        page_table = torch.arange(18, dtype=torch.int32).reshape(2, 9)
        out_indices = torch.empty((2, 512), dtype=torch.int32)

        self.assertFalse(scores.is_contiguous())
        with (
            patch.object(topk_module, "is_hip_runtime", return_value=False),
            patch.object(topk_module, "is_xpu", return_value=False),
            patch.object(topk_module, "_jit_topk_v1_module", return_value=jit_module),
        ):
            topk_module.topk_transform_512(
                scores, seq_lens, page_table, out_indices, 64
            )

        forwarded_scores = jit_module.topk_transform.call_args.args[0]
        self.assertTrue(forwarded_scores.is_contiguous())
        torch.testing.assert_close(forwarded_scores, scores)

    def test_decode_parallel_units_cap_graph_width_without_oversizing_eager(self):
        get_parallel_units = aiter_fp4_indexer._get_aiter_fp4_decode_parallel_unit_num

        self.assertEqual(get_parallel_units(32, 2048), 256)
        self.assertEqual(get_parallel_units(32, 262144), 1024)
        self.assertEqual(get_parallel_units(256, 8192), 1024)

    def _make_dispatch(self, *, mode, num_tokens, metadata_rows, batch_size):
        page_table = torch.arange(metadata_rows * 2, dtype=torch.int32).reshape(
            metadata_rows, 2
        )
        c4_seq_lens = torch.arange(11, 11 + metadata_rows, dtype=torch.int32)
        with envs.SGLANG_OPT_USE_TOPK_V2.override(False):
            indexer_metadata = PagedIndexerMetadata(
                page_size=256,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                uses_aiter_fp4_layout=True,
            )
        core_metadata = SimpleNamespace(
            positions=torch.arange(num_tokens, dtype=torch.int64),
            page_table=page_table,
            c4_sparse_page_indices=torch.full(
                (metadata_rows, 512), -1, dtype=torch.int32
            ),
        )

        q_fp4 = torch.empty((num_tokens, 64, 64), dtype=torch.float4_e2m1fn_x2)
        q_scale = torch.empty((num_tokens, 1, 4, 16, 4), dtype=torch.uint8)
        weight_storage = torch.arange(num_tokens * 64 * 2, dtype=torch.float32).to(
            torch.bfloat16
        )
        raw_weights = weight_storage.reshape(num_tokens, 128)[:, ::2]
        self.assertEqual(raw_weights.shape, (num_tokens, 64))
        self.assertFalse(raw_weights.is_contiguous())

        num_pages = metadata_rows * page_table.shape[1]
        k_payload = torch.empty((num_pages, 1, 4, 64, 16), dtype=torch.float4_e2m1fn_x2)
        k_scale = torch.empty((num_pages, 1, 4, 64), dtype=torch.uint8)
        token_pool = SimpleNamespace(
            get_index_k_fp4_payload_buffer=Mock(return_value=k_payload),
            get_index_k_fp4_scale_buffer=Mock(return_value=k_scale),
            get_index_k_with_scale_buffer=Mock(
                side_effect=AssertionError("combined accessor must not be used")
            ),
        )
        backend = C4IndexerBackendMixin.__new__(C4IndexerBackendMixin)
        backend.token_to_kv_pool = token_pool
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=indexer_metadata,
            core_metadata=core_metadata,
        )
        backend.debug_use_external_c4_sparse_indices = True
        backend._forward_prepare_normal = Mock(
            return_value=((q_fp4, q_scale), raw_weights)
        )
        backend._get_nonpaged_indexer_plan = Mock(
            side_effect=AssertionError("AITER FP4 must not use nonpaged DeepGEMM")
        )
        c4_indexer = SimpleNamespace(
            use_fp4_indexer=True,
            layer_id=7,
            weight_scale=0.03125,
        )
        forward_batch = SimpleNamespace(forward_mode=mode, batch_size=batch_size)
        return SimpleNamespace(
            backend=backend,
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            token_pool=token_pool,
            q_fp4=q_fp4,
            q_scale=q_scale,
            raw_weights=raw_weights,
            k_payload=k_payload,
            k_scale=k_scale,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
        )

    def _run_dispatch(self, case):
        flydsl, modules = _fake_flydsl()
        padded_width = max(4, (case.page_table.shape[1] + 3) // 4 * 4)
        output = torch.full(
            (case.q_fp4.shape[0], padded_width * 64),
            float("-inf"),
        )
        for row, seq_len in enumerate(case.c4_seq_lens[: case.q_fp4.shape[0]].tolist()):
            output[row, :seq_len] = row
        flydsl.flydsl_pa_mqa_logits_fp4.return_value = output
        flydsl.flydsl_pa_mqa_logits_fp4_prefill.return_value = output

        with patch.dict(sys.modules, modules):
            case.backend.forward_c4_indexer(
                x=torch.empty(case.q_fp4.shape[0], 1),
                q_lora=torch.empty(case.q_fp4.shape[0], 1),
                c4_indexer=case.c4_indexer,
                forward_batch=case.forward_batch,
            )

        case.token_pool.get_index_k_fp4_payload_buffer.assert_called_once_with(
            layer_id=7
        )
        case.token_pool.get_index_k_fp4_scale_buffer.assert_called_once_with(layer_id=7)
        case.token_pool.get_index_k_with_scale_buffer.assert_not_called()
        case.backend._get_nonpaged_indexer_plan.assert_not_called()
        return flydsl

    def _assert_common_call(self, call_args, case):
        args = call_args.args
        self.assertEqual(args[2].dtype, torch.uint8)
        self.assertEqual(args[2].data_ptr(), case.k_payload.data_ptr())
        self.assertEqual(args[2].shape, case.k_payload.shape)
        self.assertIs(args[3], case.k_scale)
        self.assertEqual(args[4].dtype, torch.int32)
        self.assertTrue(args[4].is_contiguous())
        self.assertEqual(args[5].dtype, torch.bfloat16)
        self.assertTrue(args[5].is_contiguous())
        torch.testing.assert_close(args[5], case.raw_weights)
        padded_width = max(4, (case.page_table.shape[1] + 3) // 4 * 4)
        self.assertEqual(args[4].shape, (case.q_fp4.shape[0], padded_width + 4))
        self.assertEqual(args[-1], padded_width * 64)
        self.assertEqual(
            call_args.kwargs,
            {
                "weight_scale": case.c4_indexer.weight_scale,
                "block_k": 256,
                "kv_block_size": 64,
                "num_warps": 4,
                **(
                    {"next_n": 1, "parallel_unit_num": None}
                    if case.forward_batch.forward_mode.is_decode()
                    else {"parallel_unit_num": 512}
                ),
            },
        )

    def test_decode_uses_exact_flydsl_call_and_split_cache(self):
        case = self._make_dispatch(
            mode=ForwardMode.DECODE,
            num_tokens=2,
            metadata_rows=3,
            batch_size=2,
        )
        flydsl = self._run_dispatch(case)

        flydsl.flydsl_pa_mqa_logits_fp4_prefill.assert_not_called()
        flydsl.flydsl_pa_mqa_logits_fp4.assert_called_once()
        call_args = flydsl.flydsl_pa_mqa_logits_fp4.call_args
        self._assert_common_call(call_args, case)
        args = call_args.args
        self.assertEqual(args[0].shape, (2, 1, 64, 64))
        self.assertEqual(args[0].dtype, torch.uint8)
        self.assertEqual(args[0].data_ptr(), case.q_fp4.data_ptr())
        self.assertEqual(args[1].shape, (2, 1, 1, 4, 16, 4))
        self.assertEqual(args[1].dtype, torch.uint8)
        expected_page_table = torch.zeros((2, 8), dtype=torch.int32)
        expected_page_table[:, :2] = case.page_table[:2]
        torch.testing.assert_close(args[4], expected_page_table)
        torch.testing.assert_close(args[6], case.c4_seq_lens[:2])

    def test_decode_forwards_cached_logits_metadata(self):
        case = self._make_dispatch(
            mode=ForwardMode.DECODE,
            num_tokens=2,
            metadata_rows=2,
            batch_size=2,
        )
        cached_metadata = object()
        cached_positions = torch.tensor([7, 11], dtype=torch.int64)
        case.backend.forward_metadata.aiter_fp4_logits_decode_metadata = cached_metadata
        case.backend.forward_metadata.aiter_fp4_q_positions = cached_positions

        with patch(
            "sglang.srt.layers.attention.dsv4.indexer.aiter_fp4_paged_mqa_logits",
            return_value=torch.empty((2, 128), dtype=torch.float32),
        ) as logits:
            case.backend.forward_c4_indexer(
                x=torch.empty(2, 1),
                q_lora=torch.empty(2, 1),
                c4_indexer=case.c4_indexer,
                forward_batch=case.forward_batch,
            )

        self.assertIs(logits.call_args.kwargs["decode_metadata"], cached_metadata)
        self.assertIs(
            case.backend._forward_prepare_normal.call_args.kwargs["positions"],
            cached_positions,
        )

    def test_extend_and_target_verify_use_exact_prefill_call(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                case = self._make_dispatch(
                    mode=mode,
                    num_tokens=3,
                    metadata_rows=2,
                    batch_size=1,
                )
                flydsl = self._run_dispatch(case)

                flydsl.flydsl_pa_mqa_logits_fp4.assert_not_called()
                flydsl.flydsl_pa_mqa_logits_fp4_prefill.assert_called_once()
                call_args = flydsl.flydsl_pa_mqa_logits_fp4_prefill.call_args
                self._assert_common_call(call_args, case)
                args = call_args.args
                self.assertEqual(args[0].dtype, torch.uint8)
                self.assertEqual(args[0].data_ptr(), case.q_fp4.data_ptr())
                self.assertEqual(args[0].shape, case.q_fp4.shape)
                self.assertIs(args[1], case.q_scale)
                self.assertEqual(args[0].shape, (3, 64, 64))
                self.assertEqual(args[1].shape, (3, 1, 4, 16, 4))
                expected_page_table = torch.zeros((3, 8), dtype=torch.int32)
                expected_page_table[:2, :2] = case.page_table
                torch.testing.assert_close(args[4], expected_page_table)
                torch.testing.assert_close(args[6], torch.arange(3, dtype=torch.int32))
                torch.testing.assert_close(args[7], torch.zeros(3, dtype=torch.int32))
                torch.testing.assert_close(
                    args[8],
                    torch.cat((case.c4_seq_lens, torch.zeros(1, dtype=torch.int32))),
                )
                logits = flydsl.flydsl_pa_mqa_logits_fp4_prefill.return_value
                self.assertTrue(torch.isneginf(logits[-1]).all())
                topk_output = torch.zeros((3, 512), dtype=torch.int32)
                topk_transform_512_pytorch_vectorized(
                    logits,
                    args[8],
                    args[4],
                    topk_output,
                    page_size=64,
                )
                torch.testing.assert_close(
                    topk_output[-1], torch.full((512,), -1, dtype=torch.int32)
                )

    def test_page_table_padding_preserves_rows_and_logical_output_width(self):
        num_tokens = 3
        q_fp4 = torch.empty((num_tokens, 64, 64), dtype=torch.float4_e2m1fn_x2)
        q_scale = torch.empty((num_tokens, 1, 4, 16, 4), dtype=torch.uint8)
        weights = torch.empty((num_tokens, 64), dtype=torch.bfloat16)

        for is_decode in (True, False):
            for page_table_width in (1, 2, 3, 4, 5):
                with self.subTest(
                    is_decode=is_decode, page_table_width=page_table_width
                ):
                    page_table = torch.arange(
                        1,
                        num_tokens * page_table_width + 1,
                        dtype=torch.int32,
                    ).reshape(num_tokens, page_table_width)
                    original_page_table = page_table.clone()
                    num_pages = int(page_table.max()) + 1
                    k_payload = torch.empty(
                        (num_pages, 1, 4, 64, 16),
                        dtype=torch.float4_e2m1fn_x2,
                    )
                    k_scale = torch.empty((num_pages, 1, 4, 64), dtype=torch.uint8)
                    c4_seq_lens = torch.full(
                        (num_tokens,), page_table_width * 64, dtype=torch.int32
                    )
                    padded_width = max(4, (page_table_width + 3) // 4 * 4)
                    padded_logits = torch.arange(
                        num_tokens * padded_width * 64, dtype=torch.float32
                    ).reshape(num_tokens, padded_width * 64)
                    flydsl, modules = _fake_flydsl()
                    selected_kernel = (
                        flydsl.flydsl_pa_mqa_logits_fp4
                        if is_decode
                        else flydsl.flydsl_pa_mqa_logits_fp4_prefill
                    )
                    selected_kernel.return_value = padded_logits

                    with patch.dict(sys.modules, modules):
                        logits = aiter_fp4_indexer.aiter_fp4_paged_mqa_logits(
                            q_fp4=q_fp4,
                            q_scale=q_scale,
                            k_payload=k_payload,
                            k_scale=k_scale,
                            weights=weights,
                            page_table=page_table,
                            c4_seq_lens=c4_seq_lens,
                            weight_scale=0.03125,
                            is_decode=is_decode,
                        )

                    padded_page_table = selected_kernel.call_args.args[4]
                    expected_page_table = torch.zeros(
                        (num_tokens, padded_width + 4), dtype=torch.int32
                    )
                    expected_page_table[:, :page_table_width] = page_table
                    self.assertEqual(
                        padded_page_table.shape, (num_tokens, padded_width + 4)
                    )
                    self.assertEqual(padded_page_table.dtype, torch.int32)
                    self.assertTrue(padded_page_table.is_contiguous())
                    torch.testing.assert_close(padded_page_table, expected_page_table)
                    self.assertEqual(
                        selected_kernel.call_args.args[-1], padded_width * 64
                    )
                    self.assertEqual(logits.shape, (num_tokens, page_table_width * 64))
                    self.assertEqual(logits.stride(), (padded_width * 64, 1))
                    self.assertEqual(logits.data_ptr(), padded_logits.data_ptr())
                    torch.testing.assert_close(
                        logits, padded_logits[:, : page_table_width * 64]
                    )
                    torch.testing.assert_close(page_table, original_page_table)

    def test_prefill_dispatch_reuses_cached_logits_metadata(self):
        case = self._make_dispatch(
            mode=ForwardMode.EXTEND,
            num_tokens=2,
            metadata_rows=2,
            batch_size=1,
        )
        cached_metadata = aiter_fp4_indexer.AiterFP4PagedMQAPrefillMetadata(
            padded_page_table=torch.tensor(
                [[0, 1, 0, 0, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int32,
            ),
            row_to_batch=torch.arange(2, dtype=torch.int32),
            local_starts=torch.zeros(2, dtype=torch.int32),
            cta_info=torch.empty((512, 6), dtype=torch.int32),
            n_ctas=512,
            out=torch.full((2, 256), float("-inf"), dtype=torch.float32),
            logical_max_seq_len=128,
        )
        case.backend.forward_metadata.aiter_fp4_logits_prefill_metadata = (
            cached_metadata
        )
        flydsl, modules = _fake_flydsl()
        flydsl.flydsl_pa_mqa_logits_fp4_prefill.return_value = cached_metadata.out

        with (
            patch.dict(sys.modules, modules),
            patch.object(aiter_fp4_indexer.torch, "arange") as arange,
            patch.object(aiter_fp4_indexer.torch, "zeros") as zeros,
            patch.object(aiter_fp4_indexer.torch, "full") as full,
        ):
            case.backend.forward_c4_indexer(
                x=torch.empty(2, 1),
                q_lora=torch.empty(2, 1),
                c4_indexer=case.c4_indexer,
                forward_batch=case.forward_batch,
            )

        arange.assert_not_called()
        zeros.assert_not_called()
        full.assert_not_called()
        call = flydsl.flydsl_pa_mqa_logits_fp4_prefill.call_args
        self.assertIs(call.args[4], cached_metadata.padded_page_table)
        self.assertIs(call.args[6], cached_metadata.row_to_batch)
        self.assertIs(call.args[7], cached_metadata.local_starts)
        self.assertIs(call.kwargs["out"], cached_metadata.out)
        self.assertIs(call.kwargs["cta_info"], cached_metadata.cta_info)
        self.assertEqual(call.kwargs["n_ctas"], cached_metadata.n_ctas)

    def test_target_verify_graph_refreshes_shared_prefill_metadata_per_init(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
            DSV4Metadata,
        )

        page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        c4_seq_lens = torch.tensor([63, 127], dtype=torch.int32)
        core_metadata = SimpleNamespace(
            c4_out_loc=torch.tensor([4, 8], dtype=torch.int64),
            positions=torch.tensor([7, 11], dtype=torch.int64),
        )
        indexer_metadata = SimpleNamespace(
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
        )
        metadata = DSV4Metadata(
            core_attn_metadata=core_metadata,
            indexer_metadata=indexer_metadata,
            c4_compress_metadata=object(),
        )
        backend = DeepseekV4HipRadixBackend.__new__(DeepseekV4HipRadixBackend)
        backend.enable_deepseek_v4_fp4_indexer = True
        backend.forward_metadata = metadata
        backend.aiter_fp4_max_position = 4096
        backend.device = "cpu"
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc=None,
        )
        first_prefill_metadata = object()
        second_prefill_metadata = object()

        with (
            patch(
                "sglang.kernels.ops.attention.dsv4.aiter_fp4_indexer."
                "prepare_aiter_k_indexer_fp4_cache_write_metadata",
                return_value=object(),
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.aiter_fp4_indexer."
                "prepare_aiter_fp4_paged_mqa_decode_metadata"
            ) as prepare_decode,
            patch(
                "sglang.kernels.ops.attention.dsv4.aiter_fp4_indexer."
                "prepare_aiter_fp4_paged_mqa_prefill_metadata",
                side_effect=[first_prefill_metadata, second_prefill_metadata],
            ) as prepare_prefill,
        ):
            backend.init_forward_metadata_in_graph(forward_batch)
            self.assertIs(
                metadata.aiter_fp4_logits_prefill_metadata,
                first_prefill_metadata,
            )
            backend.init_forward_metadata_in_graph(forward_batch)
            self.assertIs(
                metadata.aiter_fp4_logits_prefill_metadata,
                second_prefill_metadata,
            )

        prepare_decode.assert_not_called()
        self.assertEqual(prepare_prefill.call_count, 2)
        for call in prepare_prefill.call_args_list:
            self.assertIs(call.kwargs["page_table"], page_table)
            self.assertIs(call.kwargs["c4_seq_lens"], c4_seq_lens)

    def test_target_verify_layers_reuse_graph_prefill_metadata(self):
        case = self._make_dispatch(
            mode=ForwardMode.TARGET_VERIFY,
            num_tokens=2,
            metadata_rows=2,
            batch_size=1,
        )
        graph_prefill_metadata = object()
        case.backend.forward_metadata.aiter_fp4_logits_prefill_metadata = (
            graph_prefill_metadata
        )

        with (
            patch(
                "sglang.srt.layers.attention.dsv4.indexer."
                "prepare_aiter_fp4_paged_mqa_prefill_metadata"
            ) as prepare_prefill,
            patch(
                "sglang.srt.layers.attention.dsv4.indexer."
                "aiter_fp4_paged_mqa_logits",
                return_value=torch.empty((2, 128), dtype=torch.float32),
            ) as logits,
        ):
            for layer_id in (7, 11):
                case.c4_indexer.layer_id = layer_id
                case.backend.forward_c4_indexer(
                    x=torch.empty(2, 1),
                    q_lora=torch.empty(2, 1),
                    c4_indexer=case.c4_indexer,
                    forward_batch=case.forward_batch,
                )

        prepare_prefill.assert_not_called()
        self.assertEqual(logits.call_count, 2)
        for call in logits.call_args_list:
            self.assertIs(call.kwargs["prefill_metadata"], graph_prefill_metadata)

    def test_prefill_cache_is_built_once_after_row_reindex(self):
        case = self._make_dispatch(
            mode=ForwardMode.EXTEND,
            num_tokens=1,
            metadata_rows=2,
            batch_size=1,
        )
        reindexed_page_table = case.page_table[1:].contiguous()
        reindexed_seq_lens = case.c4_seq_lens[1:].contiguous()
        case.backend.forward_metadata.indexer_metadata.page_table = reindexed_page_table
        case.backend.forward_metadata.indexer_metadata.c4_seq_lens = reindexed_seq_lens
        case.backend.forward_metadata.core_metadata.page_table = reindexed_page_table
        case.backend.forward_metadata.aiter_fp4_logits_prefill_cache_enabled = True
        case.backend.forward_metadata.aiter_fp4_q_positions = None
        cached_metadata = object()

        with (
            patch(
                "sglang.srt.layers.attention.dsv4.indexer."
                "prepare_aiter_fp4_paged_mqa_prefill_metadata",
                return_value=cached_metadata,
            ) as prepare,
            patch(
                "sglang.srt.layers.attention.dsv4.indexer."
                "aiter_fp4_paged_mqa_logits",
                return_value=torch.empty((1, 128), dtype=torch.float32),
            ) as logits,
        ):
            for _ in range(2):
                case.backend.forward_c4_indexer(
                    x=torch.empty(1, 1),
                    q_lora=torch.empty(1, 1),
                    c4_indexer=case.c4_indexer,
                    forward_batch=case.forward_batch,
                )

        prepare.assert_called_once_with(
            page_table=reindexed_page_table,
            c4_seq_lens=reindexed_seq_lens,
        )
        self.assertIs(
            case.backend.forward_metadata.aiter_fp4_logits_prefill_metadata,
            cached_metadata,
        )
        self.assertIs(
            case.backend._forward_prepare_normal.call_args.kwargs["positions"],
            case.backend.forward_metadata.aiter_fp4_q_positions,
        )
        self.assertEqual(logits.call_count, 2)
        for call in logits.call_args_list:
            self.assertIs(call.kwargs["prefill_metadata"], cached_metadata)

    def test_cuda_fp4_dispatch_remains_deep_gemm_combined(self):
        num_tokens = 3
        page_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        indexer_metadata = PagedIndexerMetadata.__new__(PagedIndexerMetadata)
        indexer_metadata.page_size = 256
        indexer_metadata.page_table = page_table
        indexer_metadata.c4_seq_lens = torch.tensor([11, 12], dtype=torch.int32)
        indexer_metadata.force_deep_gemm_metadata = False
        indexer_metadata.use_prefill_cuda_graph = False
        indexer_metadata.uses_aiter_fp4_layout = False
        indexer_metadata.deep_gemm_metadata = object()
        indexer_metadata.topk_metadata = torch.empty(0)
        indexer_metadata.nonpaged_plan = None
        core_metadata = SimpleNamespace(
            positions=torch.arange(num_tokens, dtype=torch.int64),
            page_table=page_table,
            c4_sparse_page_indices=torch.full((num_tokens, 512), -1, dtype=torch.int32),
        )
        q_fp4 = torch.empty((num_tokens, 64, 64), dtype=torch.float4_e2m1fn_x2)
        q_scale = torch.empty((num_tokens, 4), dtype=torch.uint8)
        raw_weights = torch.randn(num_tokens, 64, 1, dtype=torch.bfloat16)
        combined = torch.empty((3, 64 * 68), dtype=torch.uint8)
        token_pool = SimpleNamespace(
            get_index_k_with_scale_buffer=Mock(return_value=combined),
            get_index_k_fp4_payload_buffer=Mock(),
            get_index_k_fp4_scale_buffer=Mock(),
        )
        backend = C4IndexerBackendMixin.__new__(C4IndexerBackendMixin)
        backend.token_to_kv_pool = token_pool
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=indexer_metadata,
            core_metadata=core_metadata,
        )
        backend.debug_use_external_c4_sparse_indices = True
        backend._forward_prepare_normal = Mock(
            return_value=((q_fp4, q_scale), raw_weights)
        )
        backend._get_nonpaged_indexer_plan = Mock(return_value=None)
        c4_indexer = SimpleNamespace(
            use_fp4_indexer=True, layer_id=7, weight_scale=0.03125, index_topk=512
        )
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_paged_mqa_logits = Mock(
            return_value=torch.empty((num_tokens, 128))
        )

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            backend.forward_c4_indexer(
                x=torch.empty(num_tokens, 1),
                q_lora=torch.empty(num_tokens, 1),
                c4_indexer=c4_indexer,
                forward_batch=SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND, batch_size=1
                ),
            )

        token_pool.get_index_k_with_scale_buffer.assert_called_once_with(layer_id=7)
        token_pool.get_index_k_fp4_payload_buffer.assert_not_called()
        token_pool.get_index_k_fp4_scale_buffer.assert_not_called()
        deep_gemm.fp8_fp4_paged_mqa_logits.assert_called_once()
        args = deep_gemm.fp8_fp4_paged_mqa_logits.call_args.args
        self.assertEqual(args[0][0].shape, (num_tokens, 1, 64, 64))
        self.assertEqual(args[0][1].shape, (num_tokens, 1, 4))
        self.assertEqual(args[1].shape, (3, 64, 1, 68))
        self.assertEqual(args[2].dtype, torch.float32)
        torch.testing.assert_close(
            args[3], torch.tensor([[11], [12], [1]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            args[4], torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int32)
        )
        self.assertIs(args[5], indexer_metadata.deep_gemm_metadata)
        self.assertEqual(args[6:], (128, False))


if __name__ == "__main__":
    unittest.main()
