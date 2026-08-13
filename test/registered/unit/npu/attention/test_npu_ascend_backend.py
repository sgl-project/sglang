"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_backend.
"""

import sys
import unittest
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMaskBuilder,
    AscendAttnMultiStepDraftBackend,
    ForwardMetadata,
    _expand_dsa_sparse_indices,
    _reshape_kv_for_fia_nz,
)
from sglang.srt.hardware_backend.npu.utils import supports_fia_mixed_split
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestExpandDsaSparseIndices(unittest.TestCase):
    def test_2d_input_adds_unsqueeze(self):
        """A [T, K] tensor becomes [T, 1, K]."""
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 1, 3))
        self.assertTrue(torch.equal(result.squeeze(1), topk))

    def test_3d_input_passthrough(self):
        topk = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertTrue(torch.equal(result, topk))

    def test_2d_shape_correctness(self):
        topk = torch.zeros(5, 8)
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.dim(), 3)
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 1)
        self.assertEqual(result.shape[2], 8)

    def test_2d_single_row(self):
        topk = torch.tensor([[1, 2, 3, 4]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (1, 1, 4))


class TestReshapeKvForFiaNz(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is (-1, 1, num_heads*head_dim//16, page_size, 16)."""
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (1, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_element_preservation(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 2 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.numel(), tensor.numel())
        self.assertTrue(torch.equal(result.flatten(), tensor))

    def test_different_parameters(self):
        num_heads = 4
        head_dim = 128
        page_size = 32
        total = 3 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.randn(total)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (3, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_view_relationship(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())


class TestForwardMetadata(unittest.TestCase):
    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(ForwardMetadata))

    def test_all_fields_default_none(self):
        metadata = ForwardMetadata()
        for f in fields(ForwardMetadata):
            self.assertIsNone(
                getattr(metadata, f.name),
                f"Field '{f.name}' should default to None",
            )

    def test_create_with_values(self):
        block_tables = torch.tensor([[1, 2], [3, 4]])
        seq_lens = torch.tensor([10, 20])
        metadata = ForwardMetadata(
            block_tables=block_tables,
            seq_lens=seq_lens,
            seq_lens_cpu_list=[10, 20],
        )
        self.assertTrue(torch.equal(metadata.block_tables, block_tables))
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertEqual(metadata.seq_lens_cpu_list, [10, 20])

    def test_partial_assignment(self):
        metadata = ForwardMetadata(swa_mask=torch.ones(3, 3))
        self.assertIsNotNone(metadata.swa_mask)
        self.assertIsNone(metadata.block_tables)
        self.assertIsNone(metadata.seq_lens)

    def test_field_names(self):
        names = {f.name for f in fields(ForwardMetadata)}
        expected = {
            "block_tables",
            "block_tables_swa",
            "swa_out_cache_loc",
            "extend_seq_lens_cpu_int",
            "seq_lens_cpu_int",
            "seq_lens_cpu_list",
            "seq_lens_list_cumsum",
            "seq_lens",
            "actual_seq_lengths_q",
            "actual_seq_lengths_q_pa",
            "actual_seq_lengths_kv",
            "mixed_num_prefill_reqs",
            "mixed_num_prefill_tokens",
            "swa_mask",
            "prefix_lens",
            "flatten_prefix_block_tables",
        }
        self.assertEqual(names, expected)


class TestSupportsFiaMixedSplit(unittest.TestCase):
    def setUp(self):
        supports_fia_mixed_split.cache_clear()
        # The module-level torch_npu stub is installed with setdefault, so on a
        # host that really has torch_npu this is the genuine module and its
        # get_soc_version is a plain function. Patch it either way.
        patcher = patch.object(sys.modules["torch_npu"].npu, "get_soc_version")
        self.get_soc_version = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(supports_fia_mixed_split.cache_clear)

    def test_a5_is_supported(self):
        self.get_soc_version.return_value = 260
        self.assertTrue(supports_fia_mixed_split())

    def test_non_a5_is_not_supported(self):
        self.get_soc_version.return_value = 220
        self.assertFalse(supports_fia_mixed_split())

    def test_query_failure_disables_split(self):
        self.get_soc_version.side_effect = RuntimeError("SoC query failed")
        self.assertFalse(supports_fia_mixed_split())


class TestFiaMixedSplitMetadata(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)
        self.backend.forward_metadata = ForwardMetadata(
            seq_lens_list_cumsum=np.array([2, 3, 4], dtype=np.int64)
        )

    @staticmethod
    def _forward_batch(**overrides):
        values = {
            "forward_mode": ForwardMode.MIXED,
            "mixed_num_prefill_reqs": 2,
            "mixed_num_prefill_tokens": 3,
            "batch_size": 3,
            "extend_num_tokens": 4,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_sets_prefill_first_boundary_with_one_token_last_chunk(self):
        self.backend._set_fia_mixed_split_metadata(self._forward_batch())
        self.assertEqual(self.backend.forward_metadata.mixed_num_prefill_reqs, 2)
        self.assertEqual(self.backend.forward_metadata.mixed_num_prefill_tokens, 3)

    def test_missing_boundary_keeps_single_call_metadata(self):
        self.backend._set_fia_mixed_split_metadata(
            self._forward_batch(
                mixed_num_prefill_reqs=None,
                mixed_num_prefill_tokens=None,
            )
        )
        self.assertIsNone(self.backend.forward_metadata.mixed_num_prefill_reqs)
        self.assertIsNone(self.backend.forward_metadata.mixed_num_prefill_tokens)

    def test_partial_boundary_raises(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.backend._set_fia_mixed_split_metadata(
                self._forward_batch(mixed_num_prefill_tokens=None)
            )

    def test_inconsistent_q_boundary_raises(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            self.backend._set_fia_mixed_split_metadata(
                self._forward_batch(mixed_num_prefill_tokens=2)
            )


def _make_mixed_layer(**overrides):
    values = {
        "attn_type": AttentionType.DECODER,
        "is_cross_attention": False,
        "sliding_window_size": -1,
        "tp_q_head_num": 4,
        "tp_k_head_num": 2,
        "tp_v_head_num": 2,
        "qk_head_dim": 2,
        "v_head_dim": 2,
        "scaling": 0.5,
        "layer_id": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestFiaMixedSplitGate(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)
        self.backend.enable_fia_mixed_split = True
        self.backend.use_fia = True
        self.backend.use_mla = False
        self.backend.is_hybrid_swa = False
        self.backend.forward_metadata = ForwardMetadata(
            mixed_num_prefill_reqs=2,
            mixed_num_prefill_tokens=3,
        )

    def test_causal_gqa_mixed_batch_is_enabled(self):
        self.assertTrue(
            self.backend._should_split_fia_mixed(_make_mixed_layer(), None)
        )

    def test_causal_mha_mixed_batch_is_enabled(self):
        self.assertTrue(
            self.backend._should_split_fia_mixed(
                _make_mixed_layer(tp_k_head_num=4), None
            )
        )

    def test_unsupported_paths_stay_single_call(self):
        cases = [
            ("feature disabled", {"enable_fia_mixed_split": False}, {}, None),
            ("FIA disabled", {"use_fia": False}, {}, None),
            ("MLA", {"use_mla": True}, {}, None),
            ("cross attention", {}, {"is_cross_attention": True}, None),
            (
                "bidirectional",
                {},
                {"attn_type": AttentionType.DECODER_BIDIRECTIONAL},
                None,
            ),
            ("non GQA heads", {}, {"tp_q_head_num": 3}, None),
            ("DSA topk", {}, {}, torch.tensor([0])),
        ]
        for name, backend_values, layer_values, topk_indices in cases:
            with self.subTest(name=name):
                backend = object.__new__(AscendAttnBackend)
                backend.enable_fia_mixed_split = backend_values.get(
                    "enable_fia_mixed_split", True
                )
                backend.use_fia = backend_values.get("use_fia", True)
                backend.use_mla = backend_values.get("use_mla", False)
                backend.is_hybrid_swa = False
                backend.forward_metadata = self.backend.forward_metadata
                self.assertFalse(
                    backend._should_split_fia_mixed(
                        _make_mixed_layer(**layer_values), topk_indices
                    )
                )

    def test_swa_and_missing_boundary_stay_single_call(self):
        self.assertFalse(
            self.backend._should_split_fia_mixed(
                _make_mixed_layer(sliding_window_size=128), None
            )
        )
        self.backend.forward_metadata = ForwardMetadata()
        self.assertFalse(
            self.backend._should_split_fia_mixed(_make_mixed_layer(), None)
        )


class TestFiaMixedSplitForward(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)
        self.backend.mix_mask = torch.zeros((8, 8), dtype=torch.int8)
        # forward_mixed re-splits the KV pool buffer by page_size; the real
        # __init__ copies it off the ModelRunner, which object.__new__ skips.
        self.backend.page_size = 8
        self.backend.forward_metadata = ForwardMetadata(
            block_tables=torch.tensor([[10], [20], [30], [40]]),
            seq_lens_list_cumsum=np.array([2, 3, 4, 5], dtype=np.int64),
            seq_lens_cpu_int=torch.tensor([11, 12, 13, 14], dtype=torch.int32),
            mixed_num_prefill_reqs=2,
            mixed_num_prefill_tokens=3,
        )
        self.layer = _make_mixed_layer()

    def test_split_slices_and_rebases_prefill_first_metadata(self):
        query = torch.arange(5 * 4 * 2, dtype=torch.float32).view(5, 4, 2)
        key = torch.zeros((4, 8, 4))
        value = torch.ones((4, 8, 4))
        prefill_output = torch.full((3, 4, 2), 11.0)
        decode_output = torch.full((2, 4, 2), 22.0)
        self.backend._run_fia_mixed = MagicMock(
            side_effect=[prefill_output, decode_output]
        )

        output = self.backend._forward_fia_mixed_split(
            query, key, value, self.layer, block_size=8
        )

        self.assertTrue(torch.equal(output[:3], prefill_output))
        self.assertTrue(torch.equal(output[3:], decode_output))
        self.assertEqual(self.backend._run_fia_mixed.call_count, 2)

        prefill_call, decode_call = self.backend._run_fia_mixed.call_args_list
        self.assertTrue(torch.equal(prefill_call.args[0], query[:3]))
        self.assertTrue(
            torch.equal(
                prefill_call.kwargs["block_table"],
                self.backend.forward_metadata.block_tables[:2],
            )
        )
        np.testing.assert_array_equal(
            prefill_call.kwargs["actual_seq_lengths"], np.array([2, 3])
        )
        self.assertTrue(
            torch.equal(
                prefill_call.kwargs["actual_seq_lengths_kv"],
                torch.tensor([11, 12], dtype=torch.int32),
            )
        )

        self.assertTrue(torch.equal(decode_call.args[0], query[3:]))
        self.assertTrue(
            torch.equal(
                decode_call.kwargs["block_table"],
                self.backend.forward_metadata.block_tables[2:],
            )
        )
        self.assertEqual(decode_call.kwargs["actual_seq_lengths"], [1, 2])
        self.assertTrue(
            torch.equal(
                decode_call.kwargs["actual_seq_lengths_kv"],
                torch.tensor([13, 14], dtype=torch.int32),
            )
        )

    def test_forward_writes_kv_once_and_preserves_output_order(self):
        self.backend.use_mla = False
        self.backend.use_fia = True
        self.backend.token_to_kv_pool = MagicMock()
        self.backend.token_to_kv_pool.get_key_buffer.return_value = torch.zeros(
            (4, 8, 1, 4)
        )
        self.backend.token_to_kv_pool.get_value_buffer.return_value = torch.zeros(
            (4, 8, 1, 4)
        )
        self.backend._should_split_fia_mixed = MagicMock(return_value=True)
        split_output = torch.arange(5 * 4 * 2, dtype=torch.float32).view(5, 4, 2)
        self.backend._forward_fia_mixed_split = MagicMock(return_value=split_output)
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.arange(5, dtype=torch.int64)
        )
        q = torch.zeros((5, 8))
        k = torch.zeros((5, 4))
        v = torch.zeros((5, 4))

        output = self.backend.forward_mixed(q, k, v, self.layer, forward_batch)

        self.backend.token_to_kv_pool.set_kv_buffer.assert_called_once()
        self.backend._forward_fia_mixed_split.assert_called_once()
        self.assertTrue(torch.equal(output, split_output.view(5, 8)))

    def test_single_call_fallback_keeps_full_metadata(self):
        self.backend.use_mla = False
        self.backend.use_fia = True
        self.backend.token_to_kv_pool = MagicMock()
        self.backend.token_to_kv_pool.get_key_buffer.return_value = torch.zeros(
            (4, 8, 1, 4)
        )
        self.backend.token_to_kv_pool.get_value_buffer.return_value = torch.zeros(
            (4, 8, 1, 4)
        )
        self.backend._should_split_fia_mixed = MagicMock(return_value=False)
        single_output = torch.ones((5, 4, 2))
        self.backend._run_fia_mixed = MagicMock(return_value=single_output)
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.arange(5, dtype=torch.int64)
        )

        output = self.backend.forward_mixed(
            torch.zeros((5, 8)),
            torch.zeros((5, 4)),
            torch.zeros((5, 4)),
            self.layer,
            forward_batch,
            save_kv_cache=False,
        )

        self.backend._run_fia_mixed.assert_called_once()
        call = self.backend._run_fia_mixed.call_args
        self.assertTrue(torch.equal(call.args[0], torch.zeros((5, 4, 2))))
        self.assertIs(
            call.kwargs["block_table"], self.backend.forward_metadata.block_tables
        )
        np.testing.assert_array_equal(
            call.kwargs["actual_seq_lengths"], np.array([2, 3, 4, 5])
        )
        self.assertTrue(torch.equal(output, single_output.view(5, 8)))


class TestGenerateMaskFlag(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_bool(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertEqual(mask.dtype, torch.bool)

    def test_upper_triangular_pattern(self):
        """generate_mask_flag returns ~tril, i.e. True above the diagonal."""
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[0, 3].item())
        self.assertTrue(mask[2, 3].item())
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[1, 0].item())
        self.assertFalse(mask[3, 3].item())

    def test_diagonal_is_false(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(5)
        for i in range(5):
            self.assertFalse(mask[i, i].item())

    def test_1x1(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(1)
        self.assertEqual(mask.shape, (1, 1))
        self.assertFalse(mask[0, 0].item())

    def test_symmetric_upper(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_mask_flag(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertTrue(mask[i, j].item())
                self.assertFalse(mask[j, i].item())


class TestGenerateAttnMask(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(8, "norm", torch.float16)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)

    def test_default_dtype_float16(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm")
        self.assertEqual(mask.dtype, torch.float16)

    def test_mix_mode_float16(self):
        """mix + float16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[1, 0].item(), 0.0)

    def test_mix_mode_bfloat16(self):
        """mix + bfloat16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[1, 1].item(), 0.0)

    def test_norm_mode_float16(self):
        """norm + float16 -> upper triangle is -inf (overflow), lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_bfloat16(self):
        """norm + bfloat16 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_float32(self):
        """norm + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_mix_mode_float32(self):
        """mix + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_lower_triangle_all_zero(self):
        """The lower triangle (including diagonal) is always zero."""
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0.0)

    def test_upper_triangle_all_masked(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1.0)

    def test_diagonal_is_zero(self):
        """Diagonal is always zero regardless of mode/dtype."""
        for mode in ("mix", "norm"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                mask = AscendAttnMaskBuilder.generate_attn_mask(4, mode, dtype)
                for i in range(4):
                    self.assertEqual(mask[i, i].item(), 0.0)


class TestGetAttentionMaskId(unittest.TestCase):
    def test_flat_tensor(self):
        """Produces a flat tensor of arange ranges concatenated."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([7, 8, 9, 15, 16, 17, 18, 19])
        self.assertEqual(result.dim(), 1)
        self.assertTrue(torch.equal(result, expected))

    def test_single_sequence(self):
        seq_lens = torch.tensor([5])
        extend_lens = torch.tensor([2])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([3, 4])
        self.assertTrue(torch.equal(result, expected))

    def test_multiple_sequences(self):
        seq_lens = torch.tensor([3, 6, 10])
        extend_lens = torch.tensor([1, 2, 4])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([2, 4, 5, 6, 7, 8, 9])
        self.assertTrue(torch.equal(result, expected))

    def test_total_length(self):
        """Result length equals sum of extend_lens per row."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected_len = 3 + 5
        self.assertEqual(result.numel(), expected_len)

    def test_values_correct(self):
        seq_lens = torch.tensor([4, 8])
        extend_lens = torch.tensor([2, 3])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        self.assertEqual(result[0].item(), 2)
        self.assertEqual(result[1].item(), 3)
        self.assertEqual(result[2].item(), 5)
        self.assertEqual(result[4].item(), 7)


class TestUpdateAttnCache(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_seqlen_greater_than_cached(self):
        """When seqlen > cached, the mask is regenerated and cache updated."""
        old_mask = torch.zeros(4, 4)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.shape, (8, 8))
        self.assertEqual(result_mask.dtype, torch.float16)

    def test_seqlen_less_equal_cached(self):
        """When seqlen <= cached, the existing mask is kept."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_dtype_change(self):
        """When dtype differs, the mask is converted but not regenerated."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.dtype, torch.float32)
        self.assertIsNot(result_mask, old_mask)

    def test_no_change(self):
        """When seqlen <= cached and dtype matches, nothing changes."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_seqlen_greater_and_dtype_change(self):
        """Both regeneration and dtype conversion happen."""
        old_mask = torch.zeros(4, 4, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=16,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 16)
        self.assertEqual(result_mask.shape, (16, 16))
        self.assertEqual(result_mask.dtype, torch.float32)

    def test_seqlen_equal_cached_no_regen(self):
        """seqlen == cached should not trigger regeneration."""
        old_mask = torch.ones(8, 8, dtype=torch.float32)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="mix",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)


class TestGetSplitfuseAttnMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        mask = self.builder.get_splitfuse_attn_mask(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_int8(self):
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask.dtype, torch.int8)

    def test_upper_triangular(self):
        """Upper triangle (excluding diagonal) is 1, rest is 0."""
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask[0, 1].item(), 1)
        self.assertEqual(mask[0, 3].item(), 1)
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertEqual(mask[1, 0].item(), 0)
        self.assertEqual(mask[1, 1].item(), 0)

    def test_lower_triangle_zero(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0)

    def test_upper_triangle_one(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1)


class TestGetSwaMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        """Output shape is (batch, 1, s2)."""
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_1d_input_unsqueezed(self):
        """1-D input of shape (B,) is handled; output still (B, 1, s2)."""
        seq_lens = torch.tensor([3, 6, 9])
        mask = self.builder.get_swa_mask(seq_lens, s2=12, left_context=512)
        self.assertEqual(mask.shape, (3, 1, 12))

    def test_2d_input(self):
        """2-D input of shape (B, 1) is also accepted."""
        seq_lens = torch.tensor([[5], [10]])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_mask_values_large_left_context(self):
        """With left_context >= max seq_len, only indices >= seq_lens are True."""
        seq_lens = torch.tensor([3, 5])
        mask = self.builder.get_swa_mask(seq_lens, s2=8, left_context=512)
        row0 = mask[0, 0]
        self.assertFalse(row0[0].item())
        self.assertFalse(row0[2].item())
        self.assertTrue(row0[3].item())
        self.assertTrue(row0[7].item())
        row1 = mask[1, 0]
        self.assertFalse(row1[4].item())
        self.assertTrue(row1[5].item())
        self.assertTrue(row1[7].item())

    def test_mask_values_small_left_context(self):
        """With a small left_context, earlier positions are also masked."""
        seq_lens = torch.tensor([10, 20])
        mask = self.builder.get_swa_mask(seq_lens, s2=30, left_context=5)
        row0 = mask[0, 0]
        self.assertTrue(row0[0].item())
        self.assertTrue(row0[4].item())
        self.assertFalse(row0[5].item())
        self.assertFalse(row0[9].item())
        self.assertTrue(row0[10].item())
        self.assertTrue(row0[29].item())
        row1 = mask[1, 0]
        self.assertTrue(row1[0].item())
        self.assertTrue(row1[14].item())
        self.assertFalse(row1[15].item())
        self.assertFalse(row1[19].item())
        self.assertTrue(row1[20].item())

    def test_clamp_to_zero(self):
        """When seq_len < left_context, start is clamped to 0."""
        seq_lens = torch.tensor([2])
        mask = self.builder.get_swa_mask(seq_lens, s2=5, left_context=10)
        row = mask[0, 0]
        self.assertFalse(row[0].item())
        self.assertFalse(row[1].item())
        self.assertTrue(row[2].item())
        self.assertTrue(row[4].item())

    def test_default_left_context(self):
        """Default left_context is 512."""
        seq_lens = torch.tensor([10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15)
        self.assertEqual(mask.shape, (1, 1, 15))
        row = mask[0, 0]
        self.assertFalse(row[9].item())
        self.assertTrue(row[10].item())

    def test_dtype_bool(self):
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.dtype, torch.bool)


class TestCanUseTnd(unittest.TestCase):
    def test_128_128(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=128)
            )
        )

    def test_192_192(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=192)
            )
        )

    def test_256_256(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=256, v_head_dim=256)
            )
        )

    def test_192_128(self):
        self.assertTrue(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=128)
            )
        )

    def test_64_64(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=64, v_head_dim=64)
            )
        )

    def test_128_256(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=256)
            )
        )

    def test_256_128(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=256, v_head_dim=128)
            )
        )

    def test_128_192(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=128, v_head_dim=192)
            )
        )

    def test_192_256(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=192, v_head_dim=256)
            )
        )

    def test_96_96(self):
        self.assertFalse(
            AscendAttnBackend._can_use_tnd(
                SimpleNamespace(qk_head_dim=96, v_head_dim=96)
            )
        )


class TestGenerateAlibiBias(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)

    def test_shape(self):
        """Output shape is (num_heads, 1, seq_len)."""
        slopes = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = self.backend._generate_alibi_bias(
            seq_len=8,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (4, 1, 8))

    def test_values(self):
        """Each element is slopes[h] * position."""
        slopes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        seq_len = 5
        result = self.backend._generate_alibi_bias(
            seq_len=seq_len,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(4):
            for p in range(seq_len):
                expected = slopes[h].item() * p
                self.assertAlmostEqual(result[h, 0, p].item(), expected, places=5)

    def test_dtype(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_single_head(self):
        slopes = torch.tensor([1.5])
        result = self.backend._generate_alibi_bias(
            seq_len=3,
            slopes=slopes,
            num_heads=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 0, 1].item(), 1.5)
        self.assertAlmostEqual(result[0, 0, 2].item(), 3.0)

    def test_zero_position_is_zero(self):
        """Position 0 always yields 0 regardless of slopes."""
        slopes = torch.tensor([1.0, 2.0, 3.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(3):
            self.assertAlmostEqual(result[h, 0, 0].item(), 0.0)

    def test_default_dtype_bfloat16(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.dtype, torch.bfloat16)


class TestGetCudaGraphSeqLenFillValue(unittest.TestCase):
    def test_returns_zero(self):
        backend = object.__new__(AscendAttnBackend)
        self.assertEqual(backend.get_cuda_graph_seq_len_fill_value(), 0)


class TestGetVerifyBuffers(unittest.TestCase):
    def test_no_verify_mask(self):
        backend = object.__new__(AscendAttnBackend)
        self.assertIsNone(backend.verify_mask)

    def test_update_is_noop(self):
        backend = object.__new__(AscendAttnBackend)
        backend.update_verify_buffers_to_fill_after_draft(None, None)
        backend.update_verify_buffers_to_fill_after_draft(MagicMock(), 4)
        backend.update_verify_buffers_to_fill_after_draft(None, 16)


class TestCommonTemplate(unittest.TestCase):
    @staticmethod
    def _make_draft_backend(speculative_num_steps):
        backend = object.__new__(AscendAttnMultiStepDraftBackend)
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_calls_fn_for_each_step(self):
        """call_fn is invoked for steps 0..speculative_num_steps-2."""
        backend = self._make_draft_backend(speculative_num_steps=4)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 3)
        for i in range(3):
            call_fn.assert_any_call(i, forward_batch)

    def test_zero_steps(self):
        """speculative_num_steps=1 -> no calls (range(0))."""
        backend = self._make_draft_backend(speculative_num_steps=1)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        call_fn.assert_not_called()

    def test_two_steps(self):
        """speculative_num_steps=2 -> exactly one call with index 0."""
        backend = self._make_draft_backend(speculative_num_steps=2)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        call_fn.assert_called_once_with(0, forward_batch)

    def test_call_indices(self):
        backend = self._make_draft_backend(speculative_num_steps=5)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        indices = []
        backend.common_template(forward_batch, lambda i, fb: indices.append(i))
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_assert_spec_info_not_none(self):
        """Raises AssertionError when forward_batch.spec_info is None."""
        backend = self._make_draft_backend(speculative_num_steps=4)
        forward_batch = MagicMock()
        forward_batch.spec_info = None
        with self.assertRaises(AssertionError):
            backend.common_template(forward_batch, MagicMock())

    def test_passes_same_forward_batch(self):
        backend = self._make_draft_backend(speculative_num_steps=3)
        forward_batch = MagicMock()
        forward_batch.spec_info = MagicMock()
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        for call in call_fn.call_args_list:
            self.assertIs(call.args[1], forward_batch)


if __name__ == "__main__":
    unittest.main()
