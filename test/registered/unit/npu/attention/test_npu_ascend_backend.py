"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_backend.
"""

import sys
import unittest
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt import runtime_context as rc
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
    ForwardMode,
    _expand_dsa_sparse_indices,
    _normalize_mla_k_rope_cache,
    _reshape_kv_for_fia_nz,
)
from sglang.srt.hardware_backend.npu.attention.dcp import (
    mask_empty_mla_dcp_shards_npu,
    merge_mla_dcp_output_npu,
)


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


class TestNormalizeMlaKRoPECache(unittest.TestCase):
    def test_dcp_token_major_cache_preserves_singleton_head(self):
        cache = torch.randn(8192, 1, 64)
        result = _normalize_mla_k_rope_cache(cache, 64)
        self.assertEqual(result.shape, (8192, 1, 64))
        self.assertEqual(result.data_ptr(), cache.data_ptr())

    def test_paged_cache_flattens_block_and_page_only(self):
        cache = torch.randn(2, 128, 1, 64)
        result = _normalize_mla_k_rope_cache(cache, 64)
        self.assertEqual(result.shape, (256, 1, 64))
        self.assertTrue(torch.equal(result.flatten(), cache.flatten()))

    def test_rejects_wrong_rope_dimension(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid MLA RoPE KV cache shape"):
            _normalize_mla_k_rope_cache(torch.empty(8, 1, 32), 64)


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
            "actual_seq_lengths_q_pa_cpu",
            "actual_seq_lengths_kv",
            "dcp_mtp_attn_mask",
            "swa_mask",
            "prefix_lens",
            "flatten_prefix_block_tables",
        }
        self.assertEqual(names, expected)

    def test_dspark_target_verify_builds_local_dcp_metadata(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.use_mla = True
        backend.is_draft_worker = False
        backend.page_size = 4
        backend.device = "cpu"
        backend.is_hybrid_swa = False
        backend.use_sliding_window_kv_pool = False

        # One widened virtual page (physical page 7, DCP=2).
        req_to_token = torch.arange(56, 64, dtype=torch.int64).view(1, 8)
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        spec_info = SimpleNamespace(draft_token_num=3)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            seq_lens=torch.tensor([5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            out_cache_loc=None,
        )

        with rc.get_parallel().override(dcp_enabled=True, dcp_size=2, dcp_rank=1):
            backend.init_forward_metadata(forward_batch)

        self.assertEqual(backend.forward_metadata.seq_lens.tolist(), [4])
        self.assertEqual(backend.forward_metadata.seq_lens_cpu_int.tolist(), [4])
        self.assertEqual(backend.forward_metadata.block_tables.tolist(), [[7]])
        mask = backend.forward_metadata.dcp_mtp_attn_mask
        self.assertEqual(mask.shape, (1, 3, 4))
        # Local positions are [1, 3, 5, 7]; global queries are [5, 6, 7].
        self.assertEqual(
            mask[0].tolist(),
            [
                [False, False, False, True],
                [False, False, False, True],
                [False, False, False, False],
            ],
        )

    def test_dspark_graph_metadata_is_fixed_shape_and_rank_local(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.use_mla = True
        backend.is_draft_worker = False
        backend.page_size = 4
        backend.device = "cpu"
        backend.max_context_len = 8
        backend.speculative_num_draft_tokens = 3
        backend.q_head_num_padding = None
        backend.is_hybrid_swa = False
        backend.use_sliding_window_kv_pool = False
        backend.req_to_token = torch.arange(56, 88, dtype=torch.int64).view(2, 16)

        with rc.get_parallel().override(dcp_enabled=True, dcp_size=2, dcp_rank=1):
            backend.init_cuda_graph_state(max_bs=2, max_num_tokens=6)
            metadata = backend._init_cuda_graph_metadata(
                2,
                ForwardMode.TARGET_VERIFY,
                torch.ones(2, dtype=torch.int32),
            )
            seq_lens_ptr = metadata.seq_lens.data_ptr()
            mask_ptr = metadata.dcp_mtp_attn_mask.data_ptr()
            backend._apply_cuda_graph_metadata(
                bs=2,
                req_pool_indices=torch.tensor([0, 1]),
                seq_lens=torch.tensor([5, 0]),
                seq_lens_cpu=torch.tensor([5, 0]),
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=None,
                num_padding=1,
                in_capture=True,
            )

        self.assertEqual(metadata.block_tables.shape, (2, 2))
        self.assertEqual(metadata.seq_lens.tolist(), [4, 0])
        self.assertEqual(metadata.seq_lens_cpu_list, [4, 0])
        self.assertTrue(metadata.dcp_mtp_attn_mask[1].all())
        self.assertEqual(metadata.seq_lens.data_ptr(), seq_lens_ptr)
        self.assertEqual(metadata.dcp_mtp_attn_mask.data_ptr(), mask_ptr)


class TestEmptyDcpTargetVerify(unittest.TestCase):
    def test_empty_shard_mask_is_dynamic_and_graph_padding_safe(self):
        output = torch.arange(24, dtype=torch.float32).view(4, 2, 3)
        lse = torch.arange(8, dtype=torch.float32).view(4, 2, 1)

        masked_output, masked_lse = mask_empty_mla_dcp_shards_npu(
            output, lse, torch.tensor([4, 0])
        )

        torch.testing.assert_close(masked_output[:2], output[:2])
        torch.testing.assert_close(masked_lse[:2], lse[:2])
        torch.testing.assert_close(masked_output[2:], torch.zeros_like(output[2:]))
        self.assertTrue(torch.isneginf(masked_lse[2:]).all())

    def test_forward_mtp_returns_empty_without_calling_fia(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.use_mla = True
        backend.is_draft_worker = False
        backend.graph_mode = False
        backend.use_fia = True
        backend.kv_lora_rank = 16
        backend.forward_metadata = SimpleNamespace(
            dcp_mtp_attn_mask=torch.empty((0, 8, 0), dtype=torch.bool)
        )
        layer = SimpleNamespace(tp_q_head_num=8)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_token_non_padded_cpu=0,
            spec_info=SimpleNamespace(draft_token_num=8),
        )

        with rc.get_parallel().override(dcp_enabled=True, dcp_size=2, dcp_rank=0):
            output, lse = backend.forward_mtp(
                torch.empty((0, 8, 16)),
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
                return_softmax_lse=True,
            )

        self.assertEqual(output.shape, (0, 128))
        self.assertEqual(lse.shape, (0, 8, 1))

    def test_merge_returns_empty_local_heads_without_collective(self):
        partial_output = torch.empty((0, 8, 16))
        partial_lse = torch.empty((0, 8, 1))

        with rc.get_parallel().override(dcp_enabled=True, dcp_size=2, dcp_rank=0):
            output = merge_mla_dcp_output_npu(partial_output, partial_lse)

        self.assertEqual(output.shape, (0, 4, 16))


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
