"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_backend.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the torch_npu mock below: importing transformers runs
# accelerate's find_spec("torch_npu"), which fails on a mock without __spec__.
import sglang.srt.utils.hf_transformers  # noqa: F401
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="base-a-test-npu")

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
    ForwardMetadata,
    _expand_dsa_sparse_indices,
    _reshape_kv_for_fia_nz,
)
from sglang.srt.model_executor.forward_mode import ForwardMode


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


class TestMlaPrefixCacheDcp(unittest.TestCase):
    def test_fp8_prefix_dequantized_before_projection_or_dcp_gather(self):
        from sglang.srt.hardware_backend.npu.attention.mla_cache import (
            build_mla_prefix_token_indices,
        )

        module = "sglang.srt.hardware_backend.npu.attention.ascend_backend"
        page = 4
        for local_lens in ([5, 3], [4, 4]):
            counts = [(length + page - 1) // page for length in local_lens]
            blocks = sum(counts)
            ids = torch.arange(blocks - 1, -1, -1)
            latent = (
                (torch.arange(blocks * page * 32).remainder(97) - 48)
                .reshape(blocks, page, 1, 32)
                .to(torch.float8_e4m3fn)
            )
            rope = (
                torch.arange(blocks * page * 32).reshape(blocks, page, 1, 32).bfloat16()
            )
            expected_k, expected_pe, offset = [], [], 0
            for length, count in zip(local_lens, counts):
                request_ids = ids[offset : offset + count]
                expected_k.append(latent.float()[request_ids].flatten(0, 1)[:length])
                expected_pe.append(rope[request_ids].flatten(0, 1)[:length])
                offset += count
            expected_k = torch.cat(expected_k)
            expected_pe = torch.cat(expected_pe)
            expected_dequantized = (expected_k * 0.25).bfloat16()

            for is_nz in (False, True):

                def encode(data):
                    if not is_nz:
                        return data
                    tile = 32 if data.dtype == torch.float8_e4m3fn else 16
                    backing = data.view(torch.uint8) if tile == 32 else data
                    return (
                        backing.reshape(blocks, page, 32 // tile, tile)
                        .permute(0, 2, 1, 3)
                        .contiguous()
                        .reshape_as(backing)
                        .view(data.dtype)
                    )

                for size in (1, 2):
                    with self.subTest(lengths=local_lens, nz=is_nz, dcp_size=size):
                        backend = AscendAttnBackend.__new__(AscendAttnBackend)
                        backend.dcp_size, backend.page_size = size, page
                        backend.kv_lora_rank = backend.qk_rope_head_dim = 32
                        global_lens = (
                            local_lens
                            if size == 1
                            else ([9, 3] if local_lens == [5, 3] else [8, 8])
                        )
                        backend.forward_metadata = ForwardMetadata(
                            prefix_lens=torch.tensor(global_lens),
                            flatten_prefix_block_tables=ids,
                            prefix_token_indices=(
                                build_mla_prefix_token_indices(
                                    local_lens, page, device=torch.device("cpu")
                                )
                                if any(length % page for length in local_lens)
                                else None
                            ),
                        )
                        k_buffer, pe_buffer = encode(latent), encode(rope)
                        backend.token_to_kv_pool = SimpleNamespace(
                            get_key_buffer=lambda _: k_buffer,
                            get_value_buffer=lambda _: pe_buffer,
                        )
                        dequantize = MagicMock(
                            side_effect=lambda value: (
                                (value.float() * 0.25).bfloat16(),
                                None,
                            )
                        )
                        full_prefix = torch.randn(sum(global_lens), 1, 64).bfloat16()
                        with (
                            patch(f"{module}.is_fia_nz", return_value=is_nz),
                            patch(
                                f"{module}.all_gather_kv_cache_for_dcp",
                                return_value=full_prefix,
                            ) as gather,
                        ):
                            result_k, result_pe = backend._load_mla_prefix_cache(
                                SimpleNamespace(
                                    layer_id=0,
                                    scheme=SimpleNamespace(dequantize_cache=dequantize),
                                )
                            )
                        dequantize.assert_called_once()
                        encoded = dequantize.call_args.args[0]
                        self.assertEqual(encoded.dtype, torch.float8_e4m3fn)
                        torch.testing.assert_close(
                            encoded.float(), expected_k, atol=0, rtol=0
                        )
                        if size == 1:
                            gather.assert_not_called()
                            torch.testing.assert_close(
                                result_k, expected_dequantized, atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                result_pe, expected_pe, atol=0, rtol=0
                            )
                        else:
                            gather.assert_called_once()
                            torch.testing.assert_close(
                                gather.call_args.args[0],
                                expected_dequantized,
                                atol=0,
                                rtol=0,
                            )
                            torch.testing.assert_close(
                                gather.call_args.args[1], expected_pe, atol=0, rtol=0
                            )
                            self.assertIs(
                                gather.call_args.args[2],
                                backend.forward_metadata.prefix_lens,
                            )
                            self.assertEqual(
                                gather.call_args.kwargs, {"interleave_size": page}
                            )
                            torch.testing.assert_close(
                                torch.cat((result_k, result_pe), -1),
                                full_prefix,
                                atol=0,
                                rtol=0,
                            )

    def _check_prefix_cache(self, lengths, page_size, size, rank, is_nz, mode):
        module = "sglang.srt.hardware_backend.npu.attention.ascend_backend"
        queries = torch.tensor([3, 1, 1], dtype=torch.int32)
        prefix_lens = torch.tensor(lengths, dtype=torch.int32)
        seq_lens = prefix_lens + queries
        stride = page_size * size
        cycles = (int(seq_lens.max()) + stride - 1) // stride
        mapping = torch.arange(3 * cycles * stride).reshape(3, -1)
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.device = torch.device("cpu")
        backend.use_mla = True
        backend.use_dsa = False
        backend.dcp_size, backend.dcp_rank = size, rank
        backend.dense_mla_dcp_target = size > 1
        backend.dsa_dcp_target = False
        backend.is_hybrid_swa = False
        backend.use_sliding_window_kv_pool = False
        backend.page_size = page_size
        backend.kv_lora_rank, backend.qk_rope_head_dim = 32, 16
        backend.req_to_token = mapping
        backend.req_to_token_pool = SimpleNamespace(req_to_token=mapping)
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            extend_seq_lens=queries,
            extend_seq_lens_cpu=queries.tolist(),
            extend_prefix_lens=prefix_lens,
            extend_prefix_lens_cpu=lengths,
            req_pool_indices=torch.tensor([2, 0, 1]),
            spec_info=None,
            out_cache_loc=None,
        )
        parallel = SimpleNamespace(attn_dcp_size=size, attn_dcp_rank=rank)
        with (
            patch(f"{module}.get_parallel", return_value=parallel),
            patch(
                f"{module}.build_mla_prefix_token_indices",
                wraps=sys.modules[module].build_mla_prefix_token_indices,
            ) as build_indices,
        ):
            backend.init_forward_metadata(batch)
        metadata = backend.forward_metadata
        self.assertEqual(metadata.prefix_lens.tolist(), lengths)

        full, local = [], []
        for request, length in enumerate(lengths):
            values = torch.arange(length * 48, dtype=torch.float32).reshape(
                length, 1, 48
            )
            values = values + request * 10000
            full.append(values)
            positions = torch.arange(length)
            local.append(values[(positions // page_size) % size == rank])
        expected_local = torch.cat(local)
        expected_full = torch.cat(full)
        needs_compaction = any(len(values) % page_size for values in local)
        if needs_compaction:
            self.assertEqual(
                metadata.prefix_token_indices.numel(), expected_local.shape[0]
            )
        else:
            self.assertIsNone(metadata.prefix_token_indices)
        self.assertEqual(build_indices.call_count, int(needs_compaction))

        blocks = metadata.flatten_prefix_block_tables.tolist()
        pages = torch.full((max(blocks, default=0) + 1, page_size, 1, 48), -999.0)
        offset = 0
        for values in local:
            count = (len(values) + page_size - 1) // page_size
            if count:
                request_pages = pages.new_full((count * page_size, 1, 48), -999)
                request_pages[: len(values)] = values
                pages[blocks[offset : offset + count]] = request_pages.reshape(
                    count, page_size, 1, 48
                )
            offset += count

        def encode(cache):
            if not is_nz:
                return cache
            return (
                cache.reshape(cache.shape[0], page_size, -1, 16)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape_as(cache)
            )

        latent, rope = encode(pages[..., :32]), encode(pages[..., 32:])
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: latent,
            get_value_buffer=lambda _: rope,
        )

        def gather(kv, pe, global_lens, *, interleave_size):
            torch.testing.assert_close(kv, expected_local[..., :32], atol=0, rtol=0)
            torch.testing.assert_close(pe, expected_local[..., 32:], atol=0, rtol=0)
            self.assertEqual(global_lens.tolist(), lengths)
            self.assertEqual(interleave_size, page_size)
            return expected_full

        with (
            patch(f"{module}.is_fia_nz", return_value=is_nz),
            patch(
                f"{module}.all_gather_kv_cache_for_dcp", side_effect=gather
            ) as collective,
        ):
            actual_kv, actual_pe = backend._load_mla_prefix_cache(
                SimpleNamespace(layer_id=0)
            )
        torch.testing.assert_close(actual_kv, expected_full[..., :32], atol=0, rtol=0)
        torch.testing.assert_close(actual_pe, expected_full[..., 32:], atol=0, rtol=0)
        self.assertEqual(collective.call_count, int(size > 1))

    def test_rank_local_compaction_and_global_prefix_contract(self):
        for lengths, page in (
            ([5, 3, 0], 4),
            ([0, 9, 2], 4),
            ([128, 255, 129], 128),
            ([4, 8, 0], 4),
            ([128, 256, 0], 128),
        ):
            for size in (1, 2, 4):
                for rank in range(size):
                    for is_nz in (False, True):
                        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
                            with self.subTest(
                                lengths=lengths,
                                size=size,
                                rank=rank,
                                nz=is_nz,
                                mode=mode,
                            ):
                                self._check_prefix_cache(
                                    lengths, page, size, rank, is_nz, mode
                                )


class TestMlaPrefixFia(unittest.TestCase):
    @staticmethod
    def _fia_reference(**kwargs):
        q, k, v = (kwargs[name] for name in ("query", "key", "value"))
        for name in ("query", "key", "value", "query_rope", "key_rope"):
            assert kwargs[name].is_contiguous(), name
        assert kwargs["input_layout"] == "TND"
        assert kwargs["return_softmax_lse"]
        assert kwargs["num_query_heads"] == q.shape[1]
        assert kwargs["num_key_value_heads"] == k.shape[1]
        q = torch.cat((q, kwargs["query_rope"]), dim=-1).float()
        k = torch.cat((k, kwargs["key_rope"]), dim=-1).float()
        output = v.new_empty(q.shape[0], q.shape[1], v.shape[-1])
        lse = q.new_empty(q.shape[0], q.shape[1], 1)
        q_start = kv_start = 0
        for q_end, kv_end in zip(kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"]):
            if kv_start == kv_end:
                output[q_start:q_end] = float("nan")
                lse[q_start:q_end] = float("inf")
            else:
                scores = (
                    torch.einsum("thd,shd->hts", q[q_start:q_end], k[kv_start:kv_end])
                    * kwargs["softmax_scale"]
                )
                if kwargs["sparse_mode"] == 3:
                    assert kwargs["next_tokens"] == 0
                    assert kwargs["atten_mask"].dtype == torch.bool
                    mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
                    scores.masked_fill_(mask, -float("inf"))
                else:
                    assert kwargs["sparse_mode"] == 0
                    assert kwargs.get("atten_mask") is None
                output[q_start:q_end] = torch.einsum(
                    "hts,shd->thd", scores.softmax(-1), v[kv_start:kv_end].float()
                ).to(v.dtype)
                lse[q_start:q_end] = scores.logsumexp(-1).T.unsqueeze(-1)
            q_start, kv_start = q_end, kv_end
        assert q_start == q.shape[0]
        assert kv_start == k.shape[0]
        return output, lse

    @staticmethod
    def _update_reference(lse, outputs, update_type):
        assert update_type == 0
        assert all(x.ndim == 1 and x.dtype == torch.float32 for x in lse)
        assert all(x.ndim == 2 and x.dtype == torch.float32 for x in outputs)
        assert all(x.is_contiguous() for x in (*lse, *outputs))
        weights = torch.stack(lse).softmax(0).unsqueeze(-1)
        return (torch.stack(outputs) * weights).sum(0), None

    def _check_prefix(
        self, query_lens, prefix_lens, dtype, padding, heads, magnitude=1, use_fia=False
    ):
        torch.manual_seed(47)
        tokens, prefix_tokens = sum(query_lens), sum(prefix_lens)
        q = torch.randn(tokens + padding, heads, 208, dtype=dtype)[..., :192]
        k = torch.randn(tokens + padding, heads, 208, dtype=dtype)[..., :192]
        v = torch.randn(tokens + padding, heads, 144, dtype=dtype)[..., :128]
        q.mul_(magnitude)
        k.mul_(magnitude)
        cached = torch.randn(prefix_tokens + 3, 1, 1, 32, dtype=dtype)
        cached_rope = torch.randn(prefix_tokens + 3, 1, 1, 64, dtype=dtype)
        indices = torch.randperm(prefix_tokens + 3)[:prefix_tokens]
        weight = torch.randn(32, heads * 256, dtype=dtype) / 32**0.5

        def project(x):
            return x @ weight, None

        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.dcp_size = 1
        backend.use_mla = True
        backend.use_fia = use_fia
        backend.is_dllm_model = False
        backend.qk_nope_head_dim = 128
        backend.qk_rope_head_dim = 64
        backend.fia_mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1)
        backend.forward_metadata = ForwardMetadata(
            extend_seq_lens_cpu_int=torch.tensor(query_lens, dtype=torch.int32),
            prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
            flatten_prefix_block_tables=indices,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: cached,
            get_value_buffer=lambda _: cached_rope,
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            tp_k_head_num=heads,
            qk_head_dim=192,
            v_head_dim=128,
            kv_b_proj=project,
            scaling=192**-0.5,
        )
        batch = SimpleNamespace(
            global_num_token_non_padded_cpu=tokens,
            extend_prefix_lens_cpu=prefix_lens,
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: False,
                is_draft_extend_v2=lambda: False,
            ),
        )
        module_name = "sglang.srt.hardware_backend.npu.attention.ascend_backend"
        with (
            patch(f"{module_name}.is_mla_preprocess_enabled", return_value=False),
            patch(
                f"{module_name}.torch_npu.npu_fused_infer_attention_score_v2",
                side_effect=self._fia_reference,
            ) as fia,
            patch(
                f"{module_name}.torch_npu.npu_attention_update",
                side_effect=self._update_reference,
            ) as update,
            patch(f"{module_name}.torch_npu.atb", None),
        ):
            actual = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=False)
        self.assertEqual(fia.call_count, 2)
        self.assertEqual(update.call_count, 1)

        prefix_k, prefix_v = (
            project(cached[indices])[0]
            .view(prefix_tokens, heads, 256)
            .split(128, dim=-1)
        )
        prefix_k = torch.cat(
            (prefix_k, cached_rope[indices].flatten(0, 1).expand(-1, heads, -1)),
            dim=-1,
        )
        expected = torch.zeros(tokens + padding, heads, 128, dtype=dtype)
        q_start = prefix_start = 0
        for q_len, prefix_len in zip(query_lens, prefix_lens):
            q_end, prefix_end = q_start + q_len, prefix_start + prefix_len
            full_k = torch.cat((prefix_k[prefix_start:prefix_end], k[q_start:q_end]))
            full_v = torch.cat((prefix_v[prefix_start:prefix_end], v[q_start:q_end]))
            allowed = torch.arange(prefix_len + q_len)[None, :] <= (
                prefix_len + torch.arange(q_len)[:, None]
            )
            expected[q_start:q_end] = (
                torch.nn.functional.scaled_dot_product_attention(
                    q[q_start:q_end].float().transpose(0, 1),
                    full_k.float().transpose(0, 1),
                    full_v.float().transpose(0, 1),
                    attn_mask=allowed,
                    scale=layer.scaling,
                )
                .transpose(0, 1)
                .to(dtype)
            )
            q_start, prefix_start = q_end, prefix_end
        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        self.assertEqual(torch.count_nonzero(actual[tokens:]).item(), 0)

    def test_prefix_matches_dense_attention(self):
        for dtype in (torch.float16, torch.bfloat16):
            for queries, prefixes, padding, heads in (
                ([3, 1, 2], [0, 5, 2], 3, 4),
                ([2, 4, 1], [5, 0, 0], 0, 2),
                ([1, 1], [2, 3], 2, 4),
                ([4], [7], 0, 2),
            ):
                with self.subTest(dtype=dtype, queries=queries, prefixes=prefixes):
                    self._check_prefix(queries, prefixes, dtype, padding, heads)

    def test_large_logits_merge_stably(self):
        self._check_prefix([3, 2], [4, 7], torch.bfloat16, 1, 4, magnitude=8)

    def test_prefix_ignores_fia_flag(self):
        for dtype in (torch.float16, torch.bfloat16):
            for use_fia in (False, True):
                with self.subTest(dtype=dtype, use_fia=use_fia):
                    self._check_prefix(
                        [3, 1, 2], [0, 5, 2], dtype, 3, 4, use_fia=use_fia
                    )


if __name__ == "__main__":
    unittest.main()
