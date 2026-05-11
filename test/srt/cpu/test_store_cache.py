import itertools
import unittest

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(42)

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# Reference implementation (from index_buf_accessor_v4.py)
def _set_k_and_s_torch(buf, loc, k_nope, k_rope, scale_k_nope, page_size):
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape

    nope_dim = k_nope.shape[1]
    rope_dim = k_rope.shape[1]
    scale_dim = scale_k_nope.shape[1]

    buf_fp8 = buf.view(fp8_dtype).flatten()
    buf_bf16 = buf.view(torch.bfloat16).flatten()
    buf_scale = buf.view(torch.uint8).flatten()

    loc_page_index = loc // page_size
    loc_token_offset_in_page = loc % page_size

    s_offset_nbytes_in_page = page_size * (nope_dim + rope_dim * 2)

    nope_offset = loc_page_index * buf_numel_per_page + loc_token_offset_in_page * (
        nope_dim + rope_dim * 2
    )

    rope_offset = (
        loc_page_index * buf_numel_per_page // 2
        + (loc_token_offset_in_page * (nope_dim + rope_dim * 2) + nope_dim) // 2
    )

    s_offset = (
        loc_page_index * buf_numel_per_page
        + s_offset_nbytes_in_page
        + loc_token_offset_in_page * (scale_dim + 1)
    )

    for i in range(num_tokens_to_write):
        buf_fp8[nope_offset[i] : nope_offset[i] + nope_dim] = k_nope[i]
        buf_bf16[rope_offset[i] : rope_offset[i] + rope_dim] = k_rope[i]
        buf_scale[s_offset[i] : s_offset[i] + scale_dim] = scale_k_nope[i]


def make_test_data(num_pages, page_size, num_tokens, nope_dim=448, rope_dim=64, scale_dim=7):
    """Create test data matching the buffer layout."""
    nope_rope_bytes_per_token = nope_dim + rope_dim * 2
    s_bytes_per_token = scale_dim + 1
    buf_numel_per_page = page_size * nope_rope_bytes_per_token + page_size * s_bytes_per_token

    buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8)

    # Generate random non-overlapping locations
    total_slots = num_pages * page_size
    assert num_tokens <= total_slots
    perm = torch.randperm(total_slots)[:num_tokens]
    loc = perm.to(torch.int64)

    k_nope = torch.randint(0, 256, (num_tokens, nope_dim), dtype=torch.uint8).view(fp8_dtype)
    k_rope = torch.randn(num_tokens, rope_dim, dtype=torch.bfloat16)
    scale_k_nope = torch.randint(0, 256, (num_tokens, scale_dim), dtype=torch.uint8)

    return buf, loc, k_nope, k_rope, scale_k_nope


class TestSetKAndS(CustomTestCase):
    num_pages_list = [4, 16]
    page_size_list = [1, 16]
    num_tokens_list = [1, 7, 32]

    def _test_set_k_and_s(self, num_pages, page_size, num_tokens):
        max_tokens = num_pages * page_size
        if num_tokens > max_tokens:
            num_tokens = max_tokens

        buf, loc, k_nope, k_rope, scale_k_nope = make_test_data(
            num_pages, page_size, num_tokens
        )

        # Reference
        buf_ref = buf.clone()
        _set_k_and_s_torch(buf_ref, loc, k_nope, k_rope, scale_k_nope, page_size)

        # C++ kernel
        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_and_s_cpu(
            buf_test, loc, k_nope, k_rope, scale_k_nope, page_size
        )

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_k_and_s(self):
        for params in itertools.product(
            self.num_pages_list, self.page_size_list, self.num_tokens_list
        ):
            with self.subTest(
                num_pages=params[0], page_size=params[1], num_tokens=params[2]
            ):
                self._test_set_k_and_s(*params)

    def test_set_k_and_s_int32_loc(self):
        """Test with int32 loc tensor."""
        buf, loc, k_nope, k_rope, scale_k_nope = make_test_data(8, 16, 20)
        loc_i32 = loc.to(torch.int32)

        buf_ref = buf.clone()
        _set_k_and_s_torch(buf_ref, loc, k_nope, k_rope, scale_k_nope, 16)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_and_s_cpu(
            buf_test, loc_i32, k_nope, k_rope, scale_k_nope, 16
        )

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_k_and_s_large(self):
        """Larger stress test."""
        num_pages, page_size, num_tokens = 64, 16, 512
        buf, loc, k_nope, k_rope, scale_k_nope = make_test_data(
            num_pages, page_size, num_tokens
        )

        buf_ref = buf.clone()
        _set_k_and_s_torch(buf_ref, loc, k_nope, k_rope, scale_k_nope, page_size)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_and_s_cpu(
            buf_test, loc, k_nope, k_rope, scale_k_nope, page_size
        )

        torch.testing.assert_close(buf_ref, buf_test)


# ===========================================================================
# Reference for quant_to_nope_fp8_rope_bf16_pack
# ===========================================================================

def _cast_scale_inv_to_ue8m0(scales_inv, out_dtype=torch.float32):
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)


def quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16):
    """Reference implementation from quant_k_cache_v4.py."""
    assert k_bf16.dtype == torch.bfloat16
    _num_tokens, hidden_dim = k_bf16.shape
    assert hidden_dim == 512
    dim_nope = 448
    dim_rope = 64

    k_nope_bf16, k_rope_bf16 = k_bf16.split([dim_nope, dim_rope], dim=-1)

    tile_size = 64
    num_tiles = dim_nope // tile_size

    x = k_nope_bf16.contiguous().view(-1, num_tiles, tile_size)
    scale = x.abs().amax(dim=-1).float() / 448.0
    scale_pow2_fp32 = _cast_scale_inv_to_ue8m0(scale, out_dtype=torch.float32)
    scale_k_nope_ue8m0 = scale_pow2_fp32.to(torch.float8_e8m0fnu)
    k_nope_fp8 = (x.float() / scale_pow2_fp32.unsqueeze(-1)).to(fp8_dtype)
    k_nope_fp8 = k_nope_fp8.view(-1, tile_size * num_tiles)
    scale_k_nope_ue8m0 = scale_k_nope_ue8m0.view(torch.uint8)

    return k_nope_fp8, k_rope_bf16.contiguous(), scale_k_nope_ue8m0


class TestQuantToNopeFp8RopeBf16Pack(CustomTestCase):
    num_tokens_list = [1, 7, 32, 128, 512]

    def _test_quant(self, num_tokens):
        k_bf16 = torch.randn(num_tokens, 512, dtype=torch.bfloat16)

        ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
        cpp_nope, cpp_rope, cpp_scale = (
            torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
        )

        torch.testing.assert_close(ref_rope, cpp_rope)
        torch.testing.assert_close(ref_scale, cpp_scale)
        torch.testing.assert_close(
            ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8)
        )

    def test_quant_various_sizes(self):
        for num_tokens in self.num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                self._test_quant(num_tokens)

    def test_quant_small_values(self):
        """Test with very small values that exercise the EPS clamp."""
        k_bf16 = torch.randn(16, 512, dtype=torch.bfloat16) * 1e-6
        ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
        cpp_nope, cpp_rope, cpp_scale = (
            torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
        )
        torch.testing.assert_close(ref_rope, cpp_rope)
        torch.testing.assert_close(ref_scale, cpp_scale)
        torch.testing.assert_close(
            ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8)
        )

    def test_quant_large_values(self):
        """Test with large values."""
        k_bf16 = torch.randn(16, 512, dtype=torch.bfloat16) * 100.0
        ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
        cpp_nope, cpp_rope, cpp_scale = (
            torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
        )
        torch.testing.assert_close(ref_rope, cpp_rope)
        torch.testing.assert_close(ref_scale, cpp_scale)
        torch.testing.assert_close(
            ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8)
        )

    def test_quant_zeros(self):
        """Test with zero input."""
        k_bf16 = torch.zeros(8, 512, dtype=torch.bfloat16)
        ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
        cpp_nope, cpp_rope, cpp_scale = (
            torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
        )
        torch.testing.assert_close(ref_rope, cpp_rope)
        torch.testing.assert_close(ref_scale, cpp_scale)
        torch.testing.assert_close(
            ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8)
        )

    def test_output_shapes_and_dtypes(self):
        """Verify output shapes and dtypes."""
        num_tokens = 16
        k_bf16 = torch.randn(num_tokens, 512, dtype=torch.bfloat16)
        cpp_nope, cpp_rope, cpp_scale = (
            torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
        )

        self.assertEqual(cpp_nope.shape, (num_tokens, 448))
        self.assertEqual(cpp_rope.shape, (num_tokens, 64))
        self.assertEqual(cpp_scale.shape, (num_tokens, 7))

        self.assertEqual(cpp_nope.dtype, fp8_dtype)
        self.assertEqual(cpp_rope.dtype, torch.bfloat16)
        self.assertEqual(cpp_scale.dtype, torch.uint8)


# ===========================================================================
# Reference for set_k (from index_buf_accessor.py SetK.torch_fast)
# ===========================================================================

def _set_k_torch(buf, loc, index_k, page_size, index_head_dim):
    """Reference implementation matching SetK.torch_fast."""
    (num_tokens_to_write,) = loc.shape
    buf_numel_per_page = buf.shape[1]
    num_k_bytes_per_token = index_head_dim

    loc_page_index = loc // page_size
    loc_token_offset_in_page = loc % page_size

    flat_buf = buf.flatten()
    flat_indices = (
        (loc_page_index * buf_numel_per_page)[:, None]
        + (loc_token_offset_in_page * num_k_bytes_per_token)[:, None]
        + torch.arange(num_k_bytes_per_token, dtype=torch.int32, device="cpu")[None, :]
    )
    num_k_bytes_total = num_tokens_to_write * num_k_bytes_per_token
    flat_indices = flat_indices.flatten()[:num_k_bytes_total]
    flat_buf[flat_indices] = index_k.view(torch.uint8).flatten()


def make_set_k_test_data(num_pages, page_size, num_tokens, index_head_dim=128):
    """Create test data for set_k_cpu."""
    buf_numel_per_page = page_size * index_head_dim + page_size * 4
    buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8)

    total_slots = num_pages * page_size
    assert num_tokens <= total_slots
    perm = torch.randperm(total_slots)[:num_tokens]
    loc = perm.to(torch.int64)

    index_k = torch.randint(0, 256, (num_tokens, index_head_dim), dtype=torch.uint8).view(fp8_dtype)

    return buf, loc, index_k


class TestSetK(CustomTestCase):
    num_pages_list = [4, 16]
    page_size_list = [1, 16, 64]
    num_tokens_list = [1, 7, 32]

    def _test_set_k(self, num_pages, page_size, num_tokens, index_head_dim=128):
        max_tokens = num_pages * page_size
        if num_tokens > max_tokens:
            num_tokens = max_tokens

        buf, loc, index_k = make_set_k_test_data(
            num_pages, page_size, num_tokens, index_head_dim
        )

        # Reference
        buf_ref = buf.clone()
        _set_k_torch(buf_ref, loc, index_k, page_size, index_head_dim)

        # C++ kernel
        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_cpu(
            buf_test, loc, index_k, page_size, index_head_dim
        )

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_k(self):
        for params in itertools.product(
            self.num_pages_list, self.page_size_list, self.num_tokens_list
        ):
            with self.subTest(
                num_pages=params[0], page_size=params[1], num_tokens=params[2]
            ):
                self._test_set_k(*params)

    def test_set_k_int32_loc(self):
        """Test with int32 loc tensor."""
        buf, loc, index_k = make_set_k_test_data(8, 64, 20)
        loc_i32 = loc.to(torch.int32)

        buf_ref = buf.clone()
        _set_k_torch(buf_ref, loc, index_k, 64, 128)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_cpu(buf_test, loc_i32, index_k, 64, 128)

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_k_large(self):
        """Larger stress test."""
        num_pages, page_size, num_tokens = 64, 64, 2048
        buf, loc, index_k = make_set_k_test_data(
            num_pages, page_size, num_tokens
        )

        buf_ref = buf.clone()
        _set_k_torch(buf_ref, loc, index_k, page_size, 128)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_k_cpu(buf_test, loc, index_k, page_size, 128)

        torch.testing.assert_close(buf_ref, buf_test)


# ===========================================================================
# Reference for set_s (from index_buf_accessor.py SetS.torch_fast)
# ===========================================================================

def _set_s_torch(buf, loc, index_k_scale, page_size, index_head_dim):
    """Reference implementation matching SetS.torch_fast."""
    (num_tokens_to_write,) = loc.shape
    buf_numel_per_page = buf.shape[1]
    num_s_bytes_per_token = 4
    s_offset_in_page = page_size * index_head_dim

    loc_page_index = loc // page_size
    loc_token_offset_in_page = loc % page_size

    flat_buf = buf.flatten()
    flat_indices = (
        (loc_page_index * buf_numel_per_page)[:, None]
        + s_offset_in_page
        + (loc_token_offset_in_page * num_s_bytes_per_token)[:, None]
        + torch.arange(num_s_bytes_per_token, dtype=torch.int32, device="cpu")[None, :]
    )
    number_s_bytes_total = num_tokens_to_write * num_s_bytes_per_token
    flat_indices = flat_indices.flatten()[:number_s_bytes_total]
    flat_buf[flat_indices] = index_k_scale.view(torch.uint8).flatten()


def make_set_s_test_data(num_pages, page_size, num_tokens, index_head_dim=128):
    """Create test data for set_s_cpu."""
    buf_numel_per_page = page_size * index_head_dim + page_size * 4
    buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8)

    total_slots = num_pages * page_size
    assert num_tokens <= total_slots
    perm = torch.randperm(total_slots)[:num_tokens]
    loc = perm.to(torch.int64)

    index_k_scale = torch.randn(num_tokens, dtype=torch.float32)

    return buf, loc, index_k_scale


class TestSetS(CustomTestCase):
    num_pages_list = [4, 16]
    page_size_list = [1, 16, 64]
    num_tokens_list = [1, 7, 32]

    def _test_set_s(self, num_pages, page_size, num_tokens, index_head_dim=128):
        max_tokens = num_pages * page_size
        if num_tokens > max_tokens:
            num_tokens = max_tokens

        buf, loc, index_k_scale = make_set_s_test_data(
            num_pages, page_size, num_tokens, index_head_dim
        )

        # Reference
        buf_ref = buf.clone()
        _set_s_torch(buf_ref, loc, index_k_scale, page_size, index_head_dim)

        # C++ kernel
        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_s_cpu(
            buf_test, loc, index_k_scale, page_size, index_head_dim
        )

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_s(self):
        for params in itertools.product(
            self.num_pages_list, self.page_size_list, self.num_tokens_list
        ):
            with self.subTest(
                num_pages=params[0], page_size=params[1], num_tokens=params[2]
            ):
                self._test_set_s(*params)

    def test_set_s_int32_loc(self):
        """Test with int32 loc tensor."""
        buf, loc, index_k_scale = make_set_s_test_data(8, 64, 20)
        loc_i32 = loc.to(torch.int32)

        buf_ref = buf.clone()
        _set_s_torch(buf_ref, loc, index_k_scale, 64, 128)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_s_cpu(buf_test, loc_i32, index_k_scale, 64, 128)

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_s_large(self):
        """Larger stress test."""
        num_pages, page_size, num_tokens = 64, 64, 2048
        buf, loc, index_k_scale = make_set_s_test_data(
            num_pages, page_size, num_tokens
        )

        buf_ref = buf.clone()
        _set_s_torch(buf_ref, loc, index_k_scale, page_size, 128)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_s_cpu(buf_test, loc, index_k_scale, page_size, 128)

        torch.testing.assert_close(buf_ref, buf_test)

    def test_set_s_2d_scale(self):
        """Test with 2D scale tensor (num_tokens, 1)."""
        num_pages, page_size, num_tokens = 8, 64, 20
        buf_numel_per_page = page_size * 128 + page_size * 4
        buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8)
        total_slots = num_pages * page_size
        perm = torch.randperm(total_slots)[:num_tokens]
        loc = perm.to(torch.int64)
        index_k_scale = torch.randn(num_tokens, 1, dtype=torch.float32)

        buf_ref = buf.clone()
        _set_s_torch(buf_ref, loc, index_k_scale.squeeze(1), page_size, 128)

        buf_test = buf.clone()
        torch.ops.sgl_kernel.set_s_cpu(buf_test, loc, index_k_scale, page_size, 128)

        torch.testing.assert_close(buf_ref, buf_test)


if __name__ == "__main__":
    unittest.main()
