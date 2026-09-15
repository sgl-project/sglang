import sys
from types import SimpleNamespace

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    CPUFP8KVCacheMethod,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=14, suite="stage-a-test-cpu-intel")

torch.manual_seed(42)

DEVICE = "cpu"
CACHE_SIZE = 4096

# for fp8 KV stored as uint8, e.g. float8_e4m3fn and float8_e5m2
DTYPES = [torch.float16, torch.bfloat16, torch.uint8]
DTYPE_IDS = ["float16", "bfloat16", "uint8"]


def _store_cache_cpu(k, v, k_cache, v_cache, indices):
    row_dim = k.size(1) * k.size(2)
    torch.ops.sgl_kernel.store_cache_cpu(k, v, k_cache, v_cache, indices, row_dim)


def _random_tensor(shape, dtype):
    """FP8 KV is stored as uint8; randn is not implemented for Byte."""
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=DEVICE)
    return torch.randn(shape, dtype=dtype, device=DEVICE)


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


def make_test_data(
    num_pages, page_size, num_tokens, nope_dim=448, rope_dim=64, scale_dim=7
):
    """Create test data matching the buffer layout."""
    nope_rope_bytes_per_token = nope_dim + rope_dim * 2
    s_bytes_per_token = scale_dim + 1
    buf_numel_per_page = (
        page_size * nope_rope_bytes_per_token + page_size * s_bytes_per_token
    )

    buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8)

    # Generate random non-overlapping locations
    total_slots = num_pages * page_size
    assert num_tokens <= total_slots
    perm = torch.randperm(total_slots)[:num_tokens]
    loc = perm.to(torch.int64)

    k_nope = torch.randint(0, 256, (num_tokens, nope_dim), dtype=torch.uint8).view(
        fp8_dtype
    )
    k_rope = torch.randn(num_tokens, rope_dim, dtype=torch.bfloat16)
    scale_k_nope = torch.randint(0, 256, (num_tokens, scale_dim), dtype=torch.uint8)

    return buf, loc, k_nope, k_rope, scale_k_nope


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 16, 32])
@pytest.mark.parametrize("batch_size", [1, 7, 133])
def test_store_cache(batch_size, num_heads, head_dim, dtype):
    shape = (batch_size, num_heads, head_dim)
    cache_shape = (CACHE_SIZE, num_heads, head_dim)
    k = _random_tensor(shape, dtype)
    v = _random_tensor(shape, dtype)
    k_cache = _random_tensor(cache_shape, dtype)
    v_cache = _random_tensor(cache_shape, dtype)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[:batch_size]

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_ref[indices] = k
    v_cache_ref[indices] = v

    _store_cache_cpu(k, v, k_cache, v_cache, indices)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("batch_size", [11])
def test_store_cache_int32_indices(batch_size, num_heads, head_dim, dtype):
    shape = (batch_size, num_heads, head_dim)
    cache_shape = (CACHE_SIZE, num_heads, head_dim)
    k = _random_tensor(shape, dtype)
    v = _random_tensor(shape, dtype)
    k_cache = _random_tensor(cache_shape, dtype)
    v_cache = _random_tensor(cache_shape, dtype)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[
        :batch_size
    ].to(torch.int32)

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_ref[indices.long()] = k
    v_cache_ref[indices.long()] = v

    _store_cache_cpu(k, v, k_cache, v_cache, indices)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_pages", [4, 16])
def test_set_k_and_s(num_pages, page_size, num_tokens):
    num_tokens = min(num_tokens, num_pages * page_size)

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


def test_set_k_and_s_int32_loc():
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


def test_set_k_and_s_large():
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


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128, 512])
def test_quant_various_sizes(num_tokens):
    k_bf16 = torch.randn(num_tokens, 512, dtype=torch.bfloat16)

    ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
    cpp_nope, cpp_rope, cpp_scale = (
        torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
    )

    torch.testing.assert_close(ref_rope, cpp_rope)
    torch.testing.assert_close(ref_scale, cpp_scale)
    torch.testing.assert_close(ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8))


def test_quant_small_values():
    """Test with very small values that exercise the EPS clamp."""
    k_bf16 = torch.randn(16, 512, dtype=torch.bfloat16) * 1e-6
    ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
    cpp_nope, cpp_rope, cpp_scale = (
        torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
    )
    torch.testing.assert_close(ref_rope, cpp_rope)
    torch.testing.assert_close(ref_scale, cpp_scale)
    torch.testing.assert_close(ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8))


def test_quant_large_values():
    """Test with large values."""
    k_bf16 = torch.randn(16, 512, dtype=torch.bfloat16) * 100.0
    ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
    cpp_nope, cpp_rope, cpp_scale = (
        torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
    )
    torch.testing.assert_close(ref_rope, cpp_rope)
    torch.testing.assert_close(ref_scale, cpp_scale)
    torch.testing.assert_close(ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8))


def test_quant_zeros():
    """Test with zero input."""
    k_bf16 = torch.zeros(8, 512, dtype=torch.bfloat16)
    ref_nope, ref_rope, ref_scale = quant_to_nope_fp8_rope_bf16_pack_ref(k_bf16)
    cpp_nope, cpp_rope, cpp_scale = (
        torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
    )
    torch.testing.assert_close(ref_rope, cpp_rope)
    torch.testing.assert_close(ref_scale, cpp_scale)
    torch.testing.assert_close(ref_nope.view(torch.uint8), cpp_nope.view(torch.uint8))


def test_output_shapes_and_dtypes():
    """Verify output shapes and dtypes."""
    num_tokens = 16
    k_bf16 = torch.randn(num_tokens, 512, dtype=torch.bfloat16)
    cpp_nope, cpp_rope, cpp_scale = (
        torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(k_bf16)
    )

    assert cpp_nope.shape == (num_tokens, 448)
    assert cpp_rope.shape == (num_tokens, 64)
    assert cpp_scale.shape == (num_tokens, 7)

    assert cpp_nope.dtype == fp8_dtype
    assert cpp_rope.dtype == torch.bfloat16
    assert cpp_scale.dtype == torch.uint8


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

    index_k = torch.randint(
        0, 256, (num_tokens, index_head_dim), dtype=torch.uint8
    ).view(fp8_dtype)

    return buf, loc, index_k


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("num_pages", [4, 16])
def test_set_k(num_pages, page_size, num_tokens, index_head_dim=128):
    num_tokens = min(num_tokens, num_pages * page_size)

    buf, loc, index_k = make_set_k_test_data(
        num_pages, page_size, num_tokens, index_head_dim
    )

    # Reference
    buf_ref = buf.clone()
    _set_k_torch(buf_ref, loc, index_k, page_size, index_head_dim)

    # C++ kernel
    buf_test = buf.clone()
    torch.ops.sgl_kernel.set_k_cpu(buf_test, loc, index_k, page_size, index_head_dim)

    torch.testing.assert_close(buf_ref, buf_test)


def test_set_k_int32_loc():
    """Test with int32 loc tensor."""
    buf, loc, index_k = make_set_k_test_data(8, 64, 20)
    loc_i32 = loc.to(torch.int32)

    buf_ref = buf.clone()
    _set_k_torch(buf_ref, loc, index_k, 64, 128)

    buf_test = buf.clone()
    torch.ops.sgl_kernel.set_k_cpu(buf_test, loc_i32, index_k, 64, 128)

    torch.testing.assert_close(buf_ref, buf_test)


def test_set_k_large():
    """Larger stress test."""
    num_pages, page_size, num_tokens = 64, 64, 2048
    buf, loc, index_k = make_set_k_test_data(num_pages, page_size, num_tokens)

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


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("num_pages", [4, 16])
def test_set_s(num_pages, page_size, num_tokens, index_head_dim=128):
    num_tokens = min(num_tokens, num_pages * page_size)

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


def test_set_s_int32_loc():
    """Test with int32 loc tensor."""
    buf, loc, index_k_scale = make_set_s_test_data(8, 64, 20)
    loc_i32 = loc.to(torch.int32)

    buf_ref = buf.clone()
    _set_s_torch(buf_ref, loc, index_k_scale, 64, 128)

    buf_test = buf.clone()
    torch.ops.sgl_kernel.set_s_cpu(buf_test, loc_i32, index_k_scale, 64, 128)

    torch.testing.assert_close(buf_ref, buf_test)


def test_set_s_large():
    """Larger stress test."""
    num_pages, page_size, num_tokens = 64, 64, 2048
    buf, loc, index_k_scale = make_set_s_test_data(num_pages, page_size, num_tokens)

    buf_ref = buf.clone()
    _set_s_torch(buf_ref, loc, index_k_scale, page_size, 128)

    buf_test = buf.clone()
    torch.ops.sgl_kernel.set_s_cpu(buf_test, loc, index_k_scale, page_size, 128)

    torch.testing.assert_close(buf_ref, buf_test)


def test_set_s_2d_scale():
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


@pytest.mark.parametrize(
    ("k_scale", "v_scale"),
    [(None, None), (0.5, 0.25)],
    ids=["unit-scale-default", "non-unit-static-scale"],
)
def test_mha_fp8_e4m3_pool_decode_numerics(k_scale, v_scale):
    seq_len = 16
    num_heads = 2
    head_dim = 64
    num_kv_splits = 8
    sm_scale = head_dim**-0.5
    pool = MHATokenToKVPool(
        size=seq_len,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        head_num=num_heads,
        head_dim=head_dim,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
        quant_method=CPUFP8KVCacheMethod(),
    )
    layer = SimpleNamespace(layer_id=0)
    loc = torch.arange(seq_len, dtype=torch.int64, device=DEVICE)
    cache_k = torch.randn(
        (seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    cache_v = torch.randn(
        (seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale=k_scale, v_scale=v_scale)

    effective_k_scale = 1.0 if k_scale is None else k_scale
    effective_v_scale = 1.0 if v_scale is None else v_scale
    k_dequant = (pool.get_key_buffer(0).float() * effective_k_scale).to(torch.bfloat16)
    v_dequant = (pool.get_value_buffer(0).float() * effective_v_scale).to(
        torch.bfloat16
    )
    query = torch.randn((1, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    output = torch.empty_like(query)
    req_to_token = loc.to(torch.int32).unsqueeze(0)
    req_pool_indices = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.full((1,), seq_len, dtype=torch.int64, device=DEVICE)
    attn_logits = torch.empty(
        (1, num_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        pool.get_key_buffer(0),
        pool.get_value_buffer(0),
        effective_k_scale,
        effective_v_scale,
        output,
        None,
        None,
        None,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        0.0,
        False,
        0,
        None,
        None,
    )

    output_ref = (
        torch.nn.functional.scaled_dot_product_attention(
            query.movedim(0, 1).unsqueeze(0),
            k_dequant[:seq_len].movedim(0, 1).unsqueeze(0),
            v_dequant[:seq_len].movedim(0, 1).unsqueeze(0),
            scale=sm_scale,
        )
        .squeeze(0)
        .movedim(1, 0)
    )
    torch.testing.assert_close(output, output_ref, atol=3e-2, rtol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
