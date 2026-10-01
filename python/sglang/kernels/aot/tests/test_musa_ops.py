import importlib
import sys

import pytest
import torch

if not (hasattr(torch, "musa") and torch.musa.is_available()):
    pytest.skip("MUSA device not available", allow_module_level=True)

sgl_kernel = importlib.import_module("sgl_kernel")
DEVICE = torch.device("musa:0")

# Each tuple is (public Python API name, torch dispatcher op name).
MUSA_OPS = [
    ("musa_rotary_embedding_contiguous", "musa_rotary_embedding_contiguous"),
    (
        "musa_batched_rotary_embedding_contiguous",
        "musa_batched_rotary_embedding_contiguous",
    ),
    ("musa_fused_mul_add", "musa_fused_mul_add"),
    ("musa_fused_gemv", "musa_fused_gemv"),
    ("musa_fused_moe_gemv", "musa_fused_moe_gemv"),
    ("top_k_top_p_sampling_from_probs", "musa_top_k_top_p_sampling_from_probs"),
]


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


def create_cos_sin_cache(
    max_position: int, rot_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device=DEVICE, dtype=dtype)


def apply_rope_reference(
    tensor: torch.Tensor,
    cache_rows: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
) -> torch.Tensor:
    half = rot_dim // 2
    cos = cache_rows[:, None, :half].float()
    sin = cache_rows[:, None, half:rot_dim].float()
    rotary = tensor[..., :rot_dim].float()

    if is_neox:
        x1, x2 = rotary[..., :half], rotary[..., half:rot_dim]
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1, x2 = rotary[..., 0::2], rotary[..., 1::2]
        rotated = torch.stack(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1
        ).flatten(-2)

    result = tensor.clone()
    result[..., :rot_dim] = rotated.to(tensor.dtype)
    return result


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-5, 1e-6
    if dtype is torch.bfloat16:
        return 1e-2, 2e-2
    return 1e-3, 1e-3


@pytest.mark.parametrize("python_name,torch_op_name", MUSA_OPS)
def test_musa_op_registration(python_name: str, torch_op_name: str):
    assert hasattr(sgl_kernel, python_name)
    assert hasattr(torch.ops.sgl_kernel, torch_op_name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize(
    "num_tokens,head_size,rot_dim",
    [(1, 64, 64), (7, 64, 64), (7, 128, 64)],
)
def test_musa_rotary_embedding_contiguous(
    dtype: torch.dtype,
    is_neox: bool,
    num_tokens: int,
    head_size: int,
    rot_dim: int,
):
    num_query_heads, num_kv_heads = 4, 2
    positions = torch.randint(0, 32, (num_tokens,), dtype=torch.int64, device=DEVICE)
    query = torch.randn(
        num_tokens, num_query_heads, head_size, dtype=dtype, device=DEVICE
    )
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=DEVICE)
    cache = create_cos_sin_cache(32, rot_dim, dtype)

    positions_before = positions.clone()
    query_before = query.clone()
    key_before = key.clone()
    cache_before = cache.clone()
    query_expected = apply_rope_reference(
        query_before, cache[positions], is_neox, rot_dim
    )
    key_expected = apply_rope_reference(key_before, cache[positions], is_neox, rot_dim)

    result = sgl_kernel.musa_rotary_embedding_contiguous(
        positions, query, key, head_size, cache, is_neox
    )
    torch.musa.synchronize()

    assert result is None
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(query, query_expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(key, key_expected, rtol=rtol, atol=atol)
    assert torch.equal(positions, positions_before)
    assert torch.equal(cache, cache_before)
    if head_size > rot_dim:
        assert torch.equal(query[..., rot_dim:], query_before[..., rot_dim:])
        assert torch.equal(key[..., rot_dim:], key_before[..., rot_dim:])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("head_size,rot_dim", [(64, 64), (128, 64)])
def test_musa_batched_rotary_embedding_contiguous(
    dtype: torch.dtype, is_neox: bool, head_size: int, rot_dim: int
):
    num_tokens, num_query_heads, num_kv_heads = 4, 4, 2
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=DEVICE)
    offsets = torch.tensor([0, 8, 16, 24], dtype=torch.int64, device=DEVICE)
    effective_positions = positions + offsets
    query = torch.randn(
        num_tokens, num_query_heads, head_size, dtype=dtype, device=DEVICE
    )
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=DEVICE)
    cache = create_cos_sin_cache(32, rot_dim, dtype)

    positions_before = positions.clone()
    offsets_before = offsets.clone()
    query_before = query.clone()
    key_before = key.clone()
    cache_before = cache.clone()
    cache_rows = cache[effective_positions]
    query_expected = apply_rope_reference(query_before, cache_rows, is_neox, rot_dim)
    key_expected = apply_rope_reference(key_before, cache_rows, is_neox, rot_dim)

    result = sgl_kernel.musa_batched_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cache,
        is_neox,
        rot_dim,
        offsets,
    )
    torch.musa.synchronize()

    assert result is None
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(query, query_expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(key, key_expected, rtol=rtol, atol=atol)
    assert torch.equal(positions, positions_before)
    assert torch.equal(offsets, offsets_before)
    assert torch.equal(cache, cache_before)
    if head_size > rot_dim:
        assert torch.equal(query[..., rot_dim:], query_before[..., rot_dim:])
        assert torch.equal(key[..., rot_dim:], key_before[..., rot_dim:])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [-0.5, 0.0, 1.25])
@pytest.mark.parametrize("shape", [(16,), (4, 128), (4608,)])
def test_musa_fused_mul_add(dtype: torch.dtype, scale: float, shape: tuple[int, ...]):
    self_tensor = torch.randn(shape, dtype=dtype, device=DEVICE)
    bias = torch.randn(shape, dtype=dtype, device=DEVICE)
    self_before = self_tensor.clone()
    bias_before = bias.clone()
    expected = (self_tensor.float() * scale + bias.float()).to(dtype)

    result = sgl_kernel.musa_fused_mul_add(self_tensor, bias, scale, accurate=True)
    torch.musa.synchronize()

    assert result is not self_tensor
    assert result is not bias
    assert result.data_ptr() != self_tensor.data_ptr()
    assert result.data_ptr() != bias.data_ptr()
    assert torch.equal(self_tensor, self_before)
    assert torch.equal(bias, bias_before)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_musa_fused_mul_add_inplace_fallback(dtype: torch.dtype):
    scale = 1.25
    self_tensor = torch.randn((3, 17), dtype=dtype, device=DEVICE)
    bias = torch.randn((3, 17), dtype=dtype, device=DEVICE)
    self_before = self_tensor.clone()
    expected = (self_tensor.float() * scale + bias.float()).to(dtype)

    result = sgl_kernel.musa_fused_mul_add(self_tensor, bias, scale, accurate=False)
    torch.musa.synchronize()

    assert result is bias
    assert torch.equal(self_tensor, self_before)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
