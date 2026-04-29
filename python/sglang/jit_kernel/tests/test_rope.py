import sys

import pytest
import torch

try:
    import triton
except ImportError:
    triton = None

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=64, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=256, suite="nightly-kernel-1-gpu", nightly=True)

# Determine device: prefer XPU if available, otherwise CUDA
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = None

DTYPE = torch.bfloat16
MAX_SEQ_LEN = 131072  # common seq length
ROPE_BASE = 10000.0
CACHE_SIZE = 1024 * 128


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    """Create cos/sin cache compatible with SGLang layout: [max_pos, rotary_dim]."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_pos, rotary_dim]
    return cache


# ---------------------------------------------------------------------------
# Implementation wrappers
# ---------------------------------------------------------------------------


def sglang_jit_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from sglang.jit_kernel.rope import apply_rope_inplace

    apply_rope_inplace(q, k, cos_sin_cache, positions, is_neox=is_neox)


def flashinfer_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    head_size = q.shape[-1]
    # flashinfer expects [nnz, num_heads * head_size]
    q_2d = q.view(q.shape[0], -1)
    k_2d = k.view(k.shape[0], -1)
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q_2d,
        key=k_2d,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def torch_impl_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> None:
    # TODO: implement a pure-PyTorch reference for extra coverage
    pass


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

BS_LIST = [2**x for x in range(12)]
BS_LIST += [x + 1 for x in BS_LIST]  # odd sizes to stress non-aligned paths
BS_LIST = get_ci_test_range(BS_LIST, [1, 129, 2048, 2049])
NUM_KV_HEADS_LIST = get_ci_test_range([1, 2, 8], [1, 8])
GQA_RATIO = get_ci_test_range([1, 4, 8], [1, 8])
ROPE_DIM_LIST = get_ci_test_range([64, 128, 256, 512], [64, 256])
IS_NEOX_LIST = [False, True]
DTYPE_LIST = get_ci_test_range(
    [torch.bfloat16, torch.float16], [torch.bfloat16, torch.float16]
)
PARTIAL_ROPE_DIM_LIST = get_ci_test_range([64, 80, 96, 128], [64, 96])
HEAD_DIM_LIST = get_ci_test_range([64, 128, 256], [64, 256])


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("gqa_ratio", GQA_RATIO)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS_LIST)
@pytest.mark.parametrize("rope_dim", ROPE_DIM_LIST)
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_rope(
    batch_size: int,
    gqa_ratio: int,
    num_kv_heads: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
) -> None:
    if DEVICE is None:
        pytest.skip("No CUDA or XPU device available")
    
    num_qo_heads = num_kv_heads * gqa_ratio
    q = torch.randn(batch_size, num_qo_heads, rope_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, rope_dim, device=DEVICE, dtype=dtype)
    positions = torch.randint(
        0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=torch.int64
    )
    cos_sin_cache = create_cos_sin_cache(rope_dim)

    q_fi, k_fi = q.clone(), k.clone()
    q_jit, k_jit = q.clone(), k.clone()

    # Use flashinfer as reference only for CUDA
    if DEVICE == "cuda":
        flashinfer_rope(q_fi, k_fi, cos_sin_cache, positions, is_neox)
        sglang_jit_rope(q_jit, k_jit, cos_sin_cache, positions, is_neox)
        
        atol = rtol = 1e-2
        if triton is not None:
            triton.testing.assert_close(q_fi, q_jit, atol=atol, rtol=rtol)
            triton.testing.assert_close(k_fi, k_jit, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(q_fi, q_jit, atol=atol, rtol=rtol)
            torch.testing.assert_close(k_fi, k_jit, atol=atol, rtol=rtol)
    else:
        # For XPU, just ensure the kernel runs without errors
        # TODO: add a PyTorch reference implementation for XPU
        sglang_jit_rope(q_jit, k_jit, cos_sin_cache, positions, is_neox)
        # Basic sanity check: output should not be all zeros or NaN
        assert not torch.isnan(q_jit).any(), "q_jit contains NaN"
        assert not torch.isnan(k_jit).any(), "k_jit contains NaN"
        assert not (q_jit == 0).all(), "q_jit is all zeros"
        assert not (k_jit == 0).all(), "k_jit is all zeros"


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_rope_position_dtypes(dtype: torch.dtype) -> None:
    """Ensure both int32 and int64 position tensors work correctly."""
    if DEVICE is None:
        pytest.skip("No CUDA or XPU device available")
    
    batch_size, num_qo_heads, num_kv_heads, rope_dim = 16384, 16, 2, 128
    is_neox = True

    q = torch.randn(batch_size, num_qo_heads, rope_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, num_kv_heads, rope_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=dtype)
    cos_sin_cache = create_cos_sin_cache(rope_dim)

    q_fi, k_fi = q.clone(), k.clone()
    q_jit, k_jit = q.clone(), k.clone()

    if DEVICE == "cuda":
        flashinfer_rope(q_fi, k_fi, cos_sin_cache, positions.long(), is_neox)
        sglang_jit_rope(q_jit, k_jit, cos_sin_cache, positions, is_neox)

        atol = rtol = 1e-2
        if triton is not None:
            triton.testing.assert_close(q_fi, q_jit, atol=atol, rtol=rtol)
            triton.testing.assert_close(k_fi, k_jit, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(q_fi, q_jit, atol=atol, rtol=rtol)
            torch.testing.assert_close(k_fi, k_jit, atol=atol, rtol=rtol)
    else:
        # For XPU, just ensure the kernel runs without errors
        sglang_jit_rope(q_jit, k_jit, cos_sin_cache, positions, is_neox)
        assert not torch.isnan(q_jit).any()
        assert not torch.isnan(k_jit).any()


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
@pytest.mark.parametrize("rope_dim", PARTIAL_ROPE_DIM_LIST)
@pytest.mark.parametrize("head_dim", HEAD_DIM_LIST)
def test_partial_rope(batch_size: int, is_neox: bool, rope_dim: int, head_dim: int):
    if DEVICE is None:
        pytest.skip("No CUDA or XPU device available")
    
    if head_dim < rope_dim:
        pytest.skip("Invalid config: head_dim must be >= rope_dim.")
    num_qo_heads, num_kv_heads = 8, 2

    q = torch.randn(batch_size, num_qo_heads, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, MAX_SEQ_LEN, (batch_size,), device=DEVICE)
    cos_sin_cache = create_cos_sin_cache(rope_dim)

    q_fi, k_fi = q.clone(), k.clone()
    q_jit, k_jit = q.clone(), k.clone()
    rope = ..., slice(rope_dim)  # NOTE: flashinfer by default apply to first rope_dim

    if DEVICE == "cuda":
        flashinfer_rope(q_fi, k_fi, cos_sin_cache, positions.long(), is_neox)
        sglang_jit_rope(q_jit[rope], k_jit[rope], cos_sin_cache, positions, is_neox)

        atol = rtol = 1e-2
        if triton is not None:
            triton.testing.assert_close(q_fi, q_jit, atol=atol, rtol=rtol)
            triton.testing.assert_close(k_fi, k_jit, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(q_fi, q_jit, atol=atol, rtol=rtol)
            torch.testing.assert_close(k_fi, k_jit, atol=atol, rtol=rtol)
    else:
        # For XPU, just ensure the kernel runs without errors
        sglang_jit_rope(q_jit[rope], k_jit[rope], cos_sin_cache, positions, is_neox)
        assert not torch.isnan(q_jit).any()
        assert not torch.isnan(k_jit).any()


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("gqa_ratio", GQA_RATIO)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS_LIST)
@pytest.mark.parametrize("rope_dim", ROPE_DIM_LIST)
@pytest.mark.parametrize("is_neox", IS_NEOX_LIST)
def test_fused_rope_store(
    batch_size: int,
    gqa_ratio: int,
    num_kv_heads: int,
    rope_dim: int,
    is_neox: bool,
) -> None:
    """Test fused RoPE + KV cache store against separate RoPE + manual store."""
    from sglang.jit_kernel.rope import apply_rope_inplace_with_kvcache
    
    if DEVICE is None:
        pytest.skip("No CUDA or XPU device available")

    num_qo_heads = num_kv_heads * gqa_ratio
    dtype = DTYPE

    q = torch.randn(batch_size, num_qo_heads, rope_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, rope_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, rope_dim, device=DEVICE, dtype=dtype)
    positions = torch.randint(
        0, MAX_SEQ_LEN, (batch_size,), device=DEVICE, dtype=torch.int64
    )
    out_loc = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[:batch_size]
    cos_sin_cache = create_cos_sin_cache(rope_dim)

    row_size = num_kv_heads * rope_dim
    k_cache_ref = torch.zeros(CACHE_SIZE, row_size, device=DEVICE, dtype=dtype)
    v_cache_ref = torch.zeros(CACHE_SIZE, row_size, device=DEVICE, dtype=dtype)
    k_cache_fused = torch.zeros(CACHE_SIZE, row_size, device=DEVICE, dtype=dtype)
    v_cache_fused = torch.zeros(CACHE_SIZE, row_size, device=DEVICE, dtype=dtype)

    if DEVICE == "cuda":
        # --- reference: separate RoPE then manual scatter ---
        q_ref, k_ref = q.clone(), k.clone()
        flashinfer_rope(q_ref, k_ref, cos_sin_cache, positions, is_neox)
        k_cache_ref[out_loc] = k_ref.view(batch_size, -1)
        v_cache_ref[out_loc] = v.view(batch_size, -1)

        # --- fused kernel ---
        q_fused, k_fused = q.clone(), k.clone()
        v_fused = v.clone()
        apply_rope_inplace_with_kvcache(
            q_fused,
            k_fused,
            v_fused,
            k_cache_fused,
            v_cache_fused,
            cos_sin_cache,
            positions,
            out_loc,
            is_neox=is_neox,
        )

        atol = rtol = 1e-2
        # q should match RoPE-only result
        if triton is not None:
            triton.testing.assert_close(q_ref, q_fused, atol=atol, rtol=rtol)
            # k_cache should contain the rotated k
            triton.testing.assert_close(
                k_cache_ref[out_loc], k_cache_fused[out_loc], atol=atol, rtol=rtol
            )
        else:
            torch.testing.assert_close(q_ref, q_fused, atol=atol, rtol=rtol)
            torch.testing.assert_close(
                k_cache_ref[out_loc], k_cache_fused[out_loc], atol=atol, rtol=rtol
            )
        # v_cache should be an exact copy
        assert torch.all(v_cache_ref[out_loc] == v_cache_fused[out_loc]), "v_cache mismatch"
    else:
        # For XPU, just ensure the kernel runs without errors
        q_fused, k_fused = q.clone(), k.clone()
        v_fused = v.clone()
        apply_rope_inplace_with_kvcache(
            q_fused,
            k_fused,
            v_fused,
            k_cache_fused,
            v_cache_fused,
            cos_sin_cache,
            positions,
            out_loc,
            is_neox=is_neox,
        )
        # Basic sanity checks
        assert not torch.isnan(q_fused).any()
        assert not torch.isnan(k_fused).any()
        assert not torch.isnan(k_cache_fused).any()
        assert not torch.isnan(v_cache_fused).any()
        # Check that cache was actually written to
        assert not (k_cache_fused[out_loc] == 0).all(), "k_cache not written"
        assert not (v_cache_fused[out_loc] == 0).all(), "v_cache not written"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
