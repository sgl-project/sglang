"""
Unit test for flashinfer batch paged prefill with native NVFP4 KV cache.

Tests that BatchPrefillWithPagedKVCacheWrapper correctly handles
float4_e2m1fn_x2 KV cache input, producing attention outputs
consistent with a BF16 dequant reference.
"""

import sys

import pytest
import torch

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    _flashinfer_available = True
except ImportError:
    _flashinfer_available = False

try:
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant

    _nvfp4_kernel_available = True
except ImportError:
    _nvfp4_kernel_available = False

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
SCALE_BLOCK_SIZE = 16  # NVFP4 per-block scale granularity


def _nvfp4_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and _flashinfer_available
        and _nvfp4_kernel_available
    )


def _quantize_kv_to_nvfp4(
    kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Quantize a KV tensor to NVFP4 format.

    Args:
        kv: BF16 tensor of shape [num_tokens, num_heads, head_dim]

    Returns:
        kv_fp4:    uint8 packed FP4, shape [num_tokens, num_heads, head_dim//2]
        kv_scale:  FP8 E4M3 per-block scales, shape [num_tokens, num_heads, head_dim//SCALE_BLOCK_SIZE]
        global_scale: float32 global (per-tensor) scale
    """
    num_tokens, num_heads, head_dim = kv.shape
    kv_flat = kv.reshape(num_tokens * num_heads, head_dim).to(torch.float32)

    tensor_amax = kv_flat.abs().max().item()
    global_scale_val = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max(tensor_amax, 1e-6)
    global_scale = torch.tensor(global_scale_val, dtype=torch.float32, device=kv.device)

    kv_fp4_flat, kv_scale_flat = scaled_fp4_quant(kv_flat.to(torch.bfloat16), global_scale)
    # kv_fp4_flat: [num_tokens*num_heads, head_dim//2] uint8
    # kv_scale_flat: swizzled FP8 scale — reshape after recovering

    kv_fp4 = kv_fp4_flat.reshape(num_tokens, num_heads, head_dim // 2)

    # Recover per-block scales in linear layout: [rows, head_dim//SCALE_BLOCK_SIZE]
    scale_cols = head_dim // SCALE_BLOCK_SIZE
    rows = num_tokens * num_heads
    rounded_rows = ((rows + 128 - 1) // 128) * 128
    rounded_scale_cols = ((scale_cols + 4 - 1) // 4) * 4
    kv_scale_swizzled = kv_scale_flat.reshape(
        1, rounded_rows // 128, rounded_scale_cols // 4, 32, 4, 4
    )
    kv_scale_linear = (
        kv_scale_swizzled.permute(0, 1, 4, 3, 2, 5)
        .reshape(rounded_rows, rounded_scale_cols)
        .to(torch.float8_e4m3fn)
    )
    kv_scale = kv_scale_linear[:rows, :scale_cols].reshape(
        num_tokens, num_heads, scale_cols
    )

    return kv_fp4, kv_scale, global_scale_val


def _dequantize_kv(
    kv_fp4: torch.Tensor,
    kv_scale: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    """
    Dequantize NVFP4 KV back to BF16 for reference computation.

    Args:
        kv_fp4:    uint8 packed FP4, shape [..., head_dim//2]
        kv_scale:  FP8 E4M3 per-block scales, shape [..., head_dim//SCALE_BLOCK_SIZE]
        global_scale: float32 global scale used during quantization

    Returns:
        BF16 tensor of same shape as original (unpacked head_dim)
    """
    E2M1_TO_FLOAT32 = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    lut = torch.tensor(E2M1_TO_FLOAT32, device=kv_fp4.device, dtype=torch.float32)

    packed = kv_fp4
    lo = (packed & 0xF).to(torch.long)
    hi = ((packed >> 4) & 0xF).to(torch.long)
    # Interleave: lo is the first token, hi is the second (little-endian packing)
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [..., head_dim]
    kv_float = lut[vals]

    # Apply per-block scales: scale shape [..., head_dim//SCALE_BLOCK_SIZE] → [..., head_dim]
    scale_f32 = kv_scale.to(torch.float32)
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK_SIZE, dim=-1)
    # global_scale encodes how fp4 values map to the original range
    inv_global = 1.0 / global_scale
    kv_dequant = (kv_float * scale_expanded * inv_global).to(torch.bfloat16)
    return kv_dequant


def _ref_attention_bfloat16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """
    Reference attention output computed in BF16 with explicit causal masking.
    Operates on the paged KV layout, gathering K/V by kv_indices.
    """
    bs = len(qo_indptr) - 1
    output_list = []

    for i in range(bs):
        q_start, q_end = int(qo_indptr[i]), int(qo_indptr[i + 1])
        kv_start, kv_end = int(kv_indptr[i]), int(kv_indptr[i + 1])
        qi = q[q_start:q_end]  # [qo_len_i, num_qo_heads, head_dim]
        slot_ids = kv_indices[kv_start:kv_end]  # [kv_len_i]
        ki = k[slot_ids]  # [kv_len_i, num_kv_heads, head_dim]
        vi = v[slot_ids]  # [kv_len_i, num_kv_heads, head_dim]

        qo_len = q_end - q_start
        kv_len = kv_end - kv_start
        num_qo_heads, num_kv_heads = qi.shape[1], ki.shape[1]
        head_dim = qi.shape[2]

        # GQA: expand kv heads
        if num_kv_heads < num_qo_heads:
            group = num_qo_heads // num_kv_heads
            ki = ki.repeat_interleave(group, dim=1)
            vi = vi.repeat_interleave(group, dim=1)

        # [qo_len, H, D] × [kv_len, H, D]^T → [H, qo_len, kv_len]
        qi_t = qi.permute(1, 0, 2).float()
        ki_t = ki.permute(1, 2, 0).float()
        scores = torch.bmm(qi_t, ki_t) * sm_scale  # [H, qo_len, kv_len]

        # Causal mask: position i in qo attends to positions ≤ (kv_len - qo_len + i)
        causal_offset = kv_len - qo_len
        mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device).tril(
            causal_offset
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        vi_t = vi.permute(1, 0, 2).float()
        out_i = torch.bmm(attn, vi_t).permute(1, 0, 2)  # [qo_len, H, D]
        output_list.append(out_i.to(torch.bfloat16))

    return torch.cat(output_list, dim=0)


@pytest.mark.skipif(
    not _nvfp4_supported(),
    reason="Requires CUDA compute >= 10.0, flashinfer, and sgl-kernel with NVFP4 support",
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
def test_nvfp4_native_prefill(
    batch_size: int, kv_len: int, qo_len: int, page_size: int
):
    """
    Verify that flashinfer BatchPrefillWithPagedKVCacheWrapper produces
    correct outputs with native NVFP4 (float4_e2m1fn_x2) KV cache input.

    The test quantizes BF16 KV to NVFP4, runs the native FP4 prefill kernel,
    then dequantizes and runs a BF16 reference to check consistency within
    the quantization error tolerance.
    """
    if qo_len > kv_len:
        pytest.skip("qo_len must be <= kv_len (prefix + new tokens)")

    torch.manual_seed(42)
    device = "cuda"
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128

    # --- Build paged KV layout ---
    # Each request has kv_len KV tokens. Allocate a flat pool.
    total_pages_per_req = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * total_pages_per_req + 1  # +1 for dummy page 0

    # Flat KV pool: [total_slots, num_kv_heads, head_dim] in BF16
    total_slots = total_pages * page_size
    k_bf16 = torch.randn(total_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    v_bf16 = torch.randn(total_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1

    # Query: [total_qo_tokens, num_qo_heads, head_dim]
    total_qo_tokens = batch_size * qo_len
    q = torch.randn(total_qo_tokens, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1

    # Build paged indices: each request occupies contiguous pages starting at page (1 + i*total_pages_per_req)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device) * kv_len
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device) * qo_len

    kv_indices_list = []
    for i in range(batch_size):
        page_offset = 1 + i * total_pages_per_req  # skip dummy page 0
        slot_ids = []
        for p in range(total_pages_per_req):
            base = (page_offset + p) * page_size
            for s in range(page_size):
                if p * page_size + s < kv_len:
                    slot_ids.append(base + s)
        kv_indices_list.extend(slot_ids[:kv_len])
    kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=device)

    last_page_tokens = kv_len % page_size or page_size
    kv_last_page_len = torch.full(
        (batch_size,), last_page_tokens, dtype=torch.int32, device=device
    )

    # --- Quantize K/V to NVFP4 ---
    k_fp4, k_block_scale, k_global_scale = _quantize_kv_to_nvfp4(k_bf16)
    v_fp4, v_block_scale, v_global_scale = _quantize_kv_to_nvfp4(v_bf16)

    # Cast to float4_e2m1fn_x2 for flashinfer
    k_fp4_typed = k_fp4.view(torch.float4_e2m1fn_x2)
    v_fp4_typed = v_fp4.view(torch.float4_e2m1fn_x2)

    # --- Run flashinfer native NVFP4 prefill ---
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

    sm_scale = head_dim ** -0.5

    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float4_e2m1fn_x2,
    )

    output_fp4 = wrapper.forward(
        q,
        (k_fp4_typed, v_fp4_typed),
        causal=True,
        sm_scale=sm_scale,
        k_scale=k_global_scale,
        v_scale=v_global_scale,
    )
    wrapper.end_forward()

    # --- Reference: dequant to BF16 and run attention ---
    k_dequant = _dequantize_kv(k_fp4, k_block_scale, k_global_scale)
    v_dequant = _dequantize_kv(v_fp4, v_block_scale, v_global_scale)

    output_ref = _ref_attention_bfloat16(
        q, k_dequant, v_dequant, qo_indptr.cpu(), kv_indptr.cpu(), kv_indices.cpu(), sm_scale
    )

    # --- Compare: allow for quantization error ---
    torch.testing.assert_close(
        output_fp4.float(),
        output_ref.float(),
        atol=0.05,
        rtol=0.05,
        msg=f"NVFP4 prefill mismatch: batch={batch_size}, kv_len={kv_len}, qo_len={qo_len}, page_size={page_size}",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
