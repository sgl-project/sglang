"""Fused FP4 dequant + GEMV Triton kernel for SM120 MoE decode.

FP4 (MXFP4/E2M1) format:
  - Two 4-bit values packed per int8 byte (low nibble first, high nibble second)
  - E2M1 magnitudes: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
  - Per-block scale (block_size=32) in float32
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_dequant(nibs):
    """Inline E2M1 dequantization using predicated tl.where."""
    sign = 1.0 - 2.0 * ((nibs >> 3) & 1).to(tl.float32)
    mag = nibs & 0x07
    val = tl.where(mag == 0, 0.0,
          tl.where(mag == 1, 0.5,
          tl.where(mag == 2, 1.0,
          tl.where(mag == 3, 1.5,
          tl.where(mag == 4, 2.0,
          tl.where(mag == 5, 3.0,
          tl.where(mag == 6, 4.0, 6.0)))))))
    return sign * val


@triton.jit
def _fp4_gemv_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_HALF: tl.constexpr,
    NUM_BLOCKS_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HALF_BK: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_start = pid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N

    acc = tl.zeros((TILE_N,), dtype=tl.float32)

    for bk in range(NUM_BLOCKS_K):
        k_start = bk * BLOCK_K

        s = tl.load(scale_ptr + n_offs * NUM_BLOCKS_K + bk, mask=n_mask, other=1.0)

        byte_offs = k_start // 2 + tl.arange(0, HALF_BK)
        w_ptrs = weight_ptr + n_offs[:, None] * K_HALF + byte_offs[None, :]
        w_bytes = tl.load(w_ptrs, mask=n_mask[:, None], other=0).to(tl.int32)

        dq_even = _e2m1_dequant(w_bytes & 0x0F) * s[:, None]
        dq_odd = _e2m1_dequant((w_bytes >> 4) & 0x0F) * s[:, None]

        half_idx = tl.arange(0, HALF_BK)
        even_k_offs = k_start + 2 * half_idx
        odd_k_offs = even_k_offs + 1
        x_even = tl.load(input_ptr + even_k_offs).to(tl.float32)
        x_odd = tl.load(input_ptr + odd_k_offs).to(tl.float32)

        acc += tl.sum(dq_even * x_even[None, :], axis=1)
        acc += tl.sum(dq_odd * x_odd[None, :], axis=1)

    tl.store(output_ptr + n_offs, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _fp4_gemv_batched_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_HALF: tl.constexpr,
    NUM_BLOCKS_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HALF_BK: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Batched GEMV: same input vector x for all E experts."""
    eid = tl.program_id(0)
    pid = tl.program_id(1)
    n_start = pid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N

    # Expert offset computed as scalar for each row
    e_off_w = eid * N * K_HALF
    e_off_s = eid * N * NUM_BLOCKS_K

    acc = tl.zeros((TILE_N,), dtype=tl.float32)

    for bk in range(NUM_BLOCKS_K):
        k_start = bk * BLOCK_K

        # Scale: scalar e_off_s + per-row offset
        s_ptrs = scale_ptr + e_off_s + n_offs * NUM_BLOCKS_K + bk
        s = tl.load(s_ptrs, mask=n_mask, other=1.0)

        byte_offs = k_start // 2 + tl.arange(0, HALF_BK)
        # Weight: scalar e_off_w + per-row offset + per-col offset
        w_ptrs = weight_ptr + e_off_w + n_offs[:, None] * K_HALF + byte_offs[None, :]
        w_bytes = tl.load(w_ptrs, mask=n_mask[:, None], other=0).to(tl.int32)

        dq_even = _e2m1_dequant(w_bytes & 0x0F) * s[:, None]
        dq_odd = _e2m1_dequant((w_bytes >> 4) & 0x0F) * s[:, None]

        half_idx = tl.arange(0, HALF_BK)
        even_k_offs = k_start + 2 * half_idx
        odd_k_offs = even_k_offs + 1
        x_even = tl.load(input_ptr + even_k_offs).to(tl.float32)
        x_odd = tl.load(input_ptr + odd_k_offs).to(tl.float32)

        acc += tl.sum(dq_even * x_even[None, :], axis=1)
        acc += tl.sum(dq_odd * x_odd[None, :], axis=1)

    tl.store(output_ptr + eid * N + n_offs, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _fp4_gemv_batched_multi_input_kernel(
    input_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_HALF: tl.constexpr,
    NUM_BLOCKS_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HALF_BK: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Batched GEMV: each expert has its own input vector.
    input layout: (E, K) row-major contiguous, so expert e's input starts at e*K.
    """
    eid = tl.program_id(0)
    pid = tl.program_id(1)
    n_start = pid * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    n_mask = n_offs < N

    # Use constexpr K as stride (input is contiguous (E, K))
    x_base = input_ptr + eid * K

    e_off_w = eid * N * K_HALF
    e_off_s = eid * N * NUM_BLOCKS_K

    acc = tl.zeros((TILE_N,), dtype=tl.float32)

    for bk in range(NUM_BLOCKS_K):
        k_start = bk * BLOCK_K

        s_ptrs = scale_ptr + e_off_s + n_offs * NUM_BLOCKS_K + bk
        s = tl.load(s_ptrs, mask=n_mask, other=1.0)

        byte_offs = k_start // 2 + tl.arange(0, HALF_BK)
        w_ptrs = weight_ptr + e_off_w + n_offs[:, None] * K_HALF + byte_offs[None, :]
        w_bytes = tl.load(w_ptrs, mask=n_mask[:, None], other=0).to(tl.int32)

        dq_even = _e2m1_dequant(w_bytes & 0x0F) * s[:, None]
        dq_odd = _e2m1_dequant((w_bytes >> 4) & 0x0F) * s[:, None]

        half_idx = tl.arange(0, HALF_BK)
        even_k_offs = k_start + 2 * half_idx
        odd_k_offs = even_k_offs + 1
        x_even = tl.load(x_base + even_k_offs).to(tl.float32)
        x_odd = tl.load(x_base + odd_k_offs).to(tl.float32)

        acc += tl.sum(dq_even * x_even[None, :], axis=1)
        acc += tl.sum(dq_odd * x_odd[None, :], axis=1)

    tl.store(output_ptr + eid * N + n_offs, acc.to(tl.bfloat16), mask=n_mask)


def fp4_gemv(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    scale: torch.Tensor,
    block_k: int = 32,
) -> torch.Tensor:
    """Fused FP4 dequant + vector-matrix multiply. Returns (N,) bf16."""
    N, K_half = w_fp4.shape
    K = K_half * 2
    BLOCK_K = block_k
    NUM_BLOCKS_K = K // BLOCK_K
    TILE_N = min(32, N)

    output = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(N, TILE_N),)
    HALF_BK = BLOCK_K // 2

    _fp4_gemv_kernel[grid](
        x, w_fp4, scale, output,
        N=N, K=K, K_HALF=K_half, NUM_BLOCKS_K=NUM_BLOCKS_K,
        BLOCK_K=BLOCK_K, HALF_BK=HALF_BK, TILE_N=TILE_N,
    )
    return output


def fp4_gemv_batched(
    x: torch.Tensor,
    w_fp4: torch.Tensor,   # (E, N, K_half) uint8
    scale: torch.Tensor,    # (E, N, NUM_BLOCKS_K) float32
    block_k: int = 32,
) -> torch.Tensor:
    """Batched FP4 GEMV: same input for all E experts. Returns (E, N) bf16."""
    E, N, K_half = w_fp4.shape
    K = K_half * 2
    BLOCK_K = block_k
    NUM_BLOCKS_K = K // BLOCK_K
    HALF_BK = BLOCK_K // 2
    TILE_N = min(32, N)

    w_fp4 = w_fp4.contiguous()
    scale = scale.contiguous()

    output = torch.empty((E, N), dtype=torch.bfloat16, device=x.device)
    grid = (E, triton.cdiv(N, TILE_N))

    _fp4_gemv_batched_kernel[grid](
        x, w_fp4, scale, output,
        N=N, K=K, K_HALF=K_half, NUM_BLOCKS_K=NUM_BLOCKS_K,
        BLOCK_K=BLOCK_K, HALF_BK=HALF_BK, TILE_N=TILE_N,
    )
    return output


def fp4_gemv_batched_multi_input(
    x: torch.Tensor,       # (E, K) bf16 - must be contiguous
    w_fp4: torch.Tensor,   # (E, N, K_half) uint8
    scale: torch.Tensor,    # (E, N, NUM_BLOCKS_K) float32
    block_k: int = 32,
) -> torch.Tensor:
    """Batched FP4 GEMV: per-expert inputs. Returns (E, N) bf16.

    Input x must be contiguous with shape (E, K).
    """
    E, N, K_half = w_fp4.shape
    K = K_half * 2
    BLOCK_K = block_k
    NUM_BLOCKS_K = K // BLOCK_K
    HALF_BK = BLOCK_K // 2
    TILE_N = min(32, N)

    w_fp4 = w_fp4.contiguous()
    scale = scale.contiguous()
    x = x.contiguous()

    output = torch.empty((E, N), dtype=torch.bfloat16, device=x.device)
    grid = (E, triton.cdiv(N, TILE_N))

    _fp4_gemv_batched_multi_input_kernel[grid](
        x, w_fp4, scale, output,
        N=N, K=K, K_HALF=K_half, NUM_BLOCKS_K=NUM_BLOCKS_K,
        BLOCK_K=BLOCK_K, HALF_BK=HALF_BK, TILE_N=TILE_N,
    )
    return output
