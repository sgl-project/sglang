"""SM120-optimized Triton MXFP4 MoE kernel — CUDA graph compatible.

Replaces the PyTorch fallback (per-expert for-loop + full dequant + matmul)
with fused Triton kernels that:
1. Fuse FP4 dequant + GEMV (no intermediate BF16 weight materialization)
2. Process each (token, expert) slot independently — no data-dependent routing
3. Respect SM120 shared memory constraint (99 KB/block)

CUDA graph compatibility:
- No .unique(), .item(), .nonzero() — all routing is tensor-level
- Fixed grid dimensions (M*topk, N_blocks) per captured batch size
- All control flow is static or within Triton kernels

SM120 constraints:
- SMEM: 99 KB/block (vs SM100 228 KB)
- No TMEM/tcgen05 — uses mma.sync.aligned via Triton
- Max warps: 48/SM
- Registers: ~128/thread practical limit

Ported from PR #24047 (AliceChenyy's SM120 support).
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _dequant_fp4_lut(nibble):
    """Decode a 4-bit FP4 E2M1 nibble to float32 using arithmetic."""
    sign_bit = (nibble >> 3) & 1
    exp_bits = (nibble >> 1) & 3
    man_bit = nibble & 1

    is_subnormal = (exp_bits == 0)
    mantissa = 1.0 + man_bit.to(tl.float32) * 0.5
    exponent = tl.math.exp2((exp_bits - 1).to(tl.float32))
    val = tl.where(is_subnormal, man_bit.to(tl.float32) * 0.5, mantissa * exponent)
    val = tl.where(sign_bit != 0, -val, val)
    return val


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _mxfp4_slot_gemv_kernel(
    # Pointers
    A_ptr,            # [M_total, K] bf16 — source rows
    B_packed_ptr,     # [E, N, K//2] uint8 — packed FP4 expert weights
    B_scale_ptr,      # [E, N, K//32] float32 — weight scales
    C_ptr,            # [num_slots, N] bf16 — output
    token_ids_ptr,    # [num_slots] int32 — which A row for each slot
    expert_ids_ptr,   # [num_slots] int32 — which expert's B for each slot
    # Dimensions
    N: tl.int32,
    K: tl.int32,
    # A strides
    stride_am: tl.int32,
    # B strides (within an expert)
    stride_bn: tl.int32,
    stride_bk2: tl.int32,
    # B_scale strides (within an expert)
    stride_bsn: tl.int32,
    stride_bsk32: tl.int32,
    # Expert strides (between experts)
    expert_b_stride: tl.int64,
    expert_s_stride: tl.int64,
    # C strides
    stride_cm: tl.int32,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-slot fused MXFP4 dequant + GEMV.

    Grid: (num_slots, cdiv(N, BLOCK_N))
    Each program computes one (token, expert) pair for a BLOCK_N slice of output.
    """
    slot_id = tl.program_id(0)
    n_block = tl.program_id(1)

    token_id = tl.load(token_ids_ptr + slot_id).to(tl.int64)
    expert_id = tl.load(expert_ids_ptr + slot_id).to(tl.int64)

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Expert weight base pointers
    b_base = expert_id * expert_b_stride
    s_base = expert_id * expert_s_stride
    a_base = token_id * stride_am

    for k_start in range(0, K, BLOCK_K):
        # Load packed B: [BLOCK_N, BLOCK_K//2]
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        b_mask = n_mask[:, None] & (offs_k2[None, :] < K // 2)
        b_packed = tl.load(
            B_packed_ptr + b_base + offs_n[:, None] * stride_bn + offs_k2[None, :] * stride_bk2,
            mask=b_mask, other=0,
        )

        # FP4 dequant
        b_u8 = b_packed.to(tl.int32)
        val_lo = _dequant_fp4_lut(b_u8 & 0x0F)
        val_hi = _dequant_fp4_lut((b_u8 >> 4) & 0x0F)

        # Load and apply scales: [BLOCK_N, BLOCK_K//2]
        group_ids = tl.arange(0, BLOCK_K // 2) // 16
        s_mask = n_mask[:, None] & ((k_start // 32 + group_ids[None, :]) < K // 32)
        scales = tl.load(
            B_scale_ptr + s_base + offs_n[:, None] * stride_bsn
            + (k_start // 32 + group_ids[None, :]) * stride_bsk32,
            mask=s_mask, other=1.0,
        )
        val_lo = val_lo * scales
        val_hi = val_hi * scales

        # Load A even/odd: [BLOCK_K//2] each
        offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
        offs_k_odd = offs_k_even + 1

        a_even = tl.load(
            A_ptr + a_base + offs_k_even,
            mask=offs_k_even < K, other=0.0,
        ).to(tl.float32)
        a_odd = tl.load(
            A_ptr + a_base + offs_k_odd,
            mask=offs_k_odd < K, other=0.0,
        ).to(tl.float32)

        # Dot product: acc[n] += sum_k(a_even[k]*B_lo[n,k] + a_odd[k]*B_hi[n,k])
        acc += tl.sum(a_even[None, :] * val_lo, axis=1)
        acc += tl.sum(a_odd[None, :] * val_hi, axis=1)

    # Store output
    tl.store(
        C_ptr + slot_id * stride_cm + offs_n,
        acc.to(tl.bfloat16),
        mask=n_mask,
    )


def _mxfp4_moe_forward_grouped(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """Prefill-optimized MoE: grouped per-expert batched GEMM.

    Sorts tokens by expert, then does batched BF16 matmul per expert.
    Higher tensor core utilization than per-slot GEMV for M > 8.
    Uses the proven _dequant_fp4_batch from Fp8MoEMethod for correctness.
    """
    import torch.nn.functional as F
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    I = intermediate_size
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Flatten and sort by expert for grouping
    flat_expert_ids = topk_ids.reshape(-1)  # [M*topk]
    flat_token_ids = (
        torch.arange(M, device=device, dtype=torch.int32)
        .unsqueeze(1).expand(M, topk).reshape(-1)
    )

    # Sort by expert ID
    sort_idx = torch.argsort(flat_expert_ids)
    sorted_experts = flat_expert_ids[sort_idx]
    sorted_tokens = flat_token_ids[sort_idx]

    # Get unique experts and their counts
    unique_experts, expert_counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    expert_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.int32),
                                expert_counts.cumsum(0)[:-1]])

    output = torch.zeros(M, K, dtype=dtype, device=device)

    # Gather hidden states for all slots
    all_hidden = hidden_states[sorted_tokens]  # [M*topk, K]

    # Process each expert group
    for i in range(unique_experts.shape[0]):
        eid = int(unique_experts[i])
        start = int(expert_offsets[i])
        count = int(expert_counts[i])

        if count == 0:
            continue

        x_group = all_hidden[start:start + count]  # [count, K]

        # Dequant w13 for this expert using proven batch dequant
        w13_bf16 = Fp8MoEMethod._dequant_fp4_batch(
            w13_packed[eid:eid+1], w13_scale[eid:eid+1]
        ).squeeze(0)  # [2I, K]

        # Batched GEMM: (count, K) @ (2I, K).T -> (count, 2I)
        gate_up = torch.mm(x_group.float(), w13_bf16.float().T).to(dtype)

        # SwiGLU
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        if clamp_limit is not None and clamp_limit > 0:
            gate = torch.clamp(gate.float(), max=clamp_limit)
            up = torch.clamp(up.float(), min=-clamp_limit, max=clamp_limit)
        activated = (F.silu(gate.float()) * up.float()).to(dtype)

        # Dequant w2 for this expert
        w2_bf16 = Fp8MoEMethod._dequant_fp4_batch(
            w2_packed[eid:eid+1], w2_scale[eid:eid+1]
        ).squeeze(0)  # [K, I]

        # Down projection: (count, I) @ (K, I).T -> (count, K)
        down = torch.mm(activated.float(), w2_bf16.float().T).to(dtype)

        # Scatter back to output with weights
        slot_weights = topk_weights.reshape(-1)[sort_idx[start:start + count]]
        token_indices = sorted_tokens[start:start + count]

        weighted = down * slot_weights.unsqueeze(1).to(dtype)
        output.scatter_add_(0, token_indices.unsqueeze(1).expand_as(weighted), weighted)

    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        output.mul_(routed_scaling_factor)

    return output


def mxfp4_moe_forward_triton(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """SM120-optimized MXFP4 MoE forward.

    Prefill path (M > 8): grouped per-expert batched GEMM for high utilization.
    Decode path (M <= 8): per-slot Triton GEMV for CUDA graph compatibility.
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]

    # Prefill: grouped per-expert batched GEMM (not CUDA graph, uses Python loop)
    # Needs M large enough that average tokens/expert > ~5 to amortize dequant.
    # With topk=6, 256 experts: M > 256 → ~6 tokens/expert average.
    # Below threshold: per-slot Triton GEMV is faster (no dequant overhead).
    if False and M > 256:
        return _mxfp4_moe_forward_grouped(
            hidden_states=hidden_states,
            w13_packed=w13_packed,
            w2_packed=w2_packed,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            routed_scaling_factor=routed_scaling_factor,
            clamp_limit=clamp_limit,
        )

    # Decode: per-slot Triton GEMV (CUDA graph compatible)
    import torch.nn.functional as F

    I = intermediate_size
    num_slots = M * topk
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Graph-safe routing: flatten topk assignments
    flat_expert_ids = topk_ids.reshape(-1).contiguous()  # [M*topk]
    token_ids = (
        torch.arange(M, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(M, topk)
        .reshape(-1)
        .contiguous()
    )  # [M*topk]

    # Ensure scales are float32
    if w13_scale.dtype != torch.float32:
        w13_scale = w13_scale.to(torch.float32)
    if w2_scale.dtype != torch.float32:
        w2_scale = w2_scale.to(torch.float32)

    # GEMM1: gate_up projection
    # hidden_states[token] @ w13[expert].T -> [num_slots, 2*I]
    intermediate = torch.empty(num_slots, 2 * I, dtype=dtype, device=device)

    w13_u8 = w13_packed.view(torch.uint8)  # [E, 2*I, K//2]
    grid1 = lambda meta: (num_slots, triton.cdiv(2 * I, meta["BLOCK_N"]))

    _mxfp4_slot_gemv_kernel[grid1](
        hidden_states,
        w13_u8,
        w13_scale,
        intermediate,
        token_ids,
        flat_expert_ids,
        2 * I,
        K,
        hidden_states.stride(0),
        w13_u8.stride(1),
        w13_u8.stride(2),
        w13_scale.stride(1),
        w13_scale.stride(2),
        w13_u8.stride(0),
        w13_scale.stride(0),
        intermediate.stride(0),
    )

    # SiLU activation (graph-safe vectorized ops)
    gate = intermediate[:, :I].float()
    up = intermediate[:, I:].float()
    if clamp_limit is not None and clamp_limit > 0:
        gate = torch.clamp(gate, max=clamp_limit)
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    activated = (F.silu(gate) * up).to(dtype)

    # GEMM2: down projection
    # activated[slot] @ w2[expert].T -> [num_slots, K]
    down = torch.empty(num_slots, K, dtype=dtype, device=device)

    slot_ids = torch.arange(num_slots, device=device, dtype=torch.int32)

    w2_u8 = w2_packed.view(torch.uint8)  # [E, K, I//2]
    grid2 = lambda meta: (num_slots, triton.cdiv(K, meta["BLOCK_N"]))

    _mxfp4_slot_gemv_kernel[grid2](
        activated,
        w2_u8,
        w2_scale,
        down,
        slot_ids,
        flat_expert_ids,
        K,
        I,
        activated.stride(0),
        w2_u8.stride(1),
        w2_u8.stride(2),
        w2_scale.stride(1),
        w2_scale.stride(2),
        w2_u8.stride(0),
        w2_scale.stride(0),
        down.stride(0),
    )

    # Weighted sum across topk slots (graph-safe)
    flat_weights = topk_weights.reshape(-1).unsqueeze(1).to(dtype)  # [M*topk, 1]
    output = (down * flat_weights).view(M, topk, K).sum(dim=1)

    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        output.mul_(routed_scaling_factor)

    return output
