"""Fused MoE gate: sigmoid + bias + top-k selection + logsigmoid renorm.

    sel  = sigmoid(logits)[:, :N] + bias        # selection score (bias optional)
    idx  = topk(sel, k)                         # top-k routed experts
    w    = logsigmoid_norm(logits[idx] ++ shared) * route_scale * global_scale

The renorm runs on the RAW logits gathered at the selected indices, so the sort
key (sigmoid+bias) is not the renorm value -> we re-gather the raw logits.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.model.inkling.inkling_gate_topk_renorm import (
    inkling_gate_topk_renorm_v2,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils.gate_topk import (
    fpval_to_key,
    indx_to_key,
    key_to_indx,
)


@triton.jit
def _sigmoid_gate_topk_renorm_kernel(
    logits_ptr,
    bias_ptr,
    stride_lm,
    routed_w_ptr,
    shared_w_ptr,
    indices_ptr,
    packed_indices_ptr,
    global_scale_ptr,
    route_scale,
    M,
    N,  # num routed experts (top-k sort dim)
    G,  # total gate experts = N + S
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    S: tl.constexpr,  # num shared experts
    A_POW2: tl.constexpr,  # next_pow2(K + S)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    RETURN_PACKED_TOPK: tl.constexpr = False,
    ENABLE_PDL: tl.constexpr = False,
):
    tl.static_assert(
        K_POW2 == A_POW2, "epilogue reuses the topk slot axis for the active axis"
    )
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M

    # --- streaming top-k by selection score sigmoid(logit)[+bias] ------------
    num_iters: tl.constexpr = N_PAD // BLOCK_SIZE_N - 1
    offs_n = num_iters * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    # first (masked) tile
    raw = tl.load(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    sel = tl.sigmoid(raw)
    sel += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    sel = tl.where(mask_n[None, :], sel, float("-inf"))  # out-of-range cols never win
    key = fpval_to_key(sel.to(tl.uint32, bitcast=True))
    x = (key.to(tl.uint64) << 16) | indx_to_key(offs_n, N_PAD)[None, :]
    acc = tl.topk(x, K_POW2, dim=1)

    # remaining tiles are fully in-range (N_PAD - N < BLOCK_SIZE_N)
    for _i in (tl.static_range if num_iters <= 4 else range)(num_iters):
        acc = tl.bitonic_merge(acc)
        offs_n -= BLOCK_SIZE_N
        raw = tl.load(
            logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        sel = tl.sigmoid(raw)
        sel += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
        key = fpval_to_key(sel.to(tl.uint32, bitcast=True))
        x = (key.to(tl.uint64) << 16) | indx_to_key(offs_n, N_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, K_POW2, dim=1))

    offs_a = tl.arange(0, A_POW2)
    mask_k = offs_a < K
    acc = tl.sort(acc, dim=1, descending=True)
    y_indices = key_to_indx((acc & 0xFFFF).to(tl.uint32), N_PAD)

    # --- renorm on RAW fp32 logits gathered at the selected indices ----------
    gather_idx = tl.where(mask_k[None, :], y_indices.to(tl.int32), 0)
    routed_vals = tl.load(
        logits_ptr + offs_m[:, None] * stride_lm + gather_idx,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    offs_s = offs_a - K
    mask_s = mask_m[:, None] & (offs_s[None, :] >= 0) & (offs_s[None, :] < S)
    shared = tl.load(
        logits_ptr + offs_m[:, None] * stride_lm + (G - S) + offs_s[None, :],
        mask=mask_s,
        other=0.0,
    ).to(tl.float32)
    active = tl.where(mask_k[None, :], routed_vals, shared)

    A: tl.constexpr = K + S
    probs = tl.sigmoid(active)
    mask_a = offs_a < A
    probs = tl.where(mask_a[None, :], probs, 0.0)
    weights = probs / tl.sum(probs, axis=1, keep_dims=True)
    weights *= (route_scale * tl.load(global_scale_ptr)).to(weights.dtype)

    mask_rk = mask_m[:, None] & mask_k[None, :]
    # PackedTopKOutput carries only packed_topk_ids, so in packed mode the separate
    # routed-weight / index stores are dead -- emit just the packed (id<<16 | bf16 w).
    if RETURN_PACKED_TOPK:
        weights_bits = weights.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        packed = (y_indices.to(tl.int32) << 16) | weights_bits
        tl.store(
            packed_indices_ptr + offs_m[:, None] * K + offs_a[None, :],
            packed,
            mask=mask_rk,
        )
    else:
        tl.store(
            routed_w_ptr + offs_m[:, None] * K + offs_a[None, :], weights, mask=mask_rk
        )
        tl.store(
            indices_ptr + offs_m[:, None] * K + offs_a[None, :], y_indices, mask=mask_rk
        )
    offs_ts = offs_m[:, None] * S + offs_s[None, :]
    tl.store(shared_w_ptr + offs_ts, weights, mask=mask_s)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def sigmoid_gate_topk_renorm(
    logits: torch.Tensor,
    k: int,
    n_shared_experts: int,
    route_scale: float,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    return_packed_topk: bool = False,
):
    """Fused top-k + logsigmoid renorm (production sigmoid+bias gate path).

    `logits` is [tokens, n_routed + n_shared]; the last `n_shared_experts`
    columns are the shared experts. Selection score is sigmoid(routed logit)
    plus `bias` (per routed expert, fp32). Returns
    (routed_weights[t,k], shared_weights[t,s], topk_indices[t,k] int32).
    """
    # Only column-stride-1 is required (the kernel reads rows via stride_lm). In
    # InklingGate the gate logits are a [t,258] slice of a padded [t,264] tensor, so
    # they are NOT contiguous but are column-contiguous -- no copy needed.
    assert (
        logits.ndim == 2 and logits.stride(1) == 1
    ), f"{logits.shape=} {logits.stride()=}"
    assert (
        logits.shape[0] * logits.stride(0) <= 2**31
    ), f"assumes int32 indexing: {logits.stride()=}"
    assert k <= 32, f"topk kernels only support k <= 32: {k=}"
    assert (
        n_shared_experts >= 0
    ), f"expected non-negative shared experts: {n_shared_experts=}"
    M, G = logits.shape
    N = G - n_shared_experts
    A = k + n_shared_experts
    assert bias.numel() == N and bias.stride(-1) == 1, f"{bias.shape=} expected [{N}]"

    # The production shape uses the specialized CUDA JIT kernel.
    if (
        k == 6
        and n_shared_experts == 2
        and G == 258
        and logits.stride(0) % 8 == 0
        and logits.data_ptr() % 32 == 0
        and torch.version.hip is None
        and envs.SGLANG_OPT_USE_GATE_TOPK_JIT.get()
    ):
        return inkling_gate_topk_renorm_v2(
            logits,
            bias,
            global_scale,
            route_scale,
            return_packed=return_packed_topk,
            enable_pdl=is_arch_support_pdl(),
        )

    shared_w = torch.empty(
        (M, n_shared_experts), dtype=logits.dtype, device=logits.device
    )
    # The kernel writes only the packed tensor in packed mode, only routed_w+indices
    # otherwise; the unused pointer args still need a valid (never-stored) address.
    if return_packed_topk:
        packed_indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)
        routed_w = indices = None
        routed_w_arg = indices_arg = packed_indices_arg = packed_indices
    else:
        routed_w = torch.empty((M, k), dtype=logits.dtype, device=logits.device)
        indices = torch.empty((M, k), dtype=torch.int32, device=logits.device)
        packed_indices = None
        routed_w_arg, indices_arg, packed_indices_arg = routed_w, indices, indices

    # Launch geometry for the production shape.
    if M <= 128:
        BLOCK_SIZE_M = 1
    elif M <= 768:
        BLOCK_SIZE_M = 2
    else:
        BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 128 if M <= 1024 else 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    kwargs = {"num_warps": 8 if M <= 1024 else 2}
    if is_arch_support_pdl():
        kwargs.update({"ENABLE_PDL": True, "launch_pdl": True})

    _sigmoid_gate_topk_renorm_kernel[grid](
        logits,
        bias,
        logits.stride(0),
        routed_w_arg,
        shared_w,
        indices_arg,
        packed_indices_arg,
        global_scale,
        route_scale,
        M=M,
        N=N,
        G=G,
        N_PAD=triton.cdiv(N, BLOCK_SIZE_N) * BLOCK_SIZE_N,
        K=k,
        K_POW2=triton.next_power_of_2(k),
        S=n_shared_experts,
        A_POW2=triton.next_power_of_2(A),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        RETURN_PACKED_TOPK=return_packed_topk,
        **kwargs,
    )
    return routed_w, indices, shared_w, packed_indices
