from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def fused_moe_router_cudacore_kernel(
    input_ptr,  # input (bs, hidden_dim)
    moe_router_weight_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    correction_bias_ptr,
    is_correction_bias: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # moe_router_weight is k major
    expert_offsets = tl.arange(0, num_experts)[:, None]
    router_mask = mask[None, :]
    w_router = tl.load(
        moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :],
        mask=router_mask,
        other=0.0,
    )

    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)

    # todo: tl.dot?
    logits = tl.sum((w_router.to(tl.float32) * x[None, :].to(tl.float32)), axis=-1)

    # logit softcap
    if moe_softcapping == 0:
        logits_softcapped = logits
    else:
        logits_scaled = logits / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        top = exped - 1
        bottom = exped + 1
        logits_softcapped = top / bottom * moe_softcapping

    # Add bias after softcapping
    if is_correction_bias:
        bias = tl.load(correction_bias_ptr + tl.arange(0, num_experts))
        logits_softcapped = logits_softcapped + bias

    # topk
    # assert 1 <= topk <= num_experts

    # 5.38 us

    top1 = tl.argmax(logits_softcapped, axis=0)
    tl.store(topk_ids_ptr + pid * topk + 0, top1)  # 5.63 us

    top1_v = tl.max(logits_softcapped, axis=0)
    invsumexp = 1.0 / tl.sum(tl.exp(logits_softcapped - top1_v), axis=0)

    tl.store(
        topk_weights_ptr + pid * topk + 0,
        invsumexp,
    )  # 5.73 us

    if topk >= 2:
        top2 = tl.argmax(
            tl.where(
                tl.arange(0, num_experts) != top1, logits_softcapped, float("-inf")
            ),
            axis=0,
        )
        tl.store(topk_ids_ptr + pid * topk + 1, top2)
        top2_v = tl.sum(logits_softcapped * (tl.arange(0, num_experts) == top2), axis=0)
        tl.store(
            topk_weights_ptr + pid * topk + 1,
            tl.exp(top2_v - top1_v) * invsumexp,
        )  # 5.95us

    # probably slow
    if topk > 2:
        topk_mask = tl.full(logits_softcapped.shape, 1.0, dtype=logits_softcapped.dtype)
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top1, topk_mask, float("-inf")
        )
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top2, topk_mask, float("-inf")
        )
        for i in range(2, topk):
            topi = tl.argmax(logits_softcapped + topk_mask, axis=0)
            topk_mask = tl.where(
                tl.arange(0, num_experts) != topi, topk_mask, float("-inf")
            )
            tl.store(topk_ids_ptr + pid * topk + i, topi)
            topi_v = tl.sum(
                logits_softcapped * (tl.arange(0, num_experts) == topi), axis=0
            )
            tl.store(
                topk_weights_ptr + pid * topk + i,
                tl.exp(topi_v - top1_v) * invsumexp,
            )
    # assert not moe_renormalize, "moe weight renormalization not implemented"


def fused_moe_router_cudacore(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor] = None,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    # router_logits = torch.empty((bs, num_experts), dtype=torch.float32, device=x.device)
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    is_correction_bias = correction_bias is not None

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_moe_router_cudacore_kernel[(bs,)](
        x,
        router_weight,
        topk_weights,
        topk_ids,
        correction_bias,
        is_correction_bias=is_correction_bias,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        hidden_dim=hidden_dim,
        **config,
    )

    return topk_weights, topk_ids


@triton.jit
def fused_moe_router_tensorcore_kernel(
    a_ptr,  # input (bs, hidden_dim)
    b_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    bs,
    num_experts: tl.constexpr,
    topk: tl.constexpr,  # only support topk <= 2
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    correction_bias_ptr,
    is_correction_bias: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    dp_attn_workaround_flag: tl.constexpr,
):
    # 1. get block id
    pid = tl.program_id(axis=0)

    # 2. create pointers for the first block of A and B
    # 2.1. setup a_ptrs with offsets in m and k
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    bs_mask = offs_m < bs
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)

    # 2.2. setup b_ptrs with offsets in k and n.
    #      Note: b matrix is k-major.
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]
    expert_mask = offs_n < num_experts
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)

    # 3. Create an accumulator of float32 of size [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #    3.1. iterate in K dimension
    #    3.2. transpose tile B
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):  # hidden_dim % BLOCK_SIZE_K == 0
        a = tl.load(
            a_ptrs,
            mask=bs_mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T
        acc += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # 4. logit softcap
    if moe_softcapping == 0:
        logits_softcapped = acc
    else:
        logits_scaled = acc / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping

    # Add bias after softcapping
    if is_correction_bias:
        bias = tl.load(
            correction_bias_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :],
            mask=expert_mask.T,
            other=0.0,
        )
        logits_softcapped = logits_softcapped + bias

    if dp_attn_workaround_flag:
        logits_softcapped = tl.where(
            logits_softcapped != logits_softcapped, -1e9, logits_softcapped
        )

    # 5. top1
    arange_block_size_n = tl.arange(0, BLOCK_SIZE_N)[None, :]
    cond_top1 = arange_block_size_n < num_experts
    top1 = tl.argmax(tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1)
    top1_v = tl.max(
        tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1, keep_dims=True
    )
    top1_invsumexp = 1.0 / tl.sum(
        tl.where(cond_top1, tl.exp(logits_softcapped - top1_v), 0.0), axis=1
    )

    # 6. store top1 to output
    offs_top1 = pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)
    top1_mask = offs_top1 < bs * topk
    tl.store(topk_ids_ptr + offs_top1, top1, mask=top1_mask)
    tl.store(
        topk_weights_ptr + offs_top1,
        top1_invsumexp,
        mask=top1_mask,
    )

    # 7. handle topk == 2
    if topk == 2:
        cond_top2 = (arange_block_size_n < num_experts) & (
            arange_block_size_n != top1[:, None]
        )
        top2 = tl.argmax(
            tl.where(cond_top2, logits_softcapped, float("-inf")),
            axis=1,
            keep_dims=True,
        )
        top2_v = tl.sum(
            logits_softcapped * (arange_block_size_n == top2), axis=1, keep_dims=True
        )
        top2_invsumexp = tl.exp(top2_v - top1_v) * top1_invsumexp[:, None]

        # store top2
        offs_top2 = (
            pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)[:, None] + 1
        )
        top2_mask = offs_top2 < bs * topk
        tl.store(topk_ids_ptr + offs_top2, top2, mask=top2_mask)
        tl.store(
            topk_weights_ptr + offs_top2,
            top2_invsumexp,
            mask=top2_mask,
        )


def fused_moe_router_tensorcore(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    correction_bias: Optional[torch.Tensor] = None,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    assert num_experts <= BLOCK_SIZE_N
    assert hidden_dim % BLOCK_SIZE_K == 0
    assert topk <= 2

    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    is_correction_bias = correction_bias is not None

    grid = (triton.cdiv(bs, BLOCK_SIZE_M) * triton.cdiv(num_experts, BLOCK_SIZE_N),)

    # TODO(ch-wan): temporary workaround for dp attention. We should support masked
    # router to skip padded tokens.
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    dp_attn_workaround_flag = is_dp_attention_enabled()

    fused_moe_router_tensorcore_kernel[grid](
        a_ptr=x,
        b_ptr=router_weight,
        topk_weights_ptr=topk_weights,
        topk_ids_ptr=topk_ids,
        bs=bs,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        K=hidden_dim,
        correction_bias_ptr=correction_bias,
        is_correction_bias=is_correction_bias,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        stride_am=hidden_dim,
        stride_bn=hidden_dim,
        dp_attn_workaround_flag=dp_attn_workaround_flag,
    )

    return topk_weights, topk_ids


def fused_moe_router_shim(
    moe_softcapping,
    hidden_states,
    gating_output,
    topk,
    renormalize,
    correction_bias: Optional[torch.Tensor] = None,
    enable_deterministic_inference: bool = False,
):
    assert not renormalize
    assert (
        len(hidden_states.shape) == 2
        and hidden_states.shape[1] == gating_output.shape[1]
    )
    bs, hidden_dim = hidden_states.shape
    num_experts = gating_output.shape[0]

    BLOCK_SIZE_M = 32

    BLOCK_SIZE_N = max(num_experts, 16)
    BLOCK_SIZE_K = (
        256 if num_experts < 256 else 64
    )  # if experts are large, need to use smaller k block or shared memory OOM

    if (
        (bs >= 512 or num_experts > 8)
        and hidden_dim % BLOCK_SIZE_K == 0
        # we keep using single kernel to avoid non-deterministic behavior
        and not enable_deterministic_inference
    ):
        # if large batch size or large expert, use kernel that uses tensorcore in matmul
        return fused_moe_router_tensorcore(
            x=hidden_states,
            router_weight=gating_output,
            topk=topk,
            moe_softcapping=moe_softcapping,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            correction_bias=correction_bias,
        )
    else:
        # if smaller, use kernel that does not use tensorcore in matmul
        return fused_moe_router_cudacore(
            x=hidden_states,
            router_weight=gating_output,
            topk=topk,
            moe_softcapping=moe_softcapping,
            correction_bias=correction_bias,
        )


@triton.jit
def router_gate_matvec_kernel(
    x_ptr,  # (M, K) bf16/fp16/fp32, row-major
    w_ptr,  # (E, K) fp32/bf16/fp16, k-major
    out_ptr,  # (M, E) fp32
    K,
    E,
    stride_xm,
    stride_we,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Router-gate logits as a single matvec launch, fp32 accumulation for
    any float weight dtype. BLOCK_K covers the whole K in one masked load
    (single iteration for K <= BLOCK_K): a cold gate weight then costs one
    HBM round trip per CTA instead of a serial dependent-load chain, which
    is what dominates in the real model where ~94MB/layer of expert traffic
    flushes L2 between gate calls.
    """
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # First K tile, weight load ahead of the PDL wait (see docstring).
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    w = tl.load(
        w_ptr + e_offs[:, None] * stride_we + k_offs[None, :],
        mask=e_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(x_ptr + pid_m * stride_xm + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    acc = tl.sum(w * x[None, :], axis=1)

    for k0 in range(BLOCK_K, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(x_ptr + pid_m * stride_xm + k_offs, mask=k_mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(
            w_ptr + e_offs[:, None] * stride_we + k_offs[None, :],
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(
        out_ptr + pid_m * E + e_offs, acc.to(out_ptr.dtype.element_ty), mask=e_mask
    )


# Cold-cache tuned on H20-3e (41 rotating gate weights so each call misses L2,
# like the real model); expected to carry to B200 (more SMs favor the wide
# grid even more) — re-tune with benchmark/kernels/bench_router_gate_matvec.py.
ROUTER_GATE_MATVEC_BLOCK_E = 4
ROUTER_GATE_MATVEC_NUM_WARPS = 8
# Beyond this M the per-M re-reads of the gate weight outgrow the library
# GEMM (cold H20-3e: bf16 wins to M=12, fp32 to M=8; cap at the lower).
ROUTER_GATE_MATVEC_MAX_M = 8


def router_gate_matvec(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Small-M router-gate logits: one triton launch replacing the library
    path — for fp32 gate weights the eager upcast + fp32 GEMM + splitKreduce
    triple, for bf16 the F.linear GEMV. Returns fp32 (M, E) logits with fp32
    accumulation (deterministic; for fp32 weights 0 top-8 routing flips over
    30104 random draws vs the fp32 reference; for bf16 weights this is
    slightly MORE precise than the library GEMV, so near-tie logits can
    round-trip differently — same order as the bf16-vs-fp32 gate change).

    Cold-cache (41 rotating weights, in-graph, H20-3e, E=513, K=2560), us/call:

        M      lib bf16   matvec bf16   lib fp32 chain   matvec fp32
        1        4.2         4.3            6.4              6.2
        2       13.8         4.5           14.6              6.4
        4       14.0         5.4           20.7             10.4
        8       14.6        10.4           20.1             18.1
        16      14.6        18.9  (lib)    20.9             30.9  (lib)

    Callers must gate on M <= ROUTER_GATE_MATVEC_MAX_M; prefill-sized M
    keeps the library GEMM."""
    assert (
        weight.dtype
        in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        )
        and weight.is_contiguous()
    )
    M, K = hidden_states.shape
    E = weight.shape[0]
    out = torch.empty((M, E), dtype=torch.float32, device=hidden_states.device)
    block_e = ROUTER_GATE_MATVEC_BLOCK_E
    # Single k-iteration whenever K fits one block: no serial dependent-load
    # chain on a cold weight.
    block_k = min(4096, triton.next_power_of_2(K))
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    router_gate_matvec_kernel[(M, triton.cdiv(E, block_e))](
        hidden_states,
        weight,
        out,
        K,
        E,
        hidden_states.stride(0),
        weight.stride(0),
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=ROUTER_GATE_MATVEC_NUM_WARPS,
        **pdl_kwargs,
    )
    return out
