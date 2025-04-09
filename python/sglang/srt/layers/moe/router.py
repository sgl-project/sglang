from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.topk import fused_topk
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def fused_moe_router_kernel(
    input_ptr,  # input (bs, hidden_dim)
    moe_router_weight_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
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
    logits_scaled = logits / moe_softcapping
    exped = tl.exp(2 * logits_scaled)
    top = exped - 1
    bottom = exped + 1
    logits_softcapped = top / bottom * moe_softcapping

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


def fused_moe_router_impl(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    # router_logits = torch.empty((bs, num_experts), dtype=torch.float32, device=x.device)
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)

    grid = lambda meta: (bs,)

    min_num_warps = 16 if _is_hip else 32

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), min_num_warps), 4
        ),
    }

    fused_moe_router_kernel[grid](
        x,
        router_weight,
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        hidden_dim=hidden_dim,
        **config,
    )

    return topk_weights, topk_ids


@triton.jit
def fused_moe_router_large_bs_kernel(
    a_ptr,  # input (bs, hidden_dim)
    b_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    bs,
    num_experts: tl.constexpr,
    topk: tl.constexpr,  # only support topk == 1
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
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
    logits_scaled = acc / moe_softcapping
    exped = tl.exp(2 * logits_scaled)
    logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping

    # 5. top1
    cond = tl.arange(0, BLOCK_SIZE_N)[None, :] < num_experts
    top1 = tl.argmax(tl.where(cond, logits_softcapped, float("-inf")), axis=1)
    top1_v = tl.max(
        tl.where(cond, logits_softcapped, float("-inf")), axis=1, keep_dims=True
    )
    invsumexp = 1.0 / tl.sum(
        tl.where(cond, tl.exp(logits_softcapped - top1_v), 0.0), axis=1
    )

    # 6. store to output
    offs_topk = pid * topk * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    topk_mask = offs_topk < bs
    tl.store(topk_ids_ptr + offs_topk, top1, mask=topk_mask)
    tl.store(
        topk_weights_ptr + offs_topk,
        invsumexp,
        mask=topk_mask,
    )


def fused_moe_router_large_bs_impl(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    assert num_experts <= BLOCK_SIZE_N
    assert hidden_dim % BLOCK_SIZE_K == 0
    assert topk == 1

    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)

    grid = (triton.cdiv(bs, BLOCK_SIZE_M) * triton.cdiv(num_experts, BLOCK_SIZE_N),)

    fused_moe_router_large_bs_kernel[grid](
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
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        stride_am=hidden_dim,
        stride_bn=hidden_dim,
    )

    return topk_weights, topk_ids


def fused_moe_router_shim(
    moe_softcapping,
    hidden_states,
    gating_output,
    topk,
    renormalize,
):
    assert not renormalize
    assert (
        len(hidden_states.shape) == 2
        and hidden_states.shape[1] == gating_output.shape[1]
    )
    bs, hidden_dim = hidden_states.shape
    num_experts = gating_output.shape[0]
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 256
    if (
        bs >= 512
        and topk == 1
        and num_experts <= BLOCK_SIZE_N
        and hidden_dim % BLOCK_SIZE_K == 0
    ):
        return fused_moe_router_large_bs_impl(
            x=hidden_states,
            router_weight=gating_output,
            topk=topk,
            moe_softcapping=moe_softcapping,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    else:
        return fused_moe_router_impl(
            x=hidden_states,
            router_weight=gating_output,
            topk=topk,
            moe_softcapping=moe_softcapping,
        )


class FusedMoeRouter:
    def __init__(self, router_linear, topk, moe_softcapping) -> None:
        self.router_linear = router_linear
        self.topk = topk
        self.moe_softcapping = moe_softcapping

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:
            return self.forward_cuda(x, residual)
        else:
            return self.forward_vllm(x, residual)

    def forward_cuda(
        self, x: torch.Tensor, autotune=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused_moe_router_shim(
            moe_softcapping=self.moe_softcapping,
            hidden_states=x,
            gating_output=self.router_linear.weight,
            topk=self.topk,
            renormalize=False,
        )

    def forward_vllm(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # g, _ = self.router_linear.forward(x)
        g = x.float() @ self.router_linear.weight.T.float()

        g = torch.tanh(g.float() / self.moe_softcapping) * self.moe_softcapping

        return fused_topk(x, g, self.topk, False)
