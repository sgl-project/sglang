from typing import Optional, Tuple

import torch

from .lib import KVScope, Testcase, TestcaseForDecode, TestParam


def _merge_two_lse(
    lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


def ref_sparse_attn_fwd(
    p: TestParam, t: Testcase
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    indices = t.indices.clone().squeeze(1)
    if t.topk_length is not None:
        mask = torch.arange(p.topk, device=t.topk_length.device).unsqueeze(
            0
        ).broadcast_to(p.s_q, p.topk) >= t.topk_length.unsqueeze(
            1
        )  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= p.s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = t.q.float()
    gathered_kv = (
        t.kv.index_select(dim=0, index=indices.flatten())
        .reshape(p.s_q, p.topk, p.d_qk)
        .float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= t.sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, t.attn_sink, p.s_q, p.h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., : p.d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(torch.bfloat16), out, max_logits, orig_lse)


def ref_sparse_attn_decode(
    p: TestParam, t: TestcaseForDecode
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation of sparse decoding attention in PyTorch
    """
    assert p.h_kv == 1
    assert p.decode is not None
    b = p.decode.b

    def process_kv_scope(kv_scope: KVScope) -> Tuple[torch.Tensor, torch.Tensor]:
        assert kv_scope.indices_in_kvcache is not None
        topk = kv_scope.indices_in_kvcache.size(-1)
        indices_in_kv_cache_fixed = torch.clamp_min(
            kv_scope.indices_in_kvcache, 0
        )  # Otherwise torch.index_select will complain
        gathered_kv = (
            kv_scope.blocked_k.view(-1, p.d_qk)
            .index_select(0, indices_in_kv_cache_fixed.view(-1))
            .view(b, p.s_q, topk, p.d_qk)
        )  # [b, s_q, topk, d]
        invalid_mask = kv_scope.indices_in_kvcache == -1
        if kv_scope.topk_length is not None:
            invalid_mask |= torch.arange(0, topk, device=invalid_mask.device).view(
                1, 1, topk
            ).broadcast_to(b, p.s_q, topk) >= kv_scope.topk_length.view(b, 1, 1)
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(t.kv_scope)
    if t.extra_kv_scope is not None:
        gathered_kv1, invalid_mask1 = process_kv_scope(t.extra_kv_scope)
        gathered_kv = torch.cat(
            [gathered_kv, gathered_kv1], dim=2
        )  # [b, s_q, topk+extra_topk, d]
        invalid_mask = torch.cat(
            [invalid_mask, invalid_mask1], dim=2
        )  # [b, s_q, topk+extra_topk]

    # may use more advanced approach

    gathered_kv = gathered_kv.view(b * p.s_q, -1, p.d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    q = t.q.float().view(b * p.s_q, p.h_q, p.d_qk)
    attn_weight = q @ gathered_kv.transpose(
        -1, -2
    )  # [t.b*t.s_q, t.h_q, topk+extra_topk]
    attn_weight *= t.sm_scale
    attn_weight[
        invalid_mask.view(b * p.s_q, 1, -1).broadcast_to(
            b * p.s_q, p.h_q, invalid_mask.size(-1)
        )
    ] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)  # [t.b*t.s_q, t.h_q]
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., : p.d_v]  # [t.b*t.s_q, t.h_q, t.dv]
    output = output.view(b, p.s_q, p.h_q, p.d_v)
    lse = lse.view(b, p.s_q, p.h_q)

    # Attention sink
    if t.attn_sink is not None:
        output *= (
            1.0 / (1.0 + torch.exp(t.attn_sink.view(1, 1, p.h_q) - lse))
        ).unsqueeze(-1)

    # Correct for q tokens which has no attendable k
    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(b, p.s_q, p.h_q, p.d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    return output.to(torch.bfloat16), lse.transpose(1, 2)
