"""The frozen vision tower's TOP program: serving's vision layers and the trainer's tower call these functions.

Forward only. The tower is frozen on both sides, so nothing here keeps autograd state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# DeepGEMM's BF16 GEMM takes K in whole multiples of this.
_GEMM_K_ALIGN = 128


def _require_top() -> None:
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    if not is_true_on_policy_enabled():
        raise RuntimeError(
            "[top] the vision tower program needs a published TOP runtime"
        )


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """A biased projection: TOP's BI DeepGEMM over the full K rounded to BF16, then the bias added in BF16.

    x [..., K]; weight [N, K] -> [..., N]. Zero columns pad K to DeepGEMM's multiple (the tower MLP's K is 4304);
    they add exact zeros after every real term, so each row keeps its sum."""
    from sglang.srt.batch_invariant_ops import matmul_persistent

    _require_top()
    lead = x.shape[:-1]
    rows = x.reshape(-1, x.shape[-1])
    pad = -rows.shape[1] % _GEMM_K_ALIGN
    if pad:
        rows = F.pad(rows, (0, pad))
        weight = F.pad(weight, (0, pad))
    out = matmul_persistent(rows.contiguous(), weight.t())
    if bias is not None:
        out = out + bias
    return out.view(*lead, out.shape[-1])


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Stands in for RMSNorm.forward_native with the same math, as a function on bare tensors: the trainer's tower
    holds plain weights, not RMSNorm modules, and calls it.

    RMSNorm in FP32 with an FP32 weight, rounded to the activation dtype once: RMSNorm.forward_native's math."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * weight).to(dtype)


def rotate(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D rotary on q/k [T, H, D] with FP32 cos/sin tables [T, D], rounded to the activation dtype once.

    Eager torch: a compiler may fuse the multiply-add, which changes the rounding of an FP32 table's products."""
    from sglang.srt.layers.rotary_embedding.utils import (
        apply_rotary_pos_emb_native_eager,
    )

    return apply_rotary_pos_emb_native_eager(q, k, cos, sin)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Packed noncausal attention on the shared FA4 forward, one key split. q/k/v [T, H, D] BF16 -> [T, H, D]."""
    from sglang.srt.true_on_policy.attention import noncausal_varlen_forward

    _require_top()
    return noncausal_varlen_forward(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        softmax_scale=softmax_scale,
    )


def rotary_table(*, head_dim: int, max_grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [max_grid, head_dim // 4] FP32 of the tower's 2D rotary, from serving's RotaryEmbedding cache.

    Under TOP the cache takes its inverse frequencies from the CPU; building it here gives the trainer that table."""
    from sglang.srt.layers.rotary_embedding import get_rope

    _require_top()
    with torch.device("cuda"):
        rope = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )
    return rope.get_cos_sin(max_grid)


def refuse_unqualified_serving(*, deepstack: bool) -> None:
    """Refuse an image or video request the tower program does not cover.

    Launch admits the text program only, since an engine cannot know it will be sent media; this runs when the
    tower first encodes. --mm-enable-dp-encoder needs no entry: a TOP engine's attention TP is 1, where the DP
    encoder encodes every image on the rank that holds it, through this same tower."""
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.runtime_context import get_mm, get_server_args
    from sglang.srt.true_on_policy.fp8 import is_block_fp8
    from sglang.srt.true_on_policy.registry import admit_serving

    server_args = get_server_args()
    hf_config = model_config_of(server_args).hf_config
    admit_serving(
        server_args,
        model_type=hf_config.model_type,
        fp8=is_block_fp8(getattr(hf_config, "quantization_config", None)),
        vision=True,
    )
    refused = [
        reason
        for reason, hit in (
            (
                "--mm-attention-backend: the tower's attention is the shared noncausal FA4 forward, whatever is named",
                get_mm().mm_attention_backend not in (None, "fa4"),
            ),
            (
                "deepstack mergers: the trainer's tower has none",
                deepstack,
            ),
        )
        if hit
    ]
    if refused:
        raise NotImplementedError(
            "[top] the vision tower program refuses:\n  " + "\n  ".join(refused)
        )
