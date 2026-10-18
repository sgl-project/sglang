"""Muse Spark TP4 NVFP4 spec-decode verify layer (B=1, T = 8 or 16 draft
tokens) on the Helion megakernels; T is a compile-time constant of every kernel.

Per layer, after the previous layer's tail computed this layer's QKVG:
    Attention           sliding-window layers: QK norm/RoPE/KV store + paged
                        causal attention over the T new tokens + norm/gate
                        (global layers keep the stock attention)
    Attention boundary  O-proj + AR + post-attn/pre-FFN norms, pooler + router
                        top-8, NVFP4 expert input
    MoE                 expert-TP4 NVFP4 MoE + post-expert norm finalize
    Tail                unpooler shard + post-FFN boundary + next layer's
                        pre-attn norm and QKVG GEMV
"""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.muse_spark.helion_megakernels.attention_boundary_verify_megakernel import (
    AttentionBoundaryVerifyScalars,
    MuseAttentionBoundaryVerifyMegakernel,
)
from sglang.kernels.ops.muse_spark.helion_megakernels.attention_verify_megakernel import (
    MuseAttentionVerifyMegakernel,
)
from sglang.kernels.ops.muse_spark.helion_megakernels.moe_tp4_verify_megakernel import (
    MoeTp4VerifyBank,
    MuseMoeTailVerifyMegakernel,
    MuseMoeTp4VerifyMegakernel,
)


class AttentionInputs(msgspec.Struct, frozen=True):
    """Live cache metadata; K/V are ``[pages, page_size * kv_width]`` views.

    A row may have a larger physical stride when the cache embeds quantization
    metadata after each page, so the attention megakernel must retain the page
    dimension.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cache_loc: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor


class LayerScalars(msgspec.Struct, frozen=True):
    """Checkpoint scalars the megakernels compile in as constants, each read
    from the module whose stock kernel it replaces."""

    attn_scale: float
    attn_norm_eps: float  # QK norm and output norm
    post_attn_eps: float
    pre_ffn_eps: float
    pre_expert_eps: float
    router_multiplier: float
    route_scale: float
    route_eps: float
    post_expert_eps: float
    post_ffn_eps: float
    next_pre_attn_eps: float


class MoeVerifyWeights(msgspec.Struct, frozen=True):
    bank: MoeTp4VerifyBank
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor
    post_expert_norm: torch.Tensor
    unpool: torch.Tensor  # [8192 / W, 4096] local row block
    alpha: torch.Tensor
    beta: torch.Tensor
    post_weight: torch.Tensor
    pre_weight: torch.Tensor
    next_qkv: torch.Tensor | None  # [Nq, 8192] the next layer's QKV rows, if fused


class MegakernelVerify:
    """One rank's T-token verify megakernel instances, shared by every layer
    with the same scalars: layers run in stream order, and the LL epochs
    advance on device."""

    def __init__(
        self,
        *,
        device: torch.device,
        group: object,
        scalars: LayerScalars,
        tokens: int,
    ):
        self.tokens = tokens
        self._sliding_attention = MuseAttentionVerifyMegakernel(
            device,
            tokens=tokens,
            sliding_window=True,
            bmm1_scale=scalars.attn_scale,
            eps=scalars.attn_norm_eps,
        )
        self.attention_boundary = MuseAttentionBoundaryVerifyMegakernel(
            device,
            group,
            AttentionBoundaryVerifyScalars(
                post_eps=scalars.post_attn_eps,
                pre_eps=scalars.pre_ffn_eps,
                quant_eps=scalars.pre_expert_eps,
                router_multiplier=scalars.router_multiplier,
                route_scale=scalars.route_scale,
                route_eps=scalars.route_eps,
            ),
            tokens=tokens,
        )
        self._moe = MuseMoeTp4VerifyMegakernel(
            device, group, tokens=tokens, norm_eps=scalars.post_expert_eps
        )
        self._tail = MuseMoeTailVerifyMegakernel(
            device,
            group,
            tokens=tokens,
            post_eps=scalars.post_ffn_eps,
            pre_eps=scalars.next_pre_attn_eps,
        )
        self.topk_ids: torch.Tensor | None = None

    def sliding_attention(
        self, *, qkv: torch.Tensor, attention: AttentionInputs
    ) -> torch.Tensor:
        """A sliding-window layer's gated attention output [T, Hq * D]; also
        stores the T new K/V rows."""
        return self._sliding_attention(
            qkv,
            attention.positions,
            attention.cos_sin_cache,
            attention.k_cache,
            attention.v_cache,
            attention.cache_loc,
            attention.block_tables,
            attention.seq_lens,
        )

    def layer(
        self,
        *,
        gated: torch.Tensor,
        stream: torch.Tensor,
        attention_boundary: object,
        moe: MoeVerifyWeights,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """From the gated attention output, returns the new FP32 residual stream,
        the next attention input and, with moe.next_qkv, the next layer's QKV
        projection."""

        inputs = self._moe.inputs
        prepared = self.attention_boundary(gated, stream, attention_boundary, inputs)
        self.topk_ids = prepared["topk_ids"]
        expert_output = self._moe(
            bank=moe.bank,
            packed_routing=inputs.packed,
            hidden_states=inputs.fp4,
            hidden_states_scale=inputs.scales,
            per_token_scale=inputs.per_token_scale,
            output1_scale_scalar=moe.output1_scale_scalar,
            output1_scale_gate_scalar=moe.output1_scale_gate_scalar,
            output2_scale_scalar=moe.output2_scale_scalar,
            norm_weight=moe.post_expert_norm,
        )
        return self._tail(
            expert_output=expert_output,
            unpool_weight=moe.unpool,
            stream=prepared["new_stream"],
            alpha=moe.alpha,
            beta=moe.beta,
            post_weight=moe.post_weight,
            pre_weight=moe.pre_weight,
            next_qkv_weight=moe.next_qkv,
        )


_INSTANCES: dict[tuple[int, object, LayerScalars, int], MegakernelVerify] = {}


def get_megakernel_verify(
    *, device: torch.device, group: object, scalars: LayerScalars, tokens: int
) -> MegakernelVerify:
    """Build once per (device, group, scalars, T); the constructors rendezvous
    symmetric buffers, so every TP rank must reach a new key together, outside
    graph capture."""

    key = (device.index, group, scalars, tokens)
    if key not in _INSTANCES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("build the Helion verify megakernels before capture")
        _INSTANCES[key] = MegakernelVerify(
            device=device, group=group, scalars=scalars, tokens=tokens
        )
    return _INSTANCES[key]


__all__ = [
    "AttentionInputs",
    "LayerScalars",
    "MegakernelVerify",
    "MoeVerifyWeights",
    "get_megakernel_verify",
]
