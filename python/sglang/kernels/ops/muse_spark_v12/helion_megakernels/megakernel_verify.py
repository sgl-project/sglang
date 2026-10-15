"""Muse Spark TP4 NVFP4 spec-decode verify layer (B=1, T = 8 or 16 draft
tokens) on the Helion megakernels; T is a compile-time constant of every kernel.

Per layer, after the stock QKVG GEMM:
    Attention           sliding-window layers: QK norm/RoPE/KV store + paged
                        causal attention over the T new tokens + norm/gate
                        (global layers keep the stock attention)
    (stock)             O-proj + all-reduce + post-attn/pre-FFN norms
    Attention boundary  pooler + router top-8, NVFP4 expert input
    MoE                 expert-TP4 NVFP4 MoE + post-expert norm finalize
    (stock)             unpooler shard + all-gather + post-FFN boundary
"""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.muse_spark_v12.helion_megakernels.attention_boundary_verify_megakernel import (
    AttentionBoundaryVerifyScalars,
    MuseAttentionBoundaryVerifyMegakernel,
)
from sglang.kernels.ops.muse_spark_v12.helion_megakernels.attention_verify_megakernel import (
    MuseAttentionVerifyMegakernel,
)
from sglang.kernels.ops.muse_spark_v12.helion_megakernels.moe_tp4_verify_megakernel import (
    MoeTp4VerifyBank,
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
    pre_expert_eps: float
    router_multiplier: float
    route_scale: float
    route_eps: float
    post_expert_eps: float


class MoeVerifyWeights(msgspec.Struct, frozen=True):
    bank: MoeTp4VerifyBank
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor
    post_expert_norm: torch.Tensor


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

    def experts(
        self, *, ffn_in: torch.Tensor, attention_boundary: object, moe: MoeVerifyWeights
    ):
        """From the pre-FFN-normed hidden state, returns the post-expert-normed
        route sum [T, H_e] (replicated); the top-8 ids land in topk_ids."""

        inputs = self._moe.inputs
        self.topk_ids = self.attention_boundary(ffn_in, attention_boundary, inputs)
        return self._moe(
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
