# SPDX-License-Identifier: Apache-2.0
"""MAGI-2-preview's 40-layer MoE DiT: video, audio and text share one packed
sequence over a 4-stream (mHC) residual; ``mm_layers`` are dense, the rest MoE.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.multimodal_gen.configs.models.dits.magi2 import (
    Magi2PreviewArchConfig,
    Magi2PreviewConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_block
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.moe_multihead import Magi2MultiHeadMoE
from sglang.multimodal_gen.runtime.layers.utils import direct_register_custom_op
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.magi2_layers import (
    Magi2ModalityLinear,
    Magi2ModalityRMSNorm,
    Magi2PostAdapter,
    Magi2PreAdapter,
    Magi2SegmentLayout,
    apply_partial_rope,
    gather_packed_rows,
    shard_packed_rows,
    sharded_cu_seqlens,
    swiglu7_interleaved,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MAGI2_PREVIEW_FORWARD_KWARGS = frozenset(
    {
        "video_latents",
        "audio_latents",
        "text_embeds",
        "layout",
        "coords",
        "timestep",
        "ref_patches",
        "ref_special",
    }
)


class Magi2PreviewDiT(CachableDiT, LayerwiseOffloadableModuleMixin):
    """MAGI-2-preview's denoiser.

    Keyword arguments only: ``DenoisingStage`` filters kwargs by signature, so a
    positional forward would silently drop the layout and coordinates.
    """

    _fsdp_shard_conditions = [is_block]
    # Per block, not the whole stack: fusing the mHC and norm chains is the win, and
    # the expert GEMMs inside stay uncompiled.
    _compile_conditions = [is_block]
    param_names_mapping: dict = {}
    # Adapters/mHC are fp32 and blocks bf16, so FSDP gathers per-parameter dtypes.
    _fsdp_mixed_dtype_params = True
    layer_names = ["blocks"]
    # Sinks and packed varlen are FA-only; other backends drop sink logits silently.
    _supported_attention_backends = {AttentionBackendEnum.FA}

    def __init__(
        self,
        config: Magi2PreviewConfig,
        hf_config: dict | None = None,
        *,
        ep_group=None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch: Magi2PreviewArchConfig = self.config

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.num_stream = arch.num_stream

        self.pre_adapter = Magi2PreAdapter(arch)
        self.blocks = nn.ModuleList(
            Magi2PreviewBlock(arch, layer_idx=i, ep_group=ep_group)
            for i in range(arch.num_layers)
        )
        self.post_adapter = Magi2PostAdapter(arch)

        self.__post_init__()

    def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        unexpected = set(kwargs) - MAGI2_PREVIEW_FORWARD_KWARGS
        if unexpected:
            raise TypeError(f"unexpected forward kwargs: {sorted(unexpected)}")

        layout: Magi2SegmentLayout = kwargs["layout"]
        rows, rope = self.pre_adapter(
            video=kwargs["video_latents"],
            audio=kwargs["audio_latents"],
            text=kwargs["text_embeds"],
            layout=layout,
            coords=kwargs["coords"],
            timestep=kwargs["timestep"],
            ref_patches=kwargs.get("ref_patches"),
            ref_special=kwargs.get("ref_special"),
        )

        streams = rows.view(-1, self.num_stream, self.hidden_size)
        streams = streams.to(self.blocks[0].attention.linear_qkv.weight.dtype)

        (streams, rope, modality_ids), plan = shard_packed_rows(
            streams, rope, layout.modality_ids
        )
        # Not layout.cu_seqlens: pad rows from an uneven split get their own varlen
        # segment so real tokens cannot attend to them.
        cu_seqlens, max_seqlen = sharded_cu_seqlens(plan=plan, device=streams.device)

        for block in self.blocks:
            streams = block(
                streams,
                rope=rope,
                modality_ids=modality_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        rows = gather_packed_rows(streams.reshape(streams.shape[0], -1), plan=plan)
        return self.post_adapter(rows, layout=layout)


class Magi2PreviewBlock(nn.Module):
    """One transformer block: mHC-gated attention then mHC-gated MLP, over
    ``[T, num_stream, hidden]``."""

    def __init__(
        self,
        config: Magi2PreviewArchConfig,
        *,
        layer_idx: int,
        ep_group=None,
    ) -> None:
        super().__init__()
        self.num_stream = config.num_stream
        is_dense = layer_idx in config.mm_layers
        block_modality = config.num_modality if is_dense else 1

        self.mhc_norm = Magi2ModalityRMSNorm(
            config.num_stream * config.hidden_size, num_modality=block_modality
        )
        self.mhc_attn = Magi2MHC(
            num_stream=config.num_stream,
            hidden_size=config.hidden_size,
            alpha_init=config.mhc_alpha_init,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
        )
        self.mhc_mlp = Magi2MHC(
            num_stream=config.num_stream,
            hidden_size=config.hidden_size,
            alpha_init=config.mhc_alpha_init,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
        )

        self.attention = Magi2PreviewAttention(config, num_modality=block_modality)
        self.mlp = (
            Magi2DenseMLP(config)
            if is_dense
            else Magi2MoEMLP(config, ep_group=ep_group)
        )

    def forward(
        self,
        streams: torch.Tensor,
        *,
        rope: torch.Tensor,
        modality_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        flat = streams.reshape(streams.shape[0], -1)
        # fp32 out: project() upcasts anyway, so bf16 here would round twice.
        normed = self.mhc_norm(flat, modality_ids, out_dtype=torch.float32)

        h_pre, h_post, h_res = self.mhc_attn.project(normed)
        attn_out = self.attention(
            self.mhc_attn.mix_input(streams, h_pre),
            rope=rope,
            modality_ids=modality_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        streams = self.mhc_attn.mix_output(streams, attn_out, h_post, h_res)

        flat = streams.reshape(streams.shape[0], -1)
        normed = self.mhc_norm(flat, modality_ids, out_dtype=torch.float32)
        h_pre, h_post, h_res = self.mhc_mlp.project(normed)
        mlp_out = self.mlp(
            self.mhc_mlp.mix_input(streams, h_pre), modality_ids=modality_ids
        )
        return self.mhc_mlp.mix_output(streams, mlp_out, h_post, h_res)


class Magi2DenseMLP(nn.Module):
    """Per-modality dense MLP used by the ``mm_layers`` blocks."""

    def __init__(self, config: Magi2PreviewArchConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.dense_intermediate_size
        modality = config.num_modality

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=modality)
        self.up_gate_proj = Magi2ModalityLinear(
            hidden, 2 * inter, num_modality=modality
        )
        self.down_proj = Magi2ModalityLinear(inter, hidden, num_modality=modality)

    def forward(self, x: torch.Tensor, *, modality_ids: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x, modality_ids)
        h = swiglu7_interleaved(self.up_gate_proj(h, modality_ids))
        return self.down_proj(h, modality_ids)


class Magi2MoEMLP(nn.Module):
    """Multi-head MoE plus two always-on shared experts, one modality-agnostic and
    one per-modality."""

    def __init__(self, config: Magi2PreviewArchConfig, *, ep_group=None) -> None:
        super().__init__()
        hidden = config.hidden_size
        modality = config.num_modality
        shared = config.moe_shared_expert_intermediate_size
        modality_shared = config.moe_modality_expert_intermediate_size

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=modality)

        self.split_linear = Magi2ModalityLinear(hidden, hidden)
        self.merge_linear = Magi2ModalityLinear(hidden, hidden)
        self.moe_mlp = Magi2MultiHeadMoE(
            num_heads=config.moe_num_heads,
            num_experts=config.moe_num_experts,
            hidden_size=hidden,
            intermediate_size=config.moe_expert_intermediate_size,
            top_k=config.moe_top_k,
            route_scale=config.moe_route_scale,
            score_func=config.moe_score_func,
            route_norm=config.moe_route_norm,
            ep_group=ep_group,
        )

        self.shared_expert_fc1 = Magi2ModalityLinear(hidden, 2 * shared)
        self.shared_expert_fc2 = Magi2ModalityLinear(shared, hidden)
        self.modality_specific_shared_expert_fc1 = Magi2ModalityLinear(
            hidden, 2 * modality_shared, num_modality=modality
        )
        self.modality_specific_shared_expert_fc2 = Magi2ModalityLinear(
            modality_shared, hidden, num_modality=modality
        )

    def forward(self, x: torch.Tensor, *, modality_ids: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x, modality_ids)

        routed = self.merge_linear(self.moe_mlp(self.split_linear(h)))

        agnostic = self.shared_expert_fc2(
            swiglu7_interleaved(self.shared_expert_fc1(h))
        )
        per_modality = self.modality_specific_shared_expert_fc2(
            swiglu7_interleaved(
                self.modality_specific_shared_expert_fc1(h, modality_ids)
            ),
            modality_ids,
        )
        return routed + agnostic + per_modality


class Magi2PreviewAttention(nn.Module):
    """Sink attention with QK-norm and per-head output gating."""

    def __init__(self, config: Magi2PreviewArchConfig, *, num_modality: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_heads = heads

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=num_modality)
        self.linear_qkv = Magi2ModalityLinear(
            hidden, 3 * hidden, num_modality=num_modality
        )
        self.q_norm = Magi2ModalityRMSNorm(self.head_dim, num_modality=num_modality)
        self.k_norm = Magi2ModalityRMSNorm(self.head_dim, num_modality=num_modality)
        self.linear_g = Magi2ModalityLinear(hidden, heads, num_modality=num_modality)
        self.linear_proj = Magi2ModalityLinear(
            hidden, hidden, num_modality=num_modality
        )

        self.attn = Magi2SinkAttention(
            num_heads=heads,
            head_dim=self.head_dim,
            sink_token_num=config.sink_token_num,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: torch.Tensor,
        modality_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        normed = self.pre_norm(x, modality_ids)
        qkv = self.linear_qkv(normed, modality_ids)
        gate = self.linear_g(normed, modality_ids)

        tokens = qkv.shape[0]
        qkv = qkv.view(tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # fp32 through the rope; the reference converts only at the attention boundary.
        q = self.q_norm(q, modality_ids, out_dtype=torch.float32)
        k = self.k_norm(k, modality_ids, out_dtype=torch.float32)

        sin, cos = rope.tensor_split(2, -1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)

        # FlashAttention requires q, k and v to share a dtype.
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        out = self.attn(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out = out * torch.sigmoid(gate).unsqueeze(-1)
        out = out.reshape(tokens, self.num_heads * self.head_dim)
        return self.linear_proj(out, modality_ids)


@triton.jit
def _mhc_mix_output_kernel(
    streams_ptr,
    block_out_ptr,
    post_ptr,
    res_ptr,
    out_ptr,
    hidden,
    NUM_STREAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < hidden
    offs_n = tl.arange(0, NUM_STREAM)

    acc = tl.zeros([NUM_STREAM, BLOCK_C], dtype=tl.float32)
    for j in tl.static_range(NUM_STREAM):
        stream = tl.load(
            streams_ptr + (token * NUM_STREAM + j) * hidden + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        # res[token, :, j]: the column every output stream mixes this stream by.
        res = tl.load(
            res_ptr + token * NUM_STREAM * NUM_STREAM + offs_n * NUM_STREAM + j
        )
        acc += res[:, None] * stream[None, :]

    written = tl.load(block_out_ptr + token * hidden + offs_c, mask=mask_c, other=0.0)
    post = tl.load(post_ptr + token * NUM_STREAM + offs_n)
    acc += post[:, None] * written.to(tl.float32)[None, :]

    tl.store(
        out_ptr + (token * NUM_STREAM + offs_n[:, None]) * hidden + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_c[None, :],
    )


def _mhc_mix_output(
    streams: torch.Tensor,
    block_out: torch.Tensor,
    post: torch.Tensor,
    res: torch.Tensor,
) -> torch.Tensor:
    """``res @ streams + post outer block_out`` in one pass, accumulating in fp32.

    The torch spelling reads the 4-stream residual three times and materializes the
    ``post`` outer product at full width only to add it.
    """
    # The kernel indexes every tensor by raw offset (token-major, last dim 1),
    # so a strided view would be read wrong with no error. Every in-tree
    # producer is contiguous; normalise the two activations and keep the
    # explicit copies of the small parameter tensors.
    if not streams.is_contiguous():
        streams = streams.contiguous()
    if not block_out.is_contiguous():
        block_out = block_out.contiguous()
    tokens, num_stream, hidden = streams.shape
    # Always contiguous so the fake and the real impl agree on output strides.
    out = torch.empty(streams.shape, dtype=streams.dtype, device=streams.device)
    block_c = min(1024, triton.next_power_of_2(hidden))
    _mhc_mix_output_kernel[(tokens, triton.cdiv(hidden, block_c))](
        streams,
        block_out,
        post.contiguous(),
        res.contiguous(),
        out,
        hidden,
        NUM_STREAM=num_stream,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _mhc_mix_output_fake(
    streams: torch.Tensor,
    block_out: torch.Tensor,
    post: torch.Tensor,
    res: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(streams)


# Registered at import, not on first call: a lazily registered op inside a compiled
# region breaks the graph and pays Triton's JIT per block. As an op inductor emits an
# opaque call, which is how the reference keeps its own kernels out of its compiler.
direct_register_custom_op(
    op_name="magi2_mhc_mix_output",
    op_func=_mhc_mix_output,
    mutates_args=[],
    fake_impl=_mhc_mix_output_fake,
)

mhc_mix_output = torch.ops.sglang.magi2_mhc_mix_output


def sinkhorn_knopp(h: torch.Tensor, *, num_iters: int, eps: float) -> torch.Tensor:
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


class Magi2MHC(nn.Module):
    """``phi_fused`` projects to ``2 * n + n * n``: pre-mix, post-mix, and a flattened stream-to-stream matrix."""

    def __init__(
        self,
        *,
        num_stream: int,
        hidden_size: int,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_stream = num_stream
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        # Scaled against the full concatenated stream, not hidden_size.
        self.matmul_scale = 1.0 / math.sqrt(num_stream * hidden_size)

        n = num_stream
        self.phi_fused = nn.Parameter(
            torch.zeros(n * hidden_size, 2 * n + n * n, dtype=torch.float32)
        )
        self.alpha_pre = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))
        self.bias_pre = nn.Parameter(torch.zeros(n))
        self.bias_post = nn.Parameter(torch.zeros(n))
        self.bias_res = nn.Parameter(torch.zeros(n, n))

    def project(self, streams_flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = self.num_stream
        fused = torch.matmul(streams_flat.float(), self.phi_fused)
        h_pre, h_post, h_res = torch.split(fused, [n, n, n * n], dim=-1)
        return h_pre, h_post, h_res.view(-1, n, n)

    def mix_input(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha_pre * self.matmul_scale * h_pre + self.bias_pre)
        # Left in torch: inductor fuses this into its consumer, where a custom op
        # is opaque and cannot be fused (measured 5s/clip slower as a kernel).
        return torch.einsum("tn,tnc->tc", gate.to(streams.dtype), streams)

    def mix_output(
        self,
        streams: torch.Tensor,
        block_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        # Scaled by 2 so the gate spans (0, 2) and can amplify, not just damp.
        post = 2.0 * torch.sigmoid(
            self.alpha_post * self.matmul_scale * h_post + self.bias_post
        )
        # Left in torch: a fused kernel measured 0.2% end to end, inside the noise.
        res = sinkhorn_knopp(
            self.alpha_res * self.matmul_scale * h_res.float() + self.bias_res,
            num_iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        if streams.is_cuda:
            return mhc_mix_output(streams, block_out, post, res)

        mixed = torch.einsum("tij,tjc->tic", res.to(streams.dtype), streams)
        written = torch.einsum("tn,tc->tnc", post.to(streams.dtype), block_out)
        return mixed + written


class Magi2SinkAttention(nn.Module):
    """``USPAttention``'s varlen paths cannot pass sink logits, so this drives ``FlashAttentionImpl`` directly."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        sink_token_num: int = 0,
        window_size: tuple[int, int] = (-1, -1),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale or head_dim**-0.5
        self.window_size = window_size

        # fp32: compared against attention logits, and the reference avoids bf16 here.
        self.sinks = (
            nn.Parameter(torch.zeros(sink_token_num, num_heads, dtype=torch.float32))
            if sink_token_num
            else None
        )
        self._impl = None
        self._impl_dtype: torch.dtype | None = None

    def _local_sinks(self) -> torch.Tensor | None:
        """``_usp_input_all_to_all_packed_qkv`` gives rank *r* heads ``[r * h_local, (r + 1) * h_local)``."""
        if self.sinks is None:
            return None
        if self.sinks.shape[0] != 1:
            raise ValueError(
                "FlashAttention takes one sink logit per head, so "
                f"sink_token_num must be 1; got {self.sinks.shape[0]}"
            )
        # The checkpoint stores [sink_token_num, num_heads]; the kernel wants flat.
        sinks = self.sinks[0]
        world_size = get_ulysses_parallel_world_size()
        if world_size <= 1:
            return sinks
        h_local = self.num_heads // world_size
        start = get_ulysses_parallel_rank() * h_local
        return sinks[start : start + h_local]

    def _ensure_impl(self, dtype: torch.dtype) -> None:
        # The impl holds a dtype-cast copy of the sinks, so a dtype change has to
        # rebuild it; reusing the old one would feed the kernel logits of the
        # wrong width.
        if self._impl is not None and self._impl_dtype == dtype:
            return
        backend = get_attn_backend(
            self.head_dim,
            dtype,
            supported_attention_backends={AttentionBackendEnum.FA},
            selected_attention_backend=AttentionBackendEnum.FA,
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )
        impl_cls = backend.get_impl_cls()
        # The platform FA resolver can still hand back SDPA, which drops sinks.
        if not issubclass(impl_cls, FlashAttentionImpl):
            raise RuntimeError(
                "MAGI-2 attention needs FlashAttention for its sink logits and "
                f"window, but the platform resolved {backend.get_enum().name}; "
                "run on a GPU with FA support"
            )
        world_size = get_ulysses_parallel_world_size()
        local_heads = self.num_heads // world_size
        local_kv_heads = max(1, self.num_kv_heads // world_size)
        sinks = self._local_sinks()
        self._impl = impl_cls(
            num_heads=local_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            num_kv_heads=local_kv_heads,
            # The parameter stays fp32, but FA3 wants sinks in the q dtype.
            sinks=None if sinks is None else sinks.to(dtype).contiguous(),
            window_size=self.window_size,
        )
        self._impl_dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all_packed_qkv,
            _usp_output_all_to_all,
        )

        ulysses = get_ulysses_parallel_world_size() > 1
        if ulysses:
            q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

        self._ensure_impl(q.dtype)
        out = self._impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if ulysses:
            out = _usp_output_all_to_all(out[None], head_dim=2)[0]
        return out
