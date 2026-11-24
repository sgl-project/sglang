# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 layers: zero-centred norms, rotary table, gated attention, dense and MoE feed-forwards, and the action expert.

Parameter names are the checkpoint key suffixes, so the checkpoint loads without a name mapping.
"""

from __future__ import annotations

import functools
import importlib.util
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from sglang.multimodal_gen.configs.models.dits.praxis import PraxisArchConfig

# One compiled callable for the process; each new sequence length compiles once more.
_compiled_flex_attention = torch.compile(flex_attention)

# The CuTe flash kernel of flex attention uses 128-wide key tiles only on Hopper and datacenter Blackwell. Other
# architectures (sm_120 consumer Blackwell among them) pick 64-wide tiles, which the 128 x 128 block mask rejects.
_FLASH_MAJORS = (9, 10)


@functools.cache
def _upstream_fa4_available() -> bool:
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except ImportError:
        return False


def _flex_kernel_options(backend: str, device: torch.device) -> dict[str, str] | None:
    if (
        backend == "flash"
        and torch.cuda.get_device_capability(device)[0] in _FLASH_MAJORS
        and _upstream_fa4_available()
    ):
        return {"BACKEND": "FLASH"}
    return None


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """``x * scale + shift`` in float32, rounded to the dtype of ``x``."""
    out = x.float() * scale.float() + shift.float()
    return out.to(x.dtype)


class Fp32RMSNorm(nn.Module):
    """Zero-centred RMSNorm (scale ``1 + weight``) that computes in float32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        weight = 1 + self.weight.float()
        out = F.rms_norm(x.float(), self.normalized_shape, weight=weight, eps=self.eps)
        return out.to(x.dtype)


class ZeroCenteredRMSNorm(nn.Module):
    """Zero-centred RMSNorm in the dtype of the input."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.dim,), weight=1 + self.weight, eps=self.eps)


class RotaryEmbedding(nn.Module):
    """Rotary (cos, sin) table of integer positions.

    Only the first ``effective_dim // 2`` frequency pairs rotate. A position splits as
    ``high * table_size + low``; each part's angle is a float64 product, and the two parts combine with
    the angle-sum identity in float64, so the table is exact for positions far beyond float32 range.
    """

    def __init__(
        self, *, dim: int, effective_dim: int, base: int, table_size: int
    ) -> None:
        super().__init__()
        self.table_size = table_size
        # Built on the CPU even when the model is created on the meta device.
        exponent = (
            torch.arange(0, effective_dim, 2, device="cpu").float() / effective_dim
        )
        theta = 1.0 / (base**exponent)
        pad = torch.zeros((dim - effective_dim) // 2, device="cpu")
        # A plain attribute, not a buffer: module casts and moves must not touch the float32 CPU copy.
        self._theta = torch.cat([theta, pad])
        self._theta_on: dict[torch.device, Tensor] = {}

    def _theta64(self, device: torch.device) -> Tensor:
        if device not in self._theta_on:
            self._theta_on[device] = self._theta.to(device).double()
        return self._theta_on[device]

    @staticmethod
    def _cos_sin(angle: Tensor) -> tuple[Tensor, Tensor]:
        return angle.cos().float().double(), angle.sin().float().double()

    def forward(self, positions: Tensor) -> Tensor:
        """Map positions ``(B, N)`` to a float32 table ``(B, N, dim // 2, 2)``."""
        theta = self._theta64(positions.device)
        high = positions // self.table_size
        low = positions % self.table_size
        high_pos = (high.to(torch.float32) * self.table_size).double()
        low_pos = low.to(torch.float32).double()
        cos_high, sin_high = self._cos_sin(high_pos[..., None] * theta)
        cos_low, sin_low = self._cos_sin(low_pos[..., None] * theta)
        cos = cos_high * cos_low - sin_high * sin_low
        sin = sin_high * cos_low + cos_high * sin_low
        return torch.stack([cos, sin], dim=-1).float()


def apply_rope(x: Tensor, table: Tensor) -> Tensor:
    """Rotate ``x`` (B, H, N, D) in float32; channel ``j`` pairs with ``j + D // 2``."""
    x0, x1 = x.float().chunk(2, dim=-1)
    seq_len, half = x0.shape[2], x0.shape[3]
    cos, sin = table.float().chunk(2, dim=-1)
    cos = cos.reshape(-1, 1, seq_len, half)
    sin = sin.reshape(-1, 1, seq_len, half)
    out = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-2)
    return out.flatten(-2).to(x.dtype)


class QKVProjection(nn.Module):
    def __init__(
        self, dim: int, head_dim: int, num_heads: int, num_kv_heads: int
    ) -> None:
        super().__init__()
        self.q_weight = nn.Parameter(torch.empty(num_heads * head_dim, dim))
        self.k_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))
        self.v_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            F.linear(x, self.q_weight),
            F.linear(x, self.k_weight),
            F.linear(x, self.v_weight),
        )


class GatedAttention(nn.Module):
    """Grouped-query attention with a sigmoid gate per head.

    The order is project, rotate q and k, RMS-normalize q and k per head, attend, gate, project out.
    """

    def __init__(
        self, *, dim: int, head_dim: int, num_kv_heads: int, backend: str
    ) -> None:
        super().__init__()
        # "flash" is the CuTe flash kernel of flex attention, "triton" the default flex kernel.
        self._backend = backend
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.in_proj = QKVProjection(dim, head_dim, self.num_heads, num_kv_heads)
        self.gate_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.q_norm = Fp32RMSNorm(head_dim)
        self.k_norm = Fp32RMSNorm(head_dim)
        self.out_proj = nn.Linear(self.num_heads * head_dim, dim)

    def forward(self, x: Tensor, *, rope: Tensor, mask: Tensor | BlockMask) -> Tensor:
        b, s, _ = x.shape
        q, k, v = self.in_proj(x)
        gate = self.gate_proj(x).sigmoid().permute(0, 2, 1).reshape(b, -1, s, 1)

        def heads(t: Tensor) -> Tensor:
            return t.reshape(b, s, -1, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(apply_rope(heads(q), rope))
        k = self.k_norm(apply_rope(heads(k), rope))
        out = self._attend(q, k, heads(v), mask)
        return self.out_proj((out * gate).permute(0, 2, 1, 3).reshape(b, s, -1))

    def _attend(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | BlockMask
    ) -> Tensor:
        if isinstance(mask, BlockMask):
            return _compiled_flex_attention(
                q,
                k,
                v,
                block_mask=mask,
                enable_gqa=True,
                kernel_options=_flex_kernel_options(self._backend, q.device),
            )
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[None, None], enable_gqa=True
        )


class SwiGLUProjection(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.a_weight = nn.Parameter(torch.empty(hidden, dim))
        self.b_weight = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return F.linear(x, self.a_weight), F.linear(x, self.b_weight)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.linear = SwiGLUProjection(dim, hidden)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.linear(x)
        return F.silu(a) * b


class SwiGLUFeedForward(nn.Sequential):
    """Dense feed-forward: ``[0]`` is the SwiGLU, ``[1]`` projects out."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__(SwiGLU(dim, hidden), nn.Linear(hidden, dim, bias=False))


class RoutedExperts(nn.Module):
    """Expert banks as stored in the checkpoint: gate|up ``(E, C, 2H)``, down ``(E, H, C)``."""

    def __init__(self, num_experts: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_weight = nn.Parameter(torch.empty(num_experts, dim, 2 * hidden))
        self.down_weight = nn.Parameter(torch.empty(num_experts, hidden, dim))


class SharedExpert(nn.Module):
    """The always-on expert, a dense SwiGLU with packed gate|up weights."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Parameter(torch.empty(dim, 2 * hidden))
        self.down = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        gate, up = (x @ self.gate_up).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down


class MoEFeedForward(nn.Module):
    """One shared expert plus the top ``top_k`` of ``num_experts`` routed experts.

    The router runs in float32 with sigmoid scores. The float32 routing bias changes which experts are
    picked, not their weights. The picked scores are normalized to sum to one and rounded to the
    activation dtype. The routed sum is scaled by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.routed = RoutedExperts(num_experts, dim, hidden)
        self.shared = SharedExpert(dim, hidden)
        self.register_buffer(
            "expert_bias", torch.zeros(num_experts, dtype=torch.float32)
        )
        self._experts_transposed = False

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        bias = self.expert_bias
        super()._apply(fn, recurse)
        # The routing bias stays float32 under module dtype casts.
        self.expert_bias = bias.to(device=self.expert_bias.device)
        return self

    def prepare_experts(self) -> None:
        """Re-lay the banks as ``(E, 2H, C)`` and ``(E, C, H)``, the layout of the fused expert kernel."""
        if self._experts_transposed:
            return
        for bank in (self.routed.gate_up_weight, self.routed.down_weight):
            bank.data = bank.data.transpose(1, 2).contiguous()
        self._experts_transposed = True

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        top_idx, gate = self._route(tokens)
        routed = self._run_routed(tokens, top_idx, gate)
        return (routed + self.shared(tokens)).reshape(shape)

    def _route(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        logits = F.linear(tokens.float(), self.router.weight.float())
        scores = torch.sigmoid(logits)
        _, top_idx = torch.topk(scores + self.expert_bias, self.top_k, dim=-1)
        picked = scores.gather(dim=-1, index=top_idx)
        return top_idx, (picked / picked.sum(dim=-1, keepdim=True)).to(tokens.dtype)

    def _run_routed(self, tokens: Tensor, top_idx: Tensor, gate: Tensor) -> Tensor:
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        if not self._experts_transposed:
            raise RuntimeError("prepare_experts() must run after the weights load")
        config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            hidden_size=tokens.shape[-1],
            intermediate_size_per_partition=self.routed.down_weight.shape[-1],
            top_k=self.top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
        )
        topk = StandardTopKOutput(
            topk_weights=gate.float() * self.routed_scaling_factor,
            topk_ids=top_idx.to(torch.int32),
            router_logits=torch.empty(0, device=tokens.device),
        )
        return fused_experts(
            tokens.contiguous(),
            self.routed.gate_up_weight,
            self.routed.down_weight,
            topk,
            config,
        )


class TransformerBlock(nn.Module):
    """Pre-norm block ``x + attn(norm(x) * s)``, then ``x + ffn(norm(x) * s)``, ``s = 1 / sqrt(layer)``."""

    def __init__(
        self,
        *,
        ffn: nn.Module,
        dim: int,
        head_dim: int,
        num_kv_heads: int,
        layer_index: int,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.register_buffer("s", torch.tensor(1 / (layer_index**0.5)))
        self.ln_in = Fp32RMSNorm(dim)
        self.mha = GatedAttention(
            dim=dim,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            backend=attention_backend,
        )
        self.ln_out = Fp32RMSNorm(dim)
        self.ffn = ffn

    def forward(self, x: Tensor, *, rope: Tensor, mask: Tensor | BlockMask) -> Tensor:
        x = x + self.mha(self.ln_in(x) * self.s, rope=rope, mask=mask)
        return x + self.ffn(self.ln_out(x) * self.s)


class Transformer(nn.Module):
    """A stack of blocks with a shared rotary table.

    The first ``num_dense`` blocks have a dense feed-forward, the rest a mixture of experts. With
    ``mod_dim`` set, each block first modulates its input with a linear map of the per-token conditioning.
    """

    def __init__(
        self,
        *,
        depth: int,
        num_dense: int,
        dim: int,
        head_dim: int,
        num_kv_heads: int,
        dense_hidden: int,
        moe: dict,
        mod_dim: int | None,
        rope: dict,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.rope = RotaryEmbedding(dim=head_dim, **rope)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                ffn=(
                    SwiGLUFeedForward(dim, dense_hidden)
                    if i < num_dense
                    else MoEFeedForward(dim=dim, **moe)
                ),
                dim=dim,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                layer_index=i + 1,
                attention_backend=attention_backend,
            )
            for i in range(depth)
        )
        self.modulators: nn.ModuleList | None = None
        if mod_dim is not None:
            self.modulators = nn.ModuleList(
                nn.Linear(mod_dim, 2 * dim) for _ in range(depth)
            )

    def forward(
        self,
        x: Tensor,
        *,
        rope_ids: Tensor,
        mask: Tensor | BlockMask,
        x_mod: Tensor | None = None,
        tap_blocks: Sequence[int] = (),
    ) -> tuple[Tensor, list[Tensor]]:
        """Run all blocks on ``x`` (B, N, C); return the output and the outputs of ``tap_blocks``."""
        rope = self.rope(rope_ids)[:, : x.size(-2)]
        taps: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            if self.modulators is not None:
                shift, scale = self.modulators[i](x_mod).chunk(2, dim=-1)
                x = modulate(x, shift, 1 + scale)
            x = block(x, rope=rope, mask=mask)
            if i in tap_blocks:
                taps[i] = x
        return x, [taps[i] for i in tap_blocks]


# Action expert: a small transformer that predicts the action velocity from backbone taps. Even layers
# cross-attend to the global tokens and the frame tokens of one tap; odd layers self-attend among the expert
# tokens [proprio | future | actions]. Parameter names are the checkpoint key suffixes.


def noised_frame_offset(
    *, num_noised_frames: int, num_clean_frames: int, chunk_size: int
) -> int:
    """Timeline position of noised frame 0 in the expert's view."""
    if num_clean_frames == 0:
        return 0
    if num_noised_frames == 1 or (chunk_size > 1 and num_noised_frames == chunk_size):
        return num_clean_frames
    if num_noised_frames == num_clean_frames + 1 or (
        chunk_size > 1 and num_noised_frames == num_clean_frames + chunk_size
    ):
        return 0
    raise ValueError(
        f"unsupported layout: noised={num_noised_frames}, "
        f"clean={num_clean_frames}, chunk={chunk_size}"
    )


def expert_query_chunks(
    *, num_action_tokens: int, num_future: int, chunk_size: int, device: torch.device
) -> Tensor:
    """Chunk index of each expert token, in the order ``[proprio | future | actions]``."""
    chunks = torch.arange(max(1, num_action_tokens // chunk_size), device=device)
    actions = torch.arange(num_action_tokens, device=device) // chunk_size
    return torch.cat([chunks, chunks.repeat_interleave(num_future), actions])


def expert_cross_mask(
    *,
    query_chunks: Tensor,
    chunk_size: int,
    num_global: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    include_noised: bool,
) -> Tensor:
    """Dense ``(Q, KV)`` mask of the expert cross-attention over ``[global | clean | noised]``."""
    device = query_chunks.device
    offset = noised_frame_offset(
        num_noised_frames=num_noised_frames,
        num_clean_frames=num_clean_frames,
        chunk_size=chunk_size,
    )
    kv_len = num_global + (num_clean_frames + num_noised_frames) * tokens_per_frame
    key = torch.arange(kv_len, device=device)
    is_global = key < num_global
    frame = (key - num_global) // tokens_per_frame
    is_clean = ~is_global & (frame < num_clean_frames)
    is_noised = ~is_global & ~is_clean
    noised_chunk = (frame - num_clean_frames) // chunk_size
    chunk_start = (query_chunks * chunk_size + offset)[:, None]
    window = num_clean_frames + num_noised_frames
    clean_ok = (
        is_clean[None]
        & (frame[None] < chunk_start)
        & (frame[None] >= chunk_start - window)
    )
    noised_ok = is_noised[None] & (noised_chunk[None] == query_chunks[:, None])
    allowed = is_global[None] | clean_ok
    return allowed | noised_ok if include_noised else allowed


def expert_self_mask(*, query_chunks: Tensor) -> Tensor:
    return query_chunks[:, None] == query_chunks[None]


def expert_query_rope_ids(
    *,
    kv_rope_ids: Tensor,
    num_global: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    num_action_tokens: int,
    num_future: int,
    chunk_size: int,
) -> Tensor:
    """``(B, Q)`` trunk rope ids of the expert tokens: each takes the id of one noised frame."""
    batch = kv_rope_ids.shape[0]
    total_frames = num_clean_frames + num_noised_frames
    video = kv_rope_ids[:, num_global:]
    frame_ids = video.view(batch, total_frames, tokens_per_frame).amin(dim=-1)
    noised_ids = frame_ids[:, num_clean_frames:]
    chunks = expert_query_chunks(
        num_action_tokens=num_action_tokens,
        num_future=num_future,
        chunk_size=chunk_size,
        device=kv_rope_ids.device,
    )
    num_prefix = chunks.numel() - num_action_tokens
    frames = torch.cat(
        [
            (chunks[:num_prefix] * chunk_size).clamp(min=0),
            torch.arange(num_action_tokens, device=kv_rope_ids.device),
        ]
    )
    return noised_ids.gather(1, frames[None].expand(batch, -1))


def rms_norm(x: Tensor, eps: float = 1e-4) -> Tensor:
    """RMS norm without a weight: the square in the input dtype, the rest in float32."""
    mean_sq = torch.mean(x**2, dim=-1, keepdim=True, dtype=torch.float32)
    return (x.to(torch.float32) * torch.rsqrt(mean_sq + eps)).to(x.dtype)


def geglu(x: Tensor) -> Tensor:
    value, gate = torch.chunk(x, 2, dim=-1)
    return F.gelu(value, approximate="tanh") * gate


class TimestepEmbedder(nn.Module):
    """Sinusoidal features ``[cos(t f), sin(t f)]`` of a flow time, then a two-layer MLP."""

    def __init__(
        self, dim: int, frequency_dim: int = 256, max_period: float = 10_000
    ) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        ).to(dtype=t.dtype)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        features = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(features.to(dtype=self.mlp[0].weight.dtype))


class AdaLN(nn.Module):
    """Scale-only adaptive RMS norm ``(1 + proj(silu(cond))) * rms_norm(x)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: Tensor, cond: Tensor) -> Tensor:
        scale = self.proj(F.silu(cond))
        return ((1 + scale.float()) * rms_norm(tokens)).to(dtype=tokens.dtype)


class ActionTauEncoder(nn.Module):
    """Action embedding with the flow time at the input: ``w3(silu(w2([w1(a) ++ tau_features])))``."""

    def __init__(self, action_dim: int, dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(action_dim, dim)
        self.tau = TimestepEmbedder(dim)
        self.w2 = nn.Linear(2 * dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, actions: Tensor, tau: Tensor) -> Tensor:
        a = self.w1(actions)
        t = self.tau(tau).to(a.dtype)
        return self.w3(F.silu(self.w2(torch.cat([a, t], dim=-1))))


class ActionExpertBlock(nn.Module):
    """Attention (cross or self) then a GEGLU MLP, each behind a scale-only AdaLN on the flow time."""

    def __init__(
        self, *, dim: int, num_heads: int, head_dim: int, is_cross: bool
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.adaln_attn = AdaLN(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.adaln_mlp = AdaLN(dim)
        self.mlp_in = nn.Linear(dim, 2 * 4 * dim)
        self.mlp_out = nn.Linear(4 * dim, dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        *,
        kv_source: Tensor | None,
        mask: Tensor,
        q_table: Tensor,
        k_table: Tensor,
    ) -> Tensor:
        b, q_len, _ = x.shape
        h = self.adaln_attn(x, cond)
        k, v = self.to_kv(kv_source if self.is_cross else h).chunk(2, dim=-1)

        def split(t: Tensor) -> Tensor:
            return t.view(b, t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q = rms_norm(apply_rope(split(self.to_q(h)), q_table))
        k = rms_norm(apply_rope(split(k), k_table))
        attn = F.scaled_dot_product_attention(
            q, k, split(v), attn_mask=mask[None, None]
        )
        x = x + self.to_out(attn.transpose(1, 2).reshape(b, q_len, -1))
        return x + self.mlp_out(geglu(self.mlp_in(self.adaln_mlp(x, cond))))


class ActionExpert(nn.Module):
    """Action transformer that reads the taps of the video backbone encoder."""

    def __init__(self, config: PraxisArchConfig) -> None:
        super().__init__()
        dim = config.expert_dim
        self.config = config
        self.rope_table = RotaryEmbedding(
            dim=config.expert_rope_dim,
            effective_dim=config.expert_rope_effective_dim,
            base=config.expert_rope_base,
            table_size=config.expert_rope_table_size,
        )
        self.action_encoder = ActionTauEncoder(config.action_dim, dim)
        self.timestep_embedder = TimestepEmbedder(dim)
        self.future_tokens = nn.Parameter(torch.zeros(config.num_future_per_chunk, dim))
        self.future_chunk_embed = nn.Embedding(config.max_chunks, dim)
        self.proprio_token = nn.Parameter(torch.zeros(1, dim))
        self.embed_proprio = nn.Linear(config.proprio_dim, dim)
        self.tap_proj = nn.Linear(config.expert_trunk_dim, dim, bias=False)
        self.text_proj = nn.Linear(config.expert_trunk_dim, dim, bias=False)
        self.tap_norm_scale = nn.Parameter(torch.ones(config.expert_trunk_dim))
        self.blocks = nn.ModuleList(
            ActionExpertBlock(
                dim=dim,
                num_heads=config.expert_heads,
                head_dim=config.expert_head_dim,
                is_cross=i % 2 == 0,
            )
            for i in range(config.expert_layers)
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, config.action_dim)

    def forward(
        self,
        *,
        taps: Sequence[Tensor],
        embeddings: Tensor,
        rope_ids: Tensor,
        num_global: int,
        num_clean_frames: int,
        num_noised_frames: int,
        tokens_per_frame: int,
        chunk_size: int,
        noised_actions: Tensor,
        action_timesteps: Tensor,
        proprio: Tensor,
    ) -> Tensor:
        """Predict the action velocity ``(1, T, action_dim)`` of a window of ``T`` action tokens.

        ``taps`` and ``embeddings`` are encoder outputs ``(1, S, C)`` over ``[global | noised | clean]``;
        ``noised_actions`` is ``(1, T, action_dim)``, ``action_timesteps`` ``(1, T)`` and ``proprio``
        ``(1, T, proprio_dim)``, with one action token per noised frame.
        """
        cfg = self.config
        t_lat = noised_actions.shape[1]
        device = noised_actions.device
        num_chunks = max(1, t_lat // chunk_size)
        kv_ids, taps_kv = self._window_kv(
            taps=taps,
            rope_ids=rope_ids,
            num_global=num_global,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
        )
        x = self._expert_tokens(
            noised_actions=noised_actions,
            action_timesteps=action_timesteps,
            proprio=proprio,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )
        query_chunks = expert_query_chunks(
            num_action_tokens=t_lat,
            num_future=cfg.num_future_per_chunk,
            chunk_size=chunk_size,
            device=device,
        )
        tok_frame = (query_chunks * chunk_size).clamp(min=0, max=t_lat - 1)
        cond = self.timestep_embedder(
            action_timesteps.gather(1, tok_frame[None].expand(1, -1))
        ).to(x.dtype)
        cross_mask = expert_cross_mask(
            query_chunks=query_chunks,
            chunk_size=chunk_size,
            num_global=num_global,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
            include_noised=not cfg.clean_only_conditioning,
        )
        self_mask = expert_self_mask(query_chunks=query_chunks)
        q_ids = expert_query_rope_ids(
            kv_rope_ids=kv_ids,
            num_global=num_global,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
            num_action_tokens=t_lat,
            num_future=cfg.num_future_per_chunk,
            chunk_size=chunk_size,
        )
        q_table, k_table = self.rope_table(q_ids), self.rope_table(kv_ids)
        text_kv = self.text_proj(embeddings[:, :num_global])
        cross_index = 0
        for block in self.blocks:
            if block.is_cross:
                feat = self._tap_features(taps_kv[cross_index])
                x = block(
                    x,
                    cond,
                    kv_source=torch.cat([text_kv.to(feat.dtype), feat], dim=1),
                    mask=cross_mask,
                    q_table=q_table,
                    k_table=k_table,
                )
                cross_index += 1
            else:
                x = block(
                    x,
                    cond,
                    kv_source=None,
                    mask=self_mask,
                    q_table=q_table,
                    k_table=q_table,
                )
        out = self.head(self.norm_out(x[:, -t_lat:]))
        return out + 0.0 if cfg.head_plus_zero else out

    def _tap_features(self, tap: Tensor) -> Tensor:
        return self.tap_proj((rms_norm(tap) * self.tap_norm_scale).to(tap.dtype))

    def _window_kv(
        self,
        *,
        taps: Sequence[Tensor],
        rope_ids: Tensor,
        num_global: int,
        num_clean_frames: int,
        num_noised_frames: int,
        tokens_per_frame: int,
    ) -> tuple[Tensor, list[Tensor]]:
        """Rope ids and per-tap frame tokens in the expert order ``[global | clean | noised]``."""
        n_noised = num_noised_frames * tokens_per_frame
        n_clean = num_clean_frames * tokens_per_frame

        def frames(t: Tensor) -> Tensor:
            noised = t[:, num_global : num_global + n_noised]
            clean = t[:, num_global + n_noised : num_global + n_noised + n_clean]
            return torch.cat([clean, noised], dim=1)

        kv_ids = torch.cat([rope_ids[:, :num_global], frames(rope_ids)], dim=1)
        return kv_ids, [frames(t) for t in taps]

    def _expert_tokens(
        self,
        *,
        noised_actions: Tensor,
        action_timesteps: Tensor,
        proprio: Tensor,
        num_chunks: int,
        chunk_size: int,
    ) -> Tensor:
        """The expert sequence ``[proprio | future | actions]``."""
        cfg = self.config
        act = self.action_encoder(noised_actions, action_timesteps)
        future = self.future_tokens[None, None].expand(1, num_chunks, -1, -1)
        future = future + self.future_chunk_embed.weight[:num_chunks][
            None, :, None, :
        ].to(future.dtype)
        future = future.reshape(
            1, num_chunks * cfg.num_future_per_chunk, cfg.expert_dim
        )
        index = torch.arange(num_chunks, device=act.device) * chunk_size
        prop = self.proprio_token[None].expand(1, num_chunks, -1).to(act.dtype)
        prop = prop + self.embed_proprio(proprio[:, index].to(prop.dtype))
        return torch.cat([prop, future.to(act.dtype), act], dim=1)
