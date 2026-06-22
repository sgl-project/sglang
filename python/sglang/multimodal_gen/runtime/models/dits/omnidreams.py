# SPDX-License-Identifier: Apache-2.0
"""OmniDreams DiT (Cosmos DiT) for SGLang.

Submodule names are checkpoint-exact: they mirror FlashDreams
``omnidreams.transformer.impl.modules`` / ``network`` so the flat ``.pt`` loads
with an identity parameter mapping.

The module is constructed with PRE-FUSION shapes to match the raw checkpoint:
- ``x_embedder`` keeps the always-zero inference padding-mask channel
  (in_features = 72 for the HDMap variant), and
- ``final_layer.linear`` keeps the Cosmos ``(kt kh kw c)`` patch-shuffle order.

Both fusions run once in :meth:`post_load_weights` after ``load_state_dict``,
matching FlashDreams ``update_parameters_after_loading_checkpoint``.

The denoising forward pass (RoPE, SDPA, KV-cache) is implemented in a later
phase; Phase-0 scaffolding only needs checkpoint-exact construction + fusion.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.runtime.distributed import divide, get_tp_world_size
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.omnidreams_kvcache import BlockKVCache
from sglang.multimodal_gen.runtime.models.dits.omnidreams_rope import apply_rope_freqs


def _sp_size() -> int:
    try:
        from sglang.multimodal_gen.runtime.distributed import get_sp_world_size

        return get_sp_world_size()
    except (ImportError, AssertionError, RuntimeError):
        return 1


# --------------------------------------------------------------------------- #
# Building blocks (checkpoint-exact module names)                             #
# --------------------------------------------------------------------------- #
class GPT2FeedForward(nn.Module):
    """GPT-2 style FFN with GELU (submodules: ``layer1``/``layer2``)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding (non-persistent buffer -> no ckpt key)."""

    SINUSOIDAL_FREQ_BASE = 10000
    emb: Tensor

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.reset_emb()

    def reset_emb(self, device: torch.device | str | None = None) -> None:
        """(Re)create the non-persistent ``emb`` frequency table.

        Carries no checkpoint key, so a meta-device construction (the production
        load path) leaves it empty; ``post_load_weights`` calls this to
        rematerialize it on the loaded param device. Kept in float32 regardless
        of model dtype, matching the original sinusoidal precision.
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(self.SINUSOIDAL_FREQ_BASE) * torch.arange(
            half_dim, dtype=torch.float32, device=device
        )
        exponent = exponent / half_dim
        self.register_buffer("emb", torch.exp(exponent), persistent=False)

    def forward(self, timesteps: Tensor) -> Tensor:
        emb = timesteps.unsqueeze(-1) * self.emb
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Timestep MLP with optional AdaLN-LoRA (submodules ``linear_1``/``linear_2``).

    When ``use_adaln_lora`` is True, ``linear_1`` has no bias and forward returns
    ``(raw_sinusoidal_input, lora_out)``.
    """

    def __init__(
        self, in_features: int, out_features: int, use_adaln_lora: bool = True
    ) -> None:
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        out_dim = 3 * out_features if use_adaln_lora else out_features
        self.linear_2 = nn.Linear(out_features, out_dim, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        # Timesteps emits float32 sinusoids for precision; cast to the MLP
        # param dtype before the linears (and before the raw embedding is
        # returned for RMSNorm / AdaLN), matching the model's running dtype.
        x = x.to(self.linear_1.weight.dtype)
        out = self.linear_2(self.activation(self.linear_1(x)))
        if self.use_adaln_lora:
            return x, out
        return out, None


class PatchEmbed(nn.Module):
    """Patch embed: ``proj = Sequential(Identity, Linear(in_features, out))``.

    The leading ``Identity`` is a placeholder kept for checkpoint key
    compatibility (the learnable linear lives at ``proj.1``).
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.proj = nn.Sequential(
            nn.Identity(),
            nn.Linear(self._compute_in_features(), out_channels, bias=False),
        )

    def _compute_in_features(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.spatial_patch_size**2

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class FinalLayer(nn.Module):
    """Final AdaLN layer (submodules ``layer_norm``/``linear``/``adaln_modulation``)."""

    NUM_ADALN_CHUNKS = 2

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora
        self.layer_norm = LayerNormScaleShift(hidden_size, eps=1e-6)
        patch_dim = spatial_patch_size**2 * temporal_patch_size * out_channels
        self.linear = nn.Linear(hidden_size, patch_dim, bias=False)
        modulation_out_dim = self.NUM_ADALN_CHUNKS * hidden_size
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, modulation_out_dim, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, modulation_out_dim, bias=False),
            )

    def forward(
        self, x: Tensor, emb: Tensor, adaln_lora: Tensor | None = None
    ) -> Tensor:
        """Final AdaLN + projection. ``x``: [B, L, D], ``emb``: [B, D]."""
        B, L, D = x.shape
        emb_ = emb.reshape(B, 1, D)
        if self.use_adaln_lora:
            assert adaln_lora is not None
            al = adaln_lora.reshape(B, 1, 3 * D)
            modulation = self.adaln_modulation(emb_) + al[..., : 2 * self.hidden_size]
            shift, scale = modulation.chunk(2, dim=-1)
        else:
            shift, scale = self.adaln_modulation(emb_).chunk(2, dim=-1)
        x = self.layer_norm(x, shift, scale)
        return self.linear(x)


class OmniDreamsAttention(nn.Module):
    """Multi-head attention block (self or cross), with optional TP support.

    Submodules: ``q_proj``/``k_proj``/``v_proj``/``output_proj`` (all bias-free)
    and per-head ``q_norm``/``k_norm`` RMSNorms. The attention op itself carries
    no parameters, so it is intentionally not a registered submodule (keeps the
    checkpoint key set identical to FlashDreams).

    When TP is active, ``q_proj``/``k_proj``/``v_proj`` use column-parallel
    projection (each rank sees ``local_num_heads`` heads) and ``output_proj``
    uses row-parallel projection (all-reduces across ranks). The per-head
    RMSNorms already operate on individual heads so they are naturally TP-local.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None,
        n_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.is_self_attn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.local_num_heads = divide(n_heads, get_tp_world_size())

        inner_dim_full = head_dim * n_heads
        if self.is_self_attn:
            # Self-attn: q/k/v all project from x -> one packed GEMM (q, k, v order).
            self.to_qkv = MergedColumnParallelLinear(
                query_dim, [inner_dim_full] * 3, bias=False, gather_output=False
            )
        else:
            # Cross-attn: q from x, k/v from the (text) context -> pack k/v.
            self.q_proj = ColumnParallelLinear(
                query_dim, inner_dim_full, bias=False, gather_output=False
            )
            self.to_kv = MergedColumnParallelLinear(
                context_dim, [inner_dim_full] * 2, bias=False, gather_output=False
            )
        self.output_proj = RowParallelLinear(
            inner_dim_full,
            query_dim,
            bias=False,
            reduce_results=True,
            input_is_parallel=True,
        )
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def _project_qkv(self, x: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V from the packed GEMM(s) (each linear returns ``(out, bias)``)."""
        if self.is_self_attn:
            qkv, _ = self.to_qkv(x)
            q_raw, k_raw, v_raw = (t.contiguous() for t in qkv.chunk(3, dim=-1))
        else:
            q_raw, _ = self.q_proj(x)
            kv, _ = self.to_kv(ctx)
            k_raw, v_raw = (t.contiguous() for t in kv.chunk(2, dim=-1))
        return q_raw, k_raw, v_raw

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        kv_cache=None,
        cross_kv: tuple[Tensor, Tensor] | None = None,
        rope_cos_sin: Tensor | None = None,
    ) -> Tensor:
        """Attention with an optional KV-cache window (autoregressive self-attn)
        and optional precomputed cross-attention K/V (Phase 6 caching).

        Q/K are per-head RMSNorm'd, then (self-attn only) rotated by RoPE before
        a full bidirectional SDPA (scale 1/sqrt(head_dim)). Cross-attention passes
        ``context`` as K/V source and no RoPE.

        ``rope_cos_sin`` is the ``[L, D]`` cos|sin cache (first D/2 cos, second
        D/2 sin) from :meth:`RotaryPositionEmbedding3D.shift_t`; self-attention
        applies NeoX 3D RoPE from it (cross-attention uses no RoPE).

        When ``kv_cache`` (a :class:`BlockKVCache`) is given, this is an AR
        self-attention step: the current chunk's post-RoPE K and V are written
        into the cache (``seq_dim=1``) and Q attends over the full cached window
        ``[sink | window]``. The cache's ``before_update``/``after_update`` (window
        roll + bookkeeping) are driven by the denoising stage, not here -- here we
        only ``update`` (write) and read ``cached_k``/``cached_v``. Cross-chunk
        causality comes solely from the window (no causal mask); each chunk's K is
        rotated by its absolute position via ``shift_t`` before being cached, so
        cached K and current Q keep the correct relative rotation.

        When ``cross_kv`` (a ``(K,V)`` tuple) is given, the K/V projections and
        K-norm are skipped for the cross-attention path and the precomputed
        tensors are used directly. The cached K already has ``k_norm`` applied.
        This avoids redundant projection of the same text context in every AR
        forward (28 blocks × num_chunks × 3 calls/chunk). ``context`` is still
        required for the Q projection from ``x``.
        """
        ctx = x if context is None else context
        B, L, _ = x.shape
        n, d = self.local_num_heads, self.head_dim

        if cross_kv is not None:
            # Precomputed cross-attn K/V — skip k_proj/v_proj/k_norm.
            # Q still needs projection since it depends on the current x.
            q_raw, _ = self.q_proj(x)
            q = self.q_norm(q_raw.reshape(B, L, n, d))
            k, v = cross_kv
        else:
            q_raw, k_raw, v_raw = self._project_qkv(x, ctx)
            Lk = ctx.shape[1]
            q = self.q_norm(q_raw.reshape(B, L, n, d))
            k = self.k_norm(k_raw.reshape(B, Lk, n, d))
            v = v_raw.reshape(B, Lk, n, d)

        if rope_cos_sin is not None:
            q = apply_rope_freqs(q, rope_cos_sin)
            k = apply_rope_freqs(k, rope_cos_sin)
        if kv_cache is not None:
            # Write this chunk's (post-RoPE) K/V, then attend over the window.
            kv_cache.update(k, v)
            k = kv_cache.cached_k()
            v = kv_cache.cached_v()
        # SDPA expects [B, n, S, d]; default scale = 1/sqrt(d), no mask.
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        out = out.transpose(1, 2).reshape(B, L, n * d)
        out, _ = self.output_proj(out)
        return out


class OmniDreamsBlock(nn.Module):
    """Cosmos transformer block: self-attn -> (cross-view-attn) -> cross-attn -> MLP.

    Cross-view attention is gated behind ``enable_cross_view_attn`` (Phase 5,
    default off for the single-view checkpoint). When enabled each camera
    view's tokens attend over all views at the same temporal position via a
    dense bidirectional attention (no RoPE, no causal mask), and per-view
    AdaLN modulation terms are added to the timestep-conditioned shift/scale/
    gate biases.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        enable_cross_view_attn: bool = False,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.use_adaln_lora = use_adaln_lora
        self.enable_cross_view_attn = enable_cross_view_attn
        head_dim = x_dim // num_heads

        self.layer_norm_self_attn = LayerNormScaleShift(x_dim, eps=1e-6)
        self.self_attn = OmniDreamsAttention(x_dim, None, num_heads, head_dim)

        # Cross-view attention (Phase 5) — learnable LayerNorm (unlike AdaLN).
        if enable_cross_view_attn:
            self.layer_norm_cross_view_attn = nn.LayerNorm(
                x_dim, elementwise_affine=True, eps=1e-6
            )
            self.cross_view_attn = OmniDreamsAttention(
                x_dim, x_dim, num_heads, head_dim
            )

        self.layer_norm_cross_attn = LayerNormScaleShift(x_dim, eps=1e-6)
        self.cross_attn = OmniDreamsAttention(x_dim, context_dim, num_heads, head_dim)

        self.layer_norm_mlp = LayerNormScaleShift(x_dim, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        def _make_adaln_mod() -> nn.Sequential:
            if use_adaln_lora:
                return nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(x_dim, adaln_lora_dim, bias=False),
                    nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
                )
            return nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.adaln_modulation_self_attn = _make_adaln_mod()
        self.adaln_modulation_cross_attn = _make_adaln_mod()
        self.adaln_modulation_mlp = _make_adaln_mod()

    @staticmethod
    def _expand_view_mod(view_tensor: Tensor, B: int, V: int, D: int) -> Tensor:
        """Expand per-view modulation ``[B, V, D]`` into ``[B, V, 1, D]``."""
        return view_tensor.reshape(B, V, 1, D)

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        adaln_lora: Tensor | None,
        context: Tensor,
        self_attn_kv_cache=None,
        cross_attn_kv: tuple[Tensor, Tensor] | None = None,
        view_embedding_proj: Tensor | None = None,
        rope_cos_sin: Tensor | None = None,
    ) -> Tensor:
        """One transformer block on a single chunk.

        Args:
            x: ``[B, L, D]`` tokens (flat across views & frames).
            emb: ``[B, D]`` timestep embedding.
            adaln_lora: ``[B, 3D]`` AdaLN-LoRA term.
            context: ``[B, Lctx, D]`` cross-attention key/value source.
            self_attn_kv_cache: optional :class:`BlockKVCache`.
            cross_attn_kv: optional precomputed ``(K,V)`` tuple for cross-attn.
            view_embedding_proj: optional ``[B, V, 9D]`` view modulation tensor
                (Phase 5 cross-view attention). ``None`` for single-view.
        """
        B, L, D = x.shape
        emb_ = emb.reshape(B, 1, D)

        if self.use_adaln_lora:
            assert adaln_lora is not None
            al = adaln_lora.reshape(B, 1, 3 * D)
            shift_s, scale_s, gate_s = (
                self.adaln_modulation_self_attn(emb_) + al
            ).chunk(3, dim=-1)
            shift_c, scale_c, gate_c = (
                self.adaln_modulation_cross_attn(emb_) + al
            ).chunk(3, dim=-1)
            shift_m, scale_m, gate_m = (self.adaln_modulation_mlp(emb_) + al).chunk(
                3, dim=-1
            )
        else:
            shift_s, scale_s, gate_s = self.adaln_modulation_self_attn(emb_).chunk(
                3, dim=-1
            )
            shift_c, scale_c, gate_c = self.adaln_modulation_cross_attn(emb_).chunk(
                3, dim=-1
            )
            shift_m, scale_m, gate_m = self.adaln_modulation_mlp(emb_).chunk(3, dim=-1)

        # Cross-view modulation (Phase 5): additive per-view AdaLN bias.
        if self.enable_cross_view_attn and view_embedding_proj is not None:
            V = view_embedding_proj.shape[1]
            (
                view_shift_s,
                view_scale_s,
                view_gate_s,
                view_shift_c,
                view_scale_c,
                view_gate_c,
                view_shift_m,
                view_scale_m,
                view_gate_m,
            ) = view_embedding_proj.chunk(9, dim=-1)
            shift_s = shift_s + self._expand_view_mod(view_shift_s, B, V, D)
            scale_s = scale_s + self._expand_view_mod(view_scale_s, B, V, D)
            gate_s = gate_s + self._expand_view_mod(view_gate_s, B, V, D)
            shift_c = shift_c + self._expand_view_mod(view_shift_c, B, V, D)
            scale_c = scale_c + self._expand_view_mod(view_scale_c, B, V, D)
            gate_c = gate_c + self._expand_view_mod(view_gate_c, B, V, D)
            shift_m = shift_m + self._expand_view_mod(view_shift_m, B, V, D)
            scale_m = scale_m + self._expand_view_mod(view_scale_m, B, V, D)
            gate_m = gate_m + self._expand_view_mod(view_gate_m, B, V, D)

        normed = self.layer_norm_self_attn(x, shift_s, scale_s)
        x = x + gate_s * self.self_attn(
            normed,
            kv_cache=self_attn_kv_cache,
            rope_cos_sin=rope_cos_sin,
        )

        # Cross-view attention (Phase 5): each view attends over all views at
        # the same temporal position (dense bidirectional, no RoPE, no gate).
        if self.enable_cross_view_attn and view_embedding_proj is not None:
            x_cv = self._cross_view_attn_forward(x, L, B, D)
            x = x + x_cv

        normed = self.layer_norm_cross_attn(x, shift_c, scale_c)
        x = x + gate_c * self.cross_attn(
            normed, context=context, cross_kv=cross_attn_kv
        )

        normed = self.layer_norm_mlp(x, shift_m, scale_m)
        x = x + gate_m * self.mlp(normed)
        return x

    def _cross_view_attn_forward(self, x: Tensor, L: int, B: int, D: int) -> Tensor:
        """Cross-view attention (Phase 5) — not yet implemented.

        The intended behavior reshapes the flat ``[B, L, D]`` tokens into
        ``[B, V, T*HW, D]`` and, for each temporal position ``t``, lets every
        view's queries attend over the concatenated K/V of all views at that
        same ``t`` (no RoPE, no causal mask). The view count ``V`` must be
        threaded in from the caller, which is not wired up yet, so we fail
        loudly instead of silently running global attention over all tokens.
        """
        raise NotImplementedError(
            "Cross-view attention (enable_cross_view_attn=True) is not yet "
            "supported: temporal-position-restricted attention is unimplemented. "
            "Run with the default enable_cross_view_attn=False."
        )


# --------------------------------------------------------------------------- #
# Top-level DiT                                                               #
# --------------------------------------------------------------------------- #
class OmniDreamsDiT(BaseDiT):
    """OmniDreams Cosmos DiT (2.06B, DiT-only, autoregressive video world model).

    Supports TP (tensor parallelism via ``ColumnParallelLinear``/``RowParallelLinear``
    head sharding), SP (sequence parallelism — guarded: SP init is detected and
    rejected with a clear error since the autoregressive chunk loop is not yet
    SP-aware), and optional cross-view attention (Phase 5, gated by config
    ``enable_cross_view_attn``).
    """

    _fsdp_shard_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]
    _compile_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]
    # Route the flat checkpoint's separate q/k/v projections into the packed
    # MergedColumnParallelLinear params: self-attn q/k/v -> to_qkv (shards 0,1,2 of
    # 3); cross-attn k/v -> to_kv (shards 0,1 of 2). Cross-attn q_proj and both
    # output_proj keep their names (no merge).
    param_names_mapping: dict = {
        r"^(blocks\.\d+\.self_attn)\.q_proj\.(.*)$": (r"\1.to_qkv.\2", 0, 3),
        r"^(blocks\.\d+\.self_attn)\.k_proj\.(.*)$": (r"\1.to_qkv.\2", 1, 3),
        r"^(blocks\.\d+\.self_attn)\.v_proj\.(.*)$": (r"\1.to_qkv.\2", 2, 3),
        r"^(blocks\.\d+\.cross_attn)\.k_proj\.(.*)$": (r"\1.to_kv.\2", 0, 2),
        r"^(blocks\.\d+\.cross_attn)\.v_proj\.(.*)$": (r"\1.to_kv.\2", 1, 2),
    }
    reverse_param_names_mapping: dict = {}

    def __init__(
        self, config: DiTConfig, hf_config: dict | None = None, **kwargs
    ) -> None:
        super().__init__(config, hf_config or {}, **kwargs)
        arch = config.arch_config
        self.arch = arch

        # +1 for the per-frame condition mask, +1 for the (training) padding mask.
        in_channels = arch.in_channels + 1
        if arch.concat_padding_mask:
            in_channels += 1

        self.x_embedder = PatchEmbed(
            arch.patch_spatial, arch.patch_temporal, in_channels, arch.model_channels
        )
        if arch.additional_concat_ch > 0:
            self.additional_patch_embedding = PatchEmbed(
                arch.patch_spatial,
                arch.patch_temporal,
                arch.additional_concat_ch,
                arch.model_channels,
            )

        self.t_embedder = nn.Sequential(
            Timesteps(arch.model_channels),
            TimestepEmbedding(
                arch.model_channels,
                arch.model_channels,
                use_adaln_lora=arch.use_adaln_lora,
            ),
        )
        self.t_embedding_norm = nn.RMSNorm(arch.model_channels, eps=1e-6)

        # Phase 5: cross-view attention (default off for single-view checkpoint).
        _cv_enabled = getattr(arch, "enable_cross_view_attn", False)
        self.blocks = nn.ModuleList(
            [
                OmniDreamsBlock(
                    x_dim=arch.model_channels,
                    context_dim=arch.crossattn_emb_channels,
                    num_heads=arch.num_heads,
                    mlp_ratio=arch.mlp_ratio,
                    use_adaln_lora=arch.use_adaln_lora,
                    adaln_lora_dim=arch.adaln_lora_dim,
                    enable_cross_view_attn=_cv_enabled,
                )
                for _ in range(arch.num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=arch.model_channels,
            spatial_patch_size=arch.patch_spatial,
            temporal_patch_size=arch.patch_temporal,
            out_channels=arch.out_channels,
            use_adaln_lora=arch.use_adaln_lora,
            adaln_lora_dim=arch.adaln_lora_dim,
        )

        if arch.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(
                    arch.crossattn_proj_in_channels,
                    arch.crossattn_emb_channels,
                    bias=True,
                ),
                nn.GELU(),
            )

        # Cross-view attention (Phase 5): embedder + projection network.
        # Stays None for the single-view checkpoint (no params, no forward cost).
        if _cv_enabled:
            n_cameras = getattr(arch, "n_cameras_emb", 7)
            self.adaln_view_embedder = nn.Embedding(n_cameras, arch.model_channels)
            self.adaln_view_proj = nn.Linear(
                arch.model_channels, arch.model_channels * 9
            )
        else:
            self.adaln_view_embedder = None
            self.adaln_view_proj = None

        self._is_shuffle_op_fused = False
        self._is_padding_mask_fused = False

        # Phase 6 SP guard: store sp_size for forward-time detection.
        self._sp_size = _sp_size()

        # BaseDiT-required instance attributes.
        self.hidden_size = arch.model_channels
        self.num_attention_heads = arch.num_heads
        self.num_channels_latents = arch.out_channels
        self.__post_init__()

    # ----- load-time weight fusions (mirror FlashDreams) ------------------- #
    def _fuse_padding_mask_into_patch_embed(self) -> None:
        """Drop the always-zero inference padding-mask channels (72 -> 68)."""
        if not self.arch.concat_padding_mask or self._is_padding_mask_fused:
            return
        self.x_embedder.in_channels -= 1
        in_channels_to_keep = self.x_embedder._compute_in_features()
        proj_linear = self.x_embedder.proj[1]
        proj_linear.weight.data = proj_linear.weight.data[
            :, :in_channels_to_keep
        ].contiguous()
        if proj_linear.bias is not None:
            proj_linear.bias.data = proj_linear.bias.data[
                :in_channels_to_keep
            ].contiguous()
        self._is_padding_mask_fused = True

    def _fuse_shuffle_op_into_last_layer(self) -> None:
        """Fold the Cosmos ``(kt kh kw c) -> (c kt kh kw)`` shuffle into the last linear."""
        if self._is_shuffle_op_fused:
            return
        self.final_layer.linear.weight.data = rearrange(
            self.final_layer.linear.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.arch.patch_temporal,
            kh=self.arch.patch_spatial,
            kw=self.arch.patch_spatial,
            c=self.arch.out_channels,
        ).contiguous()
        if self.final_layer.linear.bias is not None:
            self.final_layer.linear.bias.data = rearrange(
                self.final_layer.linear.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.arch.patch_temporal,
                kh=self.arch.patch_spatial,
                kw=self.arch.patch_spatial,
                c=self.arch.out_channels,
            ).contiguous()
        self._is_shuffle_op_fused = True

    def post_load_weights(self) -> None:
        self._fuse_padding_mask_into_patch_embed()
        self._fuse_shuffle_op_into_last_layer()
        self._materialize_nonpersistent_buffers()

    def _materialize_nonpersistent_buffers(self) -> None:
        """Rematerialize non-persistent buffers left empty by meta construction.

        Buffers registered with ``persistent=False`` (e.g. the sinusoidal
        ``Timesteps.emb`` table) have no checkpoint key, so the state-dict
        loader never fills them and they stay on the meta device after a
        meta-init load. Recompute them on the loaded parameter device.
        """
        device = self.x_embedder.proj[1].weight.device
        for module in self.modules():
            if isinstance(module, Timesteps):
                module.reset_emb(device=device)

    def init_kv_caches(
        self,
        batch_size: int,
        chunk_tokens: int,
        window_tokens: int,
        sink_tokens: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BlockKVCache]:
        """Build one :class:`BlockKVCache` per transformer block for AR rollout.

        Token counts are in *patchified tokens*, i.e. ``frames * Hp * Wp``:
        ``chunk_tokens`` = ``len_t`` latent frames, ``window_tokens`` =
        ``window_size_t`` frames, ``sink_tokens`` = ``sink_size_t`` frames (all
        already multiplied by the per-frame spatial token count). K/V are stored
        with ``seq_dim=1`` and shape ``[B, sink+window, local_heads, head_dim]``
        (TP-sharded heads when tensor parallelism is active).
        """
        n = divide(self.arch.num_heads, get_tp_world_size())
        d = self.arch.model_channels // self.arch.num_heads
        total = sink_tokens + window_tokens
        shape = (batch_size, total, n, d)
        return [
            BlockKVCache(
                k_shape=shape,
                v_shape=shape,
                seq_dim=1,
                chunk_size=chunk_tokens,
                window_size=window_tokens,
                sink_size=sink_tokens,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.arch.num_blocks)
        ]

    @torch.no_grad()
    def precompute_cross_attn_kv(
        self, context: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Precompute cross-attention K/V for every block once per prompt.

        Returns a list indexed by block, each ``(K,V)`` with per-head
        RMSNorm applied to K. When TP is active, each rank sees only its
        local heads (``local_num_heads``). This is used by the AR denoising
        stage to avoid redundant ``k_proj(ctx)``/``v_proj(ctx)`` in every
        forward call (28 blocks × num_chunks × 3 calls/chunk of wasted matmul).

        Phase 6. Invoke once before the chunk loop; pass to
        :meth:`forward` as ``cross_attn_kv=result``.
        """
        result: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            attn = block.cross_attn
            n, d = attn.local_num_heads, attn.head_dim
            kv, _ = attn.to_kv(context)
            k_raw, v_raw = (t.contiguous() for t in kv.chunk(2, dim=-1))
            k = attn.k_norm(k_raw.reshape(context.shape[0], context.shape[1], n, d))
            v = v_raw.reshape(context.shape[0], context.shape[1], n, d)
            result.append((k, v))
        return result

    def patchify(self, video: Tensor) -> Tensor:
        """[B, C, T, H, W] -> [B, T*Hp*Wp, C*kt*kh*kw] (channel-major packing)."""
        return rearrange(
            video,
            "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
            kt=self.arch.patch_temporal,
            kh=self.arch.patch_spatial,
            kw=self.arch.patch_spatial,
        )

    def unpatchify(
        self, tokens: Tensor, grid_t: int, grid_h: int, grid_w: int
    ) -> Tensor:
        """[B, L, out*kt*kh*kw] -> [B, out, T, H, W].

        Uses the simple ``(c kt kh kw)`` unpack because the Cosmos channel
        shuffle is already folded into ``final_layer.linear`` by
        :meth:`post_load_weights`.
        """
        return rearrange(
            tokens,
            "b (t h w) (c kt kh kw) -> b c (t kt) (h kh) (w kw)",
            t=grid_t,
            h=grid_h,
            w=grid_w,
            kt=self.arch.patch_temporal,
            kh=self.arch.patch_spatial,
            kw=self.arch.patch_spatial,
            c=self.arch.out_channels,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        *,
        condition_video_input_mask: torch.Tensor,
        rope_cos_sin: torch.Tensor,
        hdmap_condition: torch.Tensor | None = None,
        kv_caches: list | None = None,
        cross_attn_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        view_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Denoising forward (single-chunk, or one autoregressive chunk).

        Args:
            hidden_states: patchified latent tokens ``[B, L, in_channels*kt*kh*kw]``.
            encoder_hidden_states: text context ``[B, Lctx, crossattn_proj_in_channels]``
                (projected to ``crossattn_emb_channels`` by ``crossattn_proj``).
            timestep: scalar (warped) timestep; scaled by ``timestep_scale`` here.
            condition_video_input_mask: patchified per-frame condition mask
                ``[B, L, kt*kh*kw]`` (concatenated onto the latent channels).
            rope_cos_sin: ``[L, D]`` cos|sin RoPE cache for self-attention. For AR
                rollout pass ``shift_t(ar_idx)`` so the chunk is rotated by its
                absolute temporal position.
            hdmap_condition: patchified HDMap ``[B, L, additional_concat_ch*kt*kh*kw]``.
            kv_caches: optional per-block list of :class:`BlockKVCache` (length
                ``num_blocks``) enabling the autoregressive self-attention window.
                ``None`` runs the plain single-chunk path.
            cross_attn_kv: optional per-block list of precomputed ``(K,V)`` tuples
                (Phase 6 caching). When given, bypasses ``k_proj``/``v_proj``/
                ``k_norm`` in the cross-attention path.
            view_indices: optional ``[B, V]`` long tensor of camera view indices
                (Phase 5 cross-view attention). ``None`` for single-view mode.

        Returns:
            Patchified flow prediction ``[B, L, out_channels*kt*kh*kw]``.
            Call :meth:`unpatchify` to recover ``[B, out, T, H, W]``.
        """
        assert self._is_padding_mask_fused and self._is_shuffle_op_fused, (
            "call post_load_weights() before forward (fuses padding-mask + "
            "last-layer shuffle)"
        )
        # A single chunk's forward is compile-safe under fullgraph=False: the
        # only dynamic ops here are the per-block KV read/write, which graph-break
        # cleanly. torch.compile must use max-autotune-no-cudagraphs so inductor
        # does not install CUDA graphs that collide with OmniDreamsCUDAGraphRunner.
        if self._sp_size > 1:
            raise RuntimeError(
                "Sequence parallelism (SP) is not yet supported for OmniDreams. "
                "Run with --ulysses-degree 1 --ring-degree 1."
            )

        timestep = timestep * self.arch.timestep_scale

        x = torch.cat([hidden_states, condition_video_input_mask], dim=-1)
        x = self.x_embedder(x)
        if self.arch.additional_concat_ch > 0:
            assert hdmap_condition is not None, "HDMap variant requires hdmap_condition"
            x = x + self.additional_patch_embedding(hdmap_condition)

        t_emb, adaln_lora = self.t_embedder(timestep)
        t_emb = self.t_embedding_norm(t_emb)
        batch = x.shape[0]
        t_emb = t_emb.reshape(1, -1).expand(batch, -1)
        if adaln_lora is not None:
            adaln_lora = adaln_lora.reshape(1, -1).expand(batch, -1)

        # Phase 5: compute cross-view modulation once per forward.
        view_embedding_proj: Tensor | None = None
        if view_indices is not None and self.adaln_view_proj is not None:
            view_emb = self.adaln_view_embedder(view_indices)  # [B, V, D]
            view_embedding_proj = self.adaln_view_proj(view_emb)  # [B, V, 9D]

        context = self.crossattn_proj(encoder_hidden_states)
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                t_emb,
                adaln_lora,
                context,
                self_attn_kv_cache=None if kv_caches is None else kv_caches[i],
                cross_attn_kv=None if cross_attn_kv is None else cross_attn_kv[i],
                view_embedding_proj=view_embedding_proj,
                rope_cos_sin=rope_cos_sin,
            )
        return self.final_layer(x, t_emb, adaln_lora)


EntryClass = OmniDreamsDiT
