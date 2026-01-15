# Copied and adapted from LTX-2 and WanVideo implementations.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.ltx2_video import LTX2VideoConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_rank, get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.pipelines_core.perturbations import (
    BatchedPerturbationConfig,
    PerturbationType,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, ScaleResidual
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
    get_1d_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# Config-layer enums (kept out of runtime types to avoid torch/tensor deps in configs).
from sglang.multimodal_gen.configs.models.dits.ltx2_video import (  # noqa: E402
    LTX2AttentionFunction,
    LTX2RopeType,
    LTXModelType,
)

logger = init_logger(__name__)


# ==============================================================================
# Lightweight runtime containers (upstream-style)
# ==============================================================================


class LTX2RotaryEmbedding(NDRotaryEmbedding):
    """
    LTX-2 specific RotaryEmbedding that includes helper methods for coordinate preparation.
    Matches the interface expected by `denoising_av.py`.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 24.0,
    ) -> torch.Tensor:
        """
        Prepare video coordinates [B, 3, N, 2] where the last dim is [start, end).
        Used by `denoising_av.py`.
        """
        p_t, p_h, p_w = self.patch_size

        # Validate divisibility
        if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
             # Just a warning or strict check? ltx_2.py prepare() checks strict divisibility.
             # We assume caller passes valid latent dims.
             pass

        post_t = num_frames // p_t
        post_h = height // p_h
        post_w = width // p_w

        # Generate grid coordinates (starts)
        # indexing="ij" -> (t, h, w) order
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=post_t * p_t, step=p_t, device=device),
            torch.arange(start=0, end=post_h * p_h, step=p_h, device=device),
            torch.arange(start=0, end=post_w * p_w, step=p_w, device=device),
            indexing="ij",
        )
        
        # [3, num_tokens]
        patch_starts = (
            torch.stack(grid_coords, dim=0).reshape(3, -1).to(dtype=torch.float32)
        )
        
        # [3, 1]
        patch_size_delta = torch.tensor(
            (p_t, p_h, p_w), device=device, dtype=torch.float32
        ).view(3, 1)
        
        patch_ends = patch_starts + patch_size_delta
        
        # [3, num_tokens, 2] -> [B, 3, num_tokens, 2]
        coords = torch.stack([patch_starts, patch_ends], dim=-1)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prepare audio coordinates [B, 1, N, 2] (temporal only).
        Used by `denoising_av.py`.
        """
        # Audio usually has patch_size=(1, 1, 1) in current LTX setup, or at least 1D.
        # We use the first dim of patch_size as temporal patch size.
        p_t = self.patch_size[0]
        
        # Generate temporal starts
        # num_frames here is latent temporal dimension
        starts = torch.arange(
            start=0, end=num_frames, step=p_t, device=device, dtype=torch.float32
        )
        ends = starts + float(p_t)
        
        # [num_tokens, 2]
        coords_t = torch.stack([starts, ends], dim=-1)
        
        # [B, 1, num_tokens, 2]
        coords = coords_t.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return coords


@dataclass
class LTX2TransformerConfig:
    """
    Upstream-equivalent of `TransformerConfig` used by LTX-2 `BasicAVTransformerBlock`.

    Corresponds to:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
      - `@dataclass class TransformerConfig`
    """

    dim: int
    heads: int
    d_head: int
    context_dim: int


@dataclass
class LTX2TransformerArgs:
    """
    Upstream-style args container for `BasicAVTransformerBlock.forward`.

    Corresponds to:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
      - `TransformerArgs`

    Notes:
      - `positional_embeddings` / `cross_positional_embeddings` are represented as SGLang RoPE tuples
        `(cos, sin)` by this file / `LTX2Attention`.
      - `cross_scale_shift_timestep` / `cross_gate_timestep` are only needed for audio-video cross-attn.
    """

    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    cross_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool


# ==============================================================================
# Protocols for Preprocessors
# ==============================================================================


class AdaLNCallable(Protocol):
    def __call__(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class TPLinearCallable(Protocol):
    """
    Minimal protocol for SGLang TP linear layers used in LTX-2 (e.g. ColumnParallelLinear).

    In SGLang, TP linears typically return `(out, bias)`; we only use the tensor output.
    """

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...


class CaptionProjectionCallable(Protocol):
    def __call__(self, caption: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class LTX2PatchInfo:
    """
    Small container used by the video args preprocessor to unpatchify output tokens back to
    raw video latents.
    """

    bsz: int
    num_frames: int
    height: int
    width: int
    post_t: int
    post_h: int
    post_w: int
    patch_size: tuple[int, int, int]  # (p_t, p_h, p_w)


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # LTX-2 uses RMSNorm (no affine) heavily. We use functional rms_norm to keep it lightweight.
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)


class LTX2TextProjection(nn.Module):
    """
    LTX-2 caption projection (PixArtAlphaTextProjection-style).

    Corresponds to LTX-2:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/text_projection.py`
      - `PixArtAlphaTextProjection`
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
    ) -> None:
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = ColumnParallelLinear(
            in_features, hidden_size, bias=True, gather_output=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = ColumnParallelLinear(
            hidden_size, out_features, bias=True, gather_output=True
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LTX2TimestepEmbedder(nn.Module):
    """
    Diffusers-style TimestepEmbedding with `linear_1` / `linear_2` names (matches checkpoint).

    Corresponds to LTX-2:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`
      - `TimestepEmbedding` (the `linear_1` / `linear_2` MLP)

    Note: in LTX-2 this module is used under:
      - `PixArtAlphaCombinedTimestepSizeEmbeddings.timestep_embedder`
      - which is wrapped by `AdaLayerNormSingle.emb`
        (`LTX-2/packages/ltx-core/src/ltx_core/model/transformer/adaln.py`).
    """

    def __init__(self, embedding_dim: int, in_channels: int = 256) -> None:
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            in_channels, embedding_dim, bias=True, gather_output=True
        )
        self.linear_2 = ColumnParallelLinear(
            embedding_dim, embedding_dim, bias=True, gather_output=True
        )

    def forward(
        self,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self.linear_1(t_emb)
        x = F.silu(x)
        x, _ = self.linear_2(x)
        return x


class LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    Minimal LTX-2 `PixArtAlphaCombinedTimestepSizeEmbeddings`-like wrapper.

    Corresponds to LTX-2:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`
      - `PixArtAlphaCombinedTimestepSizeEmbeddings` (we only implement the timestep path)

    We keep `.timestep_embedder` as a named submodule so HF checkpoint keys like
    `time_embed.emb.timestep_embedder.linear_{1,2}.*` can be loaded.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        # Original LTX-2 uses Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # + TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim).
        # Here we directly generate the 256-dim sin/cos embedding via `timestep_embedding(...)`.
        self.timestep_embedder = LTX2TimestepEmbedder(embedding_dim, in_channels=256)

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # Match the original: embed scalar timesteps into a 256-dim sin/cos vector, then MLP.
        t = timestep.reshape(-1).to(dtype=torch.float32)
        t_emb = timestep_embedding(t, dim=256, max_period=10000, dtype=torch.float32)
        if hidden_dtype is not None:
            t_emb = t_emb.to(dtype=hidden_dtype)
        return self.timestep_embedder(t_emb)


class LTX2AdaLayerNormSingle(nn.Module):
    """
    LTX-2 `time_embed` module (AdaLayerNormSingle-style), matching HF checkpoint keys:
      - time_embed.emb.timestep_embedder.linear_1.{weight,bias}
      - time_embed.emb.timestep_embedder.linear_2.{weight,bias}
      - time_embed.linear.{weight,bias}

    Corresponds to LTX-2:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/adaln.py`
      - `AdaLayerNormSingle`

    Related dependencies in LTX-2:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`
      - `PixArtAlphaCombinedTimestepSizeEmbeddings` (contains `.timestep_embedder`)
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6) -> None:
        super().__init__()
        # Match LTX-2 structure: `AdaLayerNormSingle.emb` is a callable wrapper that owns
        # `.timestep_embedder` (and in the original also a `time_proj`).
        self.emb = LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = ColumnParallelLinear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
            gather_output=True,
        )

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype).to(
            dtype=self.linear.weight.dtype
        )
        out, _ = self.linear(self.silu(embedded_timestep))
        return out, embedded_timestep


class LTX2TPRMSNormAcrossHeads(nn.Module):
    """
    Implements "rms_norm_across_heads" (RMSNorm over full hidden size) in a TP-friendly way.

    - Input is sharded on the last dim: [*, local_hidden_size]
    - We compute global RMS via all-reduce of sum(x^2) across TP ranks
    - Weight is sharded to match the local slice and is loaded via a custom weight_loader

    Source mapping:
      - LTX-2 does NOT have this exact TP-sharded implementation.
      - This module is a TP-friendly adaptation of LTX-2's q/k RMSNorm behavior in:
        `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py::Attention`,
        where `q_norm` / `k_norm` are `torch.nn.RMSNorm(inner_dim, eps=norm_eps)`.

    Why it exists:
      - LTX-2 applies q/k RMSNorm over the full `inner_dim = heads * head_dim` vector
        before RoPE and attention.
      - In SGLang, q/k projections are sharded across TP ranks, so each rank only sees
        `[*, inner_dim / tp]`. To match LTX-2 numerics, we compute the RMS over the full
        hidden size via all-reduce, and apply a sharded slice of the RMSNorm weight.
    """

    def __init__(self, full_hidden_size: int, local_hidden_size: int, eps: float) -> None:
        super().__init__()
        self.full_hidden_size = full_hidden_size
        self.local_hidden_size = local_hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(local_hidden_size))

        # Let FSDP loader shard this param from the full checkpoint tensor.
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        if full_hidden_size % tp_size != 0:
            raise ValueError(
                f"full_hidden_size ({full_hidden_size}) must be divisible by tp_size ({tp_size})"
            )
        if local_hidden_size != full_hidden_size // tp_size:
            raise ValueError(
                f"local_hidden_size ({local_hidden_size}) must equal full_hidden_size/tp_size ({full_hidden_size // tp_size})"
            )

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            shard = loaded_weight.narrow(0, tp_rank * local_hidden_size, local_hidden_size)
            param.data.copy_(shard.to(dtype=param.dtype, device=param.device))

        setattr(self.weight, "weight_loader", _weight_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., local_hidden_size]
        if x.shape[-1] != self.local_hidden_size:
            raise ValueError(
                f"Expected last dim {self.local_hidden_size}, got {x.shape[-1]}"
            )

        if get_tp_world_size() == 1:
            # Exact when unsharded.
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
        else:
            # Exact global RMS over full_hidden_size.
            local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
            global_sumsq = tensor_model_parallel_all_reduce(local_sumsq)
            var = global_sumsq / float(self.full_hidden_size)

        y = x * torch.rsqrt(var + self.eps)
        return y * self.weight.to(dtype=y.dtype)


class LTX2Attention(nn.Module):
    """
    LTX-2-compatible Attention module with Wan/SGLang inference backend.

    Design goals:
    - **Looks like upstream LTX-2** (`Attention` in `ltx_core`):
      - init args: `query_dim/context_dim/heads/dim_head/norm_eps/rope_type/attention_function`
      - attributes: `q_norm/k_norm/to_q/to_k/to_v/to_out`, `heads`, `dim_head`
      - forward args: `x/context/mask/pe/k_pe`
    - **Executes like Wan/SGLang**:
      - TP-friendly linear layers
      - fastpath via `USPAttention`
      - fallback to PyTorch SDPA when an explicit mask is provided (USPAttention has no mask support)

    RoPE:
      - Upstream passes `pe` / `k_pe` objects and applies RoPE on [B, S, inner_dim].
      - In SGLang we apply RoPE on [B, S, H, D] using `(cos, sin)` (a.k.a. `freqs_cis`).
      - For compatibility, if `pe`/`k_pe` are provided as `(cos, sin)` tuples, we will use them.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: Any = None,
        attention_function: Any = None,
        qk_norm: bool = True,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Upstream-compatible fields
        self.rope_type = rope_type
        self.attention_function = attention_function

        self.query_dim = int(query_dim)
        self.context_dim = int(query_dim if context_dim is None else context_dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.norm_eps = float(norm_eps)
        self.qk_norm = bool(qk_norm)

        tp_size = get_tp_world_size()
        assert (
            self.heads % tp_size == 0
        ), f"heads {heads} must be divisible by tp world size {tp_size}"
        self.local_heads = self.heads // tp_size

        if self.inner_dim % tp_size != 0:
            raise ValueError(
                f"inner_dim ({self.inner_dim}) must be divisible by tp world size ({tp_size})"
            )
        if (self.inner_dim // tp_size) != (self.local_heads * self.dim_head):
            raise ValueError(
                "Invalid TP shape: expected (inner_dim/tp) == (local_num_heads*head_dim), "
                f"got inner_dim={self.inner_dim}, tp={tp_size}, "
                f"local_num_heads={self.local_heads}, head_dim={self.dim_head}"
            )

        # Upstream-named projection modules (TP-friendly)
        self.to_q = ColumnParallelLinear(
            self.query_dim, self.inner_dim, bias=True, gather_output=False
        )
        self.to_k = ColumnParallelLinear(
            self.context_dim, self.inner_dim, bias=True, gather_output=False
        )
        self.to_v = ColumnParallelLinear(
            self.context_dim, self.inner_dim, bias=True, gather_output=False
        )

        # Upstream-named q/k norms.
        # In TP, we need a global-RMS implementation to match upstream RMSNorm(inner_dim).
        self.q_norm: nn.Module | None = None
        self.k_norm: nn.Module | None = None
        if self.qk_norm:
            if tp_size == 1:
                self.q_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
                self.k_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
            else:
                self.q_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )
                self.k_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )

        self.to_out = nn.Sequential(
            RowParallelLinear(
                self.inner_dim, self.query_dim, bias=True, input_is_parallel=True
            ),
            nn.Identity(),
        )

        # Wan/SGLang inference backend (same style as `wanvideo.py`):
        # - Always construct `USPAttention` and let backend selection happen inside it.
        # - We still fall back to PyTorch SDPA in `forward()` when an explicit mask is provided.
        self.attn = USPAttention(
            num_heads=self.local_heads,
            head_size=self.dim_head,
            num_kv_heads=self.local_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Upstream-compatible forward:
          q = to_q(x)
          context = x if context is None else context
          k,v = to_k/to_v(context)
          q,k = q_norm/k_norm
          if pe: apply RoPE
          out = attention(q,k,v, heads, mask)
          return to_out(out)

        SGLang/Wan execution notes:
          - TP linears output local-sharded `inner_dim/tp` for q/k/v.
          - RoPE is applied on [B, S, H_local, D] using (cos, sin) tuples.
          - If `mask` is provided, we fall back to PyTorch SDPA (USPAttention has no mask support).
        """
        q, _ = self.to_q(x)
        context_ = x if context is None else context
        k, _ = self.to_k(context_)
        v, _ = self.to_v(context_)

        if self.qk_norm:
            assert self.q_norm is not None and self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [B, S, inner_dim/tp] -> [B, S, H_local, D]
        q = q.view(*q.shape[:-1], self.local_heads, self.dim_head)
        k = k.view(*k.shape[:-1], self.local_heads, self.dim_head)
        v = v.view(*v.shape[:-1], self.local_heads, self.dim_head)

        if pe is not None:
            cos, sin = pe
            k_cos, k_sin = pe if k_pe is None else k_pe

            # LTX RoPE types:
            # - INTERLEAVED corresponds to GPT-J style pairwise rotation (SGLang: interleaved=True, is_neox=False)
            # - SPLIT has different layout semantics in upstream `rope.py` and is not yet supported here.
            if self.rope_type == LTX2RopeType.SPLIT:
                raise NotImplementedError(
                    "LTX2RopeType.SPLIT is not yet supported in SGLang LTX2Attention RoPE application."
                )

            # Fast RoPE path when q/k shapes match and we share the same cache for k.
            if q.is_cuda and q.shape == k.shape and k_pe is None:
                cos_sin_cache = torch.cat(
                    [
                        cos.to(dtype=torch.float32).contiguous(),
                        sin.to(dtype=torch.float32).contiguous(),
                    ],
                    dim=-1,
                )
                q, k = apply_flashinfer_rope_qk_inplace(
                    q, k, cos_sin_cache, is_neox=False
                )
            else:
                q = _apply_rotary_emb(q, cos, sin, is_neox_style=False, interleaved=True)
                k = _apply_rotary_emb(k, k_cos, k_sin, is_neox_style=False, interleaved=True)

        # Upstream passes an additive float attention mask into SDPA.
        if mask is not None:
            # torch SDPA expects [B, H, S, D]
            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            # Accept upstream-style additive masks directly (float).
            if torch.is_floating_point(mask):
                m = mask
                if m.dim() == 2:
                    # [B, K] -> [B, 1, 1, K]
                    m = m[:, None, None, :]
                elif m.dim() == 3:
                    # [B, Q, K] -> [B, 1, Q, K]
                    m = m[:, None, :, :]
                sdpa_mask = m.to(dtype=q_.dtype, device=q_.device)
            else:
                # Key padding mask in {0,1} (or bool), where 1 means keep and 0 means mask.
                # Convert to additive mask:
                #   (mask - 1) * finfo.max  => 0 for keep, -max for masked.
                m = mask.to(dtype=q_.dtype, device=q_.device)
                if m.dim() == 2:
                    m = m[:, None, None, :]
                elif m.dim() == 3:
                    m = m[:, None, :, :]
                sdpa_mask = (m - 1.0) * torch.finfo(q_.dtype).max

            out = torch.nn.functional.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            ).transpose(1, 2)
        else:
            out = self.attn(q, k, v)

        out = out.flatten(2)
        out, _ = self.to_out[0](out)
        return out


class LTX2FeedForward(nn.Module):
    """
    LTX-2 `FeedForward` equivalent (see `ltx_core.model.transformer.feed_forward.FeedForward`).

    Upstream structure:
      - GELUApprox(dim -> inner_dim), then Linear(inner_dim -> dim_out)
    We implement an equivalent MLP using TP-friendly linear layers.
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4) -> None:
        super().__init__()
        if dim_out is None:
            dim_out = dim
        inner_dim = int(dim * mult)

        self.proj_in = ColumnParallelLinear(dim, inner_dim, bias=True, gather_output=True)
        self.act = nn.GELU(approximate="tanh")
        self.proj_out = ColumnParallelLinear(
            inner_dim, dim_out, bias=True, gather_output=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj_in(x)
        x = self.act(x)
        x, _ = self.proj_out(x)
        return x


class LTX2GELUApprox(nn.Module):
    """
    LTX-2 FFN "project_in" layer: `GELUApprox(dim_in -> dim_out)` with parameter name `proj`.

    Upstream reference:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py::GELUApprox`

    Upstream behavior:
      - `proj = Linear(dim_in, dim_out)`
      - `forward(x) = gelu(proj(x), approximate="tanh")`

    SGLang notes:
      - We use TP-friendly `ColumnParallelLinear` but keep the same parameter name (`proj`)
        for checkpoint key alignment: `...ff.net.0.proj.{weight,bias}`.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=True, gather_output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, _ = self.proj(x)
        return F.gelu(x_proj, approximate="tanh")


class LTX2TransformerBlock(nn.Module):
    """
    LTX-2 `BasicAVTransformerBlock` equivalent (video + audio + AV cross-attn).

    Upstream reference:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
      - `class BasicAVTransformerBlock`

    What we implement here:
      - **Video branch**: `attn1`, `attn2`, `ff`, `scale_shift_table`
      - **Audio branch**: `audio_attn1`, `audio_attn2`, `audio_ff`, `audio_scale_shift_table`
      - **Audio <-> Video cross-attn**:
        - `audio_to_video_attn` (Q=video, KV=audio)
        - `video_to_audio_attn` (Q=audio, KV=video)
        - `scale_shift_table_a2v_ca_audio` / `scale_shift_table_a2v_ca_video` (5×dim Ada params)

    Corresponding upstream lines (approx):
      - `__init__`:
        - video part: creates `attn1/attn2/ff/scale_shift_table`
        - audio part: creates `audio_attn1/audio_attn2/audio_ff/audio_scale_shift_table`
        - AV cross-attn part: creates `audio_to_video_attn/video_to_audio_attn` + 5×dim tables
      - `forward`:
        - video branch (self-attn + text cross-attn + FFN)
        - audio branch (self-attn + text cross-attn + FFN)
        - optional a2v/v2a cross-attn between modalities

    Differences vs upstream:
      - We don't import LTX-2's `BatchedPerturbationConfig`/`PerturbationType` here. If a compatible
        `perturbations` object is provided (has `all_in_batch()` and `mask_like()`), we will use it;
        otherwise we run without skipping/masking.
      - We use SGLang/Wan TP-friendly layers + attention backends, but keep the same math.
    """

    def __init__(
        self,
        idx: int,
        video: LTX2TransformerConfig | None = None,
        audio: LTX2TransformerConfig | None = None,
        rope_type: LTX2RopeType = LTX2RopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: Any = LTX2AttentionFunction.DEFAULT,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Upstream fields.
        self.idx = int(idx)
        self.norm_eps = float(norm_eps)
        self.rope_type = rope_type
        self.attention_function = attention_function

        self.video = video
        self.audio = audio

        # For convenience/debugging only (upstream doesn't store a unified `dim`).
        self.dim = int(video.dim) if video is not None else 0

        # ==========================
        # Video branch (upstream: `if video is not None:`)
        # ==========================
        self.attn1: LTX2Attention | None = None
        self.attn2: LTX2Attention | None = None
        self.ff: LTX2FeedForward | None = None
        self.scale_shift_table: nn.Parameter | None = None

        if video is not None:
            self.attn1 = LTX2Attention(
                query_dim=video.dim,
                context_dim=None,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=self.rope_type,
                norm_eps=self.norm_eps,
                attention_function=self.attention_function,
                qk_norm=True,
                supported_attention_backends=supported_attention_backends,
                prefix=f"blocks.{idx}.attn1",
            )
            self.attn2 = LTX2Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=self.rope_type,
                norm_eps=self.norm_eps,
                attention_function=self.attention_function,
                qk_norm=True,
                supported_attention_backends=supported_attention_backends,
                prefix=f"blocks.{idx}.attn2",
            )
            self.ff = LTX2FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = nn.Parameter(torch.empty(6, video.dim))

        # ==========================
        # Audio branch (upstream: `if audio is not None:`)
        # ==========================
        self.audio_attn1: LTX2Attention | None = None
        self.audio_attn2: LTX2Attention | None = None
        self.audio_ff: LTX2FeedForward | None = None
        self.audio_scale_shift_table: nn.Parameter | None = None

        if audio is not None:
            self.audio_attn1 = LTX2Attention(
                query_dim=audio.dim,
                context_dim=None,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=self.rope_type,
                norm_eps=self.norm_eps,
                attention_function=self.attention_function,
                qk_norm=True,
                supported_attention_backends=supported_attention_backends,
                prefix=f"blocks.{idx}.audio_attn1",
            )
            self.audio_attn2 = LTX2Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=self.rope_type,
                norm_eps=self.norm_eps,
                attention_function=self.attention_function,
                qk_norm=True,
                supported_attention_backends=supported_attention_backends,
                prefix=f"blocks.{idx}.audio_attn2",
            )
            self.audio_ff = LTX2FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = nn.Parameter(torch.empty(6, audio.dim))

        # ==========================
        # Audio <-> Video cross-attention (upstream: `if audio is not None and video is not None:`)
        # ==========================
        self.audio_to_video_attn: LTX2Attention | None = None
        self.video_to_audio_attn: LTX2Attention | None = None
        self.scale_shift_table_a2v_ca_audio: nn.Parameter | None = None
        self.scale_shift_table_a2v_ca_video: nn.Parameter | None = None

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = LTX2Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=self.rope_type,
                norm_eps=self.norm_eps,
                attention_function=self.attention_function,
                qk_norm=True,
                supported_attention_backends=supported_attention_backends,
                prefix=f"blocks.{idx}.audio_to_video_attn",
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = LTX2Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=self.rope_type,
                norm_eps=self.norm_eps,
                attention_function=self.attention_function,
                qk_norm=True,
                supported_attention_backends=supported_attention_backends,
                prefix=f"blocks.{idx}.video_to_audio_attn",
            )

            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

        # Wan-style helper. Upstream does direct `x = x + delta * gate * mask`.
        self.residual = ScaleResidual()

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        # Upstream helper:
        #   `BasicAVTransformerBlock.get_ada_values(scale_shift_table, batch, timesteps, indices)`
        #
        # timestep: [B, num_ada_params*D] or [B, 1, num_ada_params*D]
        if timestep.dim() == 2:
            timestep = timestep.unsqueeze(1)

        num_ada_params, dim = int(scale_shift_table.shape[0]), int(scale_shift_table.shape[1])
        expected_last = num_ada_params * dim
        # Upstream assumes `timestep` is produced by its preprocessor and always has the correct shape.
        # In SGLang, this function may be called from multiple pipelines, so we validate early with a
        # clear error message instead of relying on a later reshape error.
        if int(timestep.shape[-1]) != expected_last:
            raise ValueError(
                "Bad `timestep` shape for Ada params: "
                f"got {tuple(timestep.shape)}, expected last dim == {expected_last} "
                f"(= {num_ada_params} * {dim}) to match scale_shift_table {tuple(scale_shift_table.shape)}."
            )

        ada_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Upstream-equivalent helper for audio-video cross-attn Ada params.

        Corresponds to:
          - `BasicAVTransformerBlock.get_av_ca_ada_values` in upstream `transformer.py`
        """

        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )

        # Keep structure consistent with upstream helper (even though squeeze is a no-op for D>1).
        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]
        return (*scale_shift_chunks, *gate_ada_values)

    def forward(  # noqa: PLR0915
        self,
        video: LTX2TransformerArgs | None,
        audio: LTX2TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[LTX2TransformerArgs | None, LTX2TransformerArgs | None]:
        """
        Upstream-style `BasicAVTransformerBlock.forward` equivalent (audio + video + AV cross-attn).

        Upstream reference:
          - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
          - `BasicAVTransformerBlock.forward`

        Notes:
          - We currently ignore `perturbations` (LTX-2 uses it for skipping parts of a batch).
          - `positional_embeddings` / `cross_positional_embeddings` are represented as `(cos, sin)` tuples.
        """

        if video is None and audio is None:
            return None, None

        batch_size = video.x.shape[0] if video is not None else audio.x.shape[0]
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx is not None and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax is not None and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx is not None and vx.numel() > 0)

        # ==========================
        # Video branch (matches upstream)
        # ==========================
        if run_vx:
            assert self.attn1 is not None and self.attn2 is not None and self.ff is not None
            assert self.scale_shift_table is not None

            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                delta = self.attn1(norm_vx, pe=video.positional_embeddings)
                vx = self.residual(vx, delta, vgate_msa * v_mask)

            vx = vx + self.attn2(
                rms_norm(vx, eps=self.norm_eps),
                context=video.context,
                mask=video.context_mask,
            )

        # ==========================
        # Audio branch (matches upstream)
        # ==========================
        if run_ax:
            assert (
                self.audio_attn1 is not None
                and self.audio_attn2 is not None
                and self.audio_ff is not None
            )
            assert self.audio_scale_shift_table is not None

            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                delta = self.audio_attn1(norm_ax, pe=audio.positional_embeddings)
                ax = self.residual(ax, delta, agate_msa * a_mask)

            ax = ax + self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context,
                mask=audio.context_mask,
            )

        # ==========================
        # Audio - Video cross attention (matches upstream)
        # ==========================
        if (run_a2v or run_v2a) and (self.audio_to_video_attn is not None) and (self.video_to_audio_attn is not None):
            assert vx is not None and ax is not None
            assert video is not None and audio is not None
            assert self.scale_shift_table_a2v_ca_audio is not None
            assert self.scale_shift_table_a2v_ca_video is not None

            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            audio_scale_shift_t = audio.cross_scale_shift_timestep
            audio_gate_t = audio.cross_gate_timestep
            video_scale_shift_t = video.cross_scale_shift_timestep
            video_gate_t = video.cross_gate_timestep

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio_scale_shift_t,
                audio_gate_t,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video_scale_shift_t,
                video_gate_t,
            )

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                delta = self.audio_to_video_attn(
                    vx_scaled,
                    context=ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                )
                vx = self.residual(vx, delta, gate_out_a2v * a2v_mask)

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                delta = self.video_to_audio_attn(
                    ax_scaled,
                    context=vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )
                ax = self.residual(ax, delta, gate_out_v2a * v2a_mask)


        if run_vx:
            assert self.ff is not None and self.scale_shift_table is not None
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = self.residual(vx, self.ff(vx_scaled), vgate_mlp)

        if run_ax:
            assert self.audio_ff is not None and self.audio_scale_shift_table is not None
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = self.residual(ax, self.audio_ff(ax_scaled), agate_mlp)

        # Upstream returns `replace(video, x=vx)`; in SGLang we mutate for simplicity.
        if video is not None and vx is not None:
            video.x = vx
        if audio is not None and ax is not None:
            audio.x = ax
        return video, audio


# ==============================================================================
# Preprocessors
# ==============================================================================


@dataclass(frozen=True)
class LTX2VideoArgsPreprocessor:
    """
    Video-only equivalent of LTX-2 upstream `TransformerArgsPreprocessor`.

    Upstream reference:
      - `LTX-2/.../transformer/model.py::LTXModel.forward`
      - `LTX-2/.../transformer/model.py::_init_preprocessors` (video_args_preprocessor)
      - `LTX-2/.../transformer/transformer_args.py::TransformerArgsPreprocessor.prepare`

    Why this exists:
      - It makes `ltx2_video.py::LTXModel.forward` read like upstream:
          prepare(video) -> blocks -> _process_output -> unpatchify
      - It keeps the runtime signature Wan/diffusers-style (`hidden_states`, `encoder_hidden_states`, `timestep`)
        while still producing upstream-shaped `LTX2TransformerArgs` objects for blocks.
    """

    patchify_proj: TPLinearCallable
    adaln: AdaLNCallable
    caption_projection: CaptionProjectionCallable
    timestep_scale_multiplier: int
    patch_size: tuple[int, int, int]  # (p_t, p_h, p_w)
    rope_dim_list: tuple[int, int, int]  # per-axis rope dims, sum == head_dim
    positional_embedding_theta: float
    positional_embedding_max_pos: list[int]
    use_middle_indices_grid: bool
    double_precision_rope: bool

    # --------------------------------------------------------------------------
    # Upstream-aligned helpers (mirrors `TransformerArgsPreprocessor` in LTX-2)
    # --------------------------------------------------------------------------

    def _prepare_timestep(
        self, timestep: torch.Tensor, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare timestep embeddings.

        Upstream reference:
          - `transformer_args.py::TransformerArgsPreprocessor._prepare_timestep`
        """
        timestep = timestep * int(self.timestep_scale_multiplier)
        timestep, embedded_timestep = self.adaln(
            timestep.flatten(), hidden_dtype=hidden_dtype
        )
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Prepare context for transformer blocks.

        Upstream reference:
          - `transformer_args.py::TransformerArgsPreprocessor._prepare_context`
        """
        batch_size = int(x.shape[0])
        ctx = self.caption_projection(context)
        # Keep the upstream `.view(batch, -1, x.shape[-1])` shape contract.
        ctx = ctx.view(batch_size, -1, x.shape[-1])
        return ctx, attention_mask

    def _prepare_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        """
        Prepare attention mask (convert {0,1}/bool to additive float mask).

        Upstream reference:
          - `transformer_args.py::TransformerArgsPreprocessor._prepare_attention_mask`
        """
        if attention_mask is None:
            return None

        # Accept upstream-style additive masks directly (float).
        if torch.is_floating_point(attention_mask):
            m = attention_mask
            if m.dim() == 2:
                # [B, K] -> [B, 1, 1, K]
                return m.to(x_dtype)[:, None, None, :]
            if m.dim() == 3:
                # [B, Q, K] -> [B, 1, Q, K]
                return m.to(x_dtype)[:, None, :, :]
            if m.dim() == 4:
                # Already broadcasted.
                return m.to(x_dtype)
            raise ValueError(f"Unsupported attention_mask shape: {tuple(m.shape)}")

        # Key padding mask in {0,1} (or bool), where 1 means keep and 0 means mask.
        # Convert to additive float mask:
        #   (mask - 1) * finfo.max  => 0 for keep, -max for masked.
        m = attention_mask.to(dtype=x_dtype)
        if m.dim() == 2:
            m = m[:, None, None, :]
        elif m.dim() == 3:
            m = m[:, None, :, :]
        elif m.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(m.shape)}")
        return (m - 1.0) * torch.finfo(x_dtype).max

    def _prepare_positional_embeddings(
        self,
        *,
        post_t: int,
        post_h: int,
        post_w: int,
        coords: torch.Tensor | None,
        device: torch.device,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare positional embeddings (RoPE) for the post-patch grid.

        Upstream reference:
          - `transformer_args.py::TransformerArgsPreprocessor._prepare_positional_embeddings`

        LTX-2 vs SGLang:
          - Upstream LTX-2 computes RoPE from an explicit `indices_grid` (patch bounds) plus
            `max_pos/use_middle_indices_grid`, producing `freqs = indices * (fractional*2 - 1)`.
          - In SGLang we keep the native RoPE representation `(cos, sin)` (used by Wan/USPAttention),
            and map LTX's position semantics into the `pos` input of SGLang's RoPE generator.
        """
        # LTX-2 upstream uses patch bounds (`positions`) as input for RoPE.
        # We support two equivalent sources:
        # - If `coords` is provided: use it directly (cache-dit style, explicit bounds).
        # - Else: derive bounds from (post_t, post_h, post_w) and patch_size (upstream patchifier style).
        #
        # Expected `coords` shape: [B, axes, num_tokens, 2] where axes is 3 for video,
        # and may be 1 for audio (temporal-only). Bounds are [start, end) in "latent grid" units.
        if coords is not None:
            if coords.dim() != 4 or coords.shape[-1] != 2:
                raise ValueError(
                    "coords must have shape [B, axes, num_tokens, 2], "
                    f"got {tuple(coords.shape)}"
                )
            if int(coords.shape[0]) != 1:
                # We only support batch-invariant coords for now (matches common patchifier behavior).
                # If this ever becomes necessary, we can extend RoPE application to accept per-batch cos/sin.
                if not torch.all(coords == coords[:1]).item():
                    raise ValueError(
                        "coords must be identical across batch for now (batch-varying coords are unsupported)."
                    )
            coords0 = coords[0].to(device=device, dtype=torch.float32)  # [axes, num_tokens, 2]
            axes = int(coords0.shape[0])
            patch_starts = coords0[:, :, 0]
            patch_ends = coords0[:, :, 1]

            if axes == 1:
                # Temporal-only coords (audio). Pad H/W with a constant [0,1) interval so downstream
                # 3-axis RoPE logic remains unchanged.
                num_tokens = int(coords0.shape[1])
                hw_starts = torch.zeros((2, num_tokens), device=device, dtype=torch.float32)
                hw_ends = torch.ones((2, num_tokens), device=device, dtype=torch.float32)
                patch_starts = torch.cat([patch_starts, hw_starts], dim=0)
                patch_ends = torch.cat([patch_ends, hw_ends], dim=0)
            elif axes != 3:
                raise ValueError(f"coords axes must be 1 or 3, got {axes}")
        else:
            # Build per-patch bounds in latent grid coordinates (matches upstream patchifier bounds).
            # Shape: [3, num_tokens] storing [start, end) for each axis.
            p_t, p_h, p_w = self.patch_size
            grid_coords = torch.meshgrid(
                torch.arange(start=0, end=post_t * p_t, step=p_t, device=device),
                torch.arange(start=0, end=post_h * p_h, step=p_h, device=device),
                torch.arange(start=0, end=post_w * p_w, step=p_w, device=device),
                indexing="ij",
            )
            patch_starts = (
                torch.stack(grid_coords, dim=0).reshape(3, -1).to(dtype=torch.float32)
            )
            patch_size_delta = torch.tensor(
                (p_t, p_h, p_w), device=device, dtype=torch.float32
            ).view(3, 1)
            patch_ends = patch_starts + patch_size_delta

        # Choose start vs midpoint coordinates (upstream: use_middle_indices_grid selects midpoint).
        if self.use_middle_indices_grid:
            pos = (patch_starts + patch_ends) / 2.0
        else:
            pos = patch_starts

        # Map to upstream-style normalized coordinates in [-1, 1].
        # Note: upstream uses `max_pos` in "position units" (often seconds/pixels). Here we use latent-grid
        # units unless the caller config chooses `positional_embedding_max_pos` accordingly.
        if self.positional_embedding_max_pos is None:
            raise ValueError("positional_embedding_max_pos must be set for RoPE")
        max_pos = torch.tensor(
            self.positional_embedding_max_pos, device=device, dtype=torch.float32
        ).view(3, 1)
        pos = (pos / max_pos) * 2.0 - 1.0  # [3, num_tokens]

        # Generate per-axis (cos, sin) and concatenate (ND RoPE).
        rope_dtype = torch.float64 if self.double_precision_rope else torch.float32
        cos_parts: list[torch.Tensor] = []
        sin_parts: list[torch.Tensor] = []
        for axis, dim_i in enumerate(self.rope_dim_list):
            cos_i, sin_i = get_1d_rotary_pos_embed(
                dim=int(dim_i),
                pos=pos[axis],
                theta=float(self.positional_embedding_theta),
                dtype=rope_dtype,
                device=device,
            )
            cos_parts.append(cos_i)
            sin_parts.append(sin_i)

        cos = torch.cat(cos_parts, dim=-1).to(dtype=x_dtype)
        sin = torch.cat(sin_parts, dim=-1).to(dtype=x_dtype)
        return cos, sin

    def prepare(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        coords: torch.Tensor | None = None,
    ) -> LTX2TransformerArgs:
        """
        Prepare upstream-shaped video args from Wan/diffusers-style tensors.

        Upstream reference:
          - `TransformerArgsPreprocessor.prepare`
        """
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required")
        if timestep is None:
            raise ValueError("timestep is required")
        # Normalize timestep shapes (future CP/sharding friendliness):
        # - allow scalar, [B], [B,1], or [B,*] where we take the first element per batch.
        if timestep.dim() == 0:
            timestep = timestep.reshape(1)
        if timestep.dim() >= 2:
            timestep = timestep.reshape(timestep.shape[0], -1)[:, 0]

        if hidden_states.dim() != 5:
            raise ValueError(
                f"Expected hidden_states to be 5D [B,C,T,H,W], got {tuple(hidden_states.shape)}"
            )

        bsz, c_raw, num_frames, height, width = hidden_states.shape

        # 1) Patchify (VideoLatentPatchifier-style).
        #    [B, C_raw, T, H, W] -> [B, (T' H' W'), (C_raw * p_t * p_h * p_w)]
        p_t, p_h, p_w = self.patch_size
        if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
            raise ValueError(
                f"hidden_states shape {(num_frames, height, width)} must be divisible by patch_size {self.patch_size}"
            )
        post_t = num_frames // p_t
        post_h = height // p_h
        post_w = width // p_w

        x = hidden_states.reshape(bsz, c_raw, post_t, p_t, post_h, p_h, post_w, p_w).permute(
            0, 2, 4, 6, 1, 3, 5, 7
        )
        x = x.reshape(bsz, post_t * post_h * post_w, c_raw * p_t * p_h * p_w)

        # 2) Token projection (upstream: `patchify_proj`).
        x, _ = self.patchify_proj(x)

        # Upstream-style preparation sequence (see `TransformerArgsPreprocessor.prepare`).
        timestep, embedded_timestep = self._prepare_timestep(
            timestep=timestep,
            batch_size=int(x.shape[0]),
            hidden_dtype=hidden_states.dtype,
        )
        context, attention_mask = self._prepare_context(
            context=encoder_hidden_states, x=x, attention_mask=attention_mask
        )
        attention_mask = self._prepare_attention_mask(attention_mask, x_dtype=hidden_states.dtype)
        pe = self._prepare_positional_embeddings(
            post_t=post_t,
            post_h=post_h,
            post_w=post_w,
            coords=coords,
            device=hidden_states.device,
            x_dtype=x.dtype,
        )

        return LTX2TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep.to(dtype=x.dtype),
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
        )


@dataclass(frozen=True)
class LTX2MultiModalArgsPreprocessor:
    """
    Multi-modal equivalent of LTX-2 upstream `MultiModalTransformerArgsPreprocessor`.

    Used when BOTH video and audio are enabled. Wraps a simple preprocessor
    (LTX2VideoArgsPreprocessor) and adds cross-attention timestep handling.

    Upstream reference:
      - `LTX-2/.../transformer_args.py::MultiModalTransformerArgsPreprocessor`
      - `LTX-2/.../transformer_args.py::_prepare_cross_attention_timestep`
    """

    simple_preprocessor: LTX2VideoArgsPreprocessor
    cross_scale_shift_adaln: AdaLNCallable
    cross_gate_adaln: AdaLNCallable
    av_ca_timestep_scale_multiplier: int
    cross_pe_max_pos: int
    cross_rope_dim: int
    positional_embedding_theta: float
    double_precision_rope: bool

    def _prepare_cross_positional_embeddings(
        self,
        *,
        post_t: int,
        tokens_per_t: int,
        coords: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare cross positional embeddings (1D RoPE for temporal only).

        This helper is SGLang-only (upstream does not define a separate function for this).

        Upstream correspondence:
          - `LTX-2/.../transformer_args.py::MultiModalTransformerArgsPreprocessor.prepare`
            the `cross_pe = ...` block:
              `cross_pe = self.simple_preprocessor._prepare_positional_embeddings(positions=modality.positions[:, 0:1, :], ...)`

        LTX-2 vs SGLang:
          - Upstream cross PE is computed from temporal-only positions with `max_pos=[cross_pe_max_pos]` and
            `use_middle_indices_grid=True`.
          - In SGLang we keep `(cos, sin)` and map that positional semantics into the `pos` input of the
            SGLang RoPE generator.
        """
        # Upstream computes cross PE from temporal-only positions (`positions[:, 0:1, :]`).
        # If explicit coords are provided, we use them directly (temporal axis only).
        if coords is not None:
            if coords.dim() != 4 or coords.shape[-1] != 2:
                raise ValueError(
                    "coords must have shape [B, axes, num_tokens, 2], "
                    f"got {tuple(coords.shape)}"
                )
            if int(coords.shape[0]) != 1:
                if not torch.all(coords == coords[:1]).item():
                    raise ValueError(
                        "coords must be identical across batch for now (batch-varying coords are unsupported)."
                    )
            coords0 = coords[0].to(device=device, dtype=torch.float32)  # [axes, num_tokens, 2]
            if int(coords0.shape[1]) != int(seq_len):
                raise ValueError(
                    "coords token count mismatch for cross PE: "
                    f"coords has num_tokens={int(coords0.shape[1])}, expected seq_len={int(seq_len)}"
                )
            starts = coords0[0, :, 0]
            ends = coords0[0, :, 1]
            pos = (starts + ends) / 2.0
        else:
            # Temporal patch bounds in latent-grid coordinates: [start, end) for each temporal token.
            p_t = int(self.simple_preprocessor.patch_size[0])
            starts = torch.arange(post_t, device=device, dtype=torch.float32) * float(p_t)
            ends = starts + float(p_t)
            # Upstream uses `use_middle_indices_grid=True` for cross PE.
            pos_t = (starts + ends) / 2.0

            # Repeat temporal positions to match token sequence length (video has many tokens per t).
            # Upstream uses `positions[:, 0:1, :]` which is a per-token temporal coordinate.
            pos = pos_t.repeat_interleave(int(tokens_per_t), dim=0)

        # Normalize to [-1, 1] using cross_pe_max_pos (caller chooses units via config).
        pos = (pos / float(self.cross_pe_max_pos)) * 2.0 - 1.0

        rope_dtype = torch.float64 if self.double_precision_rope else torch.float32
        cross_cos, cross_sin = get_1d_rotary_pos_embed(
            dim=int(self.cross_rope_dim),
            pos=pos,
            theta=float(self.positional_embedding_theta),
            dtype=rope_dtype,
            device=device,
        )
        return cross_cos.to(dtype=x_dtype), cross_sin.to(dtype=x_dtype)

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare cross attention timestep embeddings.

        Upstream reference:
          - `MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep`
        """
        timestep_scale_multiplier = self.simple_preprocessor.timestep_scale_multiplier
        timestep = timestep * int(timestep_scale_multiplier)

        av_ca_factor = float(self.av_ca_timestep_scale_multiplier) / float(timestep_scale_multiplier)

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(), hidden_dtype=hidden_dtype
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])

        gate_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor, hidden_dtype=hidden_dtype
        )
        gate_timestep = gate_timestep.view(batch_size, -1, gate_timestep.shape[-1])

        return scale_shift_timestep, gate_timestep

    def prepare(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None,
        coords: torch.Tensor | None = None,
    ) -> LTX2TransformerArgs:
        """
        Prepare upstream-shaped multi-modal args.

        Upstream reference:
          - `MultiModalTransformerArgsPreprocessor.prepare`

        Same signature as LTX2VideoArgsPreprocessor.prepare(), but also computes
        cross_scale_shift_timestep, cross_gate_timestep, and cross_positional_embeddings
        for AV cross-attention.
        """
        # Use the simple preprocessor for base preparation.
        transformer_args = self.simple_preprocessor.prepare(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            attention_mask=attention_mask,
            coords=coords,
        )

        # Prepare cross-attention timesteps (upstream: _prepare_cross_attention_timestep).
        batch_size = int(transformer_args.x.shape[0])
        cross_scale_shift_t, cross_gate_t = self._prepare_cross_attention_timestep(
            timestep=timestep,
            batch_size=batch_size,
            hidden_dtype=transformer_args.x.dtype,
        )

        # Prepare cross positional embeddings (1D RoPE for temporal only).
        # Upstream: uses positions[:, 0:1, :] (only temporal) with max_pos=[cross_pe_max_pos].
        # SGLang: use 1D cross_rotary_emb.forward_from_grid with (post_t,).
        #
        # Get post_t from hidden_states shape.
        # hidden_states: [B, C, T, H, W]
        p_t = self.simple_preprocessor.patch_size[0]
        post_t = hidden_states.shape[2] // p_t
        seq_len = int(transformer_args.x.shape[1])
        if post_t <= 0 or seq_len % int(post_t) != 0:
            raise ValueError(
                "Invalid cross PE shape: expected seq_len divisible by post_t, "
                f"got seq_len={seq_len}, post_t={post_t}"
            )
        tokens_per_t = seq_len // int(post_t)
        cross_pe = self._prepare_cross_positional_embeddings(
            post_t=int(post_t),
            tokens_per_t=int(tokens_per_t),
            coords=coords,
            seq_len=seq_len,
            device=hidden_states.device,
            x_dtype=transformer_args.x.dtype,
        )

        # Return args with cross-attention timesteps and cross PE filled in.
        return LTX2TransformerArgs(
            x=transformer_args.x,
            context=transformer_args.context,
            context_mask=transformer_args.context_mask,
            timesteps=transformer_args.timesteps,
            embedded_timestep=transformer_args.embedded_timestep,
            positional_embeddings=transformer_args.positional_embeddings,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_t,
            cross_gate_timestep=cross_gate_t,
            enabled=transformer_args.enabled,
        )


class LTXModel(CachableDiT, OffloadableDiTMixin):
    """
    LTX-2 video transformer model (upstream-aligned structure).

    This class mirrors the structure of upstream `LTX-2/.../transformer/model.py::LTXModel`:
      - `_init_video()` initializes video-specific components
      - `_init_preprocessors()` creates the args preprocessor
      - `_init_transformer_blocks()` creates the transformer blocks
      - `_process_transformer_blocks()` runs blocks (with optional gradient checkpointing)
      - `_process_output()` applies the output head

    Differences from upstream:
      - `forward()` uses Wan/diffusers-style signature instead of `(video: Modality, audio: Modality)`.
      - Currently video-only; audio support is in `LTX2TransformerBlock` but not wired here yet.
      - Uses SGLang TP-friendly layers (`ColumnParallelLinear`, etc.).

    Upstream references:
      - `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/model.py`
    """

    _fsdp_shard_conditions = LTX2VideoConfig()._fsdp_shard_conditions
    _compile_conditions = LTX2VideoConfig()._compile_conditions
    _supported_attention_backends = LTX2VideoConfig()._supported_attention_backends
    param_names_mapping = LTX2VideoConfig().param_names_mapping
    reverse_param_names_mapping = LTX2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LTX2VideoConfig().lora_param_names_mapping

    def __init__(self, config: LTX2VideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config

        # Store config for later use.
        self._enable_gradient_checkpointing = False
        self.use_middle_indices_grid = arch.use_middle_indices_grid
        self.timestep_scale_multiplier = arch.timestep_scale_multiplier
        self.norm_eps = arch.norm_eps
        self.model_type = arch.model_type
        self.rope_type = arch.rope_type

        # Basic dims (upstream: `inner_dim = num_attention_heads * attention_head_dim`).
        self.num_attention_heads = arch.num_attention_heads
        self.hidden_size = arch.hidden_size  # = inner_dim
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels

        # Patch size handling (diffusers uses scalar patch_size/patch_size_t).
        arch_patch_size = arch.patch_size
        if isinstance(arch_patch_size, int):
            default_patch_size = int(arch_patch_size)
            default_patch_size_t = int(getattr(arch, "patch_size_t", 1))
        else:
            default_patch_size_t, default_patch_size, _ = arch_patch_size
        hf_patch_size = int(hf_config.get("patch_size", default_patch_size))
        hf_patch_size_t = int(hf_config.get("patch_size_t", default_patch_size_t))
        self.patch_size = (hf_patch_size_t, hf_patch_size, hf_patch_size)

        # Derive raw latent channel count (SGLang uses [B, C_raw, T, H, W]).
        p_t, p_h, p_w = self.patch_size
        patch_volume = p_t * p_h * p_w
        if self.in_channels % patch_volume != 0:
            raise ValueError(
                f"in_channels ({self.in_channels}) must be divisible by patch_volume ({patch_volume})"
            )
        if self.out_channels % patch_volume != 0:
            raise ValueError(
                f"out_channels ({self.out_channels}) must be divisible by patch_volume ({patch_volume})"
            )
        self.in_channels_raw = self.in_channels // patch_volume
        self.out_channels_raw = self.out_channels // patch_volume
        self.num_channels_latents = self.out_channels_raw

        # RoPE config.
        self.positional_embedding_theta = arch.positional_embedding_theta
        self.double_precision_rope = arch.double_precision_rope

        # Audio config.
        self.audio_num_attention_heads = arch.audio_num_attention_heads
        self.audio_hidden_size = arch.audio_hidden_size
        self.audio_in_channels = arch.audio_in_channels
        self.audio_out_channels = arch.audio_out_channels
        self.audio_cross_attention_dim = arch.audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = arch.av_ca_timestep_scale_multiplier

        # Initialize components based on model_type (upstream-style).
        cross_pe_max_pos = None
        if self.model_type.is_video_enabled():
            self.positional_embedding_max_pos = arch.positional_embedding_max_pos
            # Per-head RoPE dims across (t, h, w). Keep sum == head_dim (Wan-style ND RoPE).
            d = self.hidden_size // self.num_attention_heads
            d6 = d // 6
            self.rope_dim_list = (d - 4 * d6, 2 * d6, 2 * d6)
            self._init_video(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                caption_channels=arch.caption_channels,
                norm_eps=self.norm_eps,
            )
            # RoPE (grid-based) for video.
            self.rotary_emb = LTX2RotaryEmbedding(
                patch_size=self.patch_size,
                rope_dim_list=list(self.rope_dim_list),
                rope_theta=float(self.positional_embedding_theta),
                use_real=True,
                dtype=torch.float64 if self.double_precision_rope else torch.float32,
            )
            self.rope = self.rotary_emb

        if self.model_type.is_audio_enabled():
            self.audio_positional_embedding_max_pos = arch.audio_positional_embedding_max_pos
            # Audio RoPE dims: keep a 3D tuple for compatibility with the shared preprocessor
            # (t, h, w) where h=w=1 for audio-like tensors.
            ad = self.audio_hidden_size // self.audio_num_attention_heads
            ad6 = ad // 6
            self.audio_rope_dim_list = (ad - 4 * ad6, 2 * ad6, 2 * ad6)
            self._init_audio(
                in_channels=self.audio_in_channels,
                out_channels=self.audio_out_channels,
                caption_channels=arch.caption_channels,
                norm_eps=self.norm_eps,
            )
            # RoPE for audio (1D).
            self.audio_rotary_emb = LTX2RotaryEmbedding(
                patch_size=(1, 1, 1),
                rope_dim_list=list(self.audio_rope_dim_list),
                rope_theta=float(self.positional_embedding_theta),
                use_real=True,
                dtype=torch.float64 if self.double_precision_rope else torch.float32,
            )
            self.audio_rope = self.audio_rotary_emb

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        self._init_transformer_blocks(
            num_layers=arch.num_layers,
            attention_head_dim=(
                self.hidden_size // self.num_attention_heads
                if self.model_type.is_video_enabled()
                else 0
            ),
            cross_attention_dim=arch.cross_attention_dim,
            audio_attention_head_dim=(
                self.audio_hidden_size // self.audio_num_attention_heads
                if self.model_type.is_audio_enabled()
                else 0
            ),
            audio_cross_attention_dim=arch.audio_cross_attention_dim,
            norm_eps=self.norm_eps,
            attention_type=arch.attention_type,
            prefix=config.prefix,
        )

        self.layer_names = ["transformer_blocks"]
        self.__post_init__()

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """
        Initialize video-specific components.

        Upstream reference:
          - `LTX-2/.../model.py::LTXModel._init_video`
        """
        self.patchify_proj = ColumnParallelLinear(
            in_channels, self.hidden_size, bias=True, gather_output=True
        )
        self.adaln_single = LTX2AdaLayerNormSingle(self.hidden_size, embedding_coefficient=6)
        self.caption_projection = LTX2TextProjection(
            in_features=caption_channels,
            hidden_size=self.hidden_size,
        )
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.hidden_size))
        self.norm_out = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=norm_eps)
        self.proj_out = ColumnParallelLinear(
            self.hidden_size, out_channels, bias=True, gather_output=True
        )

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """
        Initialize audio-specific components.

        Upstream reference:
          - `LTX-2/.../model.py::LTXModel._init_audio`
        """
        self.audio_patchify_proj = ColumnParallelLinear(
            in_channels, self.audio_hidden_size, bias=True, gather_output=True
        )
        self.audio_adaln_single = LTX2AdaLayerNormSingle(self.audio_hidden_size, embedding_coefficient=6)
        self.audio_caption_projection = LTX2TextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_hidden_size,
        )
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_hidden_size))
        self.audio_norm_out = nn.LayerNorm(self.audio_hidden_size, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = ColumnParallelLinear(
            self.audio_hidden_size, out_channels, bias=True, gather_output=True
        )

    def _init_audio_video(self, num_scale_shift_values: int = 4) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=num_scale_shift_values
        )
        self.av_ca_a2v_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=1
        )
        self.av_ca_audio_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=num_scale_shift_values
        )
        self.av_ca_v2a_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=1
        )

    def _init_preprocessors(self, cross_pe_max_pos: int | None = None) -> None:
        """
        Initialize preprocessors for video/audio args.

        Upstream reference:
          - `LTX-2/.../model.py::LTXModel._init_preprocessors`

        Args:
            cross_pe_max_pos: Max positional embedding position for AV cross-attention.
                Only set when both video and audio are enabled.

        Structure mirrors upstream:
          - AudioVideo mode: Both use LTX2MultiModalArgsPreprocessor (with cross-attn timesteps)
          - VideoOnly mode:  video_args_preprocessor = LTX2VideoArgsPreprocessor
          - AudioOnly mode:  audio_args_preprocessor = LTX2VideoArgsPreprocessor
        """
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            # AudioVideo mode: use MultiModalArgsPreprocessor for BOTH modalities.
            #
            video_simple = LTX2VideoArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                patch_size=self.patch_size,
                rope_dim_list=self.rope_dim_list,
                positional_embedding_theta=self.positional_embedding_theta,
                positional_embedding_max_pos=self.positional_embedding_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                double_precision_rope=self.double_precision_rope,
            )
            self.video_args_preprocessor = LTX2MultiModalArgsPreprocessor(
                simple_preprocessor=video_simple,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                cross_pe_max_pos=cross_pe_max_pos,
                cross_rope_dim=self.audio_cross_attention_dim // self.num_attention_heads,
                positional_embedding_theta=self.positional_embedding_theta,
                double_precision_rope=self.double_precision_rope,
            )

            audio_simple = LTX2VideoArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                patch_size=(1, 1, 1),
                rope_dim_list=self.audio_rope_dim_list,
                positional_embedding_theta=self.positional_embedding_theta,
                positional_embedding_max_pos=[self.audio_positional_embedding_max_pos[0], 1, 1],
                use_middle_indices_grid=self.use_middle_indices_grid,
                double_precision_rope=self.double_precision_rope,
            )
            self.audio_args_preprocessor = LTX2MultiModalArgsPreprocessor(
                simple_preprocessor=audio_simple,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                cross_pe_max_pos=cross_pe_max_pos,
                cross_rope_dim=self.audio_cross_attention_dim // self.num_attention_heads,
                positional_embedding_theta=self.positional_embedding_theta,
                double_precision_rope=self.double_precision_rope,
            )

        elif self.model_type.is_video_enabled():
            # VideoOnly mode: use simple preprocessor.
            self.video_args_preprocessor = LTX2VideoArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                patch_size=self.patch_size,
                rope_dim_list=self.rope_dim_list,
                positional_embedding_theta=self.positional_embedding_theta,
                positional_embedding_max_pos=self.positional_embedding_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                double_precision_rope=self.double_precision_rope,
            )

        elif self.model_type.is_audio_enabled():
            # AudioOnly mode: use simple preprocessor.
            self.audio_args_preprocessor = LTX2VideoArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                patch_size=(1, 1, 1),
                rope_dim_list=self.audio_rope_dim_list,
                positional_embedding_theta=self.positional_embedding_theta,
                positional_embedding_max_pos=[self.audio_positional_embedding_max_pos[0], 1, 1],
                use_middle_indices_grid=self.use_middle_indices_grid,
                double_precision_rope=self.double_precision_rope,
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: LTX2AttentionFunction,
        prefix: str,
    ) -> None:
        """
        Initialize transformer blocks.

        Upstream reference:
          - `LTX-2/.../model.py::LTXModel._init_transformer_blocks`
        """
        video_cfg: LTX2TransformerConfig | None = (
            LTX2TransformerConfig(
                dim=self.hidden_size,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )

        audio_cfg: LTX2TransformerConfig | None = (
            LTX2TransformerConfig(
                dim=self.audio_hidden_size,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
            )
            if self.model_type.is_audio_enabled()
            else None
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTX2TransformerBlock(
                    idx=idx,
                    video=video_cfg,
                    audio=audio_cfg,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=f"{prefix}.transformer_blocks.{idx}",
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks."""
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: LTX2TransformerArgs | None,
        audio: LTX2TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[LTX2TransformerArgs | None, LTX2TransformerArgs | None]:
        """Process transformer blocks (with optional gradient checkpointing)."""
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(video=video, audio=audio, perturbations=perturbations)
        return video, audio

    def _process_output(
        self,
        scale_shift_table: nn.Parameter,
        norm_out: nn.LayerNorm,
        proj_out: ColumnParallelLinear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output (scale-shift modulation + projection)."""
        scale_shift_values = (
            scale_shift_table[None, :, :].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, None, :].to(dtype=x.dtype)
        )
        shift, scale = scale_shift_values[:, 0], scale_shift_values[:, 1]

        x = norm_out(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        x, _ = proj_out(x)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.LongTensor | None = None,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        # Added args for denoising stage alignment
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        # End added args
        guidance=None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass for LTX models.

        Upstream reference:
          - `LTX-2/.../model.py::LTXModel.forward`

        Args:
          hidden_states: Video latents [B, C, T, H, W], or None to skip video.
          encoder_hidden_states: Text embeddings for video.
          timestep: Diffusion timestep for video.
          audio_hidden_states (via kwargs): Audio latents, or None to skip audio.
          audio_encoder_hidden_states (via kwargs): Text embeddings for audio.
          audio_timestep (via kwargs): Diffusion timestep for audio.
          perturbations (via kwargs): Optional perturbation config.

        Returns:
          (vx, ax): Processed output tensors (patchified tokens or unpatchified latents).
        """
        # Upstream-style validation: check model capabilities vs inputs.
        if not self.model_type.is_video_enabled() and hidden_states is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and kwargs.get("audio_hidden_states") is not None:
            raise ValueError("Audio is not enabled for this model")

        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        perturbations = kwargs.get("perturbations", None)
        video_coords = kwargs.get("video_coords", None)
        audio_coords = kwargs.get("audio_coords", None)

        # Pipeline-compat flags:
        # - Upstream LTX-2 transformer/model.py returns patchified tokens.
        # - Diffusion pipelines typically expect "noise_pred" in the same latent shape.
        #   We support both by optionally unpatchifying when requested.
        #   Default to True to support denoising stage usage.
        return_latents = bool(kwargs.get("return_latents", True))

        # Accept common mask naming from pipeline stages.
        attention_mask = kwargs.get("attention_mask", kwargs.get("encoder_attention_mask"))
        audio_attention_mask = kwargs.get(
            "audio_attention_mask", kwargs.get("audio_encoder_attention_mask", attention_mask)
        )

        # Prepare args via preprocessors (upstream style).
        video_args = (
            self.video_args_preprocessor.prepare(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                attention_mask=attention_mask,
                coords=video_coords,
            )
            if hidden_states is not None
            else None
        )

        audio_hidden_states = kwargs.get("audio_hidden_states")
        original_audio_ndim = None
        if audio_hidden_states is not None and isinstance(audio_hidden_states, torch.Tensor):
            # Upstream audio latents are typically [B, C, T, F]. In SGLang pipelines we commonly
            # represent audio latents as [B, C, T]. For our current (1,1,1) patchify path, we
            # normalize to 5D [B, C, T, 1, 1].
            original_audio_ndim = int(audio_hidden_states.dim())
            if audio_hidden_states.dim() == 3:
                audio_hidden_states = audio_hidden_states[:, :, :, None, None]
            elif audio_hidden_states.dim() == 4:
                # [B, C, T, F] -> [B, C, T, F, 1]
                audio_hidden_states = audio_hidden_states[:, :, :, :, None]
        audio_args = (
            self.audio_args_preprocessor.prepare(
                hidden_states=audio_hidden_states,
                encoder_hidden_states=kwargs.get("audio_encoder_hidden_states")
                or encoder_hidden_states,
                timestep=kwargs.get("audio_timestep") or timestep,
                attention_mask=audio_attention_mask,
                coords=audio_coords,
            )
            if audio_hidden_states is not None
            else None
        )

        # Process transformer blocks.
        video_out, audio_out = self._process_transformer_blocks(video_args, audio_args, perturbations)

        # Process output (upstream: uses embedded_timestep from args).
        # Note: This produces patchified tokens (same as upstream transformer/model.py).
        vx_tokens = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out,
                video_out.x, video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        ax_tokens = (
            self._process_output(
                self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
                audio_out.x, audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )

        if not return_latents:
            return vx_tokens, ax_tokens

        # Unpatchify tokens back to latent tensors for diffusion pipelines.
        vx_latents = None
        if vx_tokens is not None and hidden_states is not None:
            bsz, _c_raw, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.patch_size
            post_t = num_frames // p_t
            post_h = height // p_h
            post_w = width // p_w
            x = vx_tokens.reshape(
                bsz,
                post_t,
                post_h,
                post_w,
                self.out_channels_raw,
                p_t,
                p_h,
                p_w,
            ).permute(0, 4, 1, 5, 2, 6, 3, 7)
            vx_latents = x.reshape(bsz, self.out_channels_raw, num_frames, height, width)

        ax_latents = None
        if ax_tokens is not None and audio_hidden_states is not None:
            # audio_hidden_states is normalized to 5D at this point.
            bsz, _c, t, f, w = audio_hidden_states.shape
            # For audio path we currently use patch_size=(1,1,1), so this is effectively a reshape.
            x = ax_tokens.reshape(
                bsz,
                int(t),
                int(f),
                int(w),
                self.audio_out_channels,
                1,
                1,
                1,
            ).permute(0, 4, 1, 5, 2, 6, 3, 7)
            ax5 = x.reshape(bsz, self.audio_out_channels, t, f, w)
            if original_audio_ndim == 4:
                ax_latents = ax5[..., 0]
            elif original_audio_ndim == 3:
                ax_latents = ax5[:, :, :, 0, 0]
            else:
                ax_latents = ax5

        return vx_latents, ax_latents


LTX2VideoTransformer3DModel = LTXModel
EntryClass = LTXModel