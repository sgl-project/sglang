# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: mossVG/mova/diffusion/models/interactionv2.py


from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
    MOVADualTowerConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@torch.no_grad()
def compute_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos/sin embeddings for given position IDs.

    This is a functional implementation that doesn't require storing buffers,
    making it compatible with FSDP meta device initialization.

    Args:
        position_ids: Position IDs tensor [B, L] or [1, L]
        head_dim: Dimension of each attention head
        base: RoPE base frequency (default: 10000.0)
        device: Target device
        dtype: Output dtype

    Returns:
        (cos, sin): Each with shape [B, L, head_dim]
    """
    device = device or position_ids.device
    dtype = dtype or torch.float32

    # Compute inverse frequencies
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    # Expand for batch computation: [B, L] -> [B, 1, L] @ [1, head_dim/2, 1] -> [B, head_dim/2, L]
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    # Compute frequencies: [B, head_dim/2, L] -> [B, L, head_dim/2]
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

    # Double the frequencies for full head_dim: [B, L, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)

    return cos, sin


class PerFrameAttentionPooling(nn.Module):
    """Per-frame multi-head attention pooling.

    Flattens the input sequence [B, L, D] and grid size (T, H, W).
    Performs single-query attention pooling on the H*W tokens for each time frame.
    Output shape: [B, T, D].
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads

        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.probe, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, L, D], where L = T * H * W.
            grid_size: Tuple of (T, H, W).

        Returns:
            Pooled tensor of shape [B, T, D].
        """
        B, L, D = x.shape
        T, H, W = grid_size
        assert (
            D == self.dim
        ), f"Input dimension D={D} does not match module dim={self.dim}"
        assert L == T * H * W, f"Flattened length L={L} does not match T*H*W={T*H*W}"

        S = H * W
        x_bt_s_d = x.view(B, T, S, D).contiguous().view(B * T, S, D)  # [B*T, S, D]
        probe = self.probe.expand(B * T, -1, -1)  # [B*T, 1, D]

        pooled_bt_1_d = self.attention(probe, x_bt_s_d, x_bt_s_d, need_weights=False)[0]
        pooled_bt_d = pooled_bt_1_d.squeeze(1)  # [B*T, D]

        pooled = pooled_bt_d.view(B, T, D)
        pooled = self.layernorm(pooled)
        return pooled


class CrossModalInteractionController:
    """Strategy class to control dual-tower interaction.

    Manages the interaction mapping between Visual DiT (e.g., 30 layers)
    and Audio DiT (e.g., 30 layers).
    """

    def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
        self.visual_layers = visual_layers
        self.audio_layers = audio_layers
        self.min_layers = min(visual_layers, audio_layers)

    def get_interaction_layers(
        self, strategy: str = "shallow_focus"
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Gets the mapping relationship of interaction layers."""
        if strategy == "shallow_focus":
            num_interact = min(10, self.min_layers // 3)
            interact_layers = list(range(0, num_interact))
        elif strategy == "distributed":
            step = 3
            interact_layers = list(range(0, self.min_layers, step))
        elif strategy == "progressive":
            shallow = list(range(0, min(8, self.min_layers)))
            if self.min_layers > 8:
                deep = list(range(8, self.min_layers, 3))
                interact_layers = shallow + deep
            else:
                interact_layers = shallow
        elif strategy == "custom":
            interact_layers = [0, 2, 4, 6, 8, 12, 16, 20]
            interact_layers = [i for i in interact_layers if i < self.min_layers]
        elif strategy == "full":
            interact_layers = list(range(0, self.min_layers))
        else:
            raise ValueError(f"Unknown interaction strategy: {strategy}")

        mapping = {
            "v2a": [(i, i) for i in interact_layers],
            "a2v": [(i, i) for i in interact_layers],
        }
        return mapping

    def should_interact(
        self, layer_idx: int, direction: str, interaction_mapping: Dict
    ) -> bool:
        """Determines if the specified layer needs to interact."""
        if direction not in interaction_mapping:
            return False
        return any(src == layer_idx for src, _ in interaction_mapping[direction])


class ConditionalCrossAttention(nn.Module):
    """
    Cross-modal attention for dual-tower bridge with Tensor Parallel support.

    This module handles attention between video and audio hidden states,
    which have different sequence lengths.
    """

    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q_dim = dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = self.q_dim // num_heads

        self.tp_size = get_tp_world_size()
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})."
            )
        self.num_heads_per_rank = self.num_heads // self.tp_size

        # TP strategy: shard Q/K/V over heads (column-parallel), then row-parallel output.
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.k = ColumnParallelLinear(kv_dim, dim, bias=True, gather_output=False)
        self.v = ColumnParallelLinear(kv_dim, dim, bias=True, gather_output=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = USPAttention(
            num_heads=self.num_heads_per_rank,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        ctx = y
        q, _ = self.q(x)
        k, _ = self.k(ctx)
        v, _ = self.v(ctx)

        # RMSNorm over sharded hidden dimension
        if self.tp_size > 1:
            q = tensor_parallel_rms_norm(q, self.norm_q)
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if x_freqs is not None:
            x_cos, x_sin = x_freqs
            q_view = rearrange(q, "b l (h d) -> b l h d", d=self.head_dim)
            x_cos = x_cos.to(q_view.dtype).to(q_view.device).squeeze(0)
            x_sin = x_sin.to(q_view.dtype).to(q_view.device).squeeze(0)
            # FlashInfer expects cos_sin_cache with shape [seqlen, head_dim],
            # where the first half is cos and the second half is sin, each with
            # head_dim//2 elements. Since compute_rope_cos_sin duplicates the
            # frequencies (cat((freqs, freqs))), we only take the first half.
            half_dim = self.head_dim // 2
            cos_sin_cache = torch.cat(
                [
                    x_cos[:, :half_dim].to(dtype=torch.float32).contiguous(),
                    x_sin[:, :half_dim].to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            q_view, _ = apply_flashinfer_rope_qk_inplace(
                q_view, q_view.clone(), cos_sin_cache, is_neox=True
            )
            q = rearrange(q_view, "b l h d -> b l (h d)")

        if y_freqs is not None:
            y_cos, y_sin = y_freqs
            k_view = rearrange(k, "b l (h d) -> b l h d", d=self.head_dim)
            y_cos = y_cos.to(k_view.dtype).to(k_view.device).squeeze(0)
            y_sin = y_sin.to(k_view.dtype).to(k_view.device).squeeze(0)
            # FlashInfer expects cos_sin_cache with shape [seqlen, head_dim],
            # where the first half is cos and the second half is sin, each with
            # head_dim//2 elements. Since compute_rope_cos_sin duplicates the
            # frequencies (cat((freqs, freqs))), we only take the first half.
            half_dim = self.head_dim // 2
            cos_sin_cache = torch.cat(
                [
                    y_cos[:, :half_dim].to(dtype=torch.float32).contiguous(),
                    y_sin[:, :half_dim].to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            k_view, _ = apply_flashinfer_rope_qk_inplace(
                k_view, k_view.clone(), cos_sin_cache, is_neox=True
            )
            k = rearrange(k_view, "b l h d -> b l (h d)")

        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads_per_rank)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads_per_rank)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads_per_rank)

        x = self.attn(q, k, v)
        x = rearrange(x, "b l h d -> b l (h d)")
        x, _ = self.o(x)
        return x


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb, _ = self.linear(self.silu(temb))

        if self.chunk_dim == 2:
            scale, shift = temb.chunk(2, dim=2)
        elif self.chunk_dim == 1:
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class ConditionalCrossAttentionBlock(nn.Module):
    """A wrapper block for ConditionalCrossAttention that applies LayerNorm to the condition input y."""

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        pooled_adaln: bool = False,
    ):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(
            dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps
        )
        self.pooled_adaln = pooled_adaln
        if pooled_adaln:
            self.per_frame_pooling = PerFrameAttentionPooling(
                kv_dim, num_heads=num_heads, eps=eps
            )
            self.adaln = AdaLayerNorm(kv_dim, output_dim=dim * 2, chunk_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if self.pooled_adaln:
            assert video_grid_size is not None, "video_grid_size cannot be None"
            pooled_y = self.per_frame_pooling(y, video_grid_size)
            if pooled_y.shape[1] != x.shape[1]:
                pooled_y = F.interpolate(
                    pooled_y.permute(0, 2, 1),
                    size=x.shape[1],
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            x = self.adaln(x, temb=pooled_y)
        y = self.y_norm(y)
        return self.inner(x=x, y=y, x_freqs=x_freqs, y_freqs=y_freqs)


class DualTowerConditionalBridge(
    CachableDiT,
    OffloadableDiTMixin,
):
    """Dual-tower conditional bridge module v2 (SGLang optimized version).

    Implements the correct architecture:
    1. Audio latents -> Audio DiT -> Audio hidden states [B, L, 1536].
    2. Visual latents -> Visual DiT -> Visual hidden states [B, L, 5120].
    3. Cross-attention interaction between the hidden states of the two DiTs.
    """

    _fsdp_shard_conditions = MOVADualTowerConfig()._fsdp_shard_conditions
    _compile_conditions = MOVADualTowerConfig()._compile_conditions
    _supported_attention_backends = MOVADualTowerConfig()._supported_attention_backends
    param_names_mapping = MOVADualTowerConfig().param_names_mapping
    reverse_param_names_mapping = MOVADualTowerConfig().reverse_param_names_mapping
    lora_param_names_mapping = MOVADualTowerConfig().lora_param_names_mapping

    def __init__(
        self,
        config: MOVADualTowerConfig | None = None,
        hf_config: dict[str, Any] | None = None,
        # Fallback parameters for from_pretrained compatibility
        visual_layers: int = 40,
        audio_layers: int = 30,
        visual_hidden_dim: int = 5120,
        audio_hidden_dim: int = 1536,
        audio_fps: float = 50.0,
        head_dim: int = 128,
        interaction_strategy: str = "full",
        apply_cross_rope: bool = True,
        apply_first_frame_bias_in_rope: bool = False,
        trainable_condition_scale: bool = False,
        pooled_adaln: bool = False,
    ):
        super().__init__(config=config, hf_config=hf_config)

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            visual_layers = config.visual_layers
            audio_layers = config.audio_layers
            visual_hidden_dim = config.visual_hidden_dim
            audio_hidden_dim = config.audio_hidden_dim
            audio_fps = config.audio_fps
            head_dim = config.head_dim
            interaction_strategy = config.interaction_strategy
            apply_cross_rope = config.apply_cross_rope
            apply_first_frame_bias_in_rope = config.apply_first_frame_bias_in_rope
            trainable_condition_scale = config.trainable_condition_scale
            pooled_adaln = config.pooled_adaln

        self.visual_hidden_dim = visual_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
        self.trainable_condition_scale = trainable_condition_scale
        self.pooled_adaln = pooled_adaln

        if self.trainable_condition_scale:
            self.condition_scale = nn.Parameter(
                torch.tensor([1.0], dtype=torch.float32)
            )
        else:
            self.condition_scale = 1.0

        self.controller = CrossModalInteractionController(visual_layers, audio_layers)
        self.interaction_mapping = self.controller.get_interaction_layers(
            interaction_strategy
        )

        # Cross-modal attention modules - interaction at DiT hidden states level
        self.audio_to_video_conditioners = nn.ModuleDict()
        self.video_to_audio_conditioners = nn.ModuleDict()

        self.rope_base = 10000.0  # RoPE base frequency hardcode. adapted from original mova implementation.

        # Audio DiT hidden states conditioning Video DiT
        for v_layer, _ in self.interaction_mapping["a2v"]:
            self.audio_to_video_conditioners[str(v_layer)] = (
                ConditionalCrossAttentionBlock(
                    dim=visual_hidden_dim,
                    kv_dim=audio_hidden_dim,
                    num_heads=visual_hidden_dim // head_dim,
                    pooled_adaln=False,
                )
            )

        # Visual DiT hidden states conditioning Audio DiT
        for a_layer, _ in self.interaction_mapping["v2a"]:
            self.video_to_audio_conditioners[str(a_layer)] = (
                ConditionalCrossAttentionBlock(
                    dim=audio_hidden_dim,
                    kv_dim=visual_hidden_dim,
                    num_heads=audio_hidden_dim // head_dim,
                    pooled_adaln=self.pooled_adaln,
                )
            )

        # Required attributes for CachableDiT/BaseDiT
        self.hidden_size = visual_hidden_dim
        self.num_attention_heads = visual_hidden_dim // head_dim
        self.num_channels_latents = (
            visual_hidden_dim  # Bridge doesn't output latents, but required by BaseDiT
        )
        self.layer_names = [
            "audio_to_video_conditioners",
            "video_to_audio_conditioners",
        ]
        self.__post_init__()

    @torch.no_grad()
    def build_aligned_freqs(
        self,
        video_fps: float,
        grid_size: Tuple[int, int, int],
        audio_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Generates aligned RoPE (cos, sin) based on video FPS, grid size, and audio length.

        Uses functional RoPE computation to avoid FSDP meta device issues.

        Args:
            video_fps: FPS of the video.
            grid_size: Tuple of (f_v, h, w).
            audio_steps: Length of the audio sequence.
            device: Target device.
            dtype: Output dtype.

        Returns:
            A tuple of ((cos_v, sin_v), (cos_a, sin_a)).
        """
        f_v, h, w = grid_size
        L_v = f_v * h * w
        L_a = int(audio_steps)

        device = device or next(self.parameters()).device
        dtype = dtype or torch.float32

        # Audio positions: 0, 1, 2, ..., L_a-1
        audio_pos = torch.arange(L_a, device=device, dtype=torch.float32).unsqueeze(0)

        # Video positions: Align video frames to audio step units
        if self.apply_first_frame_bias_in_rope:
            video_effective_fps = float(video_fps) / 4.0
            if f_v > 0:
                t_starts = torch.zeros((f_v,), device=device, dtype=torch.float32)
                if f_v > 1:
                    t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(
                        f_v - 1, device=device, dtype=torch.float32
                    ) * (1.0 / video_effective_fps)
            else:
                t_starts = torch.zeros((0,), device=device, dtype=torch.float32)
            video_pos_per_frame = t_starts * float(self.audio_fps)
        else:
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = (
                torch.arange(f_v, device=device, dtype=torch.float32) * scale
            )

        video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)

        # Use functional RoPE to compute cos/sin
        cos_v, sin_v = compute_rope_cos_sin(
            video_pos, self.head_dim, base=self.rope_base, device=device, dtype=dtype
        )
        cos_a, sin_a = compute_rope_cos_sin(
            audio_pos, self.head_dim, base=self.rope_base, device=device, dtype=dtype
        )

        return (cos_v, sin_v), (cos_a, sin_a)

    def should_interact(self, layer_idx: int, direction: str) -> bool:
        return self.controller.should_interact(
            layer_idx, direction, self.interaction_mapping
        )

    def apply_conditional_control(
        self,
        layer_idx: int,
        direction: str,
        primary_hidden_states: torch.Tensor,
        condition_hidden_states: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Applies conditional control at the DiT hidden states level."""
        if not self.controller.should_interact(
            layer_idx, direction, self.interaction_mapping
        ):
            return primary_hidden_states

        if direction == "a2v":
            conditioner = self.audio_to_video_conditioners[str(layer_idx)]
        elif direction == "v2a":
            conditioner = self.video_to_audio_conditioners[str(layer_idx)]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        conditioned_features = conditioner(
            x=primary_hidden_states,
            y=condition_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            video_grid_size=video_grid_size,
        )

        if self.trainable_condition_scale and condition_scale is not None:
            logger.warning(
                "The current model has a trainable condition_scale, but condition_scale "
                "was passed externally. Ignoring the trainable condition_scale and "
                "using the external condition_scale=%s.",
                condition_scale,
            )

        scale = condition_scale if condition_scale is not None else self.condition_scale

        primary_hidden_states = primary_hidden_states + conditioned_features * scale

        return primary_hidden_states

    def forward(
        self,
        layer_idx: int,
        visual_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs bidirectional conditional control for both visual and audio towers."""
        visual_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="a2v",
            primary_hidden_states=visual_hidden_states,
            condition_hidden_states=audio_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            condition_scale=(
                a2v_condition_scale
                if a2v_condition_scale is not None
                else condition_scale
            ),
            video_grid_size=video_grid_size,
        )

        audio_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="v2a",
            primary_hidden_states=audio_hidden_states,
            condition_hidden_states=visual_hidden_states,
            x_freqs=y_freqs,
            y_freqs=x_freqs,
            condition_scale=(
                v2a_condition_scale
                if v2a_condition_scale is not None
                else condition_scale
            ),
            video_grid_size=video_grid_size,
        )

        return visual_conditioned, audio_conditioned


EntryClass = DualTowerConditionalBridge
