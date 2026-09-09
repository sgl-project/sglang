# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams causal transformer.

Cosmos-Dreams (Cosmos3-Interactive) reuses the Cosmos3 Omni weights but runs
the GEN pathway autoregressively over latent frames. Each frame is packed as
``[action tokens | vision patches]`` (vision patches only for control-video
checkpoints) and every GEN layer takes one softmax over
``[text K/V | committed clean history K/V | current chunk]``. The current
chunk's post-norm, post-RoPE K/V are returned so the pipeline can commit them
as history after a clean refresh pass; control-video clips are committed the
same way, as clean vision frames sharing the target frames' positions.
"""

from collections.abc import Sequence
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    ACTION_CONDITIONING_MODE,
    ATTENTION_MODE_THREE_WAY,
    TEXT_TOKENS_TRAINING_MAX,
    CosmosDreamsManifest,
    load_cosmos_dreams_manifest,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3CrossAttention,
    Cosmos3GenDecoderLayer,
    Cosmos3OmniTransformer,
    _apply_qwen3_qk_norm_rope,
    compute_mrope_position_ids_text,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

KVPair = tuple[torch.Tensor, torch.Tensor]
logger = init_logger(__name__)


class CosmosDreamsForwardOutput(msgspec.Struct):
    """Velocity for the current chunk plus its per-layer ``[1, S, H_kv, D]`` K/V."""

    video: torch.Tensor
    current_kv: list[KVPair]


def interleave_action_vision_tokens(
    action_tokens: torch.Tensor, vision_tokens: torch.Tensor
) -> torch.Tensor:
    """Pack ``[B, T, A, D]`` actions and ``[B, T, P, D]`` patches as ``[B, T*(A+P), D]``."""
    if action_tokens.ndim != 4 or vision_tokens.ndim != 4:
        raise ValueError(
            "Cosmos-Dreams interleaving expects action [B,T,A,D] and vision [B,T,P,D], "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    if (
        action_tokens.shape[:2] != vision_tokens.shape[:2]
        or action_tokens.shape[-1] != vision_tokens.shape[-1]
    ):
        raise ValueError(
            "Cosmos-Dreams action/vision batch, frame, and hidden sizes must match; "
            f"got {tuple(action_tokens.shape)} and {tuple(vision_tokens.shape)}"
        )
    return torch.cat([action_tokens, vision_tokens], dim=2).flatten(1, 2)


def split_interleaved_action_vision_tokens(
    tokens: torch.Tensor,
    *,
    num_frames: int,
    action_tokens_per_frame: int,
    vision_tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_action_vision_tokens`."""
    tokens_per_frame = action_tokens_per_frame + vision_tokens_per_frame
    expected = num_frames * tokens_per_frame
    if tokens.ndim != 3 or tokens.shape[1] != expected:
        raise ValueError(
            f"Cosmos-Dreams packed tokens must have shape [B,{expected},D], "
            f"got {tuple(tokens.shape)}"
        )
    framed = tokens.view(
        tokens.shape[0], num_frames, tokens_per_frame, tokens.shape[-1]
    )
    return framed[:, :, :action_tokens_per_frame], framed[
        :, :, action_tokens_per_frame:
    ]


def null_action_token_positions(
    null_frame_indexes: Sequence[int],
    *,
    num_frames: int,
    tokens_per_frame: int,
    action_tokens_per_frame: int,
) -> list[int]:
    """Flat token positions of the action slots of null-action frames."""
    positions: list[int] = []
    for frame in null_frame_indexes:
        if frame < 0 or frame >= num_frames:
            raise ValueError(
                f"Cosmos-Dreams null action frame {frame} is outside [0, {num_frames})"
            )
        start = frame * tokens_per_frame
        positions.extend(range(start, start + action_tokens_per_frame))
    return positions


def build_cosmos_dreams_position_ids(
    *,
    frame_start: int,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    text_length: int,
    temporal_modality_margin: int,
    fps: float,
    base_fps: float,
    temporal_compression_factor: int,
    action_tokens_per_frame: int,
    null_action_frames: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """mRoPE ``(t, h, w)`` ids, shape ``[3, T*(A+P)]``, in interleaved frame order.

    Vision tokens of latent frame ``f`` sit at ``t = f * base_fps / fps`` past the
    text/modality offset with spatial ids reset per frame. The ``A`` action
    tokens of frame ``f`` cover the pixel steps ending at that frame, so at
    ``fps == base_fps`` they sit at ``f - 0.75, f - 0.5, f - 0.25, f``. When the
    chunk's first frame carries a null action, its action tokens are co-located
    with the frame's vision time; later null frames keep the real-action ids.
    FPS modulation is applied even for a single frame, unlike the Cosmos3
    helper, because per-frame clean commits must match the chunk's positions.
    With ``action_tokens_per_frame == 0`` (control-video checkpoints) only the
    vision ids are returned, shape ``[3, T*P]``.
    """
    if frame_start < 0 or num_frames <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            "Cosmos-Dreams mRoPE dimensions must be positive and frame_start non-negative; "
            f"got start={frame_start}, frames={num_frames}, grid={grid_h}x{grid_w}"
        )
    if fps <= 0 or base_fps <= 0:
        raise ValueError(
            f"Cosmos-Dreams FPS values must be positive, got fps={fps}, base_fps={base_fps}"
        )
    offset = float(text_length + temporal_modality_margin)
    patch_count = grid_h * grid_w
    action_count = action_tokens_per_frame

    frame_tps = fps / temporal_compression_factor
    base_tps = base_fps / temporal_compression_factor
    frames = torch.arange(num_frames, dtype=torch.float32, device=device) + frame_start
    vision_t = (frames / frame_tps * base_tps + offset).view(num_frames, 1)
    vision_h = torch.arange(grid_h, dtype=torch.float32, device=device).view(-1, 1)
    vision_w = torch.arange(grid_w, dtype=torch.float32, device=device).view(1, -1)
    vision_ids = torch.stack(
        [
            vision_t.expand(num_frames, patch_count),
            vision_h.expand(grid_h, grid_w)
            .reshape(1, patch_count)
            .expand(num_frames, -1),
            vision_w.expand(grid_h, grid_w)
            .reshape(1, patch_count)
            .expand(num_frames, -1),
        ],
        dim=0,
    )

    if action_count == 0:
        if null_action_frames:
            raise ValueError(
                "Cosmos-Dreams null action frames require action tokens per frame."
            )
        return vision_ids.flatten(1, 2)
    step_start = frame_start * action_count - action_count + 1
    steps = torch.arange(num_frames * action_count, dtype=torch.float32, device=device)
    action_t = (steps + step_start) / fps * base_tps + offset
    action_ids = torch.stack(
        [action_t, torch.zeros_like(action_t), torch.zeros_like(action_t)]
    )
    action_ids = action_ids.view(3, num_frames, action_count).clone()
    if 0 in set(null_action_frames):
        action_ids[0, 0] = vision_ids[0, 0, 0]
    return torch.cat([action_ids, vision_ids], dim=2).flatten(1, 2)


def _joint_attention(
    attention: Cosmos3CrossAttention,
    hidden: torch.Tensor,
    *,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_positions: torch.Tensor,
    null_token_positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One softmax over ``[prefix (text + history) | current chunk]``.

    Returns the attention output and the chunk's K (post-norm, post-RoPE) and V
    (null-action slots zeroed), which is exactly what history must store.
    """
    batch_size, seq_len = hidden.shape[:2]
    num_q_heads = attention.local_num_attention_heads
    num_kv_heads = attention.local_num_key_value_heads
    qkv, _ = attention.to_qkv(hidden)
    qkv = qkv.view(
        batch_size, seq_len, num_q_heads + 2 * num_kv_heads, attention.head_dim
    )
    q = qkv[:, :, :num_q_heads]
    k = qkv[:, :, num_q_heads : num_q_heads + num_kv_heads]
    v = qkv[:, :, num_q_heads + num_kv_heads :]
    q, k = _apply_qwen3_qk_norm_rope(
        q,
        k,
        attention.norm_q,
        attention.norm_k,
        attention.head_dim,
        cos_sin_cache,
        rope_positions,
    )
    k = k.contiguous()
    v = v.contiguous()
    if null_token_positions:
        v[:, null_token_positions] = 0.0
    out = attention.attn(
        q, torch.cat([prefix_k, k], dim=1), torch.cat([prefix_v, v], dim=1)
    )
    out, _ = attention.to_out(out.reshape(batch_size, seq_len, -1))
    return out, k, v


def _gen_layer_forward(
    layer: Cosmos3GenDecoderLayer, hidden: torch.Tensor, **attention_kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = hidden
    attention_output, k, v = _joint_attention(
        layer.cross_attention, layer.input_layernorm(hidden), **attention_kwargs
    )
    hidden = residual + attention_output
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, k, v


def _prefix_kv(text_kv: KVPair, history_kv: KVPair | None) -> KVPair:
    text_k, text_v = text_kv
    if history_kv is None or history_kv[0].shape[1] == 0:
        return text_k, text_v
    history_k, history_v = history_kv
    return torch.cat([text_k, history_k], dim=1), torch.cat([text_v, history_v], dim=1)


class CosmosDreamsTransformer(Cosmos3OmniTransformer):
    """Cosmos3 Omni weights driven as a causal world model.

    Weight names, projections, and the UND text pathway are inherited unchanged;
    only the GEN forward differs (per-frame action/vision interleaving for
    action-conditioned checkpoints, joint attention with committed history, no
    time embedding on clean tokens). Control-video checkpoints carry no action
    tokens; their control clips enter as clean vision frames.
    """

    def __init__(
        self,
        config: Cosmos3VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        self.manifest: CosmosDreamsManifest = load_cosmos_dreams_manifest(hf_config)
        self._validate_causal_arch(hf_config)
        self.action_tokens_per_frame = self.manifest.action_tokens_per_frame
        self.text_cache_max_len = self.manifest.text_cache_max_len

    def _validate_causal_arch(self, hf_config: dict[str, Any]) -> None:
        expected_flags = {
            "joint_attn_implementation": ATTENTION_MODE_THREE_WAY,
            "video_temporal_causal": True,
            "unified_3d_mrope_reset_spatial_ids": True,
        }
        for key, expected in expected_flags.items():
            if hf_config.get(key) != expected:
                raise ValueError(
                    f"Unsupported Cosmos-Dreams transformer config: {key}={hf_config.get(key)!r}; "
                    f"expected {expected!r}."
                )
        if self.sound_gen:
            raise ValueError("Cosmos-Dreams does not support joint sound generation.")
        manifest = self.manifest
        mismatches = {
            "temporal_compression_factor": (
                self.temporal_compression_factor,
                manifest.temporal_compression_factor,
            ),
            "latent_patch_size": (self.latent_patch_size, manifest.latent_patch_size),
            "temporal_modality_margin": (
                self.temporal_margin,
                manifest.temporal_modality_margin,
            ),
            "base_fps": (float(self.base_fps), float(manifest.base_fps)),
        }
        if manifest.conditioning_mode == ACTION_CONDITIONING_MODE:
            if not self.config.action_gen:
                raise ValueError(
                    "Action-conditioned Cosmos-Dreams checkpoints must enable action_gen."
                )
            mismatches["action_dim"] = (self.action_dim, manifest.max_action_dim)
            mismatches["num_embodiment_domains"] = (
                self.num_embodiment_domains,
                manifest.action_contract.num_embodiment_domains,
            )
        for name, (arch_value, manifest_value) in mismatches.items():
            if arch_value != manifest_value:
                raise ValueError(
                    f"Cosmos-Dreams {name} differs between transformer config and manifest: "
                    f"{arch_value} != {manifest_value}."
                )
        if get_sp_world_size() > 1:
            raise ValueError(
                "Cosmos-Dreams supports tensor parallelism but not sequence parallelism."
            )

    def encode_und_kv(
        self, text_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> tuple[list[KVPair], int]:
        """Run the UND pathway once; return per-layer K/V trimmed to the real prompt."""
        if (
            text_ids.ndim != 2
            or text_ids.shape[0] != 1
            or text_ids.shape != text_mask.shape
        ):
            raise ValueError(
                "Cosmos-Dreams text ids/mask must be matching [1, S] tensors, got "
                f"{tuple(text_ids.shape)} and {tuple(text_mask.shape)}."
            )
        real_len = int(text_mask[0].sum().item())
        if real_len <= 0:
            raise ValueError(
                "Cosmos-Dreams prompts must contain at least one text token."
            )
        if not bool(text_mask[0, :real_len].all()):
            raise ValueError("Cosmos-Dreams text mask must be a prefix of ones.")
        # +2 for the EOS and vision-start tokens the packer appends after truncation.
        if real_len > TEXT_TOKENS_TRAINING_MAX + 2:
            raise ValueError(
                f"Cosmos-Dreams prompt has {real_len} tokens; training truncated text at "
                f"{TEXT_TOKENS_TRAINING_MAX} tokens."
            )
        if real_len > self.text_cache_max_len:
            logger.warning(
                "Cosmos-Dreams prompt has %d tokens, above the exported text_cache_max_len=%d; "
                "the dense history here handles it, the reference paged engine would not.",
                real_len,
                self.text_cache_max_len,
            )
        text_ids = text_ids[:, :real_len]
        text_mask = text_mask[:, :real_len]
        position_ids, _ = compute_mrope_position_ids_text(
            real_len, temporal_offset=0, device=text_ids.device
        )
        text_kv = self.language_model(text_ids, text_mask, position_ids.unsqueeze(1))
        return text_kv, real_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_kv: list[KVPair],
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        history_kv: list[KVPair] | None = None,
        condition_vision: bool = False,
        null_action_frame_indexes: Sequence[int] = (),
    ) -> CosmosDreamsForwardOutput:
        """Denoise (``condition_vision=False``) or clean-refresh one current chunk.

        ``hidden_states`` is ``[1, C, T, H, W]`` for latent frames
        ``[frame_start, frame_start + T)``; ``action_latents`` is
        ``[1, T * A, action_dim]`` (normalized, zero-padded) and must be omitted
        for control-video checkpoints. Clean-refresh passes skip the time
        embedding so the returned K/V are clean history; a control clip is
        committed with ``condition_vision=True`` exactly like a clean frame.
        """
        self._validate_forward_inputs(
            hidden_states, timestep, text_kv=text_kv, history_kv=history_kv
        )
        self._validate_action_inputs(
            action_latents, action_domain_ids, null_action_frame_indexes
        )
        _, _, num_frames, latent_h, latent_w = hidden_states.shape
        grid_h, grid_w, _, _ = self._pad_to_patch_size(latent_h, latent_w)
        patch_count = grid_h * grid_w
        action_count = self.action_tokens_per_frame
        hidden = self._embed_chunk(
            hidden_states,
            timestep,
            action_latents=action_latents,
            action_domain_ids=action_domain_ids,
            condition_vision=condition_vision,
            patch_count=patch_count,
        )

        position_ids = build_cosmos_dreams_position_ids(
            frame_start=frame_start,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            text_length=text_kv[0][0].shape[1],
            temporal_modality_margin=self.temporal_margin,
            fps=fps,
            base_fps=self.base_fps,
            temporal_compression_factor=self.temporal_compression_factor,
            action_tokens_per_frame=action_count,
            null_action_frames=null_action_frame_indexes,
            device=hidden.device,
        )
        cos_sin_cache, rope_positions = (
            self.language_model.rotary_emb.build_rope_cache_inputs(
                position_ids.unsqueeze(1), cache_dtype=hidden.dtype
            )
        )
        null_positions = null_action_token_positions(
            null_action_frame_indexes,
            num_frames=num_frames,
            tokens_per_frame=action_count + patch_count,
            action_tokens_per_frame=action_count,
        )

        current_kv: list[KVPair] = []
        for layer_idx, layer in enumerate(self.gen_layers):
            prefix_k, prefix_v = _prefix_kv(
                text_kv[layer_idx],
                None if history_kv is None else history_kv[layer_idx],
            )
            hidden, k, v = _gen_layer_forward(
                layer,
                hidden,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                cos_sin_cache=cos_sin_cache,
                rope_positions=rope_positions,
                null_token_positions=null_positions,
            )
            current_kv.append((k, v))

        hidden = self.norm_moe_gen(hidden)
        _, vision_hidden = split_interleaved_action_vision_tokens(
            hidden,
            num_frames=num_frames,
            action_tokens_per_frame=action_count,
            vision_tokens_per_frame=patch_count,
        )
        video, _ = self.proj_out(vision_hidden.flatten(1, 2))
        return CosmosDreamsForwardOutput(
            video=self.unpatchify(video, num_frames, latent_h, latent_w),
            current_kv=current_kv,
        )

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_kv: list[KVPair],
        history_kv: list[KVPair] | None,
    ) -> None:
        if hidden_states.ndim != 5 or hidden_states.shape[0] != 1:
            raise ValueError(
                f"Cosmos-Dreams hidden_states must have shape [1,C,T,H,W], got {tuple(hidden_states.shape)}"
            )
        if timestep.numel() != 1:
            raise ValueError(
                f"Cosmos-Dreams timestep must be a scalar, got {tuple(timestep.shape)}"
            )
        if len(text_kv) != self.num_hidden_layers:
            raise ValueError(
                f"Cosmos-Dreams expected {self.num_hidden_layers} text KV layers, got {len(text_kv)}"
            )
        if history_kv is not None and len(history_kv) != self.num_hidden_layers:
            raise ValueError(
                f"Cosmos-Dreams expected {self.num_hidden_layers} history KV layers, got {len(history_kv)}"
            )

    def _validate_action_inputs(
        self,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
        null_action_frame_indexes: Sequence[int],
    ) -> None:
        if self.action_tokens_per_frame > 0:
            if action_latents is None or action_domain_ids is None:
                raise ValueError(
                    "Action-conditioned Cosmos-Dreams checkpoints require action_latents "
                    "and action_domain_ids on every forward."
                )
            return
        if action_latents is not None or action_domain_ids is not None:
            raise ValueError(
                "Cosmos-Dreams control-video checkpoints take no action inputs."
            )
        if null_action_frame_indexes:
            raise ValueError(
                "Cosmos-Dreams control-video checkpoints have no null-action frames."
            )

    def _embed_chunk(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
        condition_vision: bool,
        patch_count: int,
    ) -> torch.Tensor:
        """Project vision patches and actions, then pack ``[action | vision]`` per frame."""
        _, _, num_frames, latent_h, latent_w = hidden_states.shape
        action_count = self.action_tokens_per_frame
        vision_tokens, _ = self.proj_in(
            self.patchify(hidden_states, num_frames, latent_h, latent_w)
        )
        vision_tokens = vision_tokens.view(1, num_frames, patch_count, self.hidden_size)
        if not condition_vision:
            time_embed = self.time_embedder(timestep.reshape(1).float())
            vision_tokens = vision_tokens + time_embed.to(vision_tokens.dtype).view(
                1, 1, 1, -1
            )
        if action_count == 0:
            return vision_tokens.flatten(1, 2)
        expected_action_shape = (1, num_frames * action_count, self.action_dim)
        if tuple(action_latents.shape) != expected_action_shape:
            raise ValueError(
                f"Cosmos-Dreams actions must have shape {expected_action_shape}, "
                f"got {tuple(action_latents.shape)}"
            )
        action_hidden = self.action_proj_in(
            action_latents.to(vision_tokens.dtype), action_domain_ids
        )
        action_hidden = action_hidden + self.action_modality_embed.to(
            action_hidden.dtype
        )
        return interleave_action_vision_tokens(
            action_hidden.view(1, num_frames, action_count, self.hidden_size),
            vision_tokens,
        )


EntryClass = CosmosDreamsTransformer
