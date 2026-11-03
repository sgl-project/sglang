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
from sglang.srt.runtime_context import get_stream

KVPair = tuple[torch.Tensor, torch.Tensor]
# Per-layer history: legacy lists or request-local buffers.
HistoryKV = Sequence[KVPair]
logger = init_logger(__name__)


class CosmosDreamsKVCache(Sequence[KVPair]):
    """Keep text, committed history, and current K/V in one buffer per layer.

    Attention overwrites only the current suffix. A clean refresh is committed
    explicitly; denoising never changes the visible history. Sliding compaction
    uses one scratch tensor shared by all layers, rather than a second full KV
    cache. Compaction on a separate stream prepares each layer; the next
    forward waits before accessing its history or current suffix.
    """

    def __init__(
        self,
        text_kv: Sequence[KVPair],
        *,
        tokens_per_frame: int,
        sink_frames: int,
        window_frames: int,
        max_current_frames: int,
        max_history_frames: int | None = None,
    ) -> None:
        if not text_kv:
            raise ValueError("Cosmos-Dreams KV cache requires at least one layer.")
        if (
            min(tokens_per_frame, window_frames, max_current_frames) <= 0
            or sink_frames < 0
        ):
            raise ValueError("Cosmos-Dreams KV cache dimensions must be positive.")
        first = text_kv[0][0]
        if first.ndim != 4 or first.shape[0] != 1:
            raise ValueError("Cosmos-Dreams text KV must have shape [1,S,H,D].")
        self.text_tokens = first.shape[1]
        self.sink_tokens = sink_frames * tokens_per_frame
        self.window_tokens = window_frames * tokens_per_frame
        self.history_capacity = self.sink_tokens + self.window_tokens
        if max_history_frames is not None:
            if max_history_frames < 0:
                raise ValueError(
                    "Cosmos-Dreams request history capacity cannot be negative."
                )
            self.history_capacity = min(
                self.history_capacity, max_history_frames * tokens_per_frame
            )
        self.current_capacity = max_current_frames * tokens_per_frame
        self.history_tokens = 0
        capacity = self.text_tokens + self.history_capacity + self.current_capacity
        shape = (2, 1, capacity, *first.shape[-2:])
        self._buffers: list[torch.Tensor] = []
        for key, value in text_kv:
            if any(
                t.shape != first.shape
                or t.dtype != first.dtype
                or t.device != first.device
                for t in (key, value)
            ):
                raise ValueError(
                    "Cosmos-Dreams KV cache layers must have matching geometry."
                )
            buffer = first.new_empty(shape)
            buffer[0, :, : self.text_tokens].copy_(key)
            buffer[1, :, : self.text_tokens].copy_(value)
            self._buffers.append(buffer)
        scratch_tokens = (
            self.window_tokens
            if self.history_capacity == self.sink_tokens + self.window_tokens
            else 0
        )
        self._scratch = first.new_empty((2, 1, scratch_tokens, *first.shape[-2:]))
        self._stream = (
            get_stream("cosmos3_kv_compaction")
            if first.is_cuda and scratch_tokens
            else None
        )
        self._ready = (
            [torch.cuda.Event() for _ in self._buffers] if self._stream else []
        )
        self._pending = [False] * len(self)

    def __enter__(self) -> "CosmosDreamsKVCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # Drain even on failure before the request releases buffers or scratch.
        if self._stream is not None:
            self._stream.synchronize()
            self._pending = [False] * len(self)

    def _wait_layer(self, index: int) -> None:
        if self._pending[index]:
            torch.cuda.current_stream(self._buffers[index].device).wait_event(
                self._ready[index]
            )
            self._pending[index] = False

    def __len__(self) -> int:
        return len(self._buffers)

    def __getitem__(self, index: int) -> KVPair:
        """Expose only committed history, without the text or current suffix."""
        self._wait_layer(index)
        buffer = self._buffers[index]
        start = self.text_tokens
        end = start + self.history_tokens
        return buffer[0, :, start:end], buffer[1, :, start:end]

    def attention_buffer(self, layer_index: int, current_tokens: int) -> KVPair:
        """Return text + history + current views; the caller fills current K/V."""
        if not 0 < current_tokens <= self.current_capacity:
            raise ValueError("Cosmos-Dreams current KV exceeds its reserved capacity.")
        self._wait_layer(layer_index)
        end = self.text_tokens + self.history_tokens + current_tokens
        buffer = self._buffers[layer_index]
        return buffer[0, :, :end], buffer[1, :, :end]

    def load_selected(
        self,
        archive: "CosmosDreamsKVCache",
        frame_ids: Sequence[int],
        tokens_per_frame: int,
    ) -> None:
        """Copy a chunk's selected history once, ready one layer at a time."""
        ids = list(frame_ids)
        stored = archive.history_tokens // tokens_per_frame
        if ids != sorted(set(ids)) or (ids and (ids[0] < 0 or ids[-1] >= stored)):
            raise ValueError(
                "Selected frames must be ordered unique committed frames, "
                f"got {ids} with {stored} committed."
            )
        count = len(ids) * tokens_per_frame
        if count > self.history_capacity:
            raise ValueError("Cosmos-Dreams selected history exceeds its capacity.")
        device = self._buffers[0].device
        spans = []
        for frame in ids:
            if spans and frame == spans[-1][1]:
                spans[-1][1] += 1
            else:
                spans.append([frame, frame + 1])
        if device.type == "cuda":
            if self._stream is None:
                self._stream = get_stream("cosmos3_kv_gather")
                self._ready = [torch.cuda.Event() for _ in self._buffers]
            # Protect archive writes and the preceding chunk's reads before
            # reusing this working set.
            self._stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._stream):
            for index, (source, dest) in enumerate(
                zip(archive._buffers, self._buffers, strict=True)
            ):
                dest_start = self.text_tokens
                for start, end in spans:
                    source_start = archive.text_tokens + start * tokens_per_frame
                    tokens = (end - start) * tokens_per_frame
                    for kind in range(2):
                        dest[kind, :, dest_start : dest_start + tokens].copy_(
                            source[kind, :, source_start : source_start + tokens],
                            non_blocking=True,
                        )
                    dest_start += tokens
                if self._stream is not None:
                    self._ready[index].record()
        if self._stream is not None:
            self._pending = [True] * len(self)
        self.history_tokens = count

    def commit(self, frame_tokens: int) -> None:
        """Commit one clean frame already written by the preceding forward.

        That forward consumes each layer's ready dependency before writing the
        suffix. Commit only advances or compacts history; it never imports K/V.
        """
        if not 0 < frame_tokens <= min(self.current_capacity, self.window_tokens):
            raise ValueError("Cosmos-Dreams clean frame exceeds its reserved capacity.")
        if self.history_tokens + frame_tokens <= self.history_capacity:
            self.history_tokens += frame_tokens
            return
        if self.history_capacity < self.sink_tokens + self.window_tokens:
            raise ValueError("Cosmos-Dreams request history capacity exceeded.")
        tail_old = self.window_tokens - frame_tokens
        old_start = self.text_tokens + self.history_tokens - tail_old
        dest_start = self.text_tokens + self.sink_tokens
        new_start = self.text_tokens + self.history_tokens
        new_dest = dest_start + tail_old
        if self._stream is not None:
            # All layers must finish reading the old history before any move.
            self._stream.wait_stream(torch.cuda.current_stream(self._buffers[0].device))
        with torch.cuda.stream(self._stream):
            for index, buffer in enumerate(self._buffers):
                # Copy K/V separately so each slice is contiguous.
                for kind in range(2):
                    if tail_old:
                        # The source and destination overlap; stage via scratch.
                        scratch = self._scratch[kind, :, :tail_old]
                        scratch.copy_(
                            buffer[kind, :, old_start : old_start + tail_old],
                            non_blocking=True,
                        )
                        buffer[kind, :, dest_start : dest_start + tail_old].copy_(
                            scratch, non_blocking=True
                        )
                    buffer[kind, :, new_dest : new_dest + frame_tokens].copy_(
                        buffer[kind, :, new_start : new_start + frame_tokens],
                        non_blocking=True,
                    )
                if self._stream is not None:
                    self._ready[index].record()
        if self._stream is not None:
            self._pending = [True] * len(self)
        self.history_tokens = self.history_capacity


class CosmosDreamsForwardOutput(msgspec.Struct):
    """Velocity and per-layer ``[1, S, H_kv, D]`` K/V for the current chunk.

    With a request KV cache, these K/V alias its current suffix until the next
    forward or history commit.
    """

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


def null_action_token_slices(
    null_frame_indexes: Sequence[int],
    *,
    num_frames: int,
    tokens_per_frame: int,
    action_tokens_per_frame: int,
) -> list[slice]:
    """Flat token slices of the action slots of null-action frames.

    Slices, not a position list: indexing a device tensor with a list uploads the
    list with a blocking copy, and the slots are zeroed in every layer.
    """
    slices: list[slice] = []
    for frame in null_frame_indexes:
        if frame < 0 or frame >= num_frames:
            raise ValueError(
                f"Cosmos-Dreams null action frame {frame} is outside [0, {num_frames})"
            )
        start = frame * tokens_per_frame
        slices.append(slice(start, start + action_tokens_per_frame))
    return slices


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
    text_kv: KVPair,
    history_kv: KVPair | None,
    cos_sin_cache: torch.Tensor,
    rope_positions: torch.Tensor,
    null_token_slices: list[slice],
    kv_buffer: KVPair | None = None,
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
    for null_slice in null_token_slices:
        v[:, null_slice] = 0.0
    if kv_buffer is None:
        text_k, text_v = text_kv
        keys, values = [text_k], [text_v]
        if history_kv is not None:
            keys.append(history_kv[0])
            values.append(history_kv[1])
        keys.append(k)
        values.append(v)
        all_k, all_v = torch.cat(keys, dim=1), torch.cat(values, dim=1)
    else:
        all_k, all_v = kv_buffer
        current_k, current_v = all_k[:, -seq_len:], all_v[:, -seq_len:]
        current_k.copy_(k)
        current_v.copy_(v)
        k, v = current_k, current_v
    out = attention.attn(q, all_k, all_v)
    out, _ = attention.to_out(out.reshape(batch_size, seq_len, -1))
    return out, k, v


class CosmosDreamsGenDecoderLayer(Cosmos3GenDecoderLayer):
    def forward(
        self, hidden: torch.Tensor, **attention_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_output, k, v = _joint_attention(
            self.cross_attention, self.input_layernorm(hidden), **attention_kwargs
        )
        hidden = hidden + attention_output
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), k, v


def project_actions(
    projection: Any,
    action_latents: torch.Tensor,
    action_domain_ids: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Domain-aware projection of ``[1, N, action_dim]`` tokens.

    One domain id applies to every token; ``N`` ids (``[N]`` or ``[1, N]``)
    route each token through its own domain's weights, which mixed robot and
    camera rows of unified checkpoints need.
    """
    domain_ids = action_domain_ids.reshape(-1)
    if domain_ids.numel() == 1:
        return projection(action_latents, domain_ids)
    tokens = action_latents.shape[1]
    if domain_ids.numel() != tokens:
        raise ValueError(
            f"Cosmos-Dreams per-token domain ids must have one entry per action token: "
            f"{domain_ids.numel()} ids for {tokens} tokens."
        )
    flat = action_latents.reshape(tokens, action_latents.shape[-1])
    return projection(flat, domain_ids).reshape(1, tokens, hidden_size)


class CosmosDreamsTransformer(Cosmos3OmniTransformer):
    """Cosmos3 Omni weights driven as a causal world model.

    Weight names, projections, and the UND text pathway are inherited unchanged;
    only the GEN forward differs (per-frame action/vision interleaving for
    action-conditioned checkpoints, joint attention with committed history, no
    time embedding on clean tokens). Control-video checkpoints carry no action
    tokens; their control clips enter as clean vision frames.
    """

    _gen_layer_cls = CosmosDreamsGenDecoderLayer

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
        history_kv: HistoryKV | None = None,
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
        null_slices = null_action_token_slices(
            null_action_frame_indexes,
            num_frames=num_frames,
            tokens_per_frame=action_count + patch_count,
            action_tokens_per_frame=action_count,
        )

        current_kv: list[KVPair] = []
        for layer_idx, layer in enumerate(self.gen_layers):
            if isinstance(history_kv, CosmosDreamsKVCache):
                kv_buffer = history_kv.attention_buffer(layer_idx, hidden.shape[1])
                layer_history = None
            else:
                kv_buffer = None
                layer_history = None if history_kv is None else history_kv[layer_idx]
            hidden, k, v = layer(
                hidden,
                text_kv=text_kv[layer_idx],
                history_kv=layer_history,
                kv_buffer=kv_buffer,
                cos_sin_cache=cos_sin_cache,
                rope_positions=rope_positions,
                null_token_slices=null_slices,
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
        history_kv: HistoryKV | None,
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
        action_hidden = project_actions(
            self.action_proj_in,
            action_latents.to(vision_tokens.dtype),
            action_domain_ids,
            hidden_size=self.hidden_size,
        )
        action_hidden = action_hidden + self.action_modality_embed.to(
            action_hidden.dtype
        )
        return interleave_action_vision_tokens(
            action_hidden.view(1, num_frames, action_count, self.hidden_size),
            vision_tokens,
        )


EntryClass = CosmosDreamsTransformer
