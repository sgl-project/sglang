# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams causal transformer.

Cosmos-Dreams (Cosmos3-Interactive) reuses the Cosmos3 Omni weights but runs
the GEN pathway autoregressively over latent frames. Each frame is packed as
``[action tokens | vision patches]`` (vision patches only for control-video
checkpoints) and every GEN layer takes one softmax over
``[text K/V | committed clean history K/V | current chunk]``. The caller owns
that block per layer and hands every forward a slot in it: the forward writes
the chunk's post-norm, post-RoPE K/V into the slot and reads the block in place,
so the pipeline commits a clean refresh pass without copying anything.
Control-video clips are committed the same way, as clean vision frames sharing
the target frames' positions.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

import msgspec
import torch

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_qknorm_rope_pack_kv,
)
from sglang.multimodal_gen import envs
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
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

KVPair = tuple[torch.Tensor, torch.Tensor]
logger = init_logger(__name__)


class CosmosDreamsKVSlot(NamedTuple):
    """One layer's K/V views for one forward.

    ``chunk_keys`` and ``chunk_values`` are the last tokens of ``keys`` and
    ``values``, and ``chunk_kv`` is the same chunk as one ``[2, 1, chunk, H_kv, D]``
    view, keys first.
    """

    keys: torch.Tensor
    values: torch.Tensor
    chunk_keys: torch.Tensor
    chunk_values: torch.Tensor
    chunk_kv: torch.Tensor


class CosmosDreamsForwardOutput(msgspec.Struct):
    """Velocity for the current chunk; its K/V are in the forward's slots."""

    video: torch.Tensor


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


def _pack_kernel_applies(
    attention: Cosmos3CrossAttention,
    q: torch.Tensor,
    kv_slot: CosmosDreamsKVSlot,
    cos_sin_cache: torch.Tensor,
) -> bool:
    """Whether the kernel that norms and rotates Q/K can also write K/V into the slot.

    Same arithmetic as the in-place kernel ``_apply_qwen3_qk_norm_rope`` picks, so
    this only saves the two copies into the slot.
    """
    return (
        envs.SGLANG_ENABLE_FUSED_QKNORM_ROPE
        and q.device.type == "cuda"
        and q.dtype == kv_slot.chunk_kv.dtype
        and attention.norm_q.weight.dtype == q.dtype
        and attention.norm_k.weight.dtype == q.dtype
        and attention.norm_q.variance_epsilon == attention.norm_k.variance_epsilon
        and can_use_fused_inplace_qknorm_rope(
            attention.head_dim,
            cos_sin_cache.shape[-1],
            True,
            q.dtype,
            cos_sin_cache.dtype,
            pack_kv=True,
        )
    )


def _write_chunk_kv(
    attention: Cosmos3CrossAttention,
    hidden: torch.Tensor,
    *,
    kv_slot: CosmosDreamsKVSlot,
    cos_sin_cache: torch.Tensor,
    rope_positions: torch.Tensor,
    null_token_slices: list[slice],
) -> torch.Tensor:
    """Write the chunk's K (post-norm, post-RoPE) and V (null-action slots zeroed)
    into ``kv_slot``, which is exactly what history must store; returns its Q."""
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
    if _pack_kernel_applies(attention, q, kv_slot, cos_sin_cache):
        # The slot holds the chunk alone, so the kernel packs no prefix.
        no_prefix = kv_slot.chunk_keys[:, :0]
        fused_qknorm_rope_pack_kv(
            q,
            k,
            v,
            no_prefix,
            no_prefix,
            kv_slot.chunk_kv,
            attention.norm_q.weight,
            attention.norm_k.weight,
            cos_sin_cache,
            rope_positions,
            is_neox=True,
            eps=attention.norm_q.variance_epsilon,
            head_dim=attention.head_dim,
            rope_dim=cos_sin_cache.shape[-1],
        )
    else:
        q, k = _apply_qwen3_qk_norm_rope(
            q,
            k,
            attention.norm_q,
            attention.norm_k,
            attention.head_dim,
            cos_sin_cache,
            rope_positions,
        )
        kv_slot.chunk_keys.copy_(k)
        kv_slot.chunk_values.copy_(v)
    for null_slice in null_token_slices:
        kv_slot.chunk_values[:, null_slice] = 0.0
    return q


def _joint_attention(
    attention: Cosmos3CrossAttention,
    hidden: torch.Tensor,
    *,
    kv_slot: CosmosDreamsKVSlot,
    **write_kwargs: Any,
) -> torch.Tensor:
    """One softmax over ``[text | history | current chunk]``, read in place."""
    batch_size, seq_len = hidden.shape[:2]
    q = _write_chunk_kv(attention, hidden, kv_slot=kv_slot, **write_kwargs)
    out = attention.attn(q, kv_slot.keys, kv_slot.values)
    out, _ = attention.to_out(out.reshape(batch_size, seq_len, -1))
    return out


class CosmosDreamsCrossAttention(Cosmos3CrossAttention):
    def __init__(self, *args, **kwargs):
        # Backend selection still honors explicit overrides (``--attention-backend``,
        # component overrides) and falls back if the preferred backend is absent.
        super().__init__(
            *args,
            default_attention_backend=default_joint_attention_backend(),
            **kwargs,
        )


def default_joint_attention_backend() -> AttentionBackendEnum | None:
    """Preferred backend for the joint ``[text | history | chunk]`` attention.

    * SM12x: FlashInfer FA2.
    * sm_10x (B200/GB200/B300): FA, i.e. FA4. The platform default there is the
      dynamic cuDNN SDPA backend, which since sgl-project/sglang#47586 keeps GQA
      attention (this model: 32 query / 8 KV heads) on cuDNN. For the Dreams
      rollout, where a short query chunk attends to a long K/V history, cuDNN
      SDPA measured ~50% slower than FA4 on GB200 (Sim-Transfer 480p, 121 frames:
      DiT rollout 5.3 s -> 8.5 s per clip) and changes the numerics; FA4 is the
      route the model's outputs were validated on.
    * elsewhere: the platform default.
    """
    if not current_platform.is_cuda():
        return None
    if current_platform.is_sm120():
        return AttentionBackendEnum.FLASHINFER_FA2
    if current_platform.is_blackwell():
        return AttentionBackendEnum.FA
    return None


class CosmosDreamsGenDecoderLayer(Cosmos3GenDecoderLayer):
    def forward(self, hidden: torch.Tensor, **attention_kwargs: Any) -> torch.Tensor:
        hidden = hidden + _joint_attention(
            self.cross_attention, self.input_layernorm(hidden), **attention_kwargs
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def write_kv(self, hidden: torch.Tensor, **write_kwargs: Any) -> None:
        """Write this layer's K/V for ``hidden`` and stop before its attention."""
        _write_chunk_kv(
            self.cross_attention, self.input_layernorm(hidden), **write_kwargs
        )


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
    _cross_attention_cls = CosmosDreamsCrossAttention

    def __init__(
        self,
        config: Cosmos3VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        if current_platform.is_cuda() and current_platform.is_sm120():
            self._supported_attention_backends = self._supported_attention_backends | {
                AttentionBackendEnum.FLASHINFER_FA2
            }
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
        text_length: int,
        kv_slots: Sequence[CosmosDreamsKVSlot],
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        null_action_frame_indexes: Sequence[int] = (),
    ) -> CosmosDreamsForwardOutput:
        """Denoise one current chunk.

        ``hidden_states`` is ``[1, C, T, H, W]`` for latent frames
        ``[frame_start, frame_start + T)``; ``action_latents`` is
        ``[1, T * A, action_dim]`` (normalized, zero-padded) and must be omitted
        for control-video checkpoints. ``kv_slots`` holds one slot per layer whose
        block starts with the ``text_length`` prompt tokens.
        """
        if timestep.numel() != 1:
            raise ValueError(
                f"Cosmos-Dreams timestep must be a scalar, got {tuple(timestep.shape)}"
            )
        hidden, layer_kwargs = self._prepare_chunk(
            hidden_states,
            timestep,
            text_length=text_length,
            kv_slots=kv_slots,
            frame_start=frame_start,
            fps=fps,
            action_latents=action_latents,
            action_domain_ids=action_domain_ids,
            null_action_frame_indexes=null_action_frame_indexes,
        )
        for layer, kv_slot in zip(self.gen_layers, kv_slots, strict=True):
            hidden = layer(hidden, kv_slot=kv_slot, **layer_kwargs)

        _, _, num_frames, latent_h, latent_w = hidden_states.shape
        hidden = self.norm_moe_gen(hidden)
        _, vision_hidden = split_interleaved_action_vision_tokens(
            hidden,
            num_frames=num_frames,
            action_tokens_per_frame=self.action_tokens_per_frame,
            vision_tokens_per_frame=self._patch_count(latent_h, latent_w),
        )
        video, _ = self.proj_out(vision_hidden.flatten(1, 2))
        return CosmosDreamsForwardOutput(
            video=self.unpatchify(video, num_frames, latent_h, latent_w)
        )

    def write_clean_kv(
        self,
        hidden_states: torch.Tensor,
        *,
        text_length: int,
        kv_slots: Sequence[CosmosDreamsKVSlot],
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        null_action_frame_indexes: Sequence[int] = (),
    ) -> None:
        """Clean-refresh one chunk: write its K/V into ``kv_slots`` and nothing else.

        Takes what :meth:`forward` takes, without a timestep: the chunk enters
        with no time embedding, so the K/V are clean history, and a control clip
        is committed exactly like a clean frame. The last layer stops once its
        K/V are written, since nothing reads what it would compute after them.
        """
        hidden, layer_kwargs = self._prepare_chunk(
            hidden_states,
            None,
            text_length=text_length,
            kv_slots=kv_slots,
            frame_start=frame_start,
            fps=fps,
            action_latents=action_latents,
            action_domain_ids=action_domain_ids,
            null_action_frame_indexes=null_action_frame_indexes,
        )
        last_layer_idx = len(kv_slots) - 1
        for layer_idx, (layer, kv_slot) in enumerate(
            zip(self.gen_layers, kv_slots, strict=True)
        ):
            if layer_idx == last_layer_idx:
                layer.write_kv(hidden, kv_slot=kv_slot, **layer_kwargs)
            else:
                hidden = layer(hidden, kv_slot=kv_slot, **layer_kwargs)

    def _patch_count(self, latent_h: int, latent_w: int) -> int:
        grid_h, grid_w, _, _ = self._pad_to_patch_size(latent_h, latent_w)
        return grid_h * grid_w

    def _prepare_chunk(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None,
        *,
        text_length: int,
        kv_slots: Sequence[CosmosDreamsKVSlot],
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
        null_action_frame_indexes: Sequence[int],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Embed one chunk (``timestep`` None: clean, no time embedding) and build
        what every layer takes besides its slot."""
        self._validate_chunk_inputs(hidden_states, kv_slots=kv_slots)
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
            patch_count=patch_count,
        )
        position_ids = build_cosmos_dreams_position_ids(
            frame_start=frame_start,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            text_length=text_length,
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
        return hidden, {
            "cos_sin_cache": cos_sin_cache,
            "rope_positions": rope_positions,
            "null_token_slices": null_slices,
        }

    def _validate_chunk_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        kv_slots: Sequence[CosmosDreamsKVSlot],
    ) -> None:
        if hidden_states.ndim != 5 or hidden_states.shape[0] != 1:
            raise ValueError(
                f"Cosmos-Dreams hidden_states must have shape [1,C,T,H,W], got {tuple(hidden_states.shape)}"
            )
        if len(kv_slots) != self.num_hidden_layers:
            raise ValueError(
                f"Cosmos-Dreams expected {self.num_hidden_layers} K/V slots, got {len(kv_slots)}"
            )
        _, _, num_frames, latent_h, latent_w = hidden_states.shape
        chunk_tokens = num_frames * (
            self.action_tokens_per_frame + self._patch_count(latent_h, latent_w)
        )
        if kv_slots[0].chunk_keys.shape[1] != chunk_tokens:
            raise ValueError(
                f"Cosmos-Dreams K/V slots hold {kv_slots[0].chunk_keys.shape[1]} chunk "
                f"tokens but the chunk has {chunk_tokens}."
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
        timestep: torch.Tensor | None,
        *,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
        patch_count: int,
    ) -> torch.Tensor:
        """Project vision patches and actions, then pack ``[action | vision]`` per frame."""
        _, _, num_frames, latent_h, latent_w = hidden_states.shape
        action_count = self.action_tokens_per_frame
        vision_tokens, _ = self.proj_in(
            self.patchify(hidden_states, num_frames, latent_h, latent_w)
        )
        vision_tokens = vision_tokens.view(1, num_frames, patch_count, self.hidden_size)
        if timestep is not None:
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
