# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV transformer.

Same weights and layer stack as ``Cosmos3OmniTransformer``. Two things change
inside the network. The GEN cross-attention runs the block-sparse multiview
attention from ``cosmos3_multiview_attention`` whenever the pipeline hands over
a ``MultiviewLayout``. And the GEN sequence is assembled from packed sensor
items instead of one video clip: the WSM control cameras, the RGB target
cameras, and on joint checkpoints the HD-map control and LiDAR target range
maps, each patchified through its own input projection and sharing one
temporal origin. Per-camera captions are encoded by separate causal UND passes
so no caption attends another; their K/V are concatenated for the GEN layers.
The transformer also owns the request-local mask and packing-buffer caches so
36 layers and every denoising step reuse one block map and one set of padded
q/k/v buffers.
"""

from __future__ import annotations

import math
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    validate_lidar_config,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    padded_multiview_flex_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3CrossAttention,
    Cosmos3OmniTransformer,
    compute_mrope_position_ids_text,
    compute_mrope_position_ids_vision,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def pack_state(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flatten sensor latents of independent geometry into one ``[B, N]`` state."""
    return torch.cat([tensor.flatten(1) for tensor in tensors], dim=1)


def unpack_state(
    state: torch.Tensor, shapes: tuple[tuple[int, ...], ...]
) -> tuple[torch.Tensor, ...]:
    sizes = [math.prod(shape) for shape in shapes]
    if state.ndim != 2 or state.shape[1] != sum(sizes):
        raise ValueError(
            "Packed Cosmos3 multiview state does not match the declared geometries: "
            f"state={tuple(state.shape)}, shapes={list(shapes)}."
        )
    return tuple(
        part.reshape(state.shape[0], *shape)
        for part, shape in zip(state.split(sizes, dim=1), shapes, strict=True)
    )


class Cosmos3MultiviewCrossAttention(Cosmos3CrossAttention):
    """GEN cross-attention that runs the sparse multiview kernel for a layout."""

    def _forward_multiview(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        multiview_layout: Any,
    ) -> torch.Tensor:
        if not isinstance(multiview_layout, MultiviewAttentionContext):
            raise TypeError(
                "Cosmos3 multiview cross-attention expected MultiviewAttentionContext, "
                f"got {type(multiview_layout).__name__}."
            )
        return padded_multiview_flex_attention(q, k, v, k_und, v_und, multiview_layout)


class Cosmos3MultiviewTransformer(Cosmos3OmniTransformer):
    """Cosmos3 Nano weights with packed sensor items and request-local mask caching."""

    _cross_attention_cls = Cosmos3MultiviewCrossAttention

    def __init__(
        self,
        config: Cosmos3VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        backbone_type = (
            hf_config.get("backbone_type") if isinstance(hf_config, dict) else None
        )
        if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
            raise ValueError(
                "Cosmos3MultiviewTransformer requires transformer/config.json "
                f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
            )
        super().__init__(config, hf_config, quant_config)
        multiview = hf_config.get("multiview") if isinstance(hf_config, dict) else None
        lidar = multiview.get("lidar") if isinstance(multiview, dict) else None
        self.lidar_config: dict[str, Any] | None = None
        if lidar is not None:
            self.lidar_config = validate_lidar_config(lidar)
            # Joint checkpoints project LiDAR range-map latents through their
            # own input/output linears; the GEN stack is shared with cameras.
            lidar_width = self.latent_patch_size**2 * int(
                self.lidar_config["latent_channels"]
            )
            self.lidar_proj_in = ReplicatedLinear(
                lidar_width,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix="lidar_proj_in",
            )
            self.lidar_proj_out = ReplicatedLinear(
                self.hidden_size,
                lidar_width,
                bias=True,
                quant_config=quant_config,
                prefix="lidar_proj_out",
            )
        self._multiview_mask_cache: dict[tuple[Any, ...], Any] = {}
        # Padded q/k/v packing buffers, keyed by shape/dtype/device. Held on
        # the transformer rather than a per-forward context so the packed
        # tensors are allocated and zeroed once per request, not once per layer.
        self._multiview_buffer_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def reset_cache(self, cache_key: str | None = None) -> None:
        super().reset_cache(cache_key)
        if cache_key is None:
            self._multiview_mask_cache.clear()
            self._multiview_buffer_cache.clear()

    # -- Packed forward -----------------------------------------------------

    def _caption_lengths(
        self,
        text_ids: torch.Tensor,
        max_text_seq_len: int | None,
        cache_key: str,
        caption_lengths_by_cache_key: dict[str, tuple[int, ...]] | None,
    ) -> tuple[int, ...]:
        lengths = (
            caption_lengths_by_cache_key.get(cache_key)
            if caption_lengths_by_cache_key
            else None
        )
        if lengths is None:
            lengths = (int(max_text_seq_len or text_ids.shape[1]),)
        lengths = tuple(int(length) for length in lengths)
        if any(length <= 0 for length in lengths) or sum(lengths) != text_ids.shape[1]:
            raise ValueError(
                "Cosmos3 multiview caption lengths must partition the compacted text "
                f"tokens: lengths={list(lengths)}, tokens={text_ids.shape[1]}."
            )
        return lengths

    def _encode_captions(
        self, text_ids: torch.Tensor, lengths: tuple[int, ...]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """One causal UND pass per caption, K/V concatenated along the sequence.

        Every caption restarts its text positions at zero; the GEN origin is
        placed past the longest caption by ``_packed_position_ids``.
        """
        caches: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for ids in text_ids.split(list(lengths), dim=1):
            positions, _ = compute_mrope_position_ids_text(
                ids.shape[1], temporal_offset=0, device=ids.device
            )
            mask = torch.ones_like(ids)
            caches.append(self.language_model(ids, mask, positions.unsqueeze(1)))
        if len(caches) == 1:
            return caches[0]
        return [
            (
                torch.cat([cache[layer][0] for cache in caches], dim=1),
                torch.cat([cache[layer][1] for cache in caches], dim=1),
            )
            for layer in range(len(caches[0]))
        ]

    def _packed_position_ids(
        self,
        items: tuple[MaskItem, ...],
        *,
        text_origin: int,
        fps: float,
        lidar_fps: float | None,
        lidar_temporal_compression_factor: int,
        align_views: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Every sensor item starts at the shared origin; LiDAR ticks at its own rate.

        With ``base_fps`` 30 and the camera VAE's 4x compression as the unit, a
        camera latent advances 1.0 temporal position and a 10 Hz sweep 0.75.
        """
        blocks = []
        for item in items:
            if item.is_lidar:
                if lidar_fps is None:
                    raise ValueError(
                        "Cosmos3 LiDAR items need lidar_fps for their positions."
                    )
                positions, _ = compute_mrope_position_ids_vision(
                    *item.token_shape,
                    temporal_offset=text_origin,
                    device=device,
                    fps=float(lidar_fps),
                    base_fps=self.base_fps,
                    temporal_compression_factor=int(lidar_temporal_compression_factor),
                    base_temporal_compression_factor=self.temporal_compression_factor,
                )
            else:
                positions, _ = compute_mrope_position_ids_vision(
                    *item.token_shape,
                    temporal_offset=text_origin,
                    device=device,
                    fps=float(fps),
                    base_fps=self.base_fps,
                    temporal_compression_factor=self.temporal_compression_factor,
                    temporal_position_period=(
                        item.token_shape[0] // item.num_views if align_views else None
                    ),
                )
            blocks.append(positions)
        dtype = blocks[0].dtype
        for block in blocks[1:]:
            dtype = torch.promote_types(dtype, block.dtype)
        return torch.cat([block.to(dtype) for block in blocks], dim=1)

    def _frame_token_mask(
        self, frame_mask: torch.Tensor, item: MaskItem, dtype: torch.dtype
    ) -> torch.Tensor:
        """Per-token noisy mask ``[B, tokens, 1]`` from a per-frame ``[B, T]`` mask."""
        spatial = item.token_shape[1] * item.token_shape[2]
        return frame_mask.to(dtype).repeat_interleave(spatial, dim=1).unsqueeze(-1)

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        multiview_layout: MultiviewLayout,
        packed_shapes: tuple[tuple[int, ...], ...],
        fps: float,
        cache_key: str,
        noisy_frame_mask: torch.Tensor | None,
        max_text_seq_len: int | None,
        control_latents: torch.Tensor | list[torch.Tensor] | None,
        caption_lengths_by_cache_key: dict[str, tuple[int, ...]] | None,
        separate_captions: bool,
        lidar_control_latents: torch.Tensor | None,
        lidar_fps: float | None,
        lidar_temporal_compression_factor: int,
        temporal_position_period: int | None,
    ) -> torch.Tensor:
        del text_mask  # Captions arrive compacted; their boundaries are the lengths.
        targets = unpack_state(hidden_states, packed_shapes)
        camera = targets[0]
        batch_size = camera.shape[0]
        device = camera.device
        dtype = camera.dtype
        controls = (
            list(control_latents)
            if isinstance(control_latents, (list, tuple))
            else ([control_latents] if control_latents is not None else [])
        )
        if len(controls) > 1:
            raise ValueError(
                "Cosmos3 multiview packs exactly one WSM control item, "
                f"got {len(controls)}."
            )
        has_control = bool(controls)
        items = tuple(
            item
            for item in multiview_layout.items
            if has_control or not item.is_control
        )
        streams: list[torch.Tensor] = []
        if has_control:
            streams.append(controls[0])
        streams.append(camera)
        if len(targets) == 2:
            if self.lidar_config is None:
                raise ValueError(
                    "Joint camera/LiDAR requests need a checkpoint with LiDAR projections."
                )
            if has_control:
                if lidar_control_latents is None:
                    raise ValueError("Joint transfer requires LiDAR control latents.")
                streams.append(lidar_control_latents)
            streams.append(targets[1])
        elif len(targets) != 1:
            raise ValueError(
                f"Cosmos3 multiview packs one camera target and at most one LiDAR target, got {len(targets)}."
            )
        if len(streams) != len(items):
            raise ValueError(
                "Packed sensor streams do not match the layout items: "
                f"streams={len(streams)}, items={len(items)}."
            )
        for item, latent in zip(items, streams, strict=True):
            _, _, latent_t, latent_h, latent_w = latent.shape
            patch_h, patch_w, _, _ = self._pad_to_patch_size(latent_h, latent_w)
            if item.token_shape != (latent_t, patch_h, patch_w):
                raise ValueError(
                    "Packed sensor latent does not match its layout item: "
                    f"latent={tuple(latent.shape)}, item={item.token_shape}."
                )

        lengths = self._caption_lengths(
            text_ids, max_text_seq_len, cache_key, caption_lengths_by_cache_key
        )
        layout = msgspec.structs.replace(
            multiview_layout,
            items=items,
            caption_lengths=lengths if separate_captions else (),
        )
        context = MultiviewAttentionContext(
            layout, self._multiview_mask_cache, self._multiview_buffer_cache
        )

        self._ensure_cache_dicts()
        if (
            cache_key not in self.cached_kv
            or cache_key not in self.cached_gen_rope_inputs
        ):
            self.cached_kv[cache_key] = self._encode_captions(text_ids, lengths)
            positions = self._packed_position_ids(
                items,
                text_origin=max(lengths) + self.temporal_margin,
                fps=fps,
                lidar_fps=lidar_fps,
                lidar_temporal_compression_factor=lidar_temporal_compression_factor,
                align_views=temporal_position_period is not None,
                device=device,
            )
            positions = positions.unsqueeze(1).expand(-1, batch_size, -1).contiguous()
            self.cached_gen_rope_inputs[cache_key] = (
                self.language_model.rotary_emb.build_rope_cache_inputs(
                    positions, cache_dtype=dtype
                )
            )
        cos_sin_gen, gen_rope_cache_positions = self.cached_gen_rope_inputs[cache_key]
        cached_kv = self.cached_kv[cache_key]

        # Every target gets the same diffusion timestep; controls never do, and
        # anchored camera frames receive zero through the velocity mask.
        time_embed = self.time_embedder(timestep.float()).to(dtype).unsqueeze(1)
        camera_frame_mask = None
        if noisy_frame_mask is not None:
            camera_frame_mask = unpack_state(noisy_frame_mask, packed_shapes)[0][
                :, 0, :, 0, 0
            ]
        embeddings = []
        for item, latent in zip(items, streams, strict=True):
            project = self.lidar_proj_in if item.is_lidar else self.proj_in
            hidden, _ = project(
                self.patchify(latent.to(dtype), *latent.shape[2:])
                if not item.is_lidar
                else self._patchify_lidar(latent.to(dtype))
            )
            if not item.is_control:
                if not item.is_lidar and camera_frame_mask is not None:
                    hidden = hidden + time_embed * self._frame_token_mask(
                        camera_frame_mask, item, dtype
                    )
                else:
                    hidden = hidden + time_embed
            embeddings.append(hidden)
        hidden_gen = torch.cat(embeddings, dim=1)
        del embeddings

        residual: torch.Tensor | None = None
        for layer, (k_und, v_und) in zip(self.gen_layers, cached_kv, strict=True):
            hidden_gen, residual = layer(
                hidden_gen,
                k_und,
                v_und,
                cos_sin_gen,
                gen_rope_cache_positions,
                True,
                False,
                residual=residual,
                multiview_layout=context,
            )
        hidden_gen = hidden_gen + residual

        outputs = []
        parts = hidden_gen.split([item.num_tokens for item in items], dim=1)
        for item, latent, part in zip(items, streams, parts, strict=True):
            if item.is_control:
                continue
            normed = self.norm_moe_gen(part)
            if item.is_lidar:
                projected, _ = self.lidar_proj_out(normed)
                outputs.append(self._unpatchify_lidar(projected, latent.shape[1:]))
            else:
                projected, _ = self.proj_out(normed)
                outputs.append(self.unpatchify(projected, *latent.shape[2:]))
        return pack_state(outputs)

    def _patchify_lidar(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, C, T, H, W]`` range-map latents to ``[B, T*Hp*Wp, p*p*C]`` with edge padding."""
        batch, channels, frames, height, width = latent.shape
        patch = self.latent_patch_size
        patch_h, patch_w, padded_h, padded_w = self._pad_to_patch_size(height, width)
        if (padded_h, padded_w) != (height, width):
            latent = torch.nn.functional.pad(
                latent, (0, padded_w - width, 0, padded_h - height)
            )
        x = latent.reshape(batch, channels, frames, patch_h, patch, patch_w, patch)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)
        return x.reshape(batch, frames * patch_h * patch_w, patch * patch * channels)

    def _unpatchify_lidar(
        self, tokens: torch.Tensor, shape: tuple[int, ...]
    ) -> torch.Tensor:
        channels, frames, height, width = (int(dim) for dim in shape)
        patch = self.latent_patch_size
        patch_h, patch_w, padded_h, padded_w = self._pad_to_patch_size(height, width)
        x = tokens.reshape(
            tokens.shape[0], frames, patch_h, patch_w, patch, patch, channels
        )
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(tokens.shape[0], channels, frames, padded_h, padded_w)
        return x[:, :, :, :height, :width]

    def forward(
        self,
        *args,
        multiview_layout: MultiviewLayout | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if multiview_layout is None:
            return super().forward(*args, **kwargs)
        if (
            kwargs.get("action_latents") is not None
            or kwargs.get("sound_latents") is not None
        ):
            raise ValueError(
                "Cosmos3 multiview cannot be combined with action or sound streams."
            )
        if get_sp_world_size() > 1:
            raise ValueError("Cosmos3 multiview does not support sequence parallelism.")
        hidden_states = (
            kwargs.pop("hidden_states") if "hidden_states" in kwargs else args[0]
        )
        packed_shapes = kwargs.get("packed_shapes")
        if packed_shapes is None:
            raise ValueError("Cosmos3 multiview forward requires packed_shapes.")
        text_ids = kwargs["text_ids"]
        return self._forward_packed(
            hidden_states,
            kwargs["timestep"],
            text_ids,
            kwargs.get("text_mask"),
            multiview_layout=multiview_layout,
            packed_shapes=tuple(
                tuple(int(d) for d in shape) for shape in packed_shapes
            ),
            fps=float(kwargs.get("fps") or self.base_fps),
            cache_key=str(kwargs.get("cache_key", "default")),
            noisy_frame_mask=kwargs.get("noisy_frame_mask"),
            max_text_seq_len=kwargs.get("max_text_seq_len"),
            control_latents=kwargs.get("control_latents"),
            caption_lengths_by_cache_key=kwargs.get("caption_lengths_by_cache_key"),
            separate_captions=bool(kwargs.get("separate_captions", False)),
            lidar_control_latents=kwargs.get("lidar_control_latents"),
            lidar_fps=kwargs.get("lidar_fps"),
            lidar_temporal_compression_factor=int(
                kwargs.get("lidar_temporal_compression_factor", 1)
            ),
            temporal_position_period=kwargs.get("temporal_position_period"),
        )


EntryClass = Cosmos3MultiviewTransformer
