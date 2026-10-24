# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Auto transformer.

The weights are the Cosmos3 Nano multiview generator: the shared GEN stack,
the LiDAR input/output projections and the rig-view embedding table. What
changes is the GEN attention: every token of the current chunk takes one
softmax over the keys the replay predicate admits from the current chunk, the
KV ring of committed chunks and the per-camera caption documents, executed as
one block-sparse FlexAttention call planned on the host
(``cosmos3_simulation.sim_auto_attention``). Every forward returns the chunk's post-norm,
post-RoPE K and raw V, which a clean refresh commits to the ring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import msgspec
import torch
import torch.distributed as dist
import torch.nn as nn

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoManifest,
    load_cosmos3_sim_auto_manifest,
)
from sglang.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
)
from sglang.multimodal_gen.runtime.distributed import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    add_rig_view_rows,
    lidar_patch_grid,
    patchify_lidar,
    spatial_patch_hw,
    unpatchify_lidar,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_attention import (
    ReplayMask,
    replay_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_layout import (
    StreamItem,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3CrossAttention,
    Cosmos3GenDecoderLayer,
    Cosmos3OmniTransformer,
    _apply_qwen3_qk_norm_rope,
    compute_mrope_position_ids_text,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger


class UlyssesGroup(msgspec.Struct, frozen=True):
    """The rank's place in the Ulysses group; head groups follow rank order."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 2 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid Ulysses rank {self.rank} of {self.world_size}.")

    def local_heads(self, heads: int) -> int:
        if heads % self.world_size:
            raise ValueError(
                f"{heads} heads are not divisible by the Ulysses degree {self.world_size}."
            )
        return heads // self.world_size

    def head_slice(self, heads: int) -> slice:
        """This rank's contiguous group of ``heads`` (TP-local) heads."""
        size = self.local_heads(heads)
        return slice(self.rank * size, (self.rank + 1) * size)

    def shard(self, num_tokens: int) -> ChunkShard:
        return ChunkShard(group=self, num_tokens=num_tokens)


class ChunkShard(msgspec.Struct, frozen=True):
    """One forward's token split: ``ceil(num_tokens / U)`` tokens per rank, padded at the end."""

    group: UlyssesGroup
    num_tokens: int

    def __post_init__(self) -> None:
        if self.num_tokens < 1:
            raise ValueError("A Ulysses shard needs at least one token.")

    @property
    def local_tokens(self) -> int:
        return -(-self.num_tokens // self.group.world_size)

    @property
    def padded_tokens(self) -> int:
        return self.local_tokens * self.group.world_size

    def split(self, tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
        """This rank's ``local_tokens`` along ``dim``; the padding repeats the last entry.

        Repeating keeps RoPE positions valid; padded hidden rows never act as
        keys and their outputs are dropped by :meth:`gather`.
        """
        if tensor.shape[dim] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens to shard, got {tensor.shape[dim]}."
            )
        start = self.group.rank * self.local_tokens
        padded = _pad_repeat_last(tensor, self.padded_tokens, dim=dim)
        return padded.narrow(dim, start, self.local_tokens).contiguous()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather ``[B, local_tokens, ...]`` shards into ``[B, num_tokens, ...]``."""
        self._check_local(tensor)
        gathered = sequence_model_parallel_all_gather(tensor.contiguous(), dim=1)
        return gathered[:, : self.num_tokens]

    def heads_from_tokens(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, local_tokens, H, D]`` Q/K/V -> ``[B, num_tokens, H / U, D]`` of this rank's head group.

        Heads split into contiguous groups in rank order on both sides, so
        query group ``r`` still reads KV group ``r``.
        """
        for tensor in (q, k, v):
            self._check_local(tensor)
        q, k, v = (
            _usp_input_all_to_all(tensor, head_dim=2)[:, : self.num_tokens]
            for tensor in (q, k, v)
        )
        return q, k, v

    def tokens_from_heads(self, output: torch.Tensor) -> torch.Tensor:
        """``[B, num_tokens, H / U, D]`` attention output -> ``[B, local_tokens, H, D]``."""
        if output.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} attention rows, got {output.shape[1]}."
            )
        pad = self.padded_tokens - self.num_tokens
        if pad:
            output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad))
        return _usp_output_all_to_all(output, head_dim=2)

    def _check_local(self, tensor: torch.Tensor) -> None:
        if tensor.shape[1] != self.local_tokens:
            raise ValueError(
                f"Expected {self.local_tokens} local tokens ({self.num_tokens} tokens / "
                f"Ulysses {self.group.world_size}), got {tensor.shape[1]}."
            )


class SpCheck(msgspec.Struct):
    """Debug tally (``SGLANG_DIFFUSION_DEBUG_COSMOS3_MULTIVIEW_SP_CHECK``) over one forward's layers."""

    layers: int = 0
    identical: int = 0
    max_diff: float = 0.0

    def compare(
        self,
        shard: ChunkShard,
        *,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: ReplayMask,
        out: torch.Tensor,
    ) -> None:
        """Full-head attention sliced to this rank's heads must equal ``out`` bitwise (same kernel, same tokens)."""
        full = replay_attention(
            _all_heads(q, shard.group),
            _all_heads(keys, shard.group),
            _all_heads(values, shard.group),
            mask,
        )
        mine = full[:, :, shard.group.head_slice(full.shape[2])]
        self.layers += 1
        self.identical += int(torch.equal(mine, out))
        self.max_diff = max(
            self.max_diff, float((mine.float() - out.float()).abs().max())
        )

    def summary(self, group: UlyssesGroup) -> str:
        return (
            f"Cosmos3-Nano-Sim-Auto SP check (rank {group.rank}/{group.world_size}): "
            f"{self.identical}/{self.layers} layers bitwise identical to full-head attention, "
            f"max|diff| {self.max_diff:.3g}"
        )


def _all_heads(tensor: torch.Tensor, group: UlyssesGroup) -> torch.Tensor:
    """``[B, S, h / U, D]`` of every rank concatenated into ``[B, S, h, D]`` (rank order = head order)."""
    parts = [torch.empty_like(tensor) for _ in range(group.world_size)]
    dist.all_gather(parts, tensor.contiguous(), group=get_sp_group().ulysses_group)
    return torch.cat(parts, dim=2)


def _pad_repeat_last(tensor: torch.Tensor, length: int, *, dim: int) -> torch.Tensor:
    pad = length - tensor.shape[dim]
    if pad == 0:
        return tensor
    last = tensor.narrow(dim, tensor.shape[dim] - 1, 1)
    return torch.cat([tensor, *([last] * pad)], dim=dim)


def current_ulysses_group() -> UlyssesGroup | None:
    """The worker's Ulysses group, or ``None`` at degree 1."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return None
    return UlyssesGroup(rank=get_ulysses_parallel_rank(), world_size=world_size)


def local_head_group(
    layer_kv: Sequence[KVPair], group: UlyssesGroup | None
) -> list[KVPair]:
    """The rank's head group of per-layer ``[1, tokens, H_kv, D]`` K/V (identity without Ulysses)."""
    if group is None:
        return list(layer_kv)
    heads = group.head_slice(layer_kv[0][0].shape[2])
    # Copies, so the full caption K/V is not kept alive by views.
    return [
        (key[:, :, heads].contiguous(), value[:, :, heads].contiguous())
        for key, value in layer_kv
    ]


logger = init_logger(__name__)

KVPair = tuple[torch.Tensor, torch.Tensor]


class Cosmos3SimAutoForwardOutput(msgspec.Struct):
    """Predictions of the target items (in item order) plus the chunk's per-layer ``[1, S, H_kv, D]`` K/V."""

    targets: list[torch.Tensor]
    current_kv: list[KVPair]


def _replay_layer_attention(
    attention: Cosmos3CrossAttention,
    hidden: torch.Tensor,
    *,
    cos_sin_cache: torch.Tensor,
    rope_positions: torch.Tensor,
    history: KVPair,
    captions: KVPair,
    mask: ReplayMask,
    shard: ChunkShard | None,
    sp_check: SpCheck | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One softmax over ``[current | ring | captions]``; returns the output and the chunk's K/V.

    With ``shard`` the hidden states, RoPE positions and output are the rank's
    token shard, while the K/V streams and the returned K/V hold every token
    for the rank's head group.
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
    if shard is not None:
        q, k, v = shard.heads_from_tokens(q, k, v)
    k = k.contiguous()
    v = v.contiguous()
    # Key order is the plan's stream order: current chunk, ring, captions.
    keys = torch.cat([k, history[0], captions[0]], dim=1)
    values = torch.cat([v, history[1], captions[1]], dim=1)
    out = replay_attention(q, keys, values, mask)
    if shard is not None:
        if sp_check is not None:
            sp_check.compare(shard, q=q, keys=keys, values=values, mask=mask, out=out)
        out = shard.tokens_from_heads(out)
    out, _ = attention.to_out(out.reshape(batch_size, seq_len, -1))
    return out, k, v


class Cosmos3SimAutoGenDecoderLayer(Cosmos3GenDecoderLayer):
    def forward(
        self, hidden: torch.Tensor, **attention_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_output, k, v = _replay_layer_attention(
            self.cross_attention, self.input_layernorm(hidden), **attention_kwargs
        )
        hidden = hidden + attention_output
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), k, v


class Cosmos3SimAutoTransformer(Cosmos3OmniTransformer):
    """Cosmos3 Nano multiview weights driven as a chunk-causal world model."""

    _gen_layer_cls = Cosmos3SimAutoGenDecoderLayer

    def __init__(
        self,
        config: Cosmos3VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        backbone_type = hf_config.get("backbone_type")
        if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
            raise ValueError(
                "Cosmos3SimAutoTransformer requires transformer/config.json "
                f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
            )
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        self.manifest: Cosmos3SimAutoManifest = load_cosmos3_sim_auto_manifest(
            hf_config
        )
        self._validate_causal_arch(hf_config)
        manifest = self.manifest
        self.lidar_patch_hw = spatial_patch_hw(
            manifest.lidar_latent_patch_size_hw
            if manifest.lidar_latent_patch_size_hw is not None
            else self.latent_patch_size
        )
        if manifest.lidar is not None:
            lidar_width = (
                self.lidar_patch_hw[0]
                * self.lidar_patch_hw[1]
                * int(manifest.lidar["latent_channels"])
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
        rig = manifest.rig_view_embedding
        self.rig_view_embed = nn.Embedding(int(rig["num_embeddings"]), self.hidden_size)
        self.rig_lidar_id = int(rig["lidar_id"])
        self.ulysses: UlyssesGroup | None = current_ulysses_group()
        # A checkpoint without the rig table would run on random rows and no
        # error; the loader turns the missing tensor into a hard failure.
        self.rig_view_embed.weight.missing_param_init = "error"

    def _validate_causal_arch(self, hf_config: dict[str, Any]) -> None:
        expected_flags = {
            "joint_attn_implementation": "multiview",
            "video_temporal_causal": True,
            "unified_3d_mrope_reset_spatial_ids": True,
        }
        for key, expected in expected_flags.items():
            if hf_config.get(key) != expected:
                raise ValueError(
                    f"Unsupported Cosmos3-Nano-Sim-Auto transformer config: {key}={hf_config.get(key)!r}; "
                    f"expected {expected!r}."
                )
        if self.sound_gen or self.config.action_gen:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto checkpoints carry no sound or action pathway."
            )
        manifest = self.manifest
        mismatches = {
            "temporal_compression_factor": (
                self.temporal_compression_factor,
                manifest.temporal_compression_factor,
            ),
            "latent_patch_size": (self.latent_patch_size, manifest.latent_patch_size),
            "latent_channels": (self.latent_channel, manifest.latent_channels),
            "temporal_modality_margin": (
                self.temporal_margin,
                manifest.temporal_modality_margin,
            ),
            "base_fps": (float(self.base_fps), float(manifest.base_fps)),
        }
        for name, (arch_value, manifest_value) in mismatches.items():
            if arch_value != manifest_value:
                raise ValueError(
                    f"Cosmos3-Nano-Sim-Auto {name} differs between transformer config and artifact: "
                    f"{arch_value} != {manifest_value}."
                )

    def lidar_patch_grid(self, height: int, width: int) -> tuple[int, int]:
        """LiDAR token grid ``(patch_h, patch_w)`` of a latent of this height and width."""
        return lidar_patch_grid(height, width, self.lidar_patch_hw)[:2]

    # -- captions ------------------------------------------------------------

    def encode_caption(self, text_ids: torch.Tensor) -> list[KVPair]:
        """One causal UND pass over a single caption document, positions rewound to zero."""
        if text_ids.ndim != 2 or text_ids.shape[0] != 1 or text_ids.shape[1] < 1:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto captions must be [1, L] token ids, got {tuple(text_ids.shape)}."
            )
        positions, _ = compute_mrope_position_ids_text(
            text_ids.shape[1], temporal_offset=0, device=text_ids.device
        )
        return self.language_model(
            text_ids, torch.ones_like(text_ids), positions.unsqueeze(1)
        )

    # -- geometry --------------------------------------------------------------

    def chunk_position_ids(
        self, items: Sequence[StreamItem], *, text_extent: int, device: torch.device
    ) -> torch.Tensor:
        """Absolute 3D mRoPE ids of the packed chunk, ``[3, S]`` float32.

        Caption documents are rewound to ``0..len-1``, so media starts at
        ``text_extent + temporal_modality_margin``. With FPS modulation a sensor
        frame advances ``seconds_per_frame * base_fps / temporal_compression``
        positions (1.0 per RGB latent at 30 fps, 0.75 per 10 Hz LiDAR sweep);
        camera-major items repeat the temporal ids per view and spatial ids
        restart at zero per frame.
        """
        offset = float(text_extent + self.temporal_margin)
        parts = []
        for item in items:
            rate = (
                item.seconds_per_frame
                * float(self.base_fps)
                / self.temporal_compression_factor
            )
            frames = torch.tensor(item.frames, dtype=torch.float32, device=device)
            grid_h, grid_w = item.grid.grid_height, item.grid.grid_width
            spatial = grid_h * grid_w
            t_ids = (frames * rate + offset).view(-1, 1).expand(-1, spatial).reshape(-1)
            h_ids = (
                torch.arange(grid_h, dtype=torch.float32, device=device)
                .view(-1, 1)
                .expand(grid_h, grid_w)
            )
            w_ids = (
                torch.arange(grid_w, dtype=torch.float32, device=device)
                .view(1, -1)
                .expand(grid_h, grid_w)
            )
            ids = torch.stack(
                [
                    t_ids,
                    h_ids.reshape(-1).repeat(len(item.frames)),
                    w_ids.reshape(-1).repeat(len(item.frames)),
                ]
            )
            parts.append(ids.repeat(1, len(item.view_ids)))
        return torch.cat(parts, dim=1)

    # -- embedding -------------------------------------------------------------

    def embed_items(
        self,
        items: Sequence[StreamItem],
        latents: Sequence[torch.Tensor],
        *,
        timestep: torch.Tensor | None,
        rig_view_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Project every item, add its rig identity, then the timestep to generated target frames only.

        ``timestep`` is ``None`` for clean refresh forwards: controls, observed
        condition frames and committed-clean targets never carry a time embed.
        """
        if len(items) != len(latents):
            raise ValueError(f"{len(items)} items but {len(latents)} latent tensors.")
        dtype = self.proj_in.weight.dtype
        time_embed = None
        if timestep is not None:
            time_embed = (
                self.time_embedder(timestep.reshape(1).float()).to(dtype).view(1, 1, -1)
            )
        camera_rows = self.rig_view_embed.weight[
            rig_view_ids.to(self.rig_view_embed.weight.device)
        ]
        lidar_row = self.rig_view_embed.weight[self.rig_lidar_id].unsqueeze(0)
        embeddings = []
        for item, latent in zip(items, latents, strict=True):
            expected_frames = len(item.view_ids) * len(item.frames)
            if (
                latent.ndim != 5
                or latent.shape[0] != 1
                or latent.shape[2] != expected_frames
            ):
                raise ValueError(
                    f"Item {item.kind} latents must be [1, C, {expected_frames}, H, W] (camera-major), "
                    f"got {tuple(latent.shape)}."
                )
            if item.is_lidar:
                hidden, _ = self.lidar_proj_in(
                    patchify_lidar(latent.to(dtype), self.lidar_patch_hw)
                )
                rows = lidar_row
            else:
                hidden, _ = self.proj_in(
                    self.patchify(latent.to(dtype), *latent.shape[2:])
                )
                if camera_rows.shape[0] != len(item.view_ids):
                    raise ValueError(
                        f"{camera_rows.shape[0]} rig view ids for a {len(item.view_ids)}-camera {item.kind} item."
                    )
                rows = camera_rows
            if hidden.shape[1] != item.num_tokens:
                raise ValueError(
                    f"Item {item.kind} produced {hidden.shape[1]} tokens, layout expects {item.num_tokens}."
                )
            hidden = add_rig_view_rows(hidden, rows, len(item.view_ids))
            if time_embed is not None and not item.is_control:
                generated = torch.tensor(
                    [frame not in item.condition_frames for frame in item.frames],
                    dtype=torch.bool,
                    device=hidden.device,
                )
                mask = generated.repeat(len(item.view_ids)).repeat_interleave(
                    item.tokens_per_frame
                )
                hidden = hidden + time_embed * mask.view(1, -1, 1).to(dtype)
            embeddings.append(hidden)
        return torch.cat(embeddings, dim=1)

    # -- forward -----------------------------------------------------------------

    def forward(
        self,
        *,
        items: Sequence[StreamItem],
        latents: Sequence[torch.Tensor],
        timestep: torch.Tensor | None,
        text_extent: int,
        und_kv: Sequence[KVPair],
        history_kv: Sequence[KVPair],
        mask: ReplayMask,
        rig_view_ids: torch.Tensor,
    ) -> Cosmos3SimAutoForwardOutput:
        """Denoise (``timestep`` set) or clean-refresh (``timestep`` None) one chunk.

        ``und_kv`` holds the per-layer caption K/V ``[1, U, H_kv, D]`` and
        ``history_kv`` the per-layer ring K/V ``[1, R, H_kv, D]`` over the whole
        ring capacity; the plan behind ``mask`` decides which of their
        tokens a query may read. Under Ulysses every K/V argument and result
        holds the rank's head group for every token; only the GEN layers run
        on a token shard.
        """
        if (
            len(und_kv) != self.num_hidden_layers
            or len(history_kv) != self.num_hidden_layers
        ):
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto expects {self.num_hidden_layers} caption and history KV layers, "
                f"got {len(und_kv)} and {len(history_kv)}."
            )
        if timestep is not None and timestep.numel() != 1:
            raise ValueError(
                f"timestep must hold one value, got {tuple(timestep.shape)}."
            )
        hidden = self.embed_items(
            items, latents, timestep=timestep, rig_view_ids=rig_view_ids
        )
        position_ids = self.chunk_position_ids(
            items, text_extent=text_extent, device=hidden.device
        )
        cos_sin_cache, rope_positions = (
            self.language_model.rotary_emb.build_rope_cache_inputs(
                position_ids.unsqueeze(1), cache_dtype=hidden.dtype
            )
        )
        shard = None if self.ulysses is None else self.ulysses.shard(hidden.shape[1])
        sp_check = None
        if shard is not None:
            hidden = shard.split(hidden, dim=1)
            rope_positions = shard.split(rope_positions, dim=0)
            if envs.SGLANG_DIFFUSION_DEBUG_COSMOS3_MULTIVIEW_SP_CHECK:
                sp_check = SpCheck()
        current_kv: list[KVPair] = []
        for layer_idx, layer in enumerate(self.gen_layers):
            hidden, k, v = layer(
                hidden,
                cos_sin_cache=cos_sin_cache,
                rope_positions=rope_positions,
                history=history_kv[layer_idx],
                captions=und_kv[layer_idx],
                mask=mask,
                shard=shard,
                sp_check=sp_check,
            )
            current_kv.append((k, v))
        if shard is not None:
            hidden = shard.gather(hidden)
            if sp_check is not None:
                logger.info(sp_check.summary(shard.group))
        hidden = self.norm_moe_gen(hidden)
        outputs = []
        parts = hidden.split([item.num_tokens for item in items], dim=1)
        for item, latent, part in zip(items, latents, parts, strict=True):
            if item.is_control:
                continue
            if item.is_lidar:
                projected, _ = self.lidar_proj_out(part)
                outputs.append(
                    unpatchify_lidar(projected, latent.shape[1:], self.lidar_patch_hw)
                )
            else:
                projected, _ = self.proj_out(part)
                outputs.append(self.unpatchify(projected, *latent.shape[2:]))
        return Cosmos3SimAutoForwardOutput(targets=outputs, current_kv=current_kv)


EntryClass = Cosmos3SimAutoTransformer
