# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0

"""Praxis-1 DiT: the two-tower video backbone and the action expert of one chunk-causal window.

One call denoises K chunks of the window jointly. The encoder tower reads
``[registers | anchor | text | noised | clean]`` and returns conditioning tokens plus the taps the action
expert reads. The decoder tower reads ``[registers | noised]`` at full latent resolution and returns the
video velocity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Hashable

import msgspec
import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from sglang.multimodal_gen.configs.models.dits.praxis import (
    PraxisArchConfig,
    PraxisDiTConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.praxis_layers import (
    ActionExpert,
    MoEFeedForward,
    Transformer,
    ZeroCenteredRMSNorm,
    modulate,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# Rotary positions: a latent frame owns a block of FRAME_STRIDE positions and a token is placed at
# 1 + row * ROW_STRIDE + column. The encoder reads [registers | anchor | text | noised | clean], the decoder
# [registers | noised].
ROW_STRIDE = 360
FRAME_STRIDE = 129_600

GLOBAL, NOISED, CLEAN = 0, 1, 2


class LayoutCache:
    """Small LRU of tensors that depend only on the window layout."""

    def __init__(self, *, max_entries: int = 16) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[Hashable, Tensor | BlockMask] = OrderedDict()

    def get(
        self, key: Hashable, build: Callable[[], Tensor | BlockMask]
    ) -> Tensor | BlockMask:
        value = self._entries.pop(key, None)
        if value is None:
            value = build()
        self._entries[key] = value
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return value


def video_positions(
    *, frames: int, height: int, width: int, start: int, device: torch.device
) -> Tensor:
    frame = torch.arange(frames, device=device).view(-1, 1, 1) * FRAME_STRIDE
    row = torch.arange(height, device=device).view(1, -1, 1) * ROW_STRIDE
    col = torch.arange(width, device=device).view(1, 1, -1)
    return (start + 1 + frame + row + col).reshape(-1)


def encoder_positions(
    *,
    num_registers: int,
    anchor_hw: tuple[int, int],
    num_text: int,
    frames: int,
    height: int,
    width: int,
    num_clean_frames: int,
    device: torch.device,
) -> Tensor:
    """``(1, N)`` rotary positions: anchor in frame block 0, text in block 1, video from block 2."""
    registers = torch.zeros(num_registers, dtype=torch.int64, device=device)
    anchor = video_positions(
        frames=1, height=anchor_hw[0], width=anchor_hw[1], start=0, device=device
    )
    text = torch.arange(1, num_text + 1, device=device) + FRAME_STRIDE
    noised = video_positions(
        frames=frames,
        height=height,
        width=width,
        start=2 * FRAME_STRIDE,
        device=device,
    )
    clean = noised[: num_clean_frames * height * width]
    return torch.cat([registers, anchor, text, noised, clean])[None]


def decoder_positions(
    *, num_registers: int, frames: int, height: int, width: int, device: torch.device
) -> Tensor:
    registers = torch.zeros(num_registers, dtype=torch.int64, device=device)
    video = video_positions(
        frames=frames, height=height, width=width, start=0, device=device
    )
    return torch.cat([registers, video])[None]


def token_layout(
    *,
    num_global: int,
    frames: int,
    tokens_per_frame: int,
    chunk_frames: int,
    num_clean_frames: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Kind (GLOBAL, NOISED, CLEAN) and chunk index (-1 for global) of every window token."""
    noised_chunks = (
        torch.arange(frames, device=device) // chunk_frames
    ).repeat_interleave(tokens_per_frame)
    clean_chunks = (
        torch.arange(num_clean_frames, device=device) // chunk_frames
    ).repeat_interleave(tokens_per_frame)
    kinds = torch.cat(
        [
            torch.full((num_global,), GLOBAL, device=device),
            torch.full_like(noised_chunks, NOISED),
            torch.full_like(clean_chunks, CLEAN),
        ]
    )
    chunks = torch.cat(
        [torch.full((num_global,), -1, device=device), noised_chunks, clean_chunks]
    )
    return kinds, chunks


def _may_read(
    q_kind: Tensor, k_kind: Tensor, q_chunk: Tensor, k_chunk: Tensor
) -> Tensor:
    """True where a query token of ``q_kind`` in ``q_chunk`` may read a key token (broadcasting)."""
    clean_clean = (q_kind == CLEAN) & (k_kind == CLEAN) & (k_chunk <= q_chunk)
    noised_clean = (q_kind == NOISED) & (k_kind == CLEAN) & (k_chunk < q_chunk)
    noised_noised = (q_kind == NOISED) & (k_kind == NOISED) & (k_chunk == q_chunk)
    return (k_kind == GLOBAL) | clean_clean | noised_clean | noised_noised


def window_mask(*, kinds: Tensor, chunks: Tensor) -> Tensor:
    """Dense ``(N, N)`` bool mask, True where the row token may read the column token.

    Every token reads the global tokens; global tokens read only global tokens. A clean token reads
    the clean tokens of its chunk and earlier ones. A noised token reads the clean tokens of earlier
    chunks and the noised tokens of its own chunk.
    """
    return _may_read(kinds[:, None], kinds[None], chunks[:, None], chunks[None])


def window_block_mask(*, kinds: Tensor, chunks: Tensor) -> BlockMask:
    """The mask of :func:`window_mask` as a flex-attention block mask (128 x 128 tiles)."""

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return _may_read(kinds[q_idx], kinds[kv_idx], chunks[q_idx], chunks[kv_idx])

    n = kinds.numel()
    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=kinds.device
    )


TEXT = "TextEncodingTensor"
VIDEO = "VideoTensor"


class EncoderTaps(msgspec.Struct):
    """What the action expert reads from one encoder pass; spans are in the sequence order."""

    taps: list[Tensor]
    embeddings: Tensor
    rope_ids: Tensor
    num_global: int
    frames_per_chunk: int
    tokens_per_frame: int
    num_noised_frames: int
    num_clean_frames: int


class TimestepEmbedding(nn.Module):
    """Fixed random Fourier features of the flow timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.zeros(1, dim))
        self.register_buffer("phases", torch.zeros(1, dim))

    def forward(self, t: Tensor) -> Tensor:
        x = t[..., None].float() * self.freqs.float() + self.phases.float()
        return (x.to(t.dtype) * 2 * torch.pi).cos() * 0.5


class TextProjection(nn.Module):
    """MLP from text encoder states to backbone tokens."""

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(c_in, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            ZeroCenteredRMSNorm(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class VideoInputProjection(nn.Module):
    """Linear map of latent channels to tokens ``(1, T, h, w, dim)``."""

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(c_in, dim)

    def forward(self, v: Tensor) -> Tensor:
        return self.proj(v.movedim(1, -1))


class EncoderOutputProjection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DecoderOutputProjection(nn.Module):
    """RMS norm and linear map of decoder tokens to latent channels."""

    def __init__(self, dim: int, c_out: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(ZeroCenteredRMSNorm(dim), nn.Linear(dim, c_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


def unpatch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move the channel patch of ``(B, C*ph*pw, T, h, w)`` into space."""
    b, c, t, h, w = x.shape
    ph, pw = patch
    x = x.view(b, c // (ph * pw), ph, pw, t, h, w)
    return x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, -1, t, h * ph, w * pw)


def patch(x: Tensor, patch: tuple[int, int]) -> Tensor:
    """Move ``(ph, pw)`` spatial patches of ``(B, C, T, H, W)`` into channels."""
    b, c, t, hh, ww = x.shape
    ph, pw = patch
    x = x.view(b, c, t, hh // ph, ph, ww // pw, pw)
    return x.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * ph * pw, t, hh // ph, ww // pw)


def _chunk_rows(rows: Tensor, chunk_tokens: int, num_chunks: int) -> Tensor:
    """Expand row k of ``rows`` (1, K, C) over the ``chunk_tokens`` tokens of chunk k."""
    return rows[:, :num_chunks].repeat_interleave(chunk_tokens, dim=1)


def _layout_mask(
    *,
    num_global: int,
    frames: int,
    tokens_per_frame: int,
    chunk_frames: int,
    num_clean_frames: int,
    device: torch.device,
) -> Tensor | BlockMask:
    """Flex block mask on CUDA, dense bool mask elsewhere."""
    kinds, chunks = token_layout(
        num_global=num_global,
        frames=frames,
        tokens_per_frame=tokens_per_frame,
        chunk_frames=chunk_frames,
        num_clean_frames=num_clean_frames,
        device=device,
    )
    build = window_block_mask if device.type == "cuda" else window_mask
    return build(kinds=kinds, chunks=chunks)


def _core(
    config: PraxisArchConfig, *, depth: int, dense: list[int], mod_dim
) -> Transformer:
    return Transformer(
        depth=depth,
        num_dense=len(dense),
        dim=config.dim,
        head_dim=config.head_dim,
        num_kv_heads=config.num_kv_heads,
        dense_hidden=config.dense_ffn_hidden,
        moe={
            "hidden": config.expert_hidden,
            "num_experts": config.num_routed_experts,
            "top_k": config.top_k,
            "routed_scaling_factor": config.routed_scaling_factor,
        },
        mod_dim=mod_dim,
        attention_backend=config.attention_backend,
        rope={
            "effective_dim": config.rope_effective_dim,
            "base": config.rope_base,
            "table_size": config.rope_table_size,
        },
    )


class EncoderTower(nn.Module):
    def __init__(self, config: PraxisArchConfig) -> None:
        super().__init__()
        dim = config.dim
        self.config = config
        self.registers = nn.Parameter(torch.zeros(1, config.num_registers, dim))
        self.t_embedding = TimestepEmbedding(config.timestep_dim)
        self.t_proj = nn.ModuleDict({VIDEO: nn.Linear(config.timestep_dim, 2 * dim)})
        self.input_projections = nn.ModuleDict(
            {
                TEXT: TextProjection(config.text_dim, dim),
                VIDEO: VideoInputProjection(config.latent_channels, dim),
            }
        )
        self.output_projections = nn.ModuleDict({VIDEO: EncoderOutputProjection(dim)})
        self.core = _core(
            config,
            depth=config.encoder_depth,
            dense=config.encoder_dense_blocks,
            mod_dim=None,
        )
        self._masks = LayoutCache()

    def _modulated(
        self, tokens: Tensor, t: Tensor, *, num_chunks: int, chunk_tokens: int
    ) -> Tensor:
        shift, scale = self.t_proj[VIDEO](self.t_embedding(t)).chunk(2, dim=-1)
        return modulate(
            tokens,
            _chunk_rows(shift, chunk_tokens, num_chunks),
            _chunk_rows(1 + scale, chunk_tokens, num_chunks),
        )

    def forward(
        self,
        *,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, EncoderTaps]:
        """Encode K noised chunks (``timesteps`` is ``(1, K)``) with the clean copies of chunks 0 to K-2.

        Returns the conditioning tokens ``(1, K * F, h, w, dim)`` of the noised chunks and the taps.
        """
        num_chunks = int(timesteps.shape[1])
        frames, height, width = (int(d) for d in noised.shape[2:])
        chunk = frames // num_chunks
        n_clean_frames = 0 if clean is None else int(clean.shape[2])
        if frames % num_chunks or n_clean_frames != (num_chunks - 1) * chunk:
            raise ValueError(
                f"{frames} noised and {n_clean_frames} clean frames do not make {num_chunks} chunks"
            )
        tpf = height * width
        project = self.input_projections[VIDEO]
        noised_tok = project(noised)
        parts = [
            self.registers,
            project(anchor).flatten(1, -2),
            self.input_projections[TEXT](text),
            self._modulated(
                noised_tok.flatten(1, -2),
                timesteps,
                num_chunks=num_chunks,
                chunk_tokens=chunk * tpf,
            ),
        ]
        if clean is not None:
            parts.append(
                self._modulated(
                    project(clean).flatten(1, -2),
                    torch.zeros_like(timesteps),
                    num_chunks=num_chunks - 1,
                    chunk_tokens=chunk * tpf,
                )
            )
        x = torch.cat(parts, dim=1)
        num_global = x.shape[1] - (frames + n_clean_frames) * tpf
        rope_ids = encoder_positions(
            num_registers=self.registers.size(1),
            anchor_hw=(int(anchor.shape[3]), int(anchor.shape[4])),
            num_text=text.shape[1],
            frames=frames,
            height=height,
            width=width,
            num_clean_frames=n_clean_frames,
            device=x.device,
        )
        mask = self._masks.get(
            (num_global, frames, tpf, chunk, n_clean_frames, str(x.device)),
            lambda: _layout_mask(
                num_global=num_global,
                frames=frames,
                tokens_per_frame=tpf,
                chunk_frames=chunk,
                num_clean_frames=n_clean_frames,
                device=x.device,
            ),
        )
        out, taps = self.core(
            x,
            rope_ids=rope_ids,
            mask=mask,
            tap_blocks=self.config.tap_blocks,
        )
        if self.config.encoder_zero_add:
            out = out + 0.0
        noised_out = out.narrow(1, num_global, frames * tpf)
        cond = self.output_projections[VIDEO](noised_out).view(
            *noised_tok.shape[:-1], -1
        )
        return cond, EncoderTaps(
            taps=taps,
            embeddings=x,
            rope_ids=rope_ids,
            num_global=num_global,
            frames_per_chunk=chunk,
            tokens_per_frame=tpf,
            num_noised_frames=frames,
            num_clean_frames=n_clean_frames,
        )


class DecoderTower(nn.Module):
    def __init__(self, config: PraxisArchConfig) -> None:
        super().__init__()
        dim = config.dim
        self.config = config
        self.patch_size = (config.decoder_patch[0], config.decoder_patch[1])
        self.registers = nn.Parameter(torch.zeros(1, config.num_registers, dim))
        self.t_embedding = TimestepEmbedding(config.timestep_dim)
        self.t_proj = nn.ModuleDict({VIDEO: nn.Linear(config.timestep_dim, 2 * dim)})
        self.input_projections = nn.ModuleDict(
            {VIDEO: VideoInputProjection(config.decoder_channels, dim)}
        )
        self.output_projections = nn.ModuleDict(
            {VIDEO: DecoderOutputProjection(dim, config.decoder_channels)}
        )
        self.core = _core(
            config,
            depth=config.decoder_depth,
            dense=config.decoder_dense_blocks,
            mod_dim=dim,
        )
        self._masks = LayoutCache()

    def forward(self, *, noised: Tensor, cond: Tensor, timesteps: Tensor) -> Tensor:
        """Velocity ``(1, C, K * F, h, w)`` of the K noised chunks; ``cond`` is the encoder output."""
        num_chunks = int(timesteps.shape[1])
        tok = self.input_projections[VIDEO](unpatch(noised, self.patch_size))
        shape = tok.shape
        _, frames, height, width, _ = shape
        chunk = frames // num_chunks
        tpf = height * width
        shift, scale = self.t_proj[VIDEO](self.t_embedding(timesteps)).chunk(2, dim=-1)
        flat = modulate(
            tok.flatten(1, -2),
            _chunk_rows(shift, chunk * tpf, num_chunks),
            _chunk_rows(1 + scale, chunk * tpf, num_chunks),
        )
        n_reg = self.registers.size(1)
        x = torch.cat([self.registers, flat], dim=1)
        rope_ids = decoder_positions(
            num_registers=n_reg,
            frames=frames,
            height=height,
            width=width,
            device=x.device,
        )
        mask = self._masks.get(
            (n_reg, frames, tpf, chunk, str(x.device)),
            lambda: _layout_mask(
                num_global=n_reg,
                frames=frames,
                tokens_per_frame=tpf,
                chunk_frames=chunk,
                num_clean_frames=0,
                device=x.device,
            ),
        )
        out, _ = self.core(
            x,
            rope_ids=rope_ids,
            mask=mask,
            x_mod=self._modulation(cond, n_reg=n_reg),
        )
        if self.config.decoder_zero_add:
            out = out + 0.0
        v = self.output_projections[VIDEO](out.narrow(1, n_reg, flat.size(1)))
        return patch(v.view(*shape[:-1], -1).movedim(-1, 1), self.patch_size)

    def _modulation(self, cond: Tensor, *, n_reg: int) -> Tensor:
        """The encoder output upsampled to the decoder grid, zero on the registers."""
        ph, pw = self.patch_size
        b, cf, h, w, c = cond.shape
        up = cond[:, :, :, None, :, None, :].expand(b, cf, h, ph, w, pw, c)
        up = up.reshape(b, cf * h * ph * w * pw, c)
        return torch.cat([up.new_zeros(b, n_reg, c), up], dim=1)


class PraxisBackbone(nn.Module):
    def __init__(self, config: PraxisArchConfig) -> None:
        super().__init__()
        self.encoder = EncoderTower(config)
        self.decoder = DecoderTower(config)


class PraxisTransformer(BaseDiT, LayerwiseOffloadableModuleMixin):
    """Video backbone plus action expert; parameter names equal the checkpoint keys."""

    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict = {}
    reverse_param_names_mapping: dict = {}
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}
    # The block stacks that layerwise offload streams to the device one layer at a time.
    layer_names = [
        "backbone.encoder.core.blocks",
        "backbone.decoder.core.blocks",
        "action_expert.blocks",
    ]

    def __init__(
        self, config: PraxisDiTConfig, hf_config: dict[str, Any] | None = None, **kwargs
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch: PraxisArchConfig = config.arch_config
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.backbone = PraxisBackbone(arch)
        self.action_expert = ActionExpert(arch)

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, MoEFeedForward):
                module.prepare_experts()

    def validate_weight_update_source(self, *, weights_path: str | None) -> None:
        raise NotImplementedError(
            "Praxis-1 re-lays the expert banks after loading; weight updates are not supported"
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        *,
        anchor: Tensor,
        clean: Tensor | None,
        noised_actions: Tensor,
        action_timestep: Tensor,
        proprio: Tensor,
        decode: bool = True,
        act: bool = True,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Velocity of the window: ``(video, action)``, None for a part that is not computed.

        ``hidden_states`` is the noised video ``(1, C, K * F, h, w)``, ``timestep`` ``(1, K)`` in the model
        dtype, ``noised_actions`` ``(1, K * F, A)``, ``action_timestep`` ``(1, K)`` float32 and ``proprio``
        ``(1, K, P)`` the boundary pose of each chunk.
        """
        cond, taps = self.backbone.encoder(
            text=encoder_hidden_states,
            anchor=anchor,
            clean=clean,
            noised=hidden_states,
            timesteps=timestep,
        )
        video = None
        if decode:
            video = self.backbone.decoder(
                noised=hidden_states, cond=cond, timesteps=timestep
            )
        if not act:
            return video, None
        chunk = taps.frames_per_chunk
        action = self.action_expert(
            taps=taps.taps,
            embeddings=taps.embeddings,
            rope_ids=taps.rope_ids,
            num_global=taps.num_global,
            num_clean_frames=taps.num_clean_frames,
            num_noised_frames=taps.num_noised_frames,
            tokens_per_frame=taps.tokens_per_frame,
            chunk_size=chunk,
            noised_actions=noised_actions.to(taps.embeddings.dtype),
            action_timesteps=action_timestep.repeat_interleave(chunk, dim=1),
            proprio=proprio.to(taps.embeddings.dtype).repeat_interleave(chunk, dim=1),
        )
        return video, action


EntryClass = PraxisTransformer
