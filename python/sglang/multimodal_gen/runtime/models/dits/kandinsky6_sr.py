# SPDX-License-Identifier: Apache-2.0
"""Native sglang-diffusion port of the Kandinsky 6 video super-resolution DiT.

Reference: ``kandinsky_sr.core.components.model.dit.DiffusionTransformer3D`` (text-free
mode) and ``dx_dit.DXDiTWrapper`` (``n_grid`` velocity grids per token).  The building
blocks (time / visual embeddings, RoPE3D, modulation, dense attention, feed-forward) are
the ones of the K6 T2VA DiT (``kandinsky6.py``); this module only adds what the
super-resolution model differs in:

* no text tower and no cross-attention: the decoder block has a 6-parameter modulation
  (self-attention + FFN) and the pooled-text contribution to the time embedding is a
  learned constant ``pooled_bias``;
* ``instruct_type`` conditioning: the input layer is ``2 * in + 1`` channels wide for a
  ``visual_cond`` or ``channel`` / ``hybrid`` / ``hybrid_anchor`` checkpoint;
* a DX head (distilled pi-Flow checkpoints): ``out_layer`` emits ``head_width``
  channels per token, all ``n_grid`` velocity grids of it (a flow-matching head is one
  grid);
* post-load ``attribute_overrides`` (``instruct_type``, ``visual_cond``,
  ``attention_params``) that change inference behaviour but not the architecture.

Layout is batched ``[B, T, H, W, C]`` (the reference packs samples along time with
``cu_seqlens``).  Attention is always dense (never block-sparse); a checkpoint whose
``attention_params`` ask for NABLA sparse attention is run dense with a warning.

Numerics follow the reference under ``torch.autocast(bfloat16)``: parameters may be bf16,
the residual stream and all norm / modulation arithmetic are fp32, and matmul inputs are
cast to the parameter dtype.

TP shards the shared K6 attention/FFN projections. SP shards the visual tokens and
RoPE together before the decoder blocks, masks tail padding, and gathers before
the output head. The VAE, latent upscaler, and tile scheduler remain replicated.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    ATTRIBUTE_OVERRIDE_WHITELIST,
    Kandinsky6SRArchConfig,
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.kandinsky6 import (
    Kandinsky6Attention,
    Kandinsky6FeedForward,
    Kandinsky6Modulation,
    Kandinsky6OutLayer,
    Kandinsky6RoPE3D,
    Kandinsky6TimeEmbeddings,
    Kandinsky6VisualEmbeddings,
    _build_rotary_freqs,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)

_ARCH_CONFIG_DEFAULTS = Kandinsky6SRDitConfig().arch_config
_BLOCK_CONTAINERS = ("visual_transformer_blocks",)
_TIME_EMBEDDING_NAMES = ("time_embeddings", "motion_embeddings", "lq_noise_embeddings")


def _is_sr_transformer_block(name: str, module: object) -> bool:
    return is_module_list_entry_in(name, _BLOCK_CONTAINERS)


def _modulate(
    norm: LayerNormScaleShift, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """fp32 ``LN(x) * (scale + 1) + shift`` (no cast back: the caller picks the dtype)."""
    return norm(x.float(), shift=shift, scale=scale)


class Kandinsky6SRRoPE3D(Kandinsky6RoPE3D):
    """``Kandinsky6RoPE3D`` whose fp32 angle tables survive ``module.to(dtype=bf16)``.

    The component residency manager casts a whole module to its target dtype when it
    moves it to the GPU.  The RoPE tables (buffers) hold rotation angles up to position
    128; rounding them to bf16 would silently degrade the position encoding.
    """

    def _apply(self, fn, recurse=True):
        device = fn(torch.zeros(1)).device
        for name, buffer in self._buffers.items():
            if buffer is not None:
                self._buffers[name] = buffer.to(device=device)
        return self


class Kandinsky6SRDecoderBlock(nn.Module):
    """Text-free decoder block: modulated self-attention + modulated feed-forward.

    The residual stream ``visual_embed`` is kept in fp32 (like the reference, where the
    gated residual add promotes it); ``compute_dtype`` is the dtype fed to the linears.
    """

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)
        self.self_attention_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.self_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("self_attention", prefix),
            quant_config=quant_config,
        )
        self.feed_forward_norm = LayerNormScaleShift(
            model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
        )
        self.feed_forward = Kandinsky6FeedForward(
            model_dim,
            ff_dim,
            prefix=add_prefix("feed_forward", prefix),
            quant_config=quant_config,
        )

    def forward(
        self,
        visual_embed: torch.Tensor,
        time_embed: torch.Tensor,
        rope: torch.Tensor,
        compute_dtype: torch.dtype,
        attn_mask_meta: dict | None = None,
    ) -> torch.Tensor:
        modulation = self.visual_modulation(time_embed).unsqueeze(dim=1)
        self_attn_params, ff_params = torch.chunk(modulation, 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = _modulate(self.self_attention_norm, visual_embed, shift, scale)
        out = self.self_attention(
            out.to(compute_dtype), rotary_emb=rope, attn_mask_meta=attn_mask_meta
        )
        visual_embed = visual_embed.float() + gate.float() * out.float()

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = _modulate(self.feed_forward_norm, visual_embed, shift, scale)
        out = self.feed_forward(out.to(compute_dtype))
        return visual_embed + gate.float() * out.float()


class Kandinsky6SROutLayer(Kandinsky6OutLayer):
    """K6 ``OutLayer`` for an fp32 residual stream (norm in fp32, linear in weight dtype)."""

    def forward(
        self, visual_embed: torch.Tensor, time_embed: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        x = (
            self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0)
            + shift.float()[:, None, None]
        )
        x, _ = self.out_layer(x.to(self.out_layer.weight.dtype))

        batch_size, duration, height, width, _ = x.shape
        pt, ph, pw = self.patch_size
        return (
            x.view(batch_size, duration, height, width, -1, pt, ph, pw)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )


class Kandinsky6SRTransformer3DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    """Text-free Kandinsky 6 SR DiT (optionally with a DX / pi-Flow head)."""

    _fsdp_shard_conditions = [_is_sr_transformer_block]
    _compile_conditions = [_is_sr_transformer_block]
    param_names_mapping = _ARCH_CONFIG_DEFAULTS.param_names_mapping
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}
    # Dense attention only: the sparse (NABLA) path of the reference is not ported.
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }

    @staticmethod
    def _validate_tp_config(*, arch: Kandinsky6SRArchConfig, tp_size: int) -> None:
        if tp_size <= 0:
            raise ValueError("Kandinsky6SR TP size must be positive.")
        for name, value in (
            ("num_attention_heads", arch.num_attention_heads),
            ("ff_dim", arch.ff_dim),
        ):
            if value % tp_size:
                raise ValueError(
                    f"Kandinsky6SR {name}={value} must be divisible by TP size {tp_size}."
                )

    @staticmethod
    def _validate_sequence_parallel_config(
        *, arch: Kandinsky6SRArchConfig, tp_size: int, ulysses_size: int, ring_size: int
    ) -> None:
        if ulysses_size <= 0:
            raise ValueError("Kandinsky6SR Ulysses size must be positive.")
        if ring_size <= 0:
            raise ValueError("Kandinsky6SR ring size must be positive.")
        if ulysses_size == 1 and ring_size == 1:
            return
        # Ring Attention rotates whole K/V shards between ranks rather than
        # splitting heads, so only Ulysses constrains head divisibility.
        local_heads = arch.num_attention_heads // tp_size
        if local_heads % ulysses_size:
            raise ValueError(
                f"Kandinsky6SR TP-local attention heads {local_heads} must be "
                f"divisible by Ulysses size {ulysses_size} (total heads="
                f"{arch.num_attention_heads}, TP={tp_size})."
            )

    def __init__(
        self,
        config: Kandinsky6SRDitConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch: Kandinsky6SRArchConfig = self.config
        self.quant_config = quant_config
        head_dim = sum(arch.axes_dims)

        tp_size = get_tp_world_size()
        ulysses_size, _ = get_ulysses_ctx()
        ring_size, _ = get_ring_ctx()
        self._validate_tp_config(arch=arch, tp_size=tp_size)
        self._validate_sequence_parallel_config(
            arch=arch, tp_size=tp_size, ulysses_size=ulysses_size, ring_size=ring_size
        )

        self.in_visual_dim = arch.in_visual_dim
        self.base_out_visual_dim = arch.base_out_visual_dim
        self.n_grid = arch.n_grid
        self.model_dim = arch.model_dim
        self.patch_size = arch.patch_size
        self.use_motion_score = arch.use_motion_score
        self.use_lq_noise_cond = arch.use_lq_noise_cond
        # The input width is fixed by the *trained* configuration ...
        self.visual_embed_dim = (
            2 * arch.in_visual_dim + 1
            if arch.trained_wide_input
            else arch.in_visual_dim
        )
        self._build_layers(arch, head_dim=head_dim, quant_config=quant_config)

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.layer_names = list(_BLOCK_CONTAINERS)
        # ... while inference behaviour comes from the post-load overrides.
        self._apply_attribute_overrides(arch)
        self._require_all_checkpoint_keys()
        self.__post_init__()

    def _build_layers(
        self,
        arch: Kandinsky6SRArchConfig,
        *,
        head_dim: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.time_embeddings = Kandinsky6TimeEmbeddings(arch.model_dim, arch.time_dim)
        if arch.use_motion_score:
            self.motion_embeddings = Kandinsky6TimeEmbeddings(
                arch.model_dim, arch.time_dim
            )
        if arch.use_lq_noise_cond:
            # Present in the checkpoint; never used at inference (kept for loading).
            self.lq_noise_embeddings = Kandinsky6TimeEmbeddings(
                arch.model_dim, arch.time_dim
            )
        # Constant that replaces the pooled empty-caption embedding of the text DiT.
        self.pooled_bias = nn.Parameter(torch.zeros(arch.time_dim))
        self.visual_embeddings = Kandinsky6VisualEmbeddings(
            self.visual_embed_dim,
            arch.model_dim,
            arch.patch_size,
            prefix=add_prefix("visual_embeddings", self.prefix),
        )
        self.visual_rope_embeddings = Kandinsky6SRRoPE3D(arch.axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                Kandinsky6SRDecoderBlock(
                    arch.model_dim,
                    arch.time_dim,
                    arch.ff_dim,
                    head_dim,
                    self._supported_attention_backends,
                    prefix=add_prefix(f"visual_transformer_blocks.{i}", self.prefix),
                    quant_config=quant_config,
                )
                for i in range(arch.num_visual_blocks)
            ]
        )
        self.out_layer = Kandinsky6SROutLayer(
            arch.model_dim,
            arch.time_dim,
            arch.head_width,
            arch.patch_size,
        )

    def _apply_attribute_overrides(self, arch: Kandinsky6SRArchConfig) -> None:
        self.instruct_type = arch.instruct_type
        self.visual_cond = arch.visual_cond
        self.attention_params = arch.attention_params
        for name in ATTRIBUTE_OVERRIDE_WHITELIST:
            if name in arch.attribute_overrides:
                setattr(self, name, arch.attribute_overrides[name])
        self._check_input_width_matches_overrides()
        sparse_type = arch.requested_sparse_attention()
        if sparse_type is not None:
            logger.warning(
                "Kandinsky6SR: attention_params request %r sparse attention; this port "
                "runs dense attention, so outputs differ from the sparse reference.",
                sparse_type,
            )

    def _check_input_width_matches_overrides(self) -> None:
        run_wide = self.visual_cond or self.instruct_type in (
            "channel",
            "hybrid",
            "hybrid_anchor",
        )
        run_width = 2 * self.in_visual_dim + 1 if run_wide else self.in_visual_dim
        if run_width != self.visual_embed_dim:
            raise ValueError(
                f"Kandinsky6SR: the trained input layer takes {self.visual_embed_dim} "
                f"channels but instruct_type={self.instruct_type!r} with "
                f"visual_cond={self.visual_cond} feeds {run_width}; set "
                "attribute_overrides.visual_cond=true to run a wide-input checkpoint "
                "under instruct_type='noise'."
            )

    def _require_all_checkpoint_keys(self) -> None:
        """A missing tensor must fail loudly instead of being zero-filled by the loader."""
        for param in self.parameters():
            param.missing_param_init = "error"

    def preprocess_loaded_state_dict(
        self, weight_iterator: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Pass tensors through, then raise if the checkpoint held unexpected keys."""
        map_name = get_param_names_mapping(self.param_names_mapping)
        expected = set(self.state_dict())
        unexpected: list[str] = []
        for name, tensor in weight_iterator:
            if map_name(name)[0] not in expected:
                unexpected.append(name)
            yield name, tensor
        if unexpected:
            shown = ", ".join(unexpected[:10])
            raise ValueError(
                f"Kandinsky6SR checkpoint has {len(unexpected)} unexpected keys "
                f"(first: {shown}); the transformer config does not match the weights."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        visual_rope_pos: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor]
        ),
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        motion_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the velocity (flow-matching head) or the DX velocity grids.

        Args:
            hidden_states: ``[B, T, H, W, C]`` latent (channel-last, ``C`` = input width).
            timestep: ``[B]`` model times, already multiplied by 1000.
            visual_rope_pos: ``(arange(T), arange(H // ph), arange(W // pw))``.
            scale_factor: RoPE frequency scaling per (t, h, w) axis.
            motion_score: optional ``[1]`` / ``[B]`` score, used iff ``use_motion_score``.

        Returns:
            ``[B, T, H, W, head_width]``; for a DX head, grid ``g`` of channel ``c``
            sits at index ``g * base_out_visual_dim + c`` (the sampler reshapes it).
        """
        self._check_latent_shape(hidden_states)
        compute_dtype = self.visual_embeddings.in_layer.weight.dtype
        time_embed = self.time_embeddings(timestep)
        if motion_score is not None and self.use_motion_score:
            time_embed = time_embed + self.motion_embeddings(motion_score)
        time_embed = time_embed + self.pooled_bias

        visual_embed = self.visual_embeddings(hidden_states.to(compute_dtype))
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(
            visual_shape, visual_rope_pos, scale_factor
        )
        visual_embed = visual_embed.flatten(1, 3)
        visual_rope = visual_rope.flatten(1, 3)
        visual_embed, shard = shard_seq(visual_embed)
        visual_rope = shard_like(visual_rope, shard, pad_mode="repeat_last")
        attn_meta = tail_attn_meta(shard, visual_embed.shape[0], visual_embed.device)
        for block in self.visual_transformer_blocks:
            visual_embed = block(
                visual_embed,
                time_embed,
                visual_rope,
                compute_dtype,
                attn_mask_meta=attn_meta,
            )
        visual_embed = gather_seq(visual_embed, shard.orig_len)
        return self.out_layer(visual_embed.reshape(*visual_shape, -1), time_embed)

    def _check_latent_shape(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.ndim != 5 or hidden_states.shape[-1] != self.visual_embed_dim:
            raise ValueError(
                "Kandinsky6SR expects a [B, T, H, W, "
                f"{self.visual_embed_dim}] latent, got {tuple(hidden_states.shape)}"
            )
        height, width = hidden_states.shape[2:4]
        if height % self.patch_size[1] or width % self.patch_size[2]:
            raise ValueError(
                f"latent {height}x{width} is not divisible by patch {self.patch_size}"
            )

    def post_load_weights(self) -> None:
        """Rebuild RoPE / time-embedding tables that a meta-device init left on meta."""
        device = next(self.parameters()).device
        rope = self.visual_rope_embeddings
        if any(buf.is_meta for buf in rope.buffers()):
            self.visual_rope_embeddings = Kandinsky6SRRoPE3D(
                rope.axes_dims, rope.max_pos, rope.max_period
            ).to(device)
        for name in _TIME_EMBEDDING_NAMES:
            embeddings = self._modules.get(name)
            if embeddings is not None and embeddings.freqs.is_meta:
                embeddings.freqs = _build_rotary_freqs(
                    embeddings.model_dim // 2, embeddings.max_period
                ).to(device=device)


EntryClass = Kandinsky6SRTransformer3DModel
