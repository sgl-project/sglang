# SPDX-License-Identifier: Apache-2.0
#
# Lumina-Image-2.0 DiT (NextDiT). The attention block, SwiGLU feed-forward,
# timestep embedder, and axes-RoPE tables are imported from zimage.py rather
# than forked; Lumina's precision differences are opt-in keyword arguments on
# those shared classes.
#
# Reference: https://arxiv.org/abs/2503.21758
# Ported from diffusers Lumina2Transformer2DModel.

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.lumina2 import Lumina2Config
from sglang.multimodal_gen.runtime.layers.attention import (
    build_varlen_mask_meta_from_lengths,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT

# NOTE: deliberately not imported from zimage, whose value differs. Picking up
# the wrong one builds a silently wrong adaLN width, with no shape error.
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    FeedForward,
    RopeEmbedder,
    TimestepEmbedder,
    ZImageAttention,
)

# diffusers LuminaRMSNormZero modulates from min(hidden_size, 1024).
ADALN_EMBED_DIM = 1024

# Only for the class-level weight-mapping attrs; instances use their own config.
_DEFAULT_ARCH = Lumina2Config().arch_config


class FP32SiluAndMul(nn.Module):
    """SwiGLU activation with SiLU evaluated in fp32, matching Lumina.

    Injected into the shared ``FeedForward`` via its ``activation`` argument;
    Z-Image keeps the fused bf16 ``SiluAndMul``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate.float()).to(gate.dtype) * value


class Lumina2RMSNorm(nn.Module):
    """diffusers RMSNorm, cast boundary included (normalization.py:553-562).

    NOTE: the affine multiply runs in the weight's dtype, not fp32. Hoisting it
    into fp32 rounds once instead of twice and looks strictly more accurate, but
    it is not what the checkpoint was matched against -- it drifts by up to
    ~0.03 per element, in every norm of every block.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


def _ffn_hidden_dim(
    dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]
) -> int:
    """diffusers LuminaFeedForward inner dim: 4*dim, optionally scaled, rounded up.

    NOTE: a plain round-up, *not* the 8/3*dim SwiGLU rule Z-Image uses. dim=2304
    with no multiplier gives 9216, matching the published checkpoint's
    feed_forward.linear_1 = [9216, 2304].
    """
    inner = 4 * dim
    if ffn_dim_multiplier is not None:
        inner = int(ffn_dim_multiplier * inner)
    return multiple_of * ((inner + multiple_of - 1) // multiple_of)


class Lumina2TransformerBlock(nn.Module):
    """Sandwich-norm NextDiT block. ``modulation=False`` is the caption/context
    refiner variant (no adaLN). Structurally identical to ZImageTransformerBlock
    except for the adaLN input dim and fp32 normalization/activation policies."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.modulation = modulation

        self.attention = ZImageAttention(
            dim=dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            qk_norm_factory=Lumina2RMSNorm,
            allow_fused_qk_norm_rope=False,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        # Every SP rank holds the complete sequence, so USP collectives would
        # treat replicated sequences as shards. See shard_latents_for_sp.
        self.attention.attn.skip_sequence_parallel = True

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=_ffn_hidden_dim(dim, multiple_of, ffn_dim_multiplier),
            activation=FP32SiluAndMul(),
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )

        self.attention_norm1 = Lumina2RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = Lumina2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = Lumina2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = Lumina2RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                ReplicatedLinear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        adaln_input: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_mask_meta: Optional[dict] = None,
    ):
        attn_kwargs = dict(
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
        )

        if self.modulation:
            assert adaln_input is not None
            mod, _ = self.adaLN_modulation[1](self.adaLN_modulation[0](adaln_input))
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=-1)

            attn_out = self.attention(
                self.attention_norm1(x) * (1 + scale_msa), **attn_kwargs
            )
            x = x + torch.tanh(gate_msa) * self.attention_norm2(attn_out)

            ffn_out = self.feed_forward(self.ffn_norm1(x) * (1 + scale_mlp))
            x = x + torch.tanh(gate_mlp) * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(self.attention_norm1(x), **attn_kwargs)
            x = x + self.attention_norm2(attn_out)
            ffn_out = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn_out)

        return x


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    """Timestep MLP (-> min(dim,1024)) + Gemma-2 caption projection (RMSNorm+Linear).

    Mirrors diffusers Lumina2CombinedTimestepCaptionEmbedding.
    """

    def __init__(self, hidden_size: int, cap_feat_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(
            out_size=min(hidden_size, ADALN_EMBED_DIM)
        )
        self.caption_embedder = nn.Sequential(
            Lumina2RMSNorm(cap_feat_dim, eps=norm_eps),
            ReplicatedLinear(cap_feat_dim, hidden_size, bias=True),
        )

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        time_embed = self.timestep_embedder(timestep)
        cap = self.caption_embedder[0](encoder_hidden_states)
        cap, _ = self.caption_embedder[1](cap)
        return time_embed, cap


class Lumina2FinalLayer(nn.Module):
    """adaLN-continuous final norm + projection to patch*patch*out_channels.

    Mirrors diffusers LuminaLayerNormContinuous (adaLN input min(dim,1024)).
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )
        self.linear = ColumnParallelLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            gather_output=True,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, _ = self.adaLN_modulation[1](self.adaLN_modulation[0](c))
        x = self.norm_final(x) * (1 + scale.unsqueeze(1))
        x, _ = self.linear(x)
        return x


class Lumina2Transformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """Lumina-Image-2.0 NextDiT. Class name matches the diffusers ``_class_name``."""

    # NOTE: no _supports_gradient_checkpointing. forward() has no such branch,
    # so advertising it would let a caller enable a no-op.
    _no_split_modules = ["Lumina2TransformerBlock"]
    _fsdp_shard_conditions = _DEFAULT_ARCH._fsdp_shard_conditions
    param_names_mapping = _DEFAULT_ARCH.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_ARCH.reverse_param_names_mapping
    # Lets is_layer_skipped() resolve --quantization-ignored-layers entries,
    # which name checkpoint weights. Both fusions are unconditional.
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "w13": ["linear_1", "linear_3"],
    }

    def __init__(
        self,
        config: Lumina2Config,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        patch_size = arch.patch_size
        self.patch_size = patch_size
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        dim = arch.hidden_size
        self.hidden_size = dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.t_scale = arch.t_scale
        # Rows per RoPE axis: (caption/time, patch row, patch column).
        self.axes_lens = arch.axes_lens

        self.x_embedder = ReplicatedLinear(
            patch_size * patch_size * arch.in_channels, dim, bias=True
        )
        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=dim, cap_feat_dim=arch.cap_feat_dim, norm_eps=arch.norm_eps
        )
        self.rope_embedder = RopeEmbedder(
            theta=arch.rope_theta,
            axes_dims=arch.axes_dim_rope,
            axes_lens=arch.axes_lens,
            # diffusers builds the phase in fp64 (transformer_lumina2.py:245).
            # It also rotates in complex128; GQA sends us to a bf16 fallback.
            freqs_dtype=torch.float64,
        )

        def _blocks(name: str, count: int, modulation: bool) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    Lumina2TransformerBlock(
                        dim=dim,
                        n_heads=arch.num_attention_heads,
                        n_kv_heads=arch.num_kv_heads,
                        multiple_of=arch.multiple_of,
                        ffn_dim_multiplier=arch.ffn_dim_multiplier,
                        norm_eps=arch.norm_eps,
                        qk_norm=arch.qk_norm,
                        modulation=modulation,
                        quant_config=quant_config,
                        prefix=f"{name}.{i}",
                    )
                    for i in range(count)
                ]
            )

        self.noise_refiner = _blocks(
            "noise_refiner", arch.num_refiner_layers, modulation=True
        )
        self.context_refiner = _blocks(
            "context_refiner", arch.num_refiner_layers, modulation=False
        )
        self.layers = _blocks("layers", arch.num_layers, modulation=True)
        self.norm_out = Lumina2FinalLayer(dim, patch_size, arch.out_channels)
        self.layer_names = ["layers"]

    def _check_rope_axis(self, axis: int, max_index: int, what: str, knob: str) -> None:
        """Reject a position that would index past RoPE table ``axis``."""
        limit = self.axes_lens[axis]
        if max_index >= limit:
            raise ValueError(
                f"Lumina-2 {what} needs axis-{axis} RoPE position {max_index}, "
                f"but that table has only {limit} rows; lower {knob}."
            )

    def _patchify_and_rope(
        self, hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor
    ):
        """Patchify latents and build the three RoPE tables Lumina needs.

        Position ids are 3-axis: axis 0 is the caption/time axis, axes 1 and 2
        are the image patch row/column. Caption tokens occupy ids 0..cap_len on
        axis 0 with axes 1/2 pinned at 0; image tokens pin axis 0 at cap_len and
        carry their row/col on axes 1/2. Mirrors diffusers Lumina2RotaryPosEmbed.

        NOTE: captions must be right-padded. Caption extents come from
        ``encoder_attention_mask.sum()``, which is only correct if the real
        tokens come first; a left-padded mask would silently mis-assign every
        caption position and RoPE frequency. Lumina2PipelineConfig.tokenize_prompt
        pins padding_side to enforce this.
        """
        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        if height % p or width % p:
            raise ValueError(
                f"Lumina-2 latents must be divisible by patch_size {p}, got "
                f"{height}x{width}. Requested image height/width must be a "
                f"multiple of vae_scale_factor * patch_size (16 by default)."
            )
        post_patch_height, post_patch_width = height // p, width // p
        image_seq_len = post_patch_height * post_patch_width
        device = hidden_states.device

        encoder_seq_len = encoder_attention_mask.shape[1]
        cap_lens: List[int] = encoder_attention_mask.sum(dim=1).tolist()
        # Unguarded, each of these is a device-side assert in the RoPE gather.
        # Axis 0 needs cap_len itself: image tokens all sit at that position.
        self._check_rope_axis(
            0, encoder_seq_len, "caption length", "max_sequence_length"
        )
        self._check_rope_axis(
            1, post_patch_height - 1, "latent patch rows", "image height"
        )
        self._check_rope_axis(
            2, post_patch_width - 1, "latent patch columns", "image width"
        )
        seq_lens = [cap_len + image_seq_len for cap_len in cap_lens]
        max_seq_len = max(seq_lens)

        row_ids = (
            torch.arange(post_patch_height, dtype=torch.long, device=device)
            .view(-1, 1)
            .repeat(1, post_patch_width)
            .flatten()
        )
        col_ids = (
            torch.arange(post_patch_width, dtype=torch.long, device=device)
            .view(1, -1)
            .repeat(post_patch_height, 1)
            .flatten()
        )

        position_ids = torch.zeros(
            batch_size, max_seq_len, 3, dtype=torch.long, device=device
        )
        for i, (cap_len, seq_len) in enumerate(zip(cap_lens, seq_lens)):
            position_ids[i, :cap_len, 0] = torch.arange(
                cap_len, dtype=torch.long, device=device
            )
            position_ids[i, cap_len:seq_len, 0] = cap_len
            position_ids[i, cap_len:seq_len, 1] = row_ids
            position_ids[i, cap_len:seq_len, 2] = col_ids

        cos, sin = self.rope_embedder(position_ids.view(-1, 3))
        joint_cos = cos.view(batch_size, max_seq_len, -1)
        joint_sin = sin.view(batch_size, max_seq_len, -1)

        rope_dim = joint_cos.shape[-1]
        cap_cos = joint_cos.new_zeros(batch_size, encoder_seq_len, rope_dim)
        cap_sin = joint_sin.new_zeros(batch_size, encoder_seq_len, rope_dim)
        img_cos = joint_cos.new_zeros(batch_size, image_seq_len, rope_dim)
        img_sin = joint_sin.new_zeros(batch_size, image_seq_len, rope_dim)
        for i, (cap_len, seq_len) in enumerate(zip(cap_lens, seq_lens)):
            cap_cos[i, :cap_len] = joint_cos[i, :cap_len]
            cap_sin[i, :cap_len] = joint_sin[i, :cap_len]
            img_cos[i] = joint_cos[i, cap_len:seq_len]
            img_sin[i] = joint_sin[i, cap_len:seq_len]

        # (B, C, H, W) -> (B, image_seq_len, p*p*C)
        image_tokens = (
            hidden_states.view(
                batch_size, channels, post_patch_height, p, post_patch_width, p
            )
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )

        return (
            image_tokens,
            (cap_cos, cap_sin),
            (img_cos, img_sin),
            (joint_cos, joint_sin),
            cap_lens,
            seq_lens,
        )

    @staticmethod
    def _padding_mask(
        lengths: List[int], target_len: int, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        """Key-padding mask over a left-packed sequence, or (None, None).

        Uniform lengths need no mask -- the common single-prompt case. diffusers
        makes the same shortcut via ``use_mask``.
        """
        if all(length == target_len for length in lengths):
            return None, None
        length_key = tuple(int(length) for length in lengths)
        positions = torch.arange(target_len, device=device).unsqueeze(0)
        length_tensor = torch.as_tensor(
            length_key, dtype=torch.long, device=device
        ).unsqueeze(1)
        mask = positions < length_tensor
        meta = build_varlen_mask_meta_from_lengths(length_key, target_len, device)
        return mask, meta

    def _unpatchify(
        self,
        joint: torch.Tensor,
        cap_lens: List[int],
        seq_lens: List[int],
        height: int,
        width: int,
    ) -> torch.Tensor:
        p = self.patch_size
        out = []
        for i, (cap_len, seq_len) in enumerate(zip(cap_lens, seq_lens)):
            out.append(
                joint[i][cap_len:seq_len]
                .view(height // p, width // p, p, p, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        return torch.stack(out, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, _, height, width = hidden_states.shape
        device = hidden_states.device

        # Lumina treats t=0 as noise and t=1 as the image, the reverse of the
        # FlowMatchEuler schedule the denoising stage hands us. diffusers
        # reverses in the pipeline; here keeps it out of the pipeline config.
        t = 1.0 - timestep.to(torch.float32) / self.t_scale
        adaln_input, cap_feats = self.time_caption_embed(t, encoder_hidden_states)
        adaln_input = adaln_input.to(hidden_states.dtype)

        (
            image_tokens,
            cap_freqs_cis,
            img_freqs_cis,
            joint_freqs_cis,
            cap_lens,
            seq_lens,
        ) = self._patchify_and_rope(hidden_states, encoder_attention_mask)

        image_tokens, _ = self.x_embedder(image_tokens)

        cap_mask, cap_mask_meta = self._padding_mask(
            cap_lens, encoder_attention_mask.shape[1], device
        )
        for layer in self.context_refiner:
            cap_feats = layer(
                cap_feats,
                cap_freqs_cis,
                attn_mask=cap_mask,
                attn_mask_meta=cap_mask_meta,
            )

        # Image tokens are dense: every sample has the same patch count, hence no mask.
        for layer in self.noise_refiner:
            image_tokens = layer(image_tokens, img_freqs_cis, adaln_input)

        # Joint sequence, left-packed as [caption | image] per sample.
        max_seq_len = max(seq_lens)
        joint = image_tokens.new_zeros(batch_size, max_seq_len, self.hidden_size)
        for i, (cap_len, seq_len) in enumerate(zip(cap_lens, seq_lens)):
            joint[i, :cap_len] = cap_feats[i, :cap_len]
            joint[i, cap_len:seq_len] = image_tokens[i]

        joint_mask, joint_mask_meta = self._padding_mask(seq_lens, max_seq_len, device)
        for layer in self.layers:
            joint = layer(
                joint,
                joint_freqs_cis,
                adaln_input,
                attn_mask=joint_mask,
                attn_mask_meta=joint_mask_meta,
            )

        joint = self.norm_out(joint, adaln_input)
        output = self._unpatchify(joint, cap_lens, seq_lens, height, width)

        # Lumina's velocity points the opposite way from the scheduler's
        # convention. diffusers negates just before scheduler.step; doing it
        # here is equivalent, since CFG and renorm are both odd.
        return -output


EntryClass = [Lumina2Transformer2DModel]
