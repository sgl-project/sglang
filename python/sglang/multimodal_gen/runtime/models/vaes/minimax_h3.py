# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import VaeFastPathGate
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae import (
    DacAudioVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)


def _convert_diffusers_video_vae_weights(state_dict, decoder):
    """Restore the released H3 layout from a Diffusers-format checkpoint."""
    converted = {}
    for name, tensor in state_dict.items():
        if name.startswith("encoder."):
            name = (
                name.replace(".down_blocks.", ".down.")
                .replace(".resnets.", ".block.")
                .replace(".downsamplers.0.", ".downsample.")
                .replace(".conv_shortcut.", ".nin_shortcut.")
            )
        name = name.replace("decoder.proj_in.", "decoder.x_embedder.")
        name = name.replace(".attn.to_out.0.", ".attn.to_out.")
        if any(f".attn.to_{part}." in name for part in ("q", "k", "v")):
            continue
        if ".ff.net.0.proj." in name:
            # Diffusers stores [value, gate]; the native SiLU uses [gate, value].
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0)
            name = name.replace(".ff.net.0.proj.", ".ff.w1.")
        name = name.replace(".ff.net.2.", ".ff.w2.")
        converted[name] = tensor

    for layer, block in enumerate(decoder.transformer_blocks):
        prefix = f"decoder.transformer_blocks.{layer}.attn."
        for suffix in ("weight", "bias"):
            parts = [
                state_dict[f"{prefix}to_{part}.{suffix}"] for part in ("q", "k", "v")
            ]
            # Native attention reads [head, qkv, head_dim], not [qkv, head, head_dim].
            packed = torch.stack(
                [
                    part.unflatten(0, (block.attn.heads, block.attn.dim_head))
                    for part in parts
                ],
                dim=1,
            ).flatten(0, 2)
            converted[f"{prefix}to_qkv.{suffix}"] = packed

    # Diffusers omits this non-learned buffer; inference never applies masking.
    converted["decoder.mask_token"] = state_dict["decoder.proj_in.weight"].new_zeros(
        decoder.mask_token.shape
    )
    return converted


class MiniMaxH3VideoVAE(AutoencoderKLLegacy, LayerwiseOffloadableModuleMixin):
    _aliases = ["AutoencoderKLMiniMaxH3"]
    layerwise_offload_dit_group_enabled = False
    # EncoderFCN3D indexes its down containers instead of calling them, so they
    # cannot host layerwise hooks. Keep the small encoder resident.
    layer_names = ["decoder.transformer_blocks"]

    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        arch = config.arch_config
        parallel_decode_mode = config.resolved_parallel_decode_mode()
        use_tiled_decode = config.use_tiling and parallel_decode_mode == "tiled"
        super().__init__(
            in_channels=3,
            out_ch=3,
            ch=128,
            embed_dim=24,
            z_channels=24,
            use_3d_conv=True,
            zq_ch_encoder=None,
            zq_ch_decoder=None,
            num_res_blocks=2,
            num_res_blocks_decoder=None,
            ch_mult=[1, 2, 2, 4, 4, 8],
            space_down=[2, 2, 2, 2, 1, 1],
            space_up=[1, 2, 2, 2, 2, 1],
            time_down=[1, 2, 2, 1, 1, 1],
            time_up=None,
            padding_mode="reflect",
            padding_mode_t=None,
            use_t_isolated_gn=True,
            causal_encoder=True,
            causal_decoder=False,
            use_vit_decoder=True,
            vit_decoder_kwargs={
                "dim_head": 64,
                "ffn_activation_fn": "silu",
                "ffn_use_gated": True,
                "heads": 32,
                "norm_affine": True,
                "norm_type": "rms_norm",
                "num_layers": 36,
                "qk_norm_affine": False,
                "qk_norm_type": "rms_norm",
                "rope_dim_ratio": 0.75,
                "rope_theta": 100.0,
            },
            shift_factor=0.0,
            scaling_factor=1.0,
            pixel_norm_type="imagenet",
            clip_length=arch.vae_clip_length,
            token_drop=arch.vae_token_drop,
            encoder_tiling=bool(arch.vae_encoder_tiling),
            decoder_tiling=use_tiled_decode,
            parallel_tiling=use_tiled_decode
            and config.use_parallel_decode
            and config.use_parallel_tiling
            and bool(arch.vae_parallel_tiling),
            decoder_tile_batch=config.decoder_tile_batch,
            window_batch=config.window_batch,
            tile_size=int(arch.vae_tile_size),
            tile_overlap_min=int(arch.vae_tile_overlap_min),
            encoder_parallel=False,
            decoder_parallel=False,
            chunk_dim=int(arch.vae_chunk_dim),
        )
        self.sglang_config = config
        self.use_parallel_decode = config.use_parallel_decode
        self.parallel_decode_mode = parallel_decode_mode
        # The tiled decode batches under this gate; the CUDA installer registers
        # it for use_vae_fast_path and adds its attention fusion, so other
        # platforms keep the one-tile-per-forward decode.
        self._sgl_gate = VaeFastPathGate()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "decoder.proj_in.weight" in state_dict:
            state_dict = _convert_diffusers_video_vae_weights(state_dict, self.decoder)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        return self.decoder.prepare_autocast_linear_weights(dtype)


class MiniMaxH3AudioVAE(DacAudioVAE, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # BigVGAN stores each executable upsampler inside a one-element ModuleList.
    # The outer ``decoder.ups`` containers are indexed but never called, so hooks
    # must target the inner lists whose ConvTranspose1d modules run forward.
    layer_names = [
        "encoder.block",
        *(f"decoder.ups.{index}" for index in range(7)),
        "decoder.resblocks",
    ]

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__(
            encoder_dim=64,
            encoder_rates=[2, 4, 4, 5, 5],
            latent_dim=2048,
            decoder_dim=1024,
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            sample_rate=32000,
            vae_latent_channels=32,
            attn_proj=True,
            decoder_type="bigvgan",
        )
        self.config = config


EntryClass = [MiniMaxH3VideoVAE, MiniMaxH3AudioVAE]

__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"]
