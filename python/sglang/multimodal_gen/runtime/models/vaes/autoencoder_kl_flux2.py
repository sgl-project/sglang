import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.autoencoders.vae import (
    Decoder,
    DecoderOutput,
    DiagonalGaussianDistribution,
    Encoder,
)

from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.runtime.cache.conditioning import cached_vae_encode
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl import AutoencoderKLMixin
from sglang.multimodal_gen.runtime.models.vaes.common import (
    ParallelTiledVAE,
    can_install_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.diffusers_spatial import (
    enable_diffusers_decoder_spatial_parallel,
)


class AutoencoderKLFlux2(AutoencoderKLMixin, ParallelTiledVAE):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    def __init__(
        self,
        config: Flux2VAEConfig,
    ):
        super().__init__(config=config)

        self.config = config
        arch_config = config.arch_config

        in_channels: int = arch_config.in_channels
        out_channels: int = arch_config.out_channels
        down_block_types: Tuple[str, ...] = arch_config.down_block_types
        up_block_types: Tuple[str, ...] = arch_config.up_block_types
        block_out_channels: Tuple[int, ...] = arch_config.block_out_channels
        decoder_block_out_channels: Optional[Tuple[int, ...]] = getattr(
            arch_config, "decoder_block_out_channels", None
        )
        layers_per_block: int = arch_config.layers_per_block
        act_fn: str = arch_config.act_fn
        latent_channels: int = arch_config.latent_channels
        norm_num_groups: int = arch_config.norm_num_groups
        sample_size: int = arch_config.sample_size
        use_quant_conv: bool = arch_config.use_quant_conv
        use_post_quant_conv: bool = arch_config.use_post_quant_conv
        mid_block_add_attention: bool = arch_config.mid_block_add_attention
        batch_norm_eps: float = arch_config.batch_norm_eps
        batch_norm_momentum: float = arch_config.batch_norm_momentum
        patch_size: Tuple[int, int] = arch_config.patch_size
        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=decoder_block_out_channels or block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1)
            if use_post_quant_conv
            else None
        )

        self.bn = nn.BatchNorm2d(
            math.prod(patch_size) * latent_channels,
            eps=batch_norm_eps,
            momentum=batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        )

        self.use_slicing = False
        self.use_tiling = False
        self._spatial_parallel_decode_enabled = False
        self._spatial_parallel_upsample_count = 0
        if can_install_spatial_shard_parallel_decode(self.config):
            self._spatial_parallel_upsample_count = (
                enable_diffusers_decoder_spatial_parallel(self.decoder)
            )
            self._spatial_parallel_decode_enabled = True

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(
            proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS
            for proc in self.attn_processors.values()
        ):
            processor = AttnAddedKVProcessor()
        elif all(
            proc.__class__ in CROSS_ATTENTION_PROCESSORS
            for proc in self.attn_processors.values()
        ):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    @cached_vae_encode
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[DiagonalGaussianDistribution]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """

        if x.ndim == 5:
            assert x.shape[2] == 1
            x = x.squeeze(2)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        return decoded

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


EntryClass = AutoencoderKLFlux2
