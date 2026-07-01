# SPDX-License-Identifier: Apache-2.0
"""NVIDIA OmniDreams pipeline (autoregressive video world model).

The checkpoint (``single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt``) is a
flat, DiT-only ``.pt`` (570 keys, bf16) -- not a diffusers layout. This pipeline
therefore overrides ``_load_config`` (fabricates a model_index-like dict) and
``load_modules`` (loads the flat DiT directly, mirroring the Hunyuan3D
precedent). The VAE (Wan 2.1) and text encoder (Cosmos-Reason1-7B / Qwen2.5-VL)
are loaded alongside it.

Stage layout (Hybrid monolithic, autoregressive):
``BeforeDenoising -> OmniDreamsDenoising (AR rollout) -> standard DecodingStage``.
The denoising stage concatenates the AR chunks into ``batch.latents``; the
standard single-pass VAE decode lets the Wan VAE's causal temporal feature cache
flow across chunk boundaries, giving correct continuity and the FlashDreams
frame counts (chunk0 -> 1+(len_t-1)*4, each later chunk -> len_t*4).
"""

from __future__ import annotations

import glob
import os
from itertools import chain
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_omnidreams_flow_match import (  # noqa: E501
    OmniDreamsFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
    OmniDreamsBeforeDenoisingStage,
    OmniDreamsDenoisingStage,
    OmniDreamsLightTAEDecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Default in-repo location of the distilled single-view checkpoint.
_DEFAULT_CKPT_RELPATH = "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"


class OmniDreamsPipeline(ComposedPipelineBase):
    pipeline_name = "OmniDreamsPipeline"
    is_video_pipeline = True
    _required_config_modules = [
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
    ]

    def _load_config(self) -> dict[str, Any]:
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "transformer": ["sglang", "OmniDreamsDiT"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "text_encoder": ["transformers", "AutoModel"],
            "tokenizer": ["transformers", "AutoProcessor"],
            "scheduler": ["sglang", "OmniDreamsFlowMatchScheduler"],
        }

    # ----- path resolution -------------------------------------------------- #
    @staticmethod
    def _resolve_ckpt_path(model_path: str) -> str:
        """Locate the flat DiT ``.pt`` from a directory, repo, or direct file."""
        if os.path.isfile(model_path):
            return model_path
        candidate = os.path.join(model_path, _DEFAULT_CKPT_RELPATH)
        if os.path.isfile(candidate):
            return candidate
        matches = sorted(
            glob.glob(os.path.join(model_path, "**", "*.pt"), recursive=True)
        )
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"OmniDreams checkpoint (.pt) not found under {model_path}"
        )

    # ----- component loaders ------------------------------------------------ #
    @classmethod
    def _load_flat_dit(
        cls,
        dit_config: Any,
        ckpt_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Instantiate OmniDreamsDiT on meta, load the flat .pt, then fuse weights.

        The flat checkpoint keys equal the submodule names, so the param mapping
        is the identity. The custom loader bypasses the generic loader's
        post-load hook, so ``post_load_weights`` (72->68 padding-mask fuse +
        last-layer shuffle fuse) is invoked explicitly here.
        """
        with set_default_torch_dtype(dtype), torch.device("meta"):
            model = OmniDreamsDiT(config=dit_config, hf_config={})

        weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        weight_iterator = ((k, v) for k, v in weights.items())
        mapping_fn = get_param_names_mapping(model.param_names_mapping)

        load_model_from_full_model_state_dict(
            model,
            weight_iterator,
            device,
            dtype,
            strict=True,
            param_names_mapping=mapping_fn,
        )

        model.post_load_weights()

        for name, p in chain(model.named_parameters(), model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"Unexpected param/buffer {name} on meta device.")
            if isinstance(p, nn.Parameter):
                p.requires_grad = False

        return model.eval()

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        pipeline_config = server_args.pipeline_config
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]
        vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]

        # Honor the CPU-offload flags at load time: each flagged component is
        # staged on CPU so the three heavy weights (the 2B DiT, the 7B text
        # encoder, and the Wan VAE) never need to co-reside on the GPU while
        # loading. The ComponentResidencyManager then brings each one to the GPU
        # only around its use-site (declared in the stages' ``component_uses``)
        # and releases it afterwards. Without this the custom loaders push all
        # three straight to the GPU, so a small-VRAM card OOMs while loading the
        # text encoder even though --text-encoder-cpu-offload was requested.
        cpu_device = torch.device("cpu")

        def _load_device(offload_flag: object) -> torch.device:
            offload = bool(offload_flag) and not server_args.use_fsdp_inference
            return cpu_device if offload else device

        dit_device = _load_device(server_args.dit_cpu_offload)
        vae_device = _load_device(server_args.vae_cpu_offload)
        text_encoder_device = _load_device(server_args.text_encoder_cpu_offload)

        model_path = server_args.model_path
        ckpt_path = self._resolve_ckpt_path(model_path)
        logger.info("OmniDreams: loading flat DiT from %s", ckpt_path)
        transformer = self._load_flat_dit(
            pipeline_config.dit_config, ckpt_path, dit_device, dit_dtype
        )

        # A single AutoencoderKLWan has both encode + decode, so when several
        # roles use the full WanVAE we build it once and share the instance —
        # reading the ~485MB safetensors and allocating the weights up to 3x
        # (image_encoder + encoder + decoder) is pure waste.
        _wanvae_cache: dict[tuple, nn.Module] = {}

        def _setup_vae_component(cfg):
            if cfg is None:
                return None
            cfg.model_path = model_path
            cfg.device = vae_device
            cfg.dtype = vae_dtype
            if cfg.impl != "wanvae":
                return cfg.setup()
            key = (
                cfg.checkpoint_path,
                str(vae_device),
                vae_dtype,
                tuple(cfg.latents_mean),
                tuple(cfg.latents_std),
            )
            if key not in _wanvae_cache:
                _wanvae_cache[key] = cfg.setup()
            return _wanvae_cache[key]

        # image_encoder (one-shot first-frame I2V conditioning)
        image_encoder = _setup_vae_component(pipeline_config.image_encoder_config)
        # encoder (per-AR-step HDMap conditioning)
        encoder = _setup_vae_component(pipeline_config.encoder_config)
        # decoder (per-AR-step latent decode)
        decoder = _setup_vae_component(pipeline_config.decoder_config)

        # text_encoder
        if pipeline_config.text_encoder_config is not None:
            pipeline_config.text_encoder_config.device = text_encoder_device
            pipeline_config.text_encoder_config.model_path = model_path
            text_encoder, tokenizer = pipeline_config.text_encoder_config.setup()
        else:
            text_encoder, tokenizer = None, None

        scheduler = OmniDreamsFlowMatchScheduler(
            num_inference_steps=len(pipeline_config.denoising_timesteps),
            denoising_timesteps=tuple(pipeline_config.denoising_timesteps),
            shift=(
                pipeline_config.flow_shift
                if pipeline_config.flow_shift is not None
                else 5.0
            ),
            sigma_min=pipeline_config.sigma_min,
            device=device,
        )

        # Phase 6: populate memory budgets (GiB, approximate) for the
        # ComponentResidencyManager offload scheduler. Exact values are
        # TODO(gpu): measure on the target GPU with torch.cuda.memory_stats().
        impl_image = (
            pipeline_config.image_encoder_config.impl
            if pipeline_config.image_encoder_config
            else "wanvae"
        )
        impl_encoder = pipeline_config.encoder_config.impl
        impl_decoder = pipeline_config.decoder_config.impl

        self.memory_usages = {
            "transformer": 4.0,  # ~2B params in bf16 ≈ 4 GiB
            "text_encoder": 14.0,  # Cosmos-Reason1-7B ≈ 14 GiB
            "image_encoder": 1.0 if impl_image == "wanvae" else 0.2,
            "encoder": 1.0 if impl_encoder == "wanvae" else 0.2,
            "decoder": 1.0 if impl_decoder == "wanvae" else 0.2,
        }

        modules = {
            "transformer": transformer,
            "image_encoder": image_encoder,
            "encoder": encoder,
            "decoder": decoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
        }
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config

        # BeforeDenoisingStage: needs image_encoder (I2V first-frame) + encoder (HDMap)
        self.add_stage(
            stage_name="omnidreams_before_denoising",
            stage=OmniDreamsBeforeDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                image_encoder=self.get_module("image_encoder"),
                encoder=self.get_module("encoder"),
                config=config,
            ),
        )

        # DenoisingStage: no VAE (only validates use_feature_cache attribute)
        self.add_stage(
            stage_name="omnidreams_denoising",
            stage=OmniDreamsDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                decoder=self.get_module("decoder"),
                encoder=self.get_module("encoder"),
            ),
        )

        # DecodingStage: uses decoder
        if config.decoder_config.impl == "lighttae":
            # LightTAE has own latent mean/std -> skip scale_and_shift
            self.add_stage(
                stage_name="omnidreams_decoding",
                stage=OmniDreamsLightTAEDecodingStage(
                    vae=self.get_module("decoder"),
                    pipeline=self,
                    component_name="decoder",
                ),
            )
        else:
            # Standard WanVAE decode
            self.add_standard_decoding_stage(
                stage_name="omnidreams_decoding",
                vae_key="decoder",
            )


EntryClass = OmniDreamsPipeline
