# SPDX-License-Identifier: Apache-2.0

import os
from itertools import chain
from typing import Any

import torch
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    set_default_dtype,
    shard_model,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_comfyui_passthrough import (
    ComfyUIPassThroughScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ComfyUILatentPreparationStage,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE, set_mixed_precision_policy

logger = init_logger(__name__)


class ComfyUIQwenImagePipelineBase(LoRAPipeline, ComposedPipelineBase):
    """
    Base pipeline for ComfyUI QwenImage integration with only denoising stage.

    This pipeline requires pre-processed inputs:
    - prompt_embeds: Pre-encoded text embeddings (list of tensors)
    - latents: Pre-processed image latents in sequence format [B, S, D]

    Usage:
        generator = DiffGenerator.from_pretrained(
            model_path="path/to/model",
            pipeline_class_name="ComfyUIQwenImagePipeline",
            device="cuda",
        )
    """

    # Subclasses should override this
    zero_cond_t: bool = False

    pipeline_name = "ComfyUIQwenImagePipeline"

    _required_config_modules = [
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        """
        Initialize the pipeline with ComfyUI pass-through scheduler.
        This scheduler does not modify latents, allowing ComfyUI to handle denoising.
        """
        self.modules["scheduler"] = ComfyUIPassThroughScheduler(
            num_train_timesteps=1000
        )

        # Ensure VAE config is properly initialized even though we don't load the VAE model
        vae_config = server_args.pipeline_config.vae_config
        vae_config.post_init()
        logger.info(
            "Called vae_config.post_init() to set vae_scale_factor. "
            f"vae_scale_factor={vae_config.arch_config.vae_scale_factor}"
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load modules for ComfyUIQwenImagePipeline.

        If model_path is a safetensors file, load transformer directly from it
        without requiring model_index.json. Otherwise, fall back to default loading.
        """
        if os.path.isfile(self.model_path) and self.model_path.endswith(".safetensors"):
            logger.info(
                "Detected safetensors file, loading transformer directly from: %s",
                self.model_path,
            )
            return self._load_transformer_from_safetensors(server_args, loaded_modules)
        else:
            logger.info(
                "Model path is a directory, using default loading method: %s",
                self.model_path,
            )
            return super().load_modules(server_args, loaded_modules)

    def _load_transformer_from_safetensors(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load transformer directly from safetensors without model_index.json."""

        # 1) Fast path: use provided module
        if loaded_modules is not None and "transformer" in loaded_modules:
            logger.info("Using provided transformer module")
            return {
                "transformer": loaded_modules["transformer"],
                "scheduler": self.modules.get("scheduler"),
            }

        # 2) Build config and mappings
        dit_config, updated_mapping, model_cls, default_dtype = (
            self._prepare_dit_config_and_mapping(server_args)
        )
        safetensors_list = [self.model_path]
        logger.info("Loading weights from: %s", safetensors_list)

        # 3) Instantiate model (meta) and optionally shard
        model = self._instantiate_model(
            model_cls, dit_config, default_dtype, updated_mapping, server_args
        )

        # 4) Load weights
        self._load_weights_into_model(
            model, safetensors_list, default_dtype, updated_mapping, server_args
        )

        components = {
            "transformer": model,
            "scheduler": self.modules.get("scheduler"),
        }
        logger.info("Successfully loaded modules: %s", list(components.keys()))
        return components

    def _prepare_dit_config_and_mapping(self, server_args: ServerArgs):
        from sglang.multimodal_gen.configs.models.dits.qwenimage import (
            QwenImageArchConfig,
        )

        comfyui_arch_config = QwenImageArchConfig(
            patch_size=2,
            in_channels=64,
            out_channels=16,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            pooled_projection_dim=768,
            guidance_embeds=False,
            axes_dims_rope=(16, 56, 56),
            zero_cond_t=self.zero_cond_t,
        )
        dit_config = QwenImageDitConfig(arch_config=comfyui_arch_config)
        server_args.pipeline_config.dit_config = dit_config

        if dit_config.arch_config.param_names_mapping is None:
            dit_config.arch_config.param_names_mapping = {}

        comfyui_qwen_mappings = {r"^model\.diffusion_model\.(.*)$": r"\1"}
        updated_mapping = {
            **dit_config.arch_config.param_names_mapping,
            **comfyui_qwen_mappings,
        }
        dit_config.arch_config.param_names_mapping = updated_mapping
        logger.info(
            "Added ComfyUI weight name mappings to param_names_mapping. "
            f"Total mappings: {len(updated_mapping)}"
        )

        cls_name = "QwenImageTransformer2DModel"
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        logger.info("Resolved transformer class: %s", cls_name)

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        server_args.model_paths["transformer"] = os.path.dirname(self.model_path) or "."
        assert server_args.hsdp_shard_dim is not None, "hsdp_shard_dim must be set"
        logger.info(
            "Loading %s from safetensors file, default_dtype: %s",
            cls_name,
            default_dtype,
        )
        return dit_config, updated_mapping, model_cls, default_dtype

    def _instantiate_model(
        self,
        model_cls,
        dit_config,
        default_dtype,
        updated_mapping,
        server_args: ServerArgs,
    ):
        from sglang.multimodal_gen.runtime.platforms import current_platform

        hf_config = {}
        original_mapping = model_cls.param_names_mapping
        model_cls.param_names_mapping = updated_mapping
        logger.info(
            "Temporarily updated model class param_names_mapping with ComfyUI mappings. "
            f"Total mappings: {len(updated_mapping)}"
        )

        try:
            mp_policy = MixedPrecisionPolicy(
                torch.bfloat16, torch.float32, None, cast_forward_inputs=False
            )
            set_mixed_precision_policy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=None,
                mp_policy=mp_policy,
            )

            with set_default_dtype(default_dtype), torch.device("meta"):
                model = model_cls(**{"config": dit_config, "hf_config": hf_config})

            use_fsdp = server_args.use_fsdp_inference
            if current_platform.is_mps():
                use_fsdp = False
                logger.info("Disabling FSDP for MPS platform as it's not compatible")

            if use_fsdp:
                device_mesh = init_device_mesh(
                    current_platform.device_type,
                    mesh_shape=(
                        server_args.hsdp_replicate_dim,
                        server_args.hsdp_shard_dim,
                    ),
                    mesh_dim_names=("replicate", "shard"),
                )
                shard_model(
                    model,
                    cpu_offload=server_args.dit_cpu_offload,
                    reshard_after_forward=True,
                    mp_policy=mp_policy,
                    mesh=device_mesh,
                    fsdp_shard_conditions=model._fsdp_shard_conditions,
                    pin_cpu_memory=server_args.pin_cpu_memory,
                )
        finally:
            model_cls.param_names_mapping = original_mapping

        return model

    def _load_weights_into_model(
        self,
        model,
        safetensors_list,
        default_dtype,
        updated_mapping,
        server_args: ServerArgs,
    ):
        # Create weight iterator for loading
        weight_iterator = safetensors_weights_iterator(safetensors_list)

        # Load weights
        param_names_mapping_fn = get_param_names_mapping(updated_mapping)
        load_model_from_full_model_state_dict(
            model,
            weight_iterator,
            get_local_torch_device(),
            default_dtype,
            strict=True,
            cpu_offload=server_args.dit_cpu_offload,
            param_names_mapping=param_names_mapping_fn,
        )

        # Check for meta parameters
        for n, p in chain(model.named_parameters(), model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded transformer with %.2fB parameters", total_params / 1e9)

    def create_pipeline_stages(self, server_args: ServerArgs):
        logger.info(
            f"{self.__class__.__name__}.create_pipeline_stages() called - creating latent_preparation_stage and denoising_stage"
        )

        # Add ComfyUILatentPreparationStage to handle latents properly for SP
        # This stage includes device mismatch fix for ComfyUI pipelines in multi-GPU scenarios
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=ComfyUILatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # Add DenoisingStage for the actual denoising process
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        logger.info(
            f"{self.__class__.__name__} stages created: {list(self._stage_name_mapping.keys())}"
        )


class ComfyUIQwenImagePipeline(ComfyUIQwenImagePipelineBase):
    """ComfyUI QwenImage pipeline for text-to-image generation."""

    pipeline_name = "ComfyUIQwenImagePipeline"
    zero_cond_t = False

    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
        QwenImagePipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams

    pipeline_config_cls = QwenImagePipelineConfig
    sampling_params_cls = QwenImageSamplingParams


class ComfyUIQwenImageEditPipeline(ComfyUIQwenImagePipelineBase):
    """ComfyUI QwenImage pipeline for image-to-image editing."""

    pipeline_name = "ComfyUIQwenImageEditPipeline"
    zero_cond_t = True

    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
        QwenImageEditPlusPipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.qwenimage import (
        QwenImageEditPlusSamplingParams,
    )

    pipeline_config_cls = QwenImageEditPlusPipelineConfig
    sampling_params_cls = QwenImageEditPlusSamplingParams


EntryClass = [ComfyUIQwenImagePipeline, ComfyUIQwenImageEditPipeline]
