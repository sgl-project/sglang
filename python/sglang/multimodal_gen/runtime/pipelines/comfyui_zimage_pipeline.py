# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_comfyui_passthrough import (
    ComfyUIPassThroughScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ComfyUIZImagePipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Simplified pipeline for ComfyUI integration with only denoising stage.

    This pipeline requires pre-processed inputs:
    - prompt_embeds: Pre-encoded text embeddings (list of tensors)
    - negative_prompt_embeds: Pre-encoded negative prompt embeddings (if using CFG)
    - latents: Optional initial noise latents (will be generated if not provided)

    Usage:
        generator = DiffGenerator.from_pretrained(
            model_path="path/to/model",
            pipeline_class_name="ComfyUIZImagePipeline",
            device="cuda",
        )
    """

    pipeline_name = "ComfyUIZImagePipeline"
    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams

    pipeline_config_cls = ZImagePipelineConfig
    sampling_params_cls = ZImageSamplingParams

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
        # This is necessary because get_freqs_cis uses spatial_compression_ratio
        if hasattr(server_args.pipeline_config, "vae_config"):
            vae_config = server_args.pipeline_config.vae_config
            if hasattr(vae_config, "post_init") and not hasattr(
                vae_config, "_post_init_called"
            ):
                vae_config.post_init()
                logger.info(
                    "Called vae_config.post_init() to set spatial_compression_ratio. "
                    f"spatial_compression_ratio={vae_config.arch_config.spatial_compression_ratio}"
                )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load modules for ComfyUIZImagePipeline.

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
        """
        Load transformer directly from safetensors file without model_index.json.

        This method:
        1. Uses hardcoded ZImageDitConfig for zimage model
        2. Loads transformer from the safetensors file
        3. Uses ComfyUIPassThroughScheduler (already created in initialize_pipeline)
        """
        # Check if transformer is already provided
        if loaded_modules is not None and "transformer" in loaded_modules:
            logger.info("Using provided transformer module")
            components = {
                "transformer": loaded_modules["transformer"],
                "scheduler": self.modules.get("scheduler"),
            }
            return components

        if hasattr(server_args.pipeline_config, "dit_config"):
            dit_config = server_args.pipeline_config.dit_config
            if not isinstance(dit_config, ZImageDitConfig):
                logger.warning(
                    "dit_config is not ZImageDitConfig, creating new ZImageDitConfig"
                )
                dit_config = ZImageDitConfig()
                server_args.pipeline_config.dit_config = dit_config
        else:
            logger.info("Creating default ZImageDitConfig")
            dit_config = ZImageDitConfig()
            server_args.pipeline_config.dit_config = dit_config

        if dit_config.arch_config.param_names_mapping is None:
            dit_config.arch_config.param_names_mapping = {}

        # Add mappings for norm layers: map from ComfyUI format (k_norm/q_norm) to SGLang format (norm_k/norm_q)
        # The regex matches the source name from safetensors, and the tuple specifies the target name in the model
        comfyui_norm_mappings = {
            r"(.*)\.attention\.k_norm\.weight$": (
                r"\1.attention.norm_k.weight",
                None,
                None,
            ),
            r"(.*)\.attention\.q_norm\.weight$": (
                r"\1.attention.norm_q.weight",
                None,
                None,
            ),
            r"(.*)\.attention\.qkv\.weight$": (
                r"\1.attention.to_qkv.weight",
                None,
                None,
            ),
            r"(.*)\.attention\.out\.weight$": (
                r"\1.attention.to_out.0.weight",
                None,
                None,
            ),
            r"^final_layer\.(.*)$": (r"all_final_layer.2-1.\1", None, None),
            r"^x_embedder\.(.*)$": (r"all_x_embedder.2-1.\1", None, None),
        }

        # Merge ComfyUI mappings with existing mappings (ComfyUI mappings take precedence)
        updated_mapping = {
            **dit_config.arch_config.param_names_mapping,
            **comfyui_norm_mappings,
        }
        dit_config.arch_config.param_names_mapping = updated_mapping
        logger.info(
            "Added ComfyUI weight name mappings (k_norm/q_norm -> norm_k/norm_q) to param_names_mapping. "
            f"Total mappings: {len(updated_mapping)}"
        )

        cls_name = "ZImageTransformer2DModel"
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        logger.info("Resolved transformer class: %s", cls_name)
        safetensors_list = [self.model_path]
        logger.info("Loading weights from: %s", safetensors_list)

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        server_args.model_paths["transformer"] = os.path.dirname(self.model_path) or "."
        hf_config = {}

        assert server_args.hsdp_shard_dim is not None, "hsdp_shard_dim must be set"
        logger.info(
            "Loading %s from safetensors file, default_dtype: %s",
            cls_name,
            default_dtype,
        )

        original_mapping = model_cls.param_names_mapping
        model_cls.param_names_mapping = updated_mapping
        logger.info(
            "Temporarily updated model class param_names_mapping with ComfyUI mappings. "
            f"Total mappings: {len(updated_mapping)}"
        )

        try:
            model = maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params={"config": dit_config, "hf_config": hf_config},
                weight_dir_list=safetensors_list,
                device=get_local_torch_device(),
                hsdp_replicate_dim=server_args.hsdp_replicate_dim,
                hsdp_shard_dim=server_args.hsdp_shard_dim,
                cpu_offload=server_args.dit_cpu_offload,
                pin_cpu_memory=server_args.pin_cpu_memory,
                fsdp_inference=server_args.use_fsdp_inference,
                default_dtype=default_dtype,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=None,
            )
        finally:
            model_cls.param_names_mapping = original_mapping

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded transformer with %.2fB parameters", total_params / 1e9)

        components = {
            "transformer": model,
            "scheduler": self.modules.get("scheduler"),
        }

        logger.info("Successfully loaded modules: %s", list(components.keys()))
        return components

    def create_pipeline_stages(self, server_args: ServerArgs):
        logger.info(
            "ComfyUIZImagePipeline.create_pipeline_stages() called - creating only denoising_stage"
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        logger.info(
            f"ComfyUIZImagePipeline stages created: {list(self._stage_name_mapping.keys())}"
        )


EntryClass = ComfyUIZImagePipeline
