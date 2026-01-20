# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

import os
import re
from collections.abc import Generator
from itertools import chain
from typing import Any

import torch
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
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
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE, set_mixed_precision_policy

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

    def _convert_comfyui_qkv_weights(
        self,
        weight_iterator: Generator[tuple[str, torch.Tensor], None, None],
        dim: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Convert ComfyUI zimage qkv weights to SGLang format.
        Splits merged qkv.weight into separate to_q, to_k, to_v weights.

        Args:
            weight_iterator: Iterator yielding (name, tensor) pairs from safetensors
            dim: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads

        Yields:
            (name, tensor) pairs with qkv weights split into to_q, to_k, to_v
        """
        head_dim = dim // num_heads
        q_size = dim
        k_size = head_dim * num_kv_heads
        v_size = head_dim * num_kv_heads

        for name, tensor in weight_iterator:
            # Match qkv weights in layers, noise_refiner, or context_refiner
            # Pattern: (layers|noise_refiner|context_refiner).{i}.attention.qkv.(weight|bias)
            match = re.match(
                r"(layers|noise_refiner|context_refiner)\.(\d+)\.attention\.qkv\.(weight|bias)$",
                name,
            )
            if match:
                module_name, layer_idx, param_type = match.groups()
                base_name = f"{module_name}.{layer_idx}.attention"

                if param_type == "weight":
                    # Weight shape: (q_size + k_size + v_size, dim)
                    # Split into q, k, v
                    q_weight = tensor[:q_size, :]
                    k_weight = tensor[q_size : q_size + k_size, :]
                    v_weight = tensor[q_size + k_size :, :]

                    logger.debug(
                        f"Splitting {name} (shape {tensor.shape}) into "
                        f"to_q ({q_weight.shape}), to_k ({k_weight.shape}), to_v ({v_weight.shape})"
                    )

                    yield f"{base_name}.to_q.weight", q_weight
                    yield f"{base_name}.to_k.weight", k_weight
                    yield f"{base_name}.to_v.weight", v_weight
                else:  # bias
                    # Bias shape: (q_size + k_size + v_size,)
                    # Split into q, k, v
                    q_bias = tensor[:q_size]
                    k_bias = tensor[q_size : q_size + k_size]
                    v_bias = tensor[q_size + k_size :]

                    logger.debug(
                        f"Splitting {name} (shape {tensor.shape}) into "
                        f"to_q ({q_bias.shape}), to_k ({k_bias.shape}), to_v ({v_bias.shape})"
                    )

                    yield f"{base_name}.to_q.bias", q_bias
                    yield f"{base_name}.to_k.bias", k_bias
                    yield f"{base_name}.to_v.bias", v_bias
            else:
                # Pass through other weights unchanged
                yield name, tensor

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
        # Note: qkv weights are handled separately by _convert_comfyui_qkv_weights function
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
            # Create model first (same as maybe_load_fsdp_model)
            from sglang.multimodal_gen.runtime.platforms import current_platform

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

            # Check if we should use FSDP
            use_fsdp = server_args.use_fsdp_inference
            if current_platform.is_mps():
                use_fsdp = False
                logger.info("Disabling FSDP for MPS platform as it's not compatible")

            if use_fsdp:
                world_size = server_args.hsdp_replicate_dim * server_args.hsdp_shard_dim
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

            # Get model dimensions for qkv splitting
            arch_config = dit_config.arch_config
            dim = arch_config.dim
            num_heads = arch_config.num_attention_heads
            num_kv_heads = arch_config.n_kv_heads

            # Create weight iterator with qkv conversion
            base_weight_iterator = safetensors_weights_iterator(safetensors_list)
            converted_weight_iterator = self._convert_comfyui_qkv_weights(
                base_weight_iterator, dim, num_heads, num_kv_heads
            )

            # Load weights
            param_names_mapping_fn = get_param_names_mapping(updated_mapping)
            load_model_from_full_model_state_dict(
                model,
                converted_weight_iterator,
                get_local_torch_device(),
                default_dtype,
                strict=True,
                cpu_offload=server_args.dit_cpu_offload,
                param_names_mapping=param_names_mapping_fn,
            )

            # Check for meta parameters
            for n, p in chain(model.named_parameters(), model.named_buffers()):
                if p.is_meta:
                    raise RuntimeError(
                        f"Unexpected param or buffer {n} on meta device."
                    )
                if isinstance(p, torch.nn.Parameter):
                    p.requires_grad = False
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
