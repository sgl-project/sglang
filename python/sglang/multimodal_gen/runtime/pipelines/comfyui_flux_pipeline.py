# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import Any, Generator

import torch

from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
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
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ComfyUIFluxPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Simplified pipeline for ComfyUI integration with only denoising stage.

    This pipeline requires pre-processed inputs:
    - prompt_embeds: Pre-encoded text embeddings (list of tensors)
    - negative_prompt_embeds: Pre-encoded negative prompt embeddings (if using CFG)
    - latents: Optional initial noise latents (will be generated if not provided)

    Usage:
        generator = DiffGenerator.from_pretrained(
            model_path="path/to/model",
            pipeline_class_name="ComfyUIFluxPipeline",
            device="cuda",
        )
    """

    pipeline_name = "ComfyUIFluxPipeline"

    # Configuration classes for safetensors files without model_index.json
    from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
    from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams

    pipeline_config_cls = FluxPipelineConfig
    sampling_params_cls = FluxSamplingParams

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
        Load modules for ComfyUIFluxPipeline.

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

    def _load_and_convert_weights_from_safetensors(
        self,
        model_cls: type,
        dit_config: FluxConfig,
        hf_config: dict,
        safetensors_list: list[str],
        updated_mapping: dict,
        qkv_size: int,
        mlp_hidden_dim: int,
        has_guidance_embeds: bool,
        default_dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, dict]:
        """
        Load and convert weights from safetensors file, then load them into the model.
        """
        from sglang.multimodal_gen.runtime.loader.utils import (
            get_param_names_mapping,
            set_default_torch_dtype,
        )
        from sglang.multimodal_gen.runtime.loader.weight_utils import (
            safetensors_weights_iterator,
        )

        logger.info(
            "Converting ComfyUI Flux weights to SGLang format and loading model..."
        )

        # Create model on target device
        device = get_local_torch_device()
        with set_default_torch_dtype(default_dtype):
            model = model_cls(**{"config": dit_config, "hf_config": hf_config})
            model = model.to(device)

        # Verify model has guidance_embedder if config says it should
        has_guidance_embedder = hasattr(model.time_text_embed, "guidance_embedder")
        if has_guidance_embeds and not has_guidance_embedder:
            logger.warning(
                "Config has guidance_embeds=True but model doesn't have guidance_embedder. "
                "This may indicate a configuration mismatch."
            )
        elif not has_guidance_embeds and has_guidance_embedder:
            logger.warning(
                "Config has guidance_embeds=False but model has guidance_embedder. "
                "This may indicate a configuration mismatch."
            )

        # Note: guidance_in mappings are already included in comfyui_flux_mappings above.
        # If model doesn't support guidance embeddings, the weights will be filtered out
        # in _convert_comfyui_weights() based on has_guidance_embeds flag.

        param_names_mapping_fn = get_param_names_mapping(updated_mapping)

        weight_iterator = safetensors_weights_iterator(safetensors_list)
        converted_weights = self._convert_comfyui_weights(
            weight_iterator=weight_iterator,
            qkv_size=qkv_size,
            mlp_hidden_dim=mlp_hidden_dim,
            has_guidance_embeds=has_guidance_embeds,
        )

        model_state_dict = model.state_dict()
        missing_keys = set(model_state_dict.keys())
        unexpected_keys = []
        loaded_count = 0
        reverse_param_names_mapping = {}

        # Handle merged parameters (collect all parts before merging)
        from collections import defaultdict

        to_merge_params = defaultdict(dict)

        # Process weights incrementally: load immediately after conversion
        for source_name, tensor in converted_weights:
            target_name, merge_index, num_params_to_merge = param_names_mapping_fn(
                source_name
            )
            reverse_param_names_mapping[target_name] = (
                source_name,
                merge_index,
                num_params_to_merge,
            )

            if merge_index is not None:
                # Collect parts for merging
                to_merge_params[target_name][merge_index] = tensor
                if len(to_merge_params[target_name]) == num_params_to_merge:
                    # All parts collected, merge them
                    sorted_tensors = [
                        to_merge_params[target_name][i]
                        for i in range(num_params_to_merge)
                    ]
                    merged_tensor = torch.cat(sorted_tensors, dim=0)
                    # Load immediately after merging
                    if target_name in model_state_dict:
                        param = model_state_dict[target_name]
                        loaded_tensor = merged_tensor.to(
                            device=param.device, dtype=param.dtype
                        )
                        param.data.copy_(loaded_tensor)
                        missing_keys.discard(target_name)
                        loaded_count += 1
                        del merged_tensor, loaded_tensor
                    else:
                        unexpected_keys.append(target_name)
                    # Clear merged parts
                    del to_merge_params[target_name]
                    for t in sorted_tensors:
                        del t
            else:
                # Direct mapping, load immediately
                if target_name in model_state_dict:
                    param = model_state_dict[target_name]
                    # Check shape compatibility
                    if tensor.shape != param.shape:
                        logger.warning(
                            f"Shape mismatch for {target_name}: "
                            f"loaded {tensor.shape} vs model {param.shape}, skipping. "
                            f"Source: {source_name}"
                        )
                        unexpected_keys.append(target_name)
                        del tensor
                        continue

                    # Debug logging for norm_out.linear to verify mapping
                    if (
                        "norm_out.linear" in target_name
                        or "final_layer.adaLN_modulation" in source_name
                    ):
                        logger.info(
                            f"Loading norm_out.linear: {source_name} -> {target_name}, "
                            f"shape: {tensor.shape}"
                        )

                    loaded_tensor = tensor.to(device=param.device, dtype=param.dtype)
                    param.data.copy_(loaded_tensor)
                    missing_keys.discard(target_name)
                    loaded_count += 1
                    del tensor, loaded_tensor
                else:
                    # Debug logging for unmapped parameters
                    if "norm_out.linear" in target_name:
                        logger.warning(
                            f"norm_out.linear parameter {target_name} not found in model state_dict. "
                            f"Source: {source_name}"
                        )
                    unexpected_keys.append(target_name)

        optional_missing_keys = []
        required_missing_keys = []
        for key in missing_keys:
            if key.endswith(".bias"):
                # Check if corresponding weight exists (if weight exists but bias doesn't, it's optional)
                weight_key = key.replace(".bias", ".weight")
                if weight_key not in missing_keys:
                    optional_missing_keys.append(key)
                else:
                    required_missing_keys.append(key)
            else:
                required_missing_keys.append(key)

        if required_missing_keys:
            logger.warning(
                f"Required missing keys (first 10): {required_missing_keys[:10]}..."
            )
        if optional_missing_keys:
            logger.info(
                f"Optional missing keys (bias parameters, {len(optional_missing_keys)} total): "
                f"These will use default values (zeros)"
            )
        if unexpected_keys:
            logger.warning(f"Unexpected keys (first 10): {unexpected_keys[:10]}...")

        logger.info(f"Successfully loaded {loaded_count} weight tensors")

        return model, reverse_param_names_mapping

    def _convert_comfyui_weights(
        self,
        weight_iterator: Generator[tuple[str, torch.Tensor], None, None],
        qkv_size: int,
        mlp_hidden_dim: int,
        has_guidance_embeds: bool,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Convert ComfyUI Flux weights to SGLang format.
        Splits fused linear1 into separate to_qkv and proj_mlp weights.
        Filters out guidance_in weights if model doesn't support guidance embeddings.
        Handles scale/shift order difference between ComfyUI and AdaLayerNormContinuous.
        """
        for name, tensor in weight_iterator:
            if not has_guidance_embeds and name.startswith("guidance_in."):
                logger.debug(
                    f"Skipping {name} (model doesn't support guidance embeddings)"
                )
                continue

            match = re.match(r"single_blocks\.(\d+)\.linear1\.(weight|bias)$", name)
            if match:
                block_idx, param_type = match.groups()
                expected_size = qkv_size + mlp_hidden_dim

                if tensor.shape[0] < expected_size:
                    logger.warning(
                        f"linear1.{param_type} shape {tensor.shape} doesn't match "
                        f"expected size {expected_size}, skipping"
                    )
                    continue

                # Split tensor
                qkv_tensor = (
                    tensor[:qkv_size] if param_type == "bias" else tensor[:qkv_size, :]
                )
                mlp_tensor = (
                    tensor[qkv_size:] if param_type == "bias" else tensor[qkv_size:, :]
                )

                # Yield split weights
                yield f"single_transformer_blocks.{block_idx}.attn.to_qkv.{param_type}", qkv_tensor
                yield f"single_transformer_blocks.{block_idx}.proj_mlp.{param_type}", mlp_tensor
            elif name == "final_layer.adaLN_modulation.1.weight":
                # ComfyUI: output order is [shift, scale]
                # AdaLayerNormContinuous: expects [scale, shift]
                # Need to swap the first half and second half of the weight matrix
                # Weight shape: (2 * hidden_size, hidden_size)
                # Split into two halves and swap them
                half_size = tensor.shape[0] // 2
                shift_weights = tensor[:half_size, :]
                scale_weights = tensor[half_size:, :]
                # Swap: put scale first, then shift
                swapped_tensor = torch.cat([scale_weights, shift_weights], dim=0)
                logger.info(
                    f"Swapped scale/shift order for {name}: "
                    f"shape {tensor.shape} -> {swapped_tensor.shape}"
                )
                yield name, swapped_tensor
            elif name == "final_layer.adaLN_modulation.1.bias":
                # Same swap for bias: (2 * hidden_size,)
                half_size = tensor.shape[0] // 2
                shift_bias = tensor[:half_size]
                scale_bias = tensor[half_size:]
                swapped_tensor = torch.cat([scale_bias, shift_bias], dim=0)
                logger.info(
                    f"Swapped scale/shift order for {name}: "
                    f"shape {tensor.shape} -> {swapped_tensor.shape}"
                )
                yield name, swapped_tensor
            else:
                # Other weights pass through (handled by param_names_mapping)
                yield name, tensor

    def _load_transformer_from_safetensors(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load transformer directly from safetensors file without model_index.json.
        """
        if loaded_modules is not None and "transformer" in loaded_modules:
            logger.info("Using provided transformer module")
            components = {
                "transformer": loaded_modules["transformer"],
                "scheduler": self.modules.get("scheduler"),
            }
            return components

        if hasattr(server_args.pipeline_config, "dit_config"):
            dit_config = server_args.pipeline_config.dit_config
            if not isinstance(dit_config, FluxConfig):
                logger.warning("dit_config is not FluxConfig, creating new FluxConfig")
                dit_config = FluxConfig()
                server_args.pipeline_config.dit_config = dit_config
        else:
            logger.info("Creating default FluxConfig")
            dit_config = FluxConfig()
            server_args.pipeline_config.dit_config = dit_config

        # Set guidance_embeds to True for ComfyUI Flux models
        dit_config.arch_config.guidance_embeds = True
        logger.info("Set guidance_embeds=True for ComfyUI Flux model")

        if dit_config.arch_config.param_names_mapping is None:
            dit_config.arch_config.param_names_mapping = {}

        # ComfyUI Flux uses different parameter names than SGLang Flux
        # Key differences:
        # - ComfyUI: single_blocks.{i}.linear1 (fused QKV + MLP input)
        # - SGLang: single_transformer_blocks.{i}.attn.to_qkv + proj_mlp (separate)
        # - ComfyUI: single_blocks.{i}.linear2
        # - SGLang: single_transformer_blocks.{i}.proj_out
        # - ComfyUI: double_blocks.{i}.img_attn.qkv / txt_attn.qkv
        # - SGLang: transformer_blocks.{i}.attn.to_qkv / attn.to_added_qkv

        # Note: For fused layers like linear1, we need custom weight splitting logic
        # which will be handled in the weight conversion function below
        comfyui_flux_mappings = {
            # Double stream blocks - attention layers
            r"double_blocks\.(\d+)\.img_attn\.qkv\.(weight|bias)$": (
                r"transformer_blocks.\1.attn.to_qkv.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_attn\.qkv\.(weight|bias)$": (
                r"transformer_blocks.\1.attn.to_added_qkv.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.img_attn\.proj\.(weight|bias)$": (
                r"transformer_blocks.\1.attn.to_out.0.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_attn\.proj\.(weight|bias)$": (
                r"transformer_blocks.\1.attn.to_add_out.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale$": (
                r"transformer_blocks.\1.attn.norm_q.weight",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale$": (
                r"transformer_blocks.\1.attn.norm_k.weight",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale$": (
                r"transformer_blocks.\1.attn.norm_added_q.weight",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale$": (
                r"transformer_blocks.\1.attn.norm_added_k.weight",
                None,
                None,
            ),
            # Double stream blocks - MLP layers (map to net structure)
            r"double_blocks\.(\d+)\.img_mlp\.0\.(weight|bias)$": (
                r"transformer_blocks.\1.ff.net.0.proj.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.img_mlp\.2\.(weight|bias)$": (
                r"transformer_blocks.\1.ff.net.2.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_mlp\.0\.(weight|bias)$": (
                r"transformer_blocks.\1.ff_context.net.0.proj.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_mlp\.2\.(weight|bias)$": (
                r"transformer_blocks.\1.ff_context.net.2.\2",
                None,
                None,
            ),
            # Double stream blocks - modulation layers
            r"double_blocks\.(\d+)\.img_mod\.lin\.(weight|bias)$": (
                r"transformer_blocks.\1.norm1.linear.\2",
                None,
                None,
            ),
            r"double_blocks\.(\d+)\.txt_mod\.lin\.(weight|bias)$": (
                r"transformer_blocks.\1.norm1_context.linear.\2",
                None,
                None,
            ),
            # Single stream blocks - linear2 maps to proj_out
            r"single_blocks\.(\d+)\.linear2\.(weight|bias)$": (
                r"single_transformer_blocks.\1.proj_out.\2",
                None,
                None,
            ),
            # Single stream blocks - norm layers (scale -> weight)
            r"single_blocks\.(\d+)\.norm\.query_norm\.scale$": (
                r"single_transformer_blocks.\1.attn.norm_q.weight",
                None,
                None,
            ),
            r"single_blocks\.(\d+)\.norm\.key_norm\.scale$": (
                r"single_transformer_blocks.\1.attn.norm_k.weight",
                None,
                None,
            ),
            # Single stream blocks - modulation (maps to norm.linear)
            r"single_blocks\.(\d+)\.modulation\.lin\.(weight|bias)$": (
                r"single_transformer_blocks.\1.norm.linear.\2",
                None,
                None,
            ),
            # Time and guidance embeddings
            r"^time_in\.in_layer\.(weight|bias)$": (
                r"time_text_embed.timestep_embedder.linear_1.\1",
                None,
                None,
            ),
            r"^time_in\.out_layer\.(weight|bias)$": (
                r"time_text_embed.timestep_embedder.linear_2.\1",
                None,
                None,
            ),
            r"^txt_in\.(weight|bias)$": (r"context_embedder.\1", None, None),
            r"^vector_in\.in_layer\.(weight|bias)$": (
                r"time_text_embed.text_embedder.linear_1.\1",
                None,
                None,
            ),
            r"^vector_in\.out_layer\.(weight|bias)$": (
                r"time_text_embed.text_embedder.linear_2.\1",
                None,
                None,
            ),
            # Final layer mappings
            r"^final_layer\.linear\.(weight|bias)$": (r"proj_out.\1", None, None),
            r"^final_layer\.norm_final\.(weight|bias)$": (r"norm_out.\1", None, None),
            r"^final_layer\.adaLN_modulation\.1\.(weight|bias)$": (
                r"norm_out.linear.\1",
                None,
                None,
            ),
            # Image input embedding
            r"^img_in\.(weight|bias)$": (r"x_embedder.\1", None, None),
            # Guidance embeddings (if model supports guidance)
            r"^guidance_in\.in_layer\.(weight|bias)$": (
                r"time_text_embed.guidance_embedder.linear_1.\1",
                None,
                None,
            ),
            r"^guidance_in\.out_layer\.(weight|bias)$": (
                r"time_text_embed.guidance_embedder.linear_2.\1",
                None,
                None,
            ),
        }

        # Merge ComfyUI mappings with existing mappings (ComfyUI mappings take precedence)
        updated_mapping = {
            **dit_config.arch_config.param_names_mapping,
            **comfyui_flux_mappings,
        }
        dit_config.arch_config.param_names_mapping = updated_mapping
        logger.info(
            "Added ComfyUI weight name mappings for Flux model. "
            f"Total mappings: {len(updated_mapping)}"
        )

        cls_name = "FluxTransformer2DModel"
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        logger.info("Resolved transformer class: %s", cls_name)

        original_mapping = None
        if comfyui_flux_mappings:
            original_mapping = model_cls.param_names_mapping
            model_cls.param_names_mapping = updated_mapping
            logger.info(
                "Temporarily updated model class param_names_mapping with ComfyUI mappings. "
                f"Total mappings: {len(updated_mapping)}"
            )

        safetensors_list = [self.model_path]
        logger.info("Loading weights from: %s", safetensors_list)
        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        server_args.model_paths["transformer"] = os.path.dirname(self.model_path) or "."
        hf_config = {}

        hidden_size = (
            dit_config.arch_config.num_attention_heads
            * dit_config.arch_config.attention_head_dim
        )
        mlp_ratio = getattr(dit_config.arch_config, "mlp_ratio", 4.0)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        qkv_size = 3 * hidden_size
        has_guidance_embeds = True

        # Load and convert weights from safetensors file
        model, reverse_param_names_mapping = (
            self._load_and_convert_weights_from_safetensors(
                model_cls=model_cls,
                dit_config=dit_config,
                hf_config=hf_config,
                safetensors_list=safetensors_list,
                updated_mapping=updated_mapping,
                qkv_size=qkv_size,
                mlp_hidden_dim=mlp_hidden_dim,
                has_guidance_embeds=has_guidance_embeds,
                default_dtype=default_dtype,
            )
        )

        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        model.reverse_param_names_mapping = reverse_param_names_mapping

        if original_mapping is not None:
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
            "ComfyUIFluxPipeline.create_pipeline_stages() called - creating latent_preparation_stage and denoising_stage"
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
            f"ComfyUIFluxPipeline stages created: {list(self._stage_name_mapping.keys())}"
        )


EntryClass = ComfyUIFluxPipeline
