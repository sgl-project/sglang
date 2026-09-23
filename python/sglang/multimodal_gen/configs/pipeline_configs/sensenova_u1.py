# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_THINK_MODE,
    RESOLUTION_ALIGNMENT,
)


def _is_runtime_option_requested(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _is_arg_explicitly_set(server_args, option: str) -> bool:
    is_explicit = getattr(server_args, "is_arg_explicitly_set", None)
    if callable(is_explicit):
        return is_explicit(option)
    return _is_runtime_option_requested(getattr(server_args, option, None))


def _component_residency_requests_offload(value) -> bool:
    if not _is_runtime_option_requested(value):
        return False
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, str):
        values = value.split(",")
    else:
        values = value

    for raw_value in values:
        mode = str(raw_value).split("=", 1)[-1].strip().replace("_", "-").lower()
        if mode in ("component-offload", "layerwise-offload"):
            return True
    return False


def _set_compatible_runtime_defaults(server_args) -> None:
    compatible_defaults = {
        "component_residency": None,
        "cpu_offload_components": None,
        "dit_cpu_offload": False,
        "text_encoder_cpu_offload": False,
        "image_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "dit_layerwise_offload": False,
        "layerwise_offload_components": None,
        "quantization": None,
        "quantization_ignored_layers": None,
        "transformer_weights_path": None,
        "component_paths": {},
        "component_weights_paths": {},
        "component_quantizations": {},
        "component_quantization_ignored_layers": {},
        "component_precisions": {},
        "attention_backend": None,
        "component_attention_backends": {},
        "attention_backend_config": None,
    }
    for option, value in compatible_defaults.items():
        if not _is_arg_explicitly_set(server_args, option):
            setattr(server_args, option, value)


@dataclass
class SenseNovaU1PipelineConfig(PipelineConfig):
    """Native SenseNova-U1 text-to-image and image-editing pipeline configuration."""

    task_type: ModelTaskType = ModelTaskType.TI2I
    model_precision: str = "bf16"
    should_use_guidance: bool = True
    supports_cfg_parallel: bool = False

    def calculate_condition_image_size(self, image, width, height):
        del image, width, height
        return None

    def prepare_calculated_size(self, image):
        del image
        return None

    def supports_dynamic_batching(self):
        return True

    def supports_dynamic_batching_for_request(self, batch) -> bool:
        sampling_params = getattr(batch, "sampling_params", None)
        return not bool(getattr(sampling_params, "think_mode", DEFAULT_THINK_MODE))

    def estimate_request_cost(self, batch) -> float:
        image_tokens = (int(batch.width) // RESOLUTION_ALIGNMENT) * (
            int(batch.height) // RESOLUTION_ALIGNMENT
        )
        cfg_branches = 2 if float(batch.guidance_scale) > 1 else 1
        return float(
            image_tokens
            * int(batch.num_inference_steps)
            * cfg_branches
            * int(batch.num_outputs_per_prompt)
        )

    def supports_disaggregation(self) -> bool:
        return False

    def supports_sequential_multi_output_inference(self):
        return False

    @staticmethod
    def validate_parallelism(server_args) -> None:
        """Validate DP x TP x Ulysses-SP serving for the U1.5 checkpoint."""
        num_gpus = int(server_args.num_gpus)
        dp_size = int(getattr(server_args, "dp_size", 1) or 1)
        tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        sp_degree = int(getattr(server_args, "sp_degree", 1) or 1)
        ulysses_degree = int(
            getattr(server_args, "ulysses_degree", sp_degree) or sp_degree
        )
        ring_degree = int(getattr(server_args, "ring_degree", 1) or 1)
        kv_gather_degree = int(getattr(server_args, "kv_gather_degree", 1) or 1)
        if tp_size not in (1, 2, 4, 8):
            raise ValueError(
                "SenseNova-U1.5-8B-MoT supports --tp-size 1, 2, 4, or 8; "
                f"got {tp_size}."
            )
        if ring_degree != 1 or kv_gather_degree != 1 or ulysses_degree != sp_degree:
            raise ValueError(
                "SenseNova-U1.5-8B-MoT sequence parallelism currently supports "
                "Ulysses only: --ring-degree and --kv-gather-degree must be 1, "
                "and --ulysses-degree must equal --sp-degree."
            )
        # Ulysses splits the heads that remain on each TP rank. The checkpoint
        # has 32 query heads and 8 KV heads, so the latter is the tight bound.
        if 32 % (tp_size * ulysses_degree) != 0 or 8 % (tp_size * ulysses_degree) != 0:
            raise ValueError(
                "SenseNova-U1.5-8B-MoT requires both 32 attention heads and 8 "
                "KV heads to be divisible by tp_size * ulysses_degree; got "
                f"{tp_size} * {ulysses_degree}."
            )
        if (
            bool(getattr(server_args, "enable_cfg_parallel", False))
            or int(getattr(server_args, "cfg_parallel_degree", 1) or 1) != 1
        ):
            raise ValueError(
                "SenseNova-U1.5-8B-MoT does not support CFG parallelism yet."
            )
        if num_gpus != dp_size * tp_size * sp_degree:
            raise ValueError(
                "SenseNova-U1.5-8B-MoT requires "
                "num_gpus == dp_size * tp_size * sp_degree with CFG disabled; "
                f"got {num_gpus} != {dp_size} * {tp_size} * {sp_degree}."
            )

    @staticmethod
    def validate_single_gpu_replica(server_args) -> None:
        """Backward-compatible entry point for the former DP-only validation."""
        SenseNovaU1PipelineConfig.validate_parallelism(server_args)

    def validate_server_args(self, server_args) -> None:
        self.validate_parallelism(server_args)
        if getattr(server_args, "use_fsdp_inference", False):
            raise ValueError("SenseNova-U1.5-8B-MoT does not support FSDP inference.")
        if getattr(server_args, "direct_gpu_weight_loading", False):
            raise ValueError(
                "SenseNova-U1.5-8B-MoT uses rank-local safetensors loading; "
                "--direct-gpu-weight-loading is unsupported."
            )
        if getattr(server_args, "enable_torch_compile", False):
            raise ValueError(
                "SenseNovaU1Pipeline does not support torch.compile yet. "
                "Please omit --enable-torch-compile."
            )
        if getattr(server_args, "lora_target_modules", None) is None:
            # The official adapter only targets the generation branch.
            server_args.lora_target_modules = ["_mot_gen"]
        _set_compatible_runtime_defaults(server_args)
        if _is_arg_explicitly_set(
            server_args, "component_residency"
        ) and _component_residency_requests_offload(
            getattr(server_args, "component_residency", None)
        ):
            raise ValueError(
                "SenseNovaU1Pipeline does not support component residency "
                "offload modes yet. Please omit --component-residency."
            )
        unsupported_runtime_options = {
            "cpu_offload_components": "CPU offload",
            "dit_cpu_offload": "DiT CPU offload",
            "text_encoder_cpu_offload": "text encoder CPU offload",
            "image_encoder_cpu_offload": "image encoder CPU offload",
            "vae_cpu_offload": "VAE CPU offload",
            "dit_layerwise_offload": "DiT layerwise offload",
            "layerwise_offload_components": "layerwise offload",
            "quantization": "quantization",
            "quantization_ignored_layers": "quantization ignored layers",
            "transformer_weights_path": "pre-quantized transformer weights",
            "component_paths": "component path overrides",
            "component_weights_paths": "component weight path overrides",
            "component_quantizations": "component quantization",
            "component_quantization_ignored_layers": (
                "component quantization ignored layers"
            ),
            "component_precisions": "component precision overrides",
        }
        for option, description in unsupported_runtime_options.items():
            if _is_arg_explicitly_set(
                server_args, option
            ) and _is_runtime_option_requested(getattr(server_args, option, None)):
                raise ValueError(
                    f"SenseNovaU1Pipeline does not support {description} yet. "
                    f"Please omit --{option.replace('_', '-')}."
                )
        if _is_arg_explicitly_set(
            server_args, "attention_backend"
        ) and _is_runtime_option_requested(
            getattr(server_args, "attention_backend", None)
        ):
            raise ValueError(
                "SenseNovaU1Pipeline does not support custom attention backends yet. "
                "Please omit --attention-backend."
            )
        if _is_arg_explicitly_set(
            server_args, "component_attention_backends"
        ) and _is_runtime_option_requested(
            getattr(server_args, "component_attention_backends", None)
        ):
            raise ValueError(
                "SenseNovaU1Pipeline does not support component attention backends yet. "
                "Please omit --component-attention-backends."
            )
        if _is_arg_explicitly_set(
            server_args, "attention_backend_config"
        ) and _is_runtime_option_requested(
            getattr(server_args, "attention_backend_config", None)
        ):
            raise ValueError(
                "SenseNovaU1Pipeline does not support attention backend config yet. "
                "Please omit --attention-backend-config."
            )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            speed_mode_enable_torch_compile_by_default=False,
            keep_resident_min_available_gb=80,
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )
